"""Three-phase nominal force/motion controller.

Phases:
  APPROACH  Cartesian PD: hold the initial tangential position, descend
            toward the surface; switch when contact is detected or the
            pad-surface gap falls below a threshold.
  RAMP      PI normal-force regulation against the measured MuJoCo contact
            force while the desired force ramps smoothly 0 -> F_d; the
            tangential position is held.
  SLIDE     Keep regulating F_d and track x_d(t) = x_start + v_d (t - t0)
            with tangential PD.

Force-control sign convention (z up): e_F = F_d - F_n. The nonnegative
downward pushing magnitude is
    F_push = F_d + k_F e_F + k_I ∫ e_F dt
and the commanded normal Cartesian force is
    F_z_cmd = -F_push - d_F v_z.
If the measured F_n is too small, e_F > 0 increases F_push, i.e. the robot
pushes DOWN harder — verified in the tests.

Joint torques: tau_task = J_p^T F_cmd, plus MuJoCo's bias forces
(data.qfrc_bias = Coriolis + gravity; it does NOT include the model's
passive joint damping, so compensating it never fights or double-counts the
MJCF damping), minus a small controller damping D_q qdot:
    tau_nom = tau_bias + tau_task - D_q qdot.
Torque magnitude and rate limits are applied last.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import C_PROJ, NORMAL, TANGENT, U_PROJ, ControllerConfig
from .contact_extraction import ContactResult, ModelHandles, site_jacobian


class Phase(enum.IntEnum):
    APPROACH = 0
    RAMP = 1
    SLIDE = 2


@dataclass
class ControlOutput:
    """Everything the controller computed in one step (for logging)."""

    tau: np.ndarray                     # final commanded joint torques (2,)
    tau_task: np.ndarray                # J_p^T F_cmd
    tau_bias: np.ndarray                # gravity + Coriolis compensation
    tau_damping: np.ndarray             # -D_q qdot (controller damping)
    tau_c: np.ndarray                   # J_p^T C F_cmd (normal-control torques)
    tau_u: np.ndarray                   # J_p^T U F_cmd (tangential-control torques)
    f_cmd: np.ndarray                   # commanded Cartesian force (3,)
    phase: Phase = Phase.APPROACH
    f_desired: float = 0.0              # current ramped desired normal force
    f_push: float = 0.0
    e_f: float = 0.0
    integral: float = 0.0
    x_desired: float = 0.0
    v_desired: float = 0.0
    saturated: bool = False


class SlidingForceController:
    """Stateful three-phase controller. Call ``update`` once per step."""

    def __init__(self, model: mujoco.MjModel, handles: ModelHandles, cfg: ControllerConfig):
        self.model = model
        self.handles = handles
        self.cfg = cfg
        self.dt = model.opt.timestep

        self.phase = Phase.APPROACH
        self.integral = 0.0             # ∫ e_F dt
        self._tau_prev: np.ndarray | None = None
        self._x_hold: float | None = None
        self._z_target: float | None = None
        self._t_ramp_start: float | None = None
        self._t_slide_start: float | None = None
        self._x_slide_start: float | None = None

    # ------------------------------------------------------------------
    def update(
        self, data: mujoco.MjData, contact: ContactResult
    ) -> ControlOutput:
        cfg = self.cfg
        t = data.time
        p_ee = data.site_xpos[self.handles.ee_site].copy()
        jp = site_jacobian(self.model, data, self.handles.ee_site)
        v_ee = jp @ data.qvel
        x, z = p_ee[0], p_ee[2]
        vx, vz = v_ee[0], v_ee[2]

        if self._x_hold is None:
            self._x_hold = x
            self._z_target = z

        # ---------------- phase transitions ----------------
        if self.phase == Phase.APPROACH:
            if contact.active or contact.dist < cfg.contact_gap_threshold:
                self.phase = Phase.RAMP
                self._t_ramp_start = t
                self._x_hold = x
        if self.phase == Phase.RAMP:
            if t - self._t_ramp_start >= cfg.ramp_duration + cfg.settle_duration:
                self.phase = Phase.SLIDE
                self._t_slide_start = t
                self._x_slide_start = self._x_hold

        # ---------------- Cartesian force command ----------------
        f_desired = 0.0
        f_push = 0.0
        e_f = 0.0
        x_des = self._x_hold
        v_des = 0.0

        if self.phase == Phase.APPROACH:
            # Descend at constant rate; PD in both directions.
            self._z_target = max(self._z_target - cfg.approach_speed * self.dt, -0.02)
            fx = cfg.kp_cart * (self._x_hold - x) - cfg.kd_cart * vx
            fz = cfg.kp_cart * (self._z_target - z) - cfg.kd_cart * vz
        else:
            # Smooth cosine ramp of the desired normal force.
            s = np.clip((t - self._t_ramp_start) / cfg.ramp_duration, 0.0, 1.0)
            f_desired = cfg.f_desired * 0.5 * (1.0 - np.cos(np.pi * s))

            # PI force regulation on the measured MuJoCo normal force.
            e_f = f_desired - contact.f_n
            f_push = f_desired + cfg.k_f * e_f + cfg.k_i * self.integral
            f_push = max(f_push, 0.0)
            fz = -f_push - cfg.d_f * vz

            if self.phase == Phase.SLIDE:
                x_des = self._x_slide_start + cfg.v_slide * (t - self._t_slide_start)
                v_des = cfg.v_slide
                fx = cfg.kx_slide * (x_des - x) + cfg.dx_slide * (v_des - vx)
            else:
                fx = cfg.kp_cart * (self._x_hold - x) - cfg.kd_cart * vx

        f_cmd = TANGENT * fx + NORMAL * fz

        # ---------------- map to joint torques ----------------
        tau_task = jp.T @ f_cmd
        tau_c = jp.T @ (C_PROJ @ f_cmd)
        tau_u = jp.T @ (U_PROJ @ f_cmd)
        tau_bias = data.qfrc_bias.copy()
        tau_damping = -cfg.dq_joint * data.qvel.copy()
        tau = tau_bias + tau_task + tau_damping

        # ---------------- limits ----------------
        tau_unsat = tau.copy()
        tau = np.clip(tau, -cfg.tau_limit, cfg.tau_limit)
        if self._tau_prev is not None:
            dmax = cfg.tau_rate_limit * self.dt
            tau = np.clip(tau, self._tau_prev - dmax, self._tau_prev + dmax)
        saturated = bool(np.any(np.abs(tau - tau_unsat) > 1e-9))
        self._tau_prev = tau.copy()

        # ---------------- anti-windup integration ----------------
        # Integrate after computing the command; skip integration when the
        # torque command is saturated and the error would push it further
        # (conditional integration), and clamp the stored integral so the
        # integral term contribution stays within +/- integral_limit.
        if self.phase != Phase.APPROACH:
            if not (saturated and e_f * f_push > 0):
                self.integral += e_f * self.dt
            if cfg.k_i > 0:
                bound = cfg.integral_limit / cfg.k_i
                self.integral = float(np.clip(self.integral, -bound, bound))

        return ControlOutput(
            tau=tau,
            tau_task=tau_task,
            tau_bias=tau_bias,
            tau_damping=tau_damping,
            tau_c=tau_c,
            tau_u=tau_u,
            f_cmd=f_cmd,
            phase=self.phase,
            f_desired=f_desired,
            f_push=f_push,
            e_f=e_f,
            integral=self.integral,
            x_desired=x_des,
            v_desired=v_des,
            saturated=saturated,
        )
