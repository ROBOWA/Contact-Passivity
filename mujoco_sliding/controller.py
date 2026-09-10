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

from .config import (C_PROJ, GovernorConfig, NORMAL, TANGENT, U_PROJ,
                     ControllerConfig)
from .contact_extraction import ContactResult, ModelHandles, site_jacobian


class Phase(enum.IntEnum):
    APPROACH = 0
    RAMP = 1
    SLIDE = 2


class GovernorMode(enum.IntEnum):
    TRACK = 0
    INTERACT = 1          # continuous_rebase: engaged (rebase onto x(t))
    RESUME = 2            # both variants: cosine ramp back to v_d
    DECEL = 3             # stop_time_anchor: finite cosine deceleration
    HOLD = 4              # stop_time_anchor: fixed captured anchor
    RELEASE_DWELL = 5     # stop_time_anchor: post-release settle at anchor


#: Modes in which the governor actively shapes the reference away from
#: nominal tracking (used by metrics/plots to mark "governor engaged").
GOV_ENGAGED_MODES = (GovernorMode.INTERACT, GovernorMode.DECEL,
                     GovernorMode.HOLD, GovernorMode.RELEASE_DWELL)


class ReferenceGovernor:
    """Stateful, CBF-triggered tangential reference governor (x_g, v_g).

    Normal operation (TRACK): v_g -> v_d, x_g advances with v_g. The
    governor does NOT freeze merely because a human is detected: during the
    initial interaction it stays in TRACK so the passivity QP regulates the
    instantaneous human power (ROW_P_H) and the human ledger E_H visibly
    depletes. It latches into INTERACT only when the human-energy CBF row
    (ROW_CBF_H) becomes limiting AND the human is present, i.e. when
    cumulative human-energy extraction — not mere contact — starts to bind.
    The CBF-active flag is taken from the PREVIOUS QP step (see the runtime),
    so no algebraic loop is introduced.

    In INTERACT, v_g decays smoothly to zero and x_g continuously rebases
    onto the end-effector (first-order pull, never a jump); the mode stays
    LATCHED while the human is present even if ROW_CBF_H later goes inactive
    because the robot has stopped. After the human releases (plus a dwell),
    v_g cosine-ramps back to v_d from the already-rebased position, so both
    x_g and v_g stay continuous and the controller never reconnects to the
    original time-indexed reference.

    This is a task/recovery policy layered on the NOMINAL controller. It
    does not enter the passivity certificate: physical port powers keep
    using the physical end-effector velocity.
    """

    def __init__(self, cfg: GovernorConfig, v_d: float, dt: float):
        self.cfg = cfg
        self.v_d = v_d
        self.dt = dt
        self.mode = GovernorMode.RESUME
        self.x_g = 0.0
        self.v_g = 0.0
        self._ramp_t = 0.0
        self._clear_t = 0.0
        self._resume_v0 = 0.0
        self._decel_t = 0.0
        self._v_g0 = 0.0
        self.n_freeze_events = 0
        self.n_resume_events = 0
        self.trigger_time = None        # first engagement (CBF trigger)
        # stop_time_anchor bookkeeping
        self.x_a = None                 # captured fixed interaction anchor
        self.x_stop = None              # x(t) at the anchor-capture instant
        self.anchor_time = None
        self.decel_end_time = None

    def reset(self, x0: float) -> None:
        """Start governing at x0 with a gentle ramp up to v_d."""
        self.x_g = float(x0)
        self.v_g = 0.0
        self.mode = GovernorMode.RESUME
        self._ramp_t = 0.0
        self._clear_t = 0.0
        self._resume_v0 = 0.0
        self._decel_t = 0.0
        self.x_a = None
        self.x_stop = None

    def step(self, x_ee: float, human_present: bool,
             cbf_trigger: bool, t: float = 0.0) -> tuple[float, float]:
        """Advance (x_g, v_g) one control step.

        human_present : |F_H| > f_detect (perfect sensing).
        cbf_trigger   : ROW_CBF_H active on the PREVIOUS QP step.
        Engagement requires human_present AND cbf_trigger in both variants.
        """
        if self.cfg.mode == "stop_time_anchor":
            return self._step_stop_time_anchor(
                x_ee, human_present, cbf_trigger, t)
        return self._step_continuous_rebase(
            x_ee, human_present, cbf_trigger, t)

    # ------------------------------------------------------------------
    def _step_stop_time_anchor(self, x_ee: float, human_present: bool,
                               cbf_trigger: bool, t: float):
        """TRACK -> DECEL -> HOLD -> RELEASE_DWELL -> RESUME -> TRACK.

        DECEL (s = clip((t - t_g)/T_decel, 0, 1), beta = 3s^2 - 2s^3):
            v_g   = v_g0/2 (1 + cos(pi s))            -> exactly 0 at s = 1
            x_g+  = x_g + dt [v_g + beta k_rebase (x - x_g)]
        At s = 1 the governed position is captured ONCE as the fixed anchor
        x_a = x_g(t_g + T_decel) (continuous by construction, since the
        anchor IS the current governed position).
        HOLD / RELEASE_DWELL:  v_g = 0, x_g = x_a  (fixed; never re-based
            onto the retreating robot, so K_p (x_a - x) keeps position
            stiffness and the interaction settles at bounded displacement).
        RESUME (s = clip(t_r/T_resume, 0, 1)):
            v_g = v_d/2 (1 - cos(pi s)),  x_g+ = x_g + dt v_g
        """
        cfg = self.cfg
        dt = self.dt

        # ---------------- transitions ----------------
        m = self.mode
        if m in (GovernorMode.TRACK, GovernorMode.RESUME):
            if human_present and cbf_trigger:
                self._begin_decel(t)
        elif m == GovernorMode.DECEL:
            if not human_present:
                # Human vanished mid-deceleration: keep x_g and v_g exactly
                # as they are and fall through to the release sequence.
                self.mode = GovernorMode.RELEASE_DWELL
                self._clear_t = 0.0
        elif m == GovernorMode.HOLD:
            if not human_present:
                self.mode = GovernorMode.RELEASE_DWELL
                self._clear_t = 0.0
        elif m == GovernorMode.RELEASE_DWELL:
            if human_present:
                # Same interaction resumed during the dwell: keep the anchor.
                self.mode = (GovernorMode.HOLD if self.x_a is not None
                             else GovernorMode.DECEL)
            else:
                self._clear_t += dt
                if self._clear_t >= cfg.clear_dwell:
                    self.mode = GovernorMode.RESUME
                    self._ramp_t = 0.0
                    self._resume_v0 = self.v_g
                    self.n_resume_events += 1

        # ---------------- dynamics ----------------
        m = self.mode
        if m == GovernorMode.DECEL:
            self._decel_t += dt
            s = min(1.0, self._decel_t / cfg.t_decel)
            self.v_g = 0.5 * self._v_g0 * (1.0 + np.cos(np.pi * s))
            beta = s * s * (3.0 - 2.0 * s)
            self.x_g += (self.v_g
                         + beta * cfg.k_rebase * (x_ee - self.x_g)) * dt
            if s >= 1.0:
                # Scheduled stop: v_g is exactly zero here; capture the
                # governed position as the fixed anchor (no jump: x_a = x_g).
                self.v_g = 0.0
                self.x_a = self.x_g
                self.x_stop = float(x_ee)
                self.anchor_time = t
                self.decel_end_time = t
                self.mode = GovernorMode.HOLD
        elif m in (GovernorMode.HOLD, GovernorMode.RELEASE_DWELL):
            self.v_g = 0.0
            if self.x_a is not None:
                self.x_g = self.x_a
        elif m == GovernorMode.RESUME:
            self.x_g += self.v_g * dt
            self._ramp_t += dt
            s = min(1.0, self._ramp_t / cfg.t_resume)
            self.v_g = 0.5 * self.v_d * (1.0 - np.cos(np.pi * s))
            if s >= 1.0:
                self.mode = GovernorMode.TRACK
        else:  # TRACK
            self.v_g = self.v_d
            self.x_g += self.v_g * dt
        return self.x_g, self.v_g

    def _begin_decel(self, t: float) -> None:
        """Start a new interaction: record t_g and v_g0 only (no anchor yet,
        no reset of x_g or v_g)."""
        self.mode = GovernorMode.DECEL
        self._decel_t = 0.0
        self._v_g0 = self.v_g
        self.x_a = None            # captured only when DECEL completes
        self.x_stop = None
        self.n_freeze_events += 1
        if self.trigger_time is None:
            self.trigger_time = t

    # ------------------------------------------------------------------
    def _step_continuous_rebase(self, x_ee: float, human_present: bool,
                                cbf_trigger: bool, t: float):
        cfg = self.cfg
        dt = self.dt

        if self.mode == GovernorMode.INTERACT:
            # Latched: hold INTERACT while the human is present, regardless
            # of the current CBF state (the robot may have stopped, making
            # the CBF row momentarily slack). Release only after the human
            # is gone for the clear dwell.
            if human_present:
                self._clear_t = 0.0
            else:
                self._clear_t += dt
                if self._clear_t >= cfg.clear_dwell:
                    self.mode = GovernorMode.RESUME
                    self._ramp_t = 0.0
                    self._resume_v0 = self.v_g
                    self.n_resume_events += 1
        else:
            # TRACK or RESUME: enter INTERACT only when cumulative
            # human-energy extraction becomes limiting (CBF) while the human
            # is actually present.
            if human_present and cbf_trigger:
                self.mode = GovernorMode.INTERACT
                self._clear_t = 0.0
                self.n_freeze_events += 1
                if self.trigger_time is None:
                    self.trigger_time = t

        if self.mode == GovernorMode.INTERACT:
            # Explicit Euler with the pre-update governed velocity, matching
            # x_g+ = x_g + dt v_g + dt k_rebase (x - x_g).
            v_prev = self.v_g
            self.x_g += (v_prev + cfg.k_rebase * (x_ee - self.x_g)) * dt
            self.v_g += (0.0 - self.v_g) * min(1.0, dt / cfg.v_decay_tau)
        elif self.mode == GovernorMode.RESUME:
            # Integrate forward from the rebased state; never reconnect to
            # the original time-indexed reference.
            self.x_g += self.v_g * dt
            self._ramp_t += dt
            s = min(1.0, self._ramp_t / cfg.t_resume)
            blend = 0.5 * (1.0 - np.cos(np.pi * s))
            self.v_g = self._resume_v0 + (self.v_d - self._resume_v0) * blend
            if s >= 1.0:
                self.mode = GovernorMode.TRACK
        else:  # TRACK
            self.v_g = self.v_d
            self.x_g += self.v_g * dt
        return self.x_g, self.v_g


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
    # Reference-governor bookkeeping (x_desired/v_desired are the GOVERNED
    # reference when the governor is enabled; x_ref_original is the original
    # time-indexed reference for comparison).
    x_ref_original: float = 0.0
    governor_active: bool = False
    governor_mode: int = 0
    # stop_time_anchor: captured fixed anchor (NaN before capture).
    governor_anchor: float = float("nan")


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
        self.governor = (ReferenceGovernor(cfg.governor, cfg.v_slide, self.dt)
                        if cfg.governor.enabled else None)

    # ------------------------------------------------------------------
    def update(
        self, data: mujoco.MjData, contact: ContactResult,
        integrate: bool = True,
        f_h_norm: float = 0.0,
        cbf_h_active_prev: bool = False,
    ) -> ControlOutput:
        """Compute the nominal command.

        With ``integrate=True`` (default) the PI force-error integration and
        anti-windup run inside this call, exactly as before. An external
        command filter (the passivation QP) passes ``integrate=False`` and
        calls :meth:`finish_step` after deciding the applied command, so the
        integrator can be frozen when the filter modified the normal command
        (anti-windup by integrator freezing).

        ``f_h_norm`` is the perfectly sensed human-force magnitude and
        ``cbf_h_active_prev`` is the human-energy CBF row's active status on
        the PREVIOUS QP step; together they drive the optional reference
        governor (latched INTERACT on human present AND CBF active). Both are
        ignored when the governor is disabled.

        ``out.f_cmd`` is the UNLIMITED nominal Cartesian command; the nominal
        u is u_nom = B_tn^T f_cmd, and tau(u_nom) = tau0 + J^T f_cmd equals
        ``tau_bias + tau_task + tau_damping`` below (before limits).
        """
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
                if self.governor is not None:
                    self.governor.reset(self._x_hold)

        # ---------------- Cartesian force command ----------------
        f_desired = 0.0
        f_push = 0.0
        e_f = 0.0
        x_des = self._x_hold
        v_des = 0.0
        x_ref_orig = self._x_hold

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
                x_ref_orig = (self._x_slide_start
                              + cfg.v_slide * (t - self._t_slide_start))
                if self.governor is not None:
                    human_present = f_h_norm > cfg.governor.f_detect
                    x_des, v_des = self.governor.step(
                        x, human_present, cbf_h_active_prev, t=t)
                else:
                    x_des, v_des = x_ref_orig, cfg.v_slide
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
        if integrate:
            self._integrate(e_f, f_push, saturated)

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
            x_ref_original=x_ref_orig,
            governor_active=(self.governor is not None
                            and self.phase == Phase.SLIDE),
            governor_mode=(int(self.governor.mode)
                          if self.governor is not None else 0),
            governor_anchor=(
                float(self.governor.x_a)
                if self.governor is not None and self.governor.x_a is not None
                else float("nan")),
        )

    # ------------------------------------------------------------------
    def _integrate(self, e_f: float, f_push: float, saturated: bool) -> None:
        """PI integration with anti-windup.

        Skips integration when the torque command is saturated and the error
        would push it further (conditional integration), and clamps the
        stored integral so the integral-term contribution stays within
        +/- integral_limit.
        """
        cfg = self.cfg
        if self.phase != Phase.APPROACH:
            if not (saturated and e_f * f_push > 0):
                self.integral += e_f * self.dt
            if cfg.k_i > 0:
                bound = cfg.integral_limit / cfg.k_i
                self.integral = float(np.clip(self.integral, -bound, bound))

    def finish_step(self, out: ControlOutput, frozen: bool,
                    tau_applied: np.ndarray | None = None) -> None:
        """Complete a step started with ``update(..., integrate=False)``.

        ``frozen=True`` freezes the force integrator for this step (used
        when an external filter modified the normal command — anti-windup by
        integrator freezing). ``tau_applied`` records the torque actually
        sent to the plant so the internal rate-limit reference tracks it.
        """
        if not frozen:
            self._integrate(out.e_f, out.f_push, out.saturated)
        if tau_applied is not None:
            self._tau_prev = np.asarray(tau_applied, dtype=float).copy()
