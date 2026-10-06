"""Nominal Cartesian force/motion controller for the 7-DoF Panda."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import ControllerConfig
from .ports import ContactPort, PandaHandles, site_jacobian


class Phase(enum.IntEnum):
    APPROACH = 0
    RAMP = 1
    SLIDE = 2


def orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Small-angle world-frame error taking current orientation to desired."""
    return 0.5 * sum((np.cross(current[:, i], desired[:, i]) for i in range(3)),
                     start=np.zeros(3))


@dataclass
class ControlOutput:
    tau_nominal: np.ndarray
    tau_bias: np.ndarray
    tau_translation: np.ndarray
    tau_orientation: np.ndarray
    tau_posture: np.ndarray
    f_command: np.ndarray
    moment_command: np.ndarray
    desired_position: np.ndarray
    desired_velocity: np.ndarray
    desired_force: float
    force_integral: float
    phase: Phase
    orientation_error: float
    original_progress: float
    governed_progress: float
    governor_scale: float


class PandaController:
    """APPROACH -> force RAMP -> SLIDE/wipe complete-torque controller."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 handles: PandaHandles, cfg: ControllerConfig):
        self.model, self.h, self.cfg = model, handles, cfg
        self.dt = float(model.opt.timestep)
        self.phase = Phase.APPROACH
        self.q_home = data.qpos[handles.qpos_ids].copy()
        self.p0 = data.site_xpos[handles.tool_site].copy()
        self.r_des = data.site_xmat[handles.tool_site].reshape(3, 3).copy()
        self.z_target = float(self.p0[2])
        self.hold_xy = self.p0[:2].copy()
        self.t_ramp = 0.0
        self.t_slide = 0.0
        self.t_path = 0.0
        self.t_path_original = 0.0
        self.motion_scale = 0.0
        self.force_integral = 0.0
        self.filtered_normal_force = 0.0
        self.governor_scale = 1.0
        self._release_clock = 0.0

    def _update_governor(self, human_present: bool, constrain: bool) -> None:
        g = self.cfg.governor
        if not g.enabled:
            self.governor_scale = 1.0
            return
        if human_present and constrain:
            self._release_clock = 0.0
            self.governor_scale = max(0.0, self.governor_scale - self.dt / g.decel_time)
        elif not human_present:
            self._release_clock += self.dt
            if self._release_clock >= g.release_dwell:
                self.governor_scale = min(1.0, self.governor_scale + self.dt / g.resume_time)

    def _path_reference(self):
        c = self.cfg
        if c.trajectory == "straight":
            s = min(c.straight_length, c.slide_speed * self.t_path)
            moving = s < c.straight_length - 1e-12
            p = self.hold_xy + np.array([s, 0.0])
            v = (np.array([c.slide_speed if moving else 0.0, 0.0])
                 * self.motion_scale * self.governor_scale)
            return p, v
        if c.trajectory != "wipe":
            raise ValueError(f"unknown trajectory {c.trajectory!r}")
        ax, ay = c.wipe_radii
        w = 2.0 * np.pi / c.wipe_period
        th = w * self.t_path
        p = self.hold_xy + np.array([ax * np.sin(th), ay * (1.0 - np.cos(th))])
        v = (np.array([ax * w * np.cos(th), ay * w * np.sin(th)])
             * self.motion_scale * self.governor_scale)
        return p, v

    def update(self, data: mujoco.MjData, contact: ContactPort,
               human_present: bool = False, protect_active: bool = False) -> ControlOutput:
        c, dt = self.cfg, self.dt
        jp, jr = site_jacobian(self.model, data, self.h.tool_site)
        p = data.site_xpos[self.h.tool_site].copy()
        v, omega = jp @ data.qvel, jr @ data.qvel
        r = data.site_xmat[self.h.tool_site].reshape(3, 3)
        e_rot = orientation_error(r, self.r_des)
        alpha = min(1.0, dt / c.force_filter_time)
        self.filtered_normal_force += alpha * (
            contact.normal_force - self.filtered_normal_force)

        if self.phase == Phase.APPROACH:
            self.z_target -= c.approach_speed * dt
            p_des = np.r_[self.hold_xy, self.z_target]
            v_des = np.array([0.0, 0.0, -c.approach_speed])
            f_cmd = c.kp_approach * (p_des - p) + c.kd_translation * (v_des - v)
            f_des = 0.0
            if contact.active or contact.gap <= c.contact_gap:
                self.phase = Phase.RAMP
                self.hold_xy = p[:2].copy()
                self.z_target = float(p[2])
        else:
            if self.phase == Phase.RAMP:
                self.t_ramp += dt
                s = min(1.0, self.t_ramp / c.ramp_duration)
                f_des = c.f_desired * 0.5 * (1.0 - np.cos(np.pi * s))
                if self.t_ramp >= c.ramp_duration + c.settle_duration:
                    self.phase = Phase.SLIDE
                    self.t_slide = 0.0
                    self.t_path = 0.0
                    self.t_path_original = 0.0
                    self.motion_scale = 0.0
                    self.hold_xy = p[:2].copy()
            else:
                f_des = c.f_desired
                self.t_slide += dt

            self._update_governor(human_present, protect_active)
            if self.phase == Phase.SLIDE:
                # Progress is explicit, making original versus governed task
                # progress auditable instead of silently rebasing references.
                ramp_s = (1.0 if c.slide_ramp_duration <= 0
                          else min(1.0, self.t_slide / c.slide_ramp_duration))
                self.motion_scale = 0.5 * (1.0 - np.cos(np.pi * ramp_s))
                self.t_path_original += dt * self.motion_scale
                self.t_path += dt * self.motion_scale * self.governor_scale
                xy_des, xy_vel = self._path_reference()
            else:
                xy_des, xy_vel = self.hold_xy.copy(), np.zeros(2)

            # MuJoCo's point contact can flicker for a few integration steps
            # at near-zero distance; the PI loop consumes a causal low-pass
            # task-force sample rather than amplifying that solver ripple.
            ef = f_des - self.filtered_normal_force
            candidate = np.clip(self.force_integral + ef * dt,
                                -c.integral_limit / c.ki_force,
                                c.integral_limit / c.ki_force)
            push_candidate = f_des + c.k_force * ef + c.ki_force * candidate
            # Conditional integration is the anti-windup policy.
            if 0.0 <= push_candidate <= 25.0:
                self.force_integral = float(candidate)
            push = np.clip(f_des + c.k_force * ef + c.ki_force * self.force_integral,
                           0.0, 25.0)
            f_xy = c.kp_tangent * (xy_des - p[:2]) + c.kd_tangent * (xy_vel - v[:2])
            f_cmd = np.array([f_xy[0], f_xy[1], -push - c.d_force * v[2]])
            p_des = np.r_[xy_des, self.z_target]
            v_des = np.r_[xy_vel, 0.0]

        m_cmd = c.kp_orientation * e_rot - c.kd_orientation * omega
        tau_trans = jp.T @ f_cmd
        tau_rot = jr.T @ m_cmd
        j6 = np.vstack((jp, jr))[:, self.h.dof_ids]
        null = np.eye(7) - j6.T @ np.linalg.pinv(j6.T)
        q = data.qpos[self.h.qpos_ids]
        qd = data.qvel[self.h.dof_ids]
        tau_posture = null @ (c.kp_posture * (self.q_home - q))
        tau_posture -= c.kd_joint * qd
        tau_bias = data.qfrc_bias[self.h.dof_ids].copy()
        tau = tau_bias + tau_trans[self.h.dof_ids] + tau_rot[self.h.dof_ids] + tau_posture
        if c.trajectory == "straight":
            original_progress = min(c.straight_length,
                                    c.slide_speed * self.t_path_original)
            governed_progress = min(c.straight_length, c.slide_speed * self.t_path)
        else:
            original_progress = c.slide_speed * self.t_path_original
            governed_progress = c.slide_speed * self.t_path
        return ControlOutput(
            tau, tau_bias, tau_trans[self.h.dof_ids], tau_rot[self.h.dof_ids],
            tau_posture, f_cmd, m_cmd, p_des, v_des, f_des,
            self.force_integral, self.phase, float(np.linalg.norm(e_rot)),
            original_progress, governed_progress,
            self.governor_scale)
