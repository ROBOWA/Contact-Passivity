"""Approach the table along its normal and hold a constant contact force.

The controller works in operational space at the `tcp` site (tool tip centre):

* tangential translation (table plane) and orientation: inertia-weighted PD
  with unit-mass gains,  F = Lambda * (Kp e - Kd v),  Lambda = (J M^-1 J^T)^-1;
* normal direction, phase APPROACH: the z set-point descends at a constant
  speed and the same PD tracks it;
* normal direction, phase CONTACT (entered when the sensor-based normal force
  exceeds a threshold): force command along -n,
      f_cmd = f_d(t) + kp_f (f_d - f_n) + ki_f * integral(f_d - f_n)  -  d_n * v_n,
  with f_d ramped linearly to the target after contact and an anti-windup
  clamp on the integral term;
* null space: posture spring/damper toward q_home, projected with the
  dynamically consistent projector N = I - J^T Jbar^T;
* gravity and Coriolis compensation through qfrc_bias, applied once.

The measured normal force comes from the FT sensor only. MuJoCo's force sensor
at `ft_site` reports the wrench the parent body applies to the `ft_sensor`
subtree (sensor housing + shaft + tip, 0.50 kg), expressed in the site frame.
With gravity g and quasi-static motion,

    S_world = -F_ext + m_tool * (-g)      =>    F_ext = -R S - m_tool * g

where F_ext is the environment force on the tool and g = model.opt.gravity. The
sensor value available to the controller is the one computed at the end of the
previous physics step (one timestep of latency).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .holding import full_mass_matrix
from .scene import ArmIndex, TCP_SITE, TOOL_TIP_GEOM

FT_FORCE_SENSOR = "ft_force"
FT_SITE = "ft_site"
FT_BODY = "ft_sensor"
TABLETOP_GEOM = "tabletop"

APPROACH = "approach"
CONTACT = "contact"


@dataclass(frozen=True)
class ForceControlConfig:
    force_desired: float = 10.0       # N, normal force to hold (positive = pressing into the table)
    approach_speed: float = 0.05      # m/s, descent speed of the normal set-point before contact
    contact_threshold: float = 1.0    # N, sensor-based normal force that ends the approach
    contact_samples: int = 3          # consecutive control updates above the threshold required
    release_threshold: float = 0.2    # N, below this for release_time the controller re-approaches
    release_time: float = 0.3         # s
    force_ramp_time: float = 1.0      # s, linear ramp of the desired force after contact
    kp_force: float = 0.5             # proportional gain on the force error (dimensionless)
    ki_force: float = 5.0             # integral gain (1/s)
    integral_limit: float = 10.0      # N, clamp on the integral contribution ki * integral
    normal_damping: float = 40.0      # N*s/m, damping along the normal while in contact
    motion_freq_hz: float = 2.0       # unit-mass impedance natural frequency (translation and rotation)
    motion_zeta: float = 1.0
    posture_freq_hz: float = 1.0      # null-space posture spring natural frequency
    posture_zeta: float = 1.0
    lambda_reg: float = 1e-3          # regularisation added to J M^-1 J^T before inversion
    max_penetration: float = 0.01     # m, the approach set-point never goes deeper than this below the surface


class ForceApproachController:
    def __init__(self, model: mujoco.MjModel, arm: ArmIndex, control_period: float,
                 config: ForceControlConfig | None = None) -> None:
        self.model = model
        self.arm = arm
        self.dt = control_period
        self.cfg = config or ForceControlConfig()
        self.torque_limit = arm.torque_limit
        self.tcp = model.site(TCP_SITE).id
        self.ft_site = model.site(FT_SITE).id
        sensor = model.sensor(FT_FORCE_SENSOR)
        if model.sensor_type[sensor.id] != mujoco.mjtSensor.mjSENS_FORCE or model.sensor_dim[sensor.id] != 3:
            raise ValueError(f"{FT_FORCE_SENSOR} is not a 3-axis force sensor")
        self.ft_adr = int(model.sensor_adr[sensor.id])
        self.tool_mass = float(model.body_subtreemass[model.body(FT_BODY).id])
        self.gravity = model.opt.gravity.copy()
        self.tip_radius = float(model.geom_size[model.geom(TOOL_TIP_GEOM).id][0])
        self.normal = np.array([0.0, 0.0, 1.0])  # tabletop normal, pointing toward the tool
        wn = 2.0 * np.pi * self.cfg.motion_freq_hz
        self.kp_m, self.kd_m = wn ** 2, 2.0 * self.cfg.motion_zeta * wn
        wq = 2.0 * np.pi * self.cfg.posture_freq_hz
        self.kp_q, self.kd_q = wq ** 2, 2.0 * self.cfg.posture_zeta * wq
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self.saturated = np.zeros(7, dtype=bool)
        # state (set in reset)
        self.q_home = np.zeros(7)
        self.x_ref = np.zeros(3)
        self.quat_d = np.array([1.0, 0, 0, 0])
        self.z_d = 0.0
        self.z_floor = 0.0
        self.phase = APPROACH
        self.t_contact: float | None = None
        self.t_last_contact = 0.0
        self.f_int = 0.0
        self.f_n = 0.0
        self.f_d = 0.0
        self.f_cmd = 0.0
        self.n_releases = 0
        self.contact_count = 0

    # ------------------------------------------------------------------ helpers
    def contact_force_world(self, data: mujoco.MjData) -> np.ndarray:
        """Environment force on the tool from the FT sensor, bias-compensated for the tool weight."""
        s = data.sensordata[self.ft_adr:self.ft_adr + 3]
        rot = data.site_xmat[self.ft_site].reshape(3, 3)
        return -(rot @ s) - self.tool_mass * self.gravity

    def normal_force(self, data: mujoco.MjData) -> float:
        """Sensor-based normal contact force, positive when pressing into the table."""
        return float(self.contact_force_world(data) @ self.normal)

    def orientation_error(self, rot: np.ndarray) -> np.ndarray:
        """Rotation vector (world frame) taking the current tool orientation to the desired one."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.reshape(9))
        quat_conj = np.zeros(4)
        mujoco.mju_negQuat(quat_conj, quat)
        quat_err = np.zeros(4)
        mujoco.mju_mulQuat(quat_err, self.quat_d, quat_conj)
        err = np.zeros(3)
        mujoco.mju_quat2Vel(err, quat_err, 1.0)
        return err

    def status(self) -> str:
        return (f"{self.phase:8s} f_n={self.f_n:6.2f} N  f_d={self.f_d:5.2f} N  "
                f"f_cmd={self.f_cmd:6.2f} N  z_d={self.z_d:.4f} m")

    # ------------------------------------------------------------------ interface
    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, q_home: np.ndarray) -> None:
        """Called after mj_forward on the home keyframe."""
        self.q_home = q_home.copy()
        self.x_ref = data.site_xpos[self.tcp].copy()
        self.z_d = float(self.x_ref[2])
        mujoco.mju_mat2Quat(self.quat_d, data.site_xmat[self.tcp])
        tabletop = model.geom(TABLETOP_GEOM).id
        top_z = float(data.geom_xpos[tabletop][2] + model.geom_size[tabletop][2])
        self.z_floor = top_z + self.tip_radius - self.cfg.max_penetration
        self.phase = APPROACH
        self.t_contact = None
        self.t_last_contact = 0.0
        self.f_int = 0.0
        self.f_n = self.f_d = self.f_cmd = 0.0
        self.n_releases = 0
        self.contact_count = 0
        self.saturated[:] = False

    def compute(self, data: mujoco.MjData) -> np.ndarray:
        m, arm, cfg = self.model, self.arm, self.cfg
        t = float(data.time)
        q = data.qpos[arm.qpos_adr]
        qd = data.qvel[arm.dof_adr]
        bias = data.qfrc_bias[arm.dof_adr]
        n = self.normal

        # --- measurement and phase logic -------------------------------------------------
        self.f_n = f_n = self.normal_force(data)
        x = data.site_xpos[self.tcp]
        if self.phase == APPROACH:
            self.z_d = max(self.z_d - cfg.approach_speed * self.dt, self.z_floor)
            self.contact_count = self.contact_count + 1 if f_n > cfg.contact_threshold else 0
            if self.contact_count >= cfg.contact_samples:
                self.phase = CONTACT
                self.t_contact = t
                self.t_last_contact = t
                self.f_int = 0.0
        else:
            if f_n > cfg.release_threshold:
                self.t_last_contact = t
            elif t - self.t_last_contact > cfg.release_time:
                self.phase = APPROACH          # contact lost: re-approach from the current height
                self.z_d = float(x[2])
                self.n_releases += 1

        # --- kinematics and dynamics at the current state -----------------------------------
        mujoco.mj_jacSite(m, data, self._jacp, self._jacr, self.tcp)
        jac = np.vstack((self._jacp, self._jacr))[:, arm.dof_adr]      # 6 x 7
        rot = data.site_xmat[self.tcp].reshape(3, 3)
        v = jac[:3] @ qd
        w = jac[3:] @ qd
        mass = full_mass_matrix(m, data)[np.ix_(arm.dof_adr, arm.dof_adr)]
        mass_inv = np.linalg.inv(mass)
        lambda_inv = jac @ mass_inv @ jac.T + cfg.lambda_reg * np.eye(6)
        lam = np.linalg.inv(lambda_inv)

        # --- operational-space PD (unit-mass gains, inertia-weighted) -------------------------
        x_d = np.array([self.x_ref[0], self.x_ref[1], self.z_d])
        a_p = self.kp_m * (x_d - x) - self.kd_m * v
        a_o = self.kp_m * self.orientation_error(rot) - self.kd_m * w
        if self.phase == CONTACT:
            a_p = a_p - (a_p @ n) * n                 # the normal direction is force-controlled
        force = lam @ np.concatenate((a_p, a_o))

        # --- normal force control ---------------------------------------------------------------
        if self.phase == CONTACT:
            ramp = min(1.0, (t - self.t_contact) / cfg.force_ramp_time) if cfg.force_ramp_time > 0 else 1.0
            self.f_d = cfg.force_desired * ramp
            err = self.f_d - f_n
            limit = cfg.integral_limit / cfg.ki_force if cfg.ki_force > 0 else 0.0
            self.f_int = float(np.clip(self.f_int + err * self.dt, -limit, limit))
            self.f_cmd = self.f_d + cfg.kp_force * err + cfg.ki_force * self.f_int
            f_p = force[:3] - (force[:3] @ n) * n
            force[:3] = f_p - self.f_cmd * n - cfg.normal_damping * (v @ n) * n
        else:
            self.f_d = self.f_cmd = 0.0

        # --- null-space posture and final torque -------------------------------------------------
        tau_task = jac.T @ force
        jbar = mass_inv @ jac.T @ lam                       # dynamically consistent generalised inverse
        null_t = np.eye(7) - jac.T @ jbar.T                 # torque-space projector: J M^-1 null_t = 0
        tau_posture = mass @ (-self.kp_q * (q - self.q_home) - self.kd_q * qd)
        tau = bias + tau_task + null_t @ tau_posture
        self.saturated = np.abs(tau) >= self.torque_limit
        return np.clip(tau, -self.torque_limit, self.torque_limit)
