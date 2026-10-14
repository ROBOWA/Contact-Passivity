"""Approach the table along its normal and hold a constant contact force.

The controller works in operational space at the `tcp` site (tool tip centre):

* tangential translation (table plane) and orientation: inertia-weighted PD
  with unit-mass gains,  F = Lambda * (Kp e - Kd v),  Lambda = (J M^-1 J^T)^-1;
* normal direction, phase APPROACH: the set-point follows an acceleration-
  continuous profile (ApproachProfile): a quintic (minimum-jerk) segment from
  rest down to a creep point just above the expected surface, arriving there
  at the creep speed with zero acceleration, then constant creep speed until
  the sensor detects contact; the PD tracks it with velocity and acceleration
  feedforward;
* normal direction, phase CONTACT (entered when the sensor-based normal force
  exceeds a threshold): force command along -n,
      f_cmd = f_d(t) + kp_f (f_d - f_n) + ki_f * integral(f_d - f_n)  -  d_n * v_n,
  with f_d ramped linearly to the target after contact and an anti-windup
  clamp on the integral term;
* phase SLIDE (optional, after the force ramp): the tangential set-point
  moves back and forth along a fixed in-plane direction over a fixed distance
  with rest-to-rest quintic strokes (SlideProfile), so position, velocity and
  acceleration of the reference are continuous; the normal force loop keeps
  running, integral terms on the in-plane position and the orientation absorb
  the friction force and moment, and a nominal friction feedforward (viscous
  c_hat * v_d and/or Coulomb mu_hat * f_n * sign(v_d)) removes most of the lag;
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
from .ik import orientation_error
from .scene import ArmIndex, TCP_SITE, TOOL_TIP_GEOM

FT_FORCE_SENSOR = "ft_force"
FT_SITE = "ft_site"
FT_BODY = "ft_sensor"
TABLETOP_GEOM = "tabletop"

APPROACH = "approach"
CONTACT = "contact"
SLIDE = "slide"


@dataclass(frozen=True)
class ForceControlConfig:
    force_desired: float = 10.0       # N, normal force to hold (positive = pressing into the table)
    approach_speed: float = 0.05      # m/s, peak descent speed of the smooth approach segment
    creep_speed: float = 0.005        # m/s, constant speed of the final approach until contact
    creep_distance: float = 0.01      # m, height above the expected surface where the creep begins
    contact_threshold: float = 1.0    # N, sensor-based normal force that ends the approach
    contact_samples: int = 3          # consecutive control updates above the threshold required
    release_threshold: float = 0.2    # N, below this for release_time the controller re-approaches
    release_time: float = 0.3         # s
    force_ramp_time: float = 1.0      # s, linear ramp of the desired force after contact
    kp_force: float = 0.5             # proportional gain on the force error (dimensionless)
    ki_force: float = 5.0             # integral gain (1/s)
    integral_limit: float = 10.0      # N, clamp on the integral contribution ki * integral
    normal_damping: float = 40.0      # N*s/m, damping along the normal while in contact
    motion_freq_hz: float = 4.0       # unit-mass impedance natural frequency, translation
    motion_zeta: float = 1.0
    rotation_freq_hz: float = 6.0     # unit-mass impedance natural frequency, orientation
    ki_tangential: float = 300.0      # 1/s^3, unit-mass integral gain on the in-plane position error while in contact
    tangential_integral_limit: float = 5.0   # m/s^2, clamp on the unit-mass integral contribution
    ki_rotation: float = 300.0        # 1/s^3, unit-inertia integral gain on the orientation error while in contact
    rotation_integral_limit: float = 20.0    # rad/s^2, clamp on the unit-inertia integral contribution
    friction_ff: float = 0.0          # nominal Coulomb coefficient mu_hat for the tangential feedforward (mu_hat * f_n * sign)
    friction_ff_speed: float = 0.005  # m/s, width of the tanh(v_d / v_eps) sign blend of the Coulomb feedforward
    friction_viscous_ff: float = 80.0 # N*s/m, nominal viscous coefficient c_hat for the tangential feedforward (c_hat * v_d);
                                      # MuJoCo's soft friction rows act viscously below the cone limit (see the scene file)
    slide_distance: float = 0.10      # m, stroke length; 0 disables sliding
    slide_direction: tuple = (1.0, 0.0, 0.0)  # world direction, projected onto the table plane
    slide_speed: float = 0.05         # m/s, peak speed of each stroke
    slide_cycles: int = 0             # number of out-and-back cycles, 0 = repeat until the run ends
    slide_dwell: float = 0.0          # s, pause at each end of a stroke
    slide_start_delay: float = 1.0    # s, after the force ramp completes
    posture_freq_hz: float = 1.0      # null-space posture spring natural frequency
    posture_zeta: float = 1.0
    lambda_reg: float = 1e-3          # regularisation added to J M^-1 J^T before inversion
    max_penetration: float = 0.01     # m, the approach set-point never goes deeper than this below the surface


class ApproachProfile:
    """1-D set-point z(t) with continuous acceleration.

    Segment 1 (0 <= t < T): quintic polynomial from (z0, 0, 0) to (z_creep, -v_creep, 0);
    T is chosen so that the peak speed equals v_peak. Segment 2 (t >= T): constant
    speed -v_creep, clamped at z_floor (where it stops with zero velocity).
    """

    def __init__(self, z0: float, z_creep: float, z_floor: float, v_peak: float, v_creep: float) -> None:
        self.z0, self.z_creep, self.z_floor, self.v_creep = float(z0), float(z_creep), float(z_floor), float(v_creep)
        dz = self.z_creep - self.z0
        if dz >= 0.0 or v_peak <= 0.0:
            self.T, self.coef = 0.0, None
            self.z_creep = self.z0
            return
        lo, hi = abs(dz) / v_peak, 2.0 * abs(dz) / max(v_peak, 1e-9)    # bracket: quintic peak speed <= 1.875 x mean
        for _ in range(60):                                              # bisection on the duration
            mid = 0.5 * (lo + hi)
            if self._peak_speed(self._coefficients(mid), mid) > v_peak:
                lo = mid
            else:
                hi = mid
        self.T = hi
        self.coef = self._coefficients(self.T)

    def _coefficients(self, T: float) -> np.ndarray:
        p0, v0, a0, p1, v1, a1 = self.z0, 0.0, 0.0, self.z_creep, -self.v_creep, 0.0
        c3 = (20 * (p1 - p0) - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T ** 2) / (2 * T ** 3)
        c4 = (30 * (p0 - p1) + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T ** 2) / (2 * T ** 4)
        c5 = (12 * (p1 - p0) - 6 * (v1 + v0) * T - (a0 - a1) * T ** 2) / (2 * T ** 5)
        return np.array([p0, v0, a0 / 2, c3, c4, c5])

    @staticmethod
    def _peak_speed(c: np.ndarray, T: float) -> float:
        t = np.linspace(0.0, T, 400)
        v = c[1] + 2 * c[2] * t + 3 * c[3] * t ** 2 + 4 * c[4] * t ** 3 + 5 * c[5] * t ** 4
        return float(np.max(np.abs(v)))

    def __call__(self, t: float) -> tuple[float, float, float]:
        """(z, dz/dt, d2z/dt2) at time t since the start of the approach."""
        if self.coef is not None and t < self.T:
            c = self.coef
            z = c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3 + c[4] * t ** 4 + c[5] * t ** 5
            v = c[1] + 2 * c[2] * t + 3 * c[3] * t ** 2 + 4 * c[4] * t ** 3 + 5 * c[5] * t ** 4
            a = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t ** 2 + 20 * c[5] * t ** 3
            return float(z), float(v), float(a)
        z = self.z_creep - self.v_creep * max(0.0, t - self.T)
        if z <= self.z_floor:
            return self.z_floor, 0.0, 0.0
        return float(z), -self.v_creep, 0.0


class SlideProfile:
    """Scalar s(t) in [0, D]: rest-to-rest quintic strokes 0 -> D -> 0 -> ..., with continuous
    position, velocity and acceleration (both are zero at every stroke end), optional dwell at
    the ends, and a cycle limit (0 = unlimited). Each stroke lasts T = 1.875 D / v_peak, the
    duration for which a rest-to-rest quintic reaches peak speed v_peak."""

    def __init__(self, distance: float, v_peak: float, cycles: int = 0, dwell: float = 0.0) -> None:
        self.D = float(distance)
        self.T = 1.875 * self.D / v_peak if v_peak > 0 and self.D > 0 else 0.0
        self.cycles = int(cycles)
        self.dwell = float(dwell)

    def strokes_completed(self, t: float) -> int:
        if self.T <= 0 or t <= 0:
            return 0
        n = int(t // (self.T + self.dwell))
        return min(n, 2 * self.cycles) if self.cycles > 0 else n

    def __call__(self, t: float) -> tuple[float, float, float]:
        """(s, ds/dt, d2s/dt2) at time t since the slide started."""
        if self.T <= 0 or t <= 0:
            return 0.0, 0.0, 0.0
        period = self.T + self.dwell
        n = int(t // period)
        if self.cycles > 0 and n >= 2 * self.cycles:
            return 0.0, 0.0, 0.0                    # finished: rest at the start point
        tau = t - n * period
        forward = n % 2 == 0
        if tau >= self.T:                           # dwell
            return (self.D if forward else 0.0), 0.0, 0.0
        u = tau / self.T
        p = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5           # rest-to-rest quintic, 0 -> 1
        v = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / self.T
        a = (60 * u - 180 * u ** 2 + 120 * u ** 3) / self.T ** 2
        if forward:
            return self.D * p, self.D * v, self.D * a
        return self.D * (1 - p), -self.D * v, -self.D * a


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
        wr = 2.0 * np.pi * self.cfg.rotation_freq_hz
        self.kp_r, self.kd_r = wr ** 2, 2.0 * self.cfg.motion_zeta * wr
        direction = np.asarray(self.cfg.slide_direction, dtype=float).reshape(3)
        direction = direction - (direction @ self.normal) * self.normal
        if self.cfg.slide_distance > 0 and np.linalg.norm(direction) < 1e-9:
            raise ValueError("slide_direction must have a component in the table plane")
        self.slide_dir = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-9 else np.zeros(3)
        self.slide = SlideProfile(self.cfg.slide_distance, self.cfg.slide_speed,
                                  self.cfg.slide_cycles, self.cfg.slide_dwell)
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
        self.zd_dot = 0.0
        self.zd_ddot = 0.0
        self.z_floor = 0.0
        self.z_surface = 0.0            # expected tip-centre height at first contact (surface + tip radius)
        self.profile: ApproachProfile | None = None
        self.t_approach = 0.0
        self.phase = APPROACH
        self.t_contact: float | None = None
        self.t_last_contact = 0.0
        self.f_int = 0.0
        self.f_n = 0.0
        self.f_d = 0.0
        self.f_cmd = 0.0
        self.n_releases = 0
        self.contact_count = 0
        self.t_slide: float | None = None
        self.s_d = self.sd_dot = self.sd_ddot = 0.0
        self.tan_int = np.zeros(3)
        self.rot_int = np.zeros(3)
        self.f_friction_ff = 0.0

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
        return orientation_error(self.quat_d, rot)

    def status(self) -> str:
        return (f"{self.phase:8s} f_n={self.f_n:6.2f} N  f_d={self.f_d:5.2f} N  "
                f"f_cmd={self.f_cmd:6.2f} N  z_d={self.z_d:.4f} m  dz_d={self.zd_dot:+.4f} m/s  "
                f"s_d={self.s_d:.4f} m  ds_d={self.sd_dot:+.4f} m/s")

    @property
    def in_contact(self) -> bool:
        return self.phase in (CONTACT, SLIDE)

    def strokes_completed(self, t: float) -> int:
        return self.slide.strokes_completed(t - self.t_slide) if self.t_slide is not None else 0

    def start_approach(self, z0: float, t: float) -> None:
        """Begin a new smooth descent from height z0 at time t."""
        cfg = self.cfg
        self.profile = ApproachProfile(z0, self.z_surface + cfg.creep_distance, self.z_floor,
                                       cfg.approach_speed, cfg.creep_speed)
        self.t_approach = t
        self.z_d, self.zd_dot, self.zd_ddot = self.profile(0.0)

    # ------------------------------------------------------------------ interface
    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, q_home: np.ndarray) -> None:
        """Called after mj_forward on the home keyframe."""
        self.q_home = q_home.copy()
        self.x_ref = data.site_xpos[self.tcp].copy()
        self.z_d = float(self.x_ref[2])
        mujoco.mju_mat2Quat(self.quat_d, data.site_xmat[self.tcp])
        tabletop = model.geom(TABLETOP_GEOM).id
        top_z = float(data.geom_xpos[tabletop][2] + model.geom_size[tabletop][2])
        self.z_surface = top_z + self.tip_radius
        self.z_floor = self.z_surface - self.cfg.max_penetration
        self.start_approach(self.z_d, float(data.time))
        self.phase = APPROACH
        self.t_contact = None
        self.t_last_contact = 0.0
        self.f_int = 0.0
        self.f_n = self.f_d = self.f_cmd = 0.0
        self.n_releases = 0
        self.contact_count = 0
        self.t_slide = None
        self.s_d = self.sd_dot = self.sd_ddot = 0.0
        self.tan_int[:] = 0.0
        self.rot_int[:] = 0.0
        self.f_friction_ff = 0.0
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
            self.z_d, self.zd_dot, self.zd_ddot = self.profile(t - self.t_approach)
            self.contact_count = self.contact_count + 1 if f_n > cfg.contact_threshold else 0
            if self.contact_count >= cfg.contact_samples:
                self.phase = CONTACT
                self.zd_dot = self.zd_ddot = 0.0    # the set-point stops where contact was found
                self.t_contact = t
                self.t_last_contact = t
                self.f_int = 0.0
        else:
            if f_n > cfg.release_threshold:
                self.t_last_contact = t
            elif t - self.t_last_contact > cfg.release_time:
                self.phase = APPROACH          # contact lost: re-approach from the current height
                self.start_approach(float(x[2]), t)
                self.n_releases += 1
                self.t_slide = None
                self.s_d = self.sd_dot = self.sd_ddot = 0.0
            if self.phase == CONTACT and cfg.slide_distance > 0 and \
                    t - self.t_contact >= cfg.force_ramp_time + cfg.slide_start_delay:
                self.phase = SLIDE
                self.t_slide = t
            if self.phase == SLIDE:
                self.s_d, self.sd_dot, self.sd_ddot = self.slide(t - self.t_slide)

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
        x_d = self.x_ref + self.s_d * self.slide_dir
        xd_dot = self.sd_dot * self.slide_dir
        xd_ddot = self.sd_ddot * self.slide_dir
        if self.phase == APPROACH:
            x_d = x_d - (x_d @ n) * n + self.z_d * n
            xd_dot = xd_dot + self.zd_dot * n
            xd_ddot = xd_ddot + self.zd_ddot * n
        e_p = x_d - x
        e_o = self.orientation_error(rot)
        a_p = xd_ddot + self.kp_m * e_p + self.kd_m * (xd_dot - v)
        a_o = self.kp_r * e_o - self.kd_r * w
        if self.in_contact:
            a_p = a_p - (a_p @ n) * n                 # the normal direction is force-controlled
            if cfg.ki_tangential > 0:                 # integral on the in-plane error (friction mismatch)
                e_t = e_p - (e_p @ n) * n
                self.tan_int = self.tan_int + e_t * self.dt
                lim = cfg.tangential_integral_limit / cfg.ki_tangential
                norm = np.linalg.norm(self.tan_int)
                if norm > lim:
                    self.tan_int *= lim / norm
                a_p = a_p + cfg.ki_tangential * self.tan_int
            if cfg.ki_rotation > 0:                   # integral on the orientation error (friction moment)
                self.rot_int = self.rot_int + e_o * self.dt
                lim = cfg.rotation_integral_limit / cfg.ki_rotation
                norm = np.linalg.norm(self.rot_int)
                if norm > lim:
                    self.rot_int *= lim / norm
                a_o = a_o + cfg.ki_rotation * self.rot_int
        force = lam @ np.concatenate((a_p, a_o))
        # Nominal Coulomb friction feedforward along the commanded tangential velocity (a real
        # force, not inertia-weighted); the tanh blend keeps it continuous through the reversals.
        self.f_friction_ff = 0.0
        if self.in_contact and self.sd_dot != 0.0:
            if cfg.friction_ff > 0:
                self.f_friction_ff += cfg.friction_ff * max(f_n, 0.0) * float(np.tanh(self.sd_dot / cfg.friction_ff_speed))
            if cfg.friction_viscous_ff > 0:
                self.f_friction_ff += cfg.friction_viscous_ff * self.sd_dot
            force[:3] = force[:3] + self.f_friction_ff * self.slide_dir

        # --- normal force control ---------------------------------------------------------------
        if self.in_contact:
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
