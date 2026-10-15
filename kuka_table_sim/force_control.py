"""Approach the table along its normal, hold a constant contact force, slide, and recover from contact loss.

The controller works in operational space at the `tcp` site (tool tip centre).

Measurement path
    The normal force comes only from the FT sensor: F_ext = -R S - m g + m a_com (see ft_sensor.py), computed
    from ONE time-aligned FTSample (force, sensor-frame rotation and ideal COM acceleration all from the same
    physics evaluation). The sample consumed at a control update at time t_k was evaluated at the end of the
    previous physics step, so it is one physics step (timestep) old. This corrected estimate is used for contact
    entry, the contact-deficit / loss logic, the force PI loop and the friction feedforward. No contact truth is used.

Phases
    APPROACH  quintic descent to a creep point above the expected surface, then creep, until the corrected
              normal force exceeds a threshold for a few consecutive updates.
    CONTACT   normal direction under force control (f_d ramped up); tangential/orientation under position control.
    SLIDE     as CONTACT plus rest-to-rest quintic strokes along a fixed in-plane direction.
    SEARCH    contact is in doubt (a force deficit) or was lost: sliding is suspended, the normal channel is under position
              control and a bounded, acceleration-limited reference creeps down from the last good contact height
              until contact is confirmed again.
    HOLD      the search failed within its time/distance bounds: controlled hold at a standoff above the surface
              until request_resume().

Recovery policy
    Entry.      In CONTACT/SLIDE the force deficit f_d - f_n exceeds max(deficit_threshold, deficit_fraction * f_d)
                for `deficit_samples` consecutive updates (5 N at the full 10 N target, never during the first
                second of the ramp, where f_n legitimately lags f_d by a few tenths of a newton) -> SEARCH.
    On entry.   The stroke reference is stopped by a bounded quintic from its current (s, v, a); the normal
                reference is planned from the MEASURED state (z, v) to the height of the last healthy contact
                sample (known kinematics) and then creeps down at creep_speed; the normal channel blends from
                force to position control (smootherstep over mode_blend_time, no command step); integrators freeze.
    LOST.       f_n <= release_threshold continuously for release_time (the existing 0.3 s persistence) is an
                escalation inside SEARCH: f_int is reset, tan_int/rot_int are tapered to zero, the search budget
                keeps running.
    Governor.   While SEARCH runs, the descent reference's clock is slowed (critically damped, continuous in velocity
                and acceleration) whenever the tool lags the reference by more than search_lag_hold, and paused at
                search_lag_max. A disturbance that holds the tool away therefore cannot wind the reference up below
                the surface, and a released tool meets a reference near the surface instead of being driven into it.
    Bounds.     SEARCH ends in HOLD after search_time_max, or after resting search_floor_dwell at its floor
                (the lower of surface - max_penetration and start - search_distance_max). HOLD retreats with a
                bounded quintic to a standoff of creep_distance above the surface (never snaps to a fixed height)
                and waits for request_resume(), which restarts the search from the hold reference.
    Exit.       Contact is CONFIRMED when the corrected f_n exceeds contact_threshold for contact_samples updates
                (the same test as the first touchdown). The controller returns to CONTACT with a new force ramp
                that starts at the measured force (0 for the very first contact), the force loop takes over
                with an integrator reset, and sliding resumes toward the endpoint of the interrupted stroke once
                the ramp has completed (the first stroke also waits slide_start_delay).

Integral states (f_int normal force, tan_int in-plane position, rot_int orientation)
    * frozen in SEARCH/HOLD and while the previous update's output saturated any actuator (conditional
      integration: the final clipped torque is what counts, not only the integral clamp); f_int additionally
      only runs while the force loop is fully active;
    * f_int is reset to 0 on every entry into CONTACT (a new force ramp starts the loop afresh);
    * tan_int / rot_int keep their frozen values through a short interruption (they encode the friction
      compensation of the surface, which is still there) and are tapered to zero over integral_taper_time
      after LOST (smootherstep, no command step), re-accumulating from zero after reacquisition.

Reference continuity
    Every reference comes from a Trajectory1D / StrokeSequencer planned from the CURRENT reference state
    (position, velocity, acceleration) or, where no reference existed, from the measured state, with declared
    speed/acceleration limits. The stroke is never reset to its origin. Force <-> position changes of the
    normal channel are blended with a smootherstep weight; the return to force control starts the force ramp
    at the measured force, so the commanded force is continuous across that switch as well.

Null space
    The posture torque is projected with N^T = I - J^T Jbar^T built from the REGULARISED operational-space
    inertia Lambda = (J M^-1 J^T + lambda I)^-1. The projector is therefore only approximately dynamically
    consistent: J M^-1 N^T = lambda Lambda J M^-1 != 0. The resulting task-space leakage is logged
    (`null_leak`) and is not claimed to be zero.

Gravity and Coriolis compensation through qfrc_bias is applied exactly once, here.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .ft_sensor import FTSample, FTSensor
from .holding import full_mass_matrix
from .ik import orientation_error
from .scene import ArmIndex, TCP_SITE, TOOL_TIP_GEOM
from .trajectories import StrokeSequencer, Trajectory1D, descent_trajectory, move_to_trajectory

FT_FORCE_SENSOR = "ft_force"
FT_SITE = "ft_site"
FT_BODY = "ft_sensor"
TABLETOP_GEOM = "tabletop"

APPROACH = "approach"
CONTACT = "contact"
SLIDE = "slide"
SEARCH = "search"
HOLD = "hold"
PHASE_CODE = {APPROACH: 0, CONTACT: 1, SLIDE: 2, SEARCH: 3, HOLD: 4}
CODE_PHASE = {v: k for k, v in PHASE_CODE.items()}
_CONTACT_PHASES = (CONTACT, SLIDE)


def smootherstep(u: float) -> float:
    u = min(1.0, max(0.0, float(u)))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


@dataclass(frozen=True)
class ForceControlConfig:
    force_desired: float = 10.0       # N, normal force to hold (positive = pressing into the table)
    approach_speed: float = 0.05      # m/s, peak descent speed of the smooth approach segment
    creep_speed: float = 0.005        # m/s, constant speed of the final approach until contact
    creep_distance: float = 0.01      # m, height above the expected surface where the creep begins
    contact_threshold: float = 1.0    # N, corrected normal force that ends the approach / search
    contact_samples: int = 3          # consecutive control updates above the threshold required
    release_threshold: float = 0.2    # N, corrected normal force below which contact counts as absent
    release_time: float = 0.3         # s, persistence before contact is declared LOST
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
    max_penetration: float = 0.01     # m, the approach/search reference never goes deeper than this below the surface
    # --- contact-quality and recovery policy. These limits are DECLARED before any recovery result is judged. ---
    deficit_threshold: float = 1.0    # recovery starts when the force deficit f_d - f_n exceeds max(this (N), fraction * f_d) ...
    deficit_fraction: float = 0.5     # ... (the absolute part keeps the ramp start, where f_n lags f_d by a few tenths of N, quiet)
    deficit_samples: int = 10         # ... for this many consecutive control updates
    search_creep_distance: float = 0.002  # m, height above the last good contact height where a recovery search creeps
    search_lag_hold: float = 0.0005   # m, SEARCH reference progress is full speed while the tool lags it by less than this ...
    search_lag_max: float = 0.003     # m, ... and paused once it lags by this much (smooth in between): no reference windup
    search_gate_freq: float = 30.0    # rad/s, natural frequency of the critically damped filter that smooths the progress rate
    mode_blend_time: float = 0.1      # s, smootherstep blend of the normal channel between force and position control
    integral_taper_time: float = 0.5  # s, tan_int / rot_int are tapered to zero over this time after LOST
    recovery_stop_time: float = 0.6   # s, minimum duration of the bounded stop of a moving reference
    recovery_speed_max: float = 0.05  # m/s, declared peak reference speed of recovery motion
    recovery_accel_max: float = 0.30  # m/s^2, declared peak reference acceleration of recovery motion
    search_time_max: float = 4.0      # s, SEARCH gives up (-> HOLD) after this long
    search_distance_max: float = 0.03 # m, the search reference never descends further than this below its start
    search_floor_dwell: float = 0.5   # s, SEARCH gives up after resting this long at its floor without contact


class ForceApproachController:
    def __init__(self, model: mujoco.MjModel, arm: ArmIndex, control_period: float,
                 config: ForceControlConfig | None = None) -> None:
        self.model = model
        self.arm = arm
        self.dt = control_period
        self.cfg = config or ForceControlConfig()
        self.torque_limit = arm.torque_limit.copy()
        self.tcp = model.site(TCP_SITE).id
        self.ft = FTSensor(model, FT_FORCE_SENSOR, FT_SITE, FT_BODY)
        self.ft_latest: FTSample | None = None
        self.tool_mass = self.ft.mass
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
        c = self.cfg
        self.stroke = StrokeSequencer(c.slide_distance, c.slide_speed, c.slide_cycles, c.slide_dwell,
                                      stop_time=c.recovery_stop_time, a_max=c.recovery_accel_max)
        wq = 2.0 * np.pi * self.cfg.posture_freq_hz
        self.kp_q, self.kd_q = wq ** 2, 2.0 * self.cfg.posture_zeta * wq
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._init_state()

    def _init_state(self) -> None:
        self.saturated = np.zeros(7, dtype=bool)
        self._sat_prev = False
        self.q_home = np.zeros(7)
        self.x_ref = np.zeros(3)
        self.quat_d = np.array([1.0, 0, 0, 0])
        self.z_surface = 0.0            # expected tip-centre height at first contact (surface + tip radius)
        self.z_floor = 0.0
        self.phase = APPROACH
        self.lost = False                               # LOST escalation active inside SEARCH
        self.normal_traj: Trajectory1D | None = None
        self.profile: Trajectory1D | None = None        # the initial approach trajectory (for reports)
        self.w_pos = 1.0                                # normal-channel position weight (before smootherstep)
        self.t_contact: float | None = None
        self.t_last_good = 0.0
        self.contact_count = 0
        self.z_good = 0.0                               # tip height at the last healthy contact sample
        self.f_start = 0.0                              # force at which the current ramp started
        self._deficit_count = 0
        self.t_lost: float | None = None
        self.search_t0 = 0.0
        self.search_floor = 0.0
        self._t_at_floor: float | None = None
        self._resume_requested = False
        self._t_eff = 0.0                               # effective trajectory time of the SEARCH reference (progress governor)
        self._rate = 1.0
        self._rate_dot = 0.0
        # counters of the controller's own decisions: recoveries started by a contact deficit, LOST escalations,
        # reacquisitions, HOLD entries (not physical contact loss)
        self.n_uncertain = self.n_lost = self.n_reacquired = self.n_hold = 0
        # integrators
        self.f_int = 0.0
        self.tan_int = np.zeros(3)
        self.rot_int = np.zeros(3)
        self.int_scale = 1.0
        self.taper_active = False
        # last-update diagnostics
        self.f_n = self.f_d = self.f_cmd = 0.0
        self.f_friction_ff = 0.0
        self.z_d = self.zd_dot = self.zd_ddot = 0.0
        self.s_d = self.sd_dot = self.sd_ddot = 0.0
        self.x_d = np.full(3, np.nan)
        self.v_d = np.full(3, np.nan)
        self.a_d = np.full(3, np.nan)
        self.sample_t = float("nan")
        self.sample: FTSample | None = None
        self.f_corrected = np.zeros(3)
        self.f_gravity_only = np.zeros(3)
        self.tau_cmd = np.zeros(7)
        self.tau_bias = np.zeros(7)
        self.tau_task = np.zeros(7)
        self.tau_null = np.zeros(7)
        self.null_leak = np.zeros(6)
        self.e_o = np.zeros(3)
        self.w_blend = 1.0

    # ------------------------------------------------------------------ measurement
    def observe(self, model: mujoco.MjModel, data: mujoco.MjData, t_eval: float) -> None:
        """Capture the FT chain right after mj_step2 (or mj_forward); `t_eval` is the instant that
        evaluation describes (the START time of the step that just ran)."""
        self.ft_latest = self.ft.sample(data, t_eval)

    def contact_force_world(self, data: mujoco.MjData) -> np.ndarray:
        """Corrected external force on the tool (N) from the latest time-aligned sample."""
        s = self.ft_latest if self.ft_latest is not None else self.ft.sample(data, data.time)
        return s.corrected

    def normal_force(self, data: mujoco.MjData) -> float:
        return float(self.contact_force_world(data) @ self.normal)

    def orientation_error(self, rot: np.ndarray) -> np.ndarray:
        return orientation_error(self.quat_d, rot)

    # ------------------------------------------------------------------ status
    def status(self) -> str:
        tag = f"{self.phase}{'!' if self.lost else ''}"
        return (f"{tag:9s} f_n={self.f_n:6.2f} N  f_d={self.f_d:5.2f} N  f_cmd={self.f_cmd:6.2f} N  "
                f"z_d={self.z_d:.4f} m  s_d={self.s_d:.4f} m  ds_d={self.sd_dot:+.4f} m/s  "
                f"unc/lost/reacq/hold={self.n_uncertain}/{self.n_lost}/{self.n_reacquired}/{self.n_hold}")

    @property
    def in_contact(self) -> bool:
        return self.phase in _CONTACT_PHASES

    @property
    def n_releases(self) -> int:
        """Number of FSM LOST events (a counter of the controller's decision, not of physical contact loss)."""
        return self.n_lost

    def strokes_completed(self, t: float = 0.0) -> int:
        return self.stroke.strokes

    def request_resume(self) -> None:
        """Ask a controller in HOLD to search for the surface again (processed at the next update)."""
        self._resume_requested = True

    # ------------------------------------------------------------------ interface
    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, q_home: np.ndarray) -> None:
        """Called after mj_forward on the home keyframe."""
        self._init_state()
        self.q_home = q_home.copy()
        self.x_ref = data.site_xpos[self.tcp].copy()
        mujoco.mju_mat2Quat(self.quat_d, data.site_xmat[self.tcp])
        tabletop = model.geom(TABLETOP_GEOM).id
        top_z = float(data.geom_xpos[tabletop][2] + model.geom_size[tabletop][2])
        self.z_surface = top_z + self.tip_radius
        self.z_floor = self.z_surface - self.cfg.max_penetration
        c = self.cfg
        self.normal_traj = descent_trajectory(float(data.time), float(self.x_ref @ self.normal), 0.0, 0.0,
                                              self.z_surface + c.creep_distance, self.z_floor,
                                              v_peak=c.approach_speed, v_creep=c.creep_speed)
        self.profile = self.normal_traj
        self.z_d = float(self.x_ref @ self.normal)
        self.stroke.reset()
        self.ft_latest = None

    # ------------------------------------------------------------------ finite-state logic
    def _force_desired(self, t: float) -> float:
        c = self.cfg
        if self.t_contact is None:
            return 0.0
        ramp = min(1.0, (t - self.t_contact) / c.force_ramp_time) if c.force_ramp_time > 0 else 1.0
        return self.f_start + (c.force_desired - self.f_start) * ramp

    def _enter_contact(self, t: float, z: float, f_n: float) -> None:
        reacquired = self.phase == SEARCH
        if reacquired:
            self.n_reacquired += 1
        self.f_start = max(f_n, 0.0) if reacquired else 0.0    # the first touchdown ramps from 0, as validated
        self.phase = CONTACT
        self.t_contact = t
        self.t_last_good = t
        self.z_good = z
        self.w_pos = 0.0           # hard switch to force control; the ramp starts at the measured force
        self.normal_traj = None
        self.zd_dot = self.zd_ddot = 0.0
        self.f_int = 0.0
        self.lost = False
        self._deficit_count = 0

    def _begin_search(self, t: float, z: float, vz: float, from_hold: bool = False) -> None:
        """Start the bounded recovery search: from the measured state toward the last good contact height
        (from contact), or from the active hold reference (from HOLD)."""
        c = self.cfg
        if from_hold and self.normal_traj is not None:
            z0, v0, a0 = self.normal_traj(t)
            z_creep = self.z_surface + c.creep_distance
            floor0 = z0
        else:
            z0, v0, a0 = z, vz, 0.0
            z_creep = self.z_good + c.search_creep_distance
            floor0 = self.z_good
        self.search_floor = max(self.z_floor, floor0 - c.search_distance_max)
        self.normal_traj = descent_trajectory(t, z0, v0, a0, z_creep, self.search_floor,
                                              v_peak=min(c.approach_speed, c.recovery_speed_max), v_creep=c.creep_speed,
                                              a_max=c.recovery_accel_max)
        self.search_t0 = t
        self._t_eff, self._rate, self._rate_dot = t, 1.0, 0.0
        self._t_at_floor = None
        self.phase = SEARCH
        if not from_hold:                  # a search resumed from HOLD continues the same, already declared, loss
            self.lost = False
            self.t_last_good = t
        self.contact_count = 0

    def _enter_hold(self, t: float) -> None:
        """Search exhausted: retreat with a bounded quintic to a standoff above the expected surface and hold."""
        c = self.cfg
        z0, v0, a0 = self.z_d, self.zd_dot, self.zd_ddot          # the reference that is active now (after the governor)
        self.normal_traj = move_to_trajectory(t, z0, v0, a0, self.z_surface + c.creep_distance, v_max=c.recovery_speed_max,
                                              a_max=c.recovery_accel_max, T_min=c.recovery_stop_time)
        self.phase = HOLD
        self.n_hold += 1

    def _update_fsm(self, t: float, f_n: float, z: float, vz: float) -> None:
        c = self.cfg
        if self._resume_requested:
            self._resume_requested = False
            if self.phase == HOLD:
                self._begin_search(t, z, vz, from_hold=True)
        if self.phase == HOLD:
            return
        if self.phase in (APPROACH, SEARCH):
            self.contact_count = self.contact_count + 1 if f_n > c.contact_threshold else 0
            if self.contact_count >= c.contact_samples:
                self._enter_contact(t, z, f_n)
                return
            if self.phase == SEARCH:
                if f_n > c.release_threshold:
                    self.t_last_good = t
                elif not self.lost and t - self.t_last_good > c.release_time:
                    self.lost = True                       # escalation: contact is declared LOST
                    self.n_lost += 1
                    self.t_lost = t
                    self.f_int = 0.0
                    self.taper_active = True
                z_ref, v_ref, _ = self.normal_traj(self._t_eff)
                if z_ref <= self.search_floor + 1e-6 and abs(v_ref) < 1e-9:
                    self._t_at_floor = t if self._t_at_floor is None else self._t_at_floor
                if t - self.search_t0 > c.search_time_max or \
                        (self._t_at_floor is not None and t - self._t_at_floor > c.search_floor_dwell):
                    self._enter_hold(t)
            return
        # CONTACT / SLIDE
        f_d = self._force_desired(t)
        deficit = f_d - f_n
        if f_n > c.release_threshold and deficit <= max(c.deficit_threshold, c.deficit_fraction * f_d):
            self.z_good = z                              # healthy contact: remember the surface height (kinematics only)
        self._deficit_count = self._deficit_count + 1 if deficit > max(c.deficit_threshold, c.deficit_fraction * f_d) else 0
        if self._deficit_count >= c.deficit_samples:
            self.n_uncertain += 1
            self.stroke.suspend(t)
            self._begin_search(t, z, vz)
            return
        if f_n > c.release_threshold:
            self.t_last_good = t
        if self.phase == CONTACT and c.slide_distance > 0:
            first = self.stroke.state == StrokeSequencer.IDLE
            if t - self.t_contact >= c.force_ramp_time + (c.slide_start_delay if first else 0.0) - 1e-9:   # tolerance: t accumulates rounding
                self.phase = SLIDE
                self.stroke.start(t) if first else self.stroke.resume(t)

    # ------------------------------------------------------------------ control law
    def compute(self, data: mujoco.MjData) -> np.ndarray:
        m, arm, cfg = self.model, self.arm, self.cfg
        t = float(data.time)
        dt = self.dt
        q = data.qpos[arm.qpos_adr]
        qd = data.qvel[arm.dof_adr]
        bias = data.qfrc_bias[arm.dof_adr]
        n = self.normal

        # --- measurement: ONE time-aligned FT sample, corrected for the inertia of the supported tool -------
        sample = self.ft_latest if self.ft_latest is not None else self.ft.sample(data, t)
        self.sample = sample
        self.sample_t = sample.t
        self.f_gravity_only = sample.gravity_only
        self.f_corrected = sample.corrected
        self.f_n = f_n = float(self.f_corrected @ n)

        # --- kinematics and dynamics at the current state (x_k) ----------------------------------------------
        mujoco.mj_jacSite(m, data, self._jacp, self._jacr, self.tcp)
        jac = np.vstack((self._jacp, self._jacr))[:, arm.dof_adr]      # 6 x 7
        x = data.site_xpos[self.tcp].copy()
        rot = data.site_xmat[self.tcp].reshape(3, 3)
        v = jac[:3] @ qd
        w = jac[3:] @ qd
        mass = full_mass_matrix(m, data)[np.ix_(arm.dof_adr, arm.dof_adr)]
        mass_inv = np.linalg.inv(mass)
        lam = np.linalg.inv(jac @ mass_inv @ jac.T + cfg.lambda_reg * np.eye(6))

        # --- phase logic and references ----------------------------------------------------------------------
        self._update_fsm(t, f_n, float(x @ n), float(v @ n))
        normal_active = self.normal_traj is not None
        if normal_active and self.phase == SEARCH:
            # Reference progress governor: the descent clock slows (smoothly) while the tool lags the reference, so
            # a disturbance that holds the tool away cannot wind the reference up below the surface.
            z_now = float(x @ n)
            z_ref, _, _ = self.normal_traj(self._t_eff)
            lag = z_now - z_ref
            u = (lag - cfg.search_lag_hold) / max(cfg.search_lag_max - cfg.search_lag_hold, 1e-9)
            target = 1.0 - smootherstep(u)
            wg = cfg.search_gate_freq
            self._rate_dot += (wg * wg * (target - self._rate) - 2.0 * wg * self._rate_dot) * dt
            self._rate = float(np.clip(self._rate + self._rate_dot * dt, 0.0, 1.0))
            self._t_eff += self._rate * dt
            z_ref, v_tr, a_tr = self.normal_traj(self._t_eff)
            self.z_d, self.zd_dot, self.zd_ddot = z_ref, v_tr * self._rate, a_tr * self._rate ** 2 + v_tr * self._rate_dot
        elif normal_active:
            self.z_d, self.zd_dot, self.zd_ddot = self.normal_traj(t)
        self.s_d, self.sd_dot, self.sd_ddot = self.stroke.eval(t)
        x_plane = self.x_ref - (self.x_ref @ n) * n
        x_d = x_plane + self.s_d * self.slide_dir
        xd_dot = self.sd_dot * self.slide_dir
        xd_ddot = self.sd_ddot * self.slide_dir
        if normal_active:
            x_d = x_d + self.z_d * n
            xd_dot = xd_dot + self.zd_dot * n
            xd_ddot = xd_ddot + self.zd_ddot * n
        else:                                     # force mode: the normal part of the PD is discarded below
            x_d = x_d + (x @ n) * n
            xd_dot = xd_dot + (v @ n) * n
        # Active reference sequence exposed to the diagnostics: the normal component is NaN while the normal
        # channel is under pure force control (no position reference exists there).
        if normal_active:
            self.x_d, self.v_d, self.a_d = x_d.copy(), xd_dot.copy(), xd_ddot.copy()
        else:
            nan_n = np.where(np.abs(n) > 0.5, np.nan, 0.0)
            self.x_d = (x_d - (x_d @ n) * n) + nan_n
            self.v_d = (xd_dot - (xd_dot @ n) * n) + nan_n
            self.a_d = (xd_ddot - (xd_ddot @ n) * n) + nan_n
        e_p = x_d - x
        self.e_o = e_o = self.orientation_error(rot)

        # --- normal-channel mode weight -------------------------------------------------------------------------
        w_target = 1.0 if self.phase in (APPROACH, SEARCH, HOLD) else 0.0
        step = dt / cfg.mode_blend_time if cfg.mode_blend_time > 0 else 1.0
        self.w_pos += float(np.clip(w_target - self.w_pos, -step, step))
        w_blend = smootherstep(self.w_pos)
        self.w_blend = w_blend

        # --- integrators: freeze / taper / anti-windup ------------------------------------------------------------
        in_contact_phase = self.phase in _CONTACT_PHASES
        can_integrate = in_contact_phase and not self.taper_active and not self._sat_prev
        if self.taper_active:
            self.int_scale -= dt / cfg.integral_taper_time
            if self.int_scale <= 0.0:
                self.int_scale = 1.0
                self.taper_active = False
                self.tan_int[:] = 0.0
                self.rot_int[:] = 0.0
        m_int = smootherstep(self.int_scale) if self.taper_active else 1.0
        e_t = e_p - (e_p @ n) * n
        if can_integrate and cfg.ki_tangential > 0:
            self.tan_int = self.tan_int + e_t * dt
            lim = cfg.tangential_integral_limit / cfg.ki_tangential
            norm = np.linalg.norm(self.tan_int)
            if norm > lim:
                self.tan_int *= lim / norm
        if can_integrate and cfg.ki_rotation > 0:
            self.rot_int = self.rot_int + e_o * dt
            lim = cfg.rotation_integral_limit / cfg.ki_rotation
            norm = np.linalg.norm(self.rot_int)
            if norm > lim:
                self.rot_int *= lim / norm

        # --- operational-space PD (unit-mass gains, inertia-weighted) with feedforward ------------------------------
        a_p = xd_ddot + self.kp_m * e_p + self.kd_m * (xd_dot - v)
        a_o = self.kp_r * e_o - self.kd_r * w
        a_int = np.zeros(3)
        a_o_int = np.zeros(3)
        if self.phase != APPROACH:
            a_int = cfg.ki_tangential * self.tan_int * m_int
            a_o_int = cfg.ki_rotation * self.rot_int * m_int
        a_tan = a_p - (a_p @ n) * n + a_int
        a_full = (a_p + a_int) if normal_active else a_tan
        wrench_pos = lam @ np.concatenate((a_full, a_o + a_o_int))
        wrench_tan = lam @ np.concatenate((a_tan, a_o + a_o_int))

        # nominal friction feedforward along the commanded tangential velocity
        self.f_friction_ff = 0.0
        if in_contact_phase and self.sd_dot != 0.0:
            if cfg.friction_ff > 0:
                self.f_friction_ff += cfg.friction_ff * max(f_n, 0.0) * float(np.tanh(self.sd_dot / cfg.friction_ff_speed))
            if cfg.friction_viscous_ff > 0:
                self.f_friction_ff += cfg.friction_viscous_ff * self.sd_dot
        wrench_tan[:3] = wrench_tan[:3] + self.f_friction_ff * self.slide_dir

        # --- normal force law (force-control mode) -------------------------------------------------------------------
        if in_contact_phase:
            self.f_d = self._force_desired(t)
            err = self.f_d - f_n
            if can_integrate and self.w_pos <= 0.0 and cfg.ki_force > 0:
                limit = cfg.integral_limit / cfg.ki_force
                self.f_int = float(np.clip(self.f_int + err * dt, -limit, limit))
            self.f_cmd = self.f_d + cfg.kp_force * err + cfg.ki_force * self.f_int
        else:
            self.f_d = self.f_cmd = 0.0
        wrench_force = wrench_tan.copy()
        f_p = wrench_force[:3] - (wrench_force[:3] @ n) * n
        wrench_force[:3] = f_p - (self.f_cmd + cfg.normal_damping * (v @ n)) * n
        W = (1.0 - w_blend) * wrench_force + w_blend * wrench_pos

        # --- null-space posture and final torque ------------------------------------------------------------------------
        tau_task = jac.T @ W
        jbar = mass_inv @ jac.T @ lam                       # generalised inverse built from the REGULARISED Lambda
        null_t = np.eye(7) - jac.T @ jbar.T                 # approximate projector: J M^-1 null_t = lambda Lambda J M^-1
        tau_posture = mass @ (-self.kp_q * (q - self.q_home) - self.kd_q * qd)
        tau_null = null_t @ tau_posture
        tau = bias + tau_task + tau_null
        self.tau_bias, self.tau_task, self.tau_null, self.tau_cmd = bias.copy(), tau_task, tau_null, tau.copy()
        self.null_leak = jac @ mass_inv @ tau_null           # task-space acceleration the posture torque leaks into
        self.saturated = np.abs(tau) >= self.torque_limit
        self._sat_prev = bool(np.any(self.saturated))
        return np.clip(tau, -self.torque_limit, self.torque_limit)

    # ------------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict:
        """Snapshot of the last compute() for the recorder (plain numbers and small arrays)."""
        s = self.sample
        return dict(
            phase=PHASE_CODE[self.phase], lost=float(self.lost), w_blend=self.w_blend, w_pos=self.w_pos,
            t_ft=self.sample_t, ft_raw=s.raw_site if s is not None else np.zeros(3), ft_grav=self.f_gravity_only,
            ft_corr=self.f_corrected, f_n=self.f_n, f_d=self.f_d, f_cmd=self.f_cmd, f_int=self.f_int,
            tan_int=self.tan_int.copy(), rot_int=self.rot_int.copy(), int_scale=self.int_scale if self.taper_active else 1.0,
            x_d=self.x_d, v_d=self.v_d, a_d=self.a_d, s_d=self.s_d, z_d=self.z_d if self.normal_traj is not None else np.nan,
            e_o=self.e_o, friction_ff=self.f_friction_ff,
            tau_cmd=self.tau_cmd, tau_bias=self.tau_bias, tau_task=self.tau_task, tau_null=self.tau_null,
            null_leak=self.null_leak, sat_any=float(np.any(self.saturated)),
            n_uncertain=self.n_uncertain, n_lost=self.n_lost, n_reacquired=self.n_reacquired, n_hold=self.n_hold,
            search_rate=self._rate if self.phase == SEARCH else 1.0,
            stroke_state=self.stroke.state, strokes=self.stroke.strokes,
        )
