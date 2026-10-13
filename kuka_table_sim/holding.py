"""Joint-space holding controller and the stepping loop that schedules it.

Any controller used by `Simulation` provides reset(model, data, q_home),
compute(data) -> 7 torques, a `saturated` flag array and status() -> str.

    tau = qfrc_bias + Kp * (q_home - q) - Kd * qvel

qfrc_bias is MuJoCo's C(q, qvel) * qvel + g(q) for the state the step is about
to integrate; it is applied exactly once, here, and nowhere else. This controller
only keeps the arm still for scene checks. It is not the research controller.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import ArmIndex, TimingConfig, HOME_KEYFRAME, home_qpos

# Default holding gains are scaled per joint by the mass-matrix diagonal at q_home:
#   Kp_i = M_ii(q_home) * wn^2,   Kd_i = 2 * zeta * M_ii(q_home) * wn
# so every joint closes the loop at the same natural frequency wn (rad/s) and damping
# ratio zeta, regardless of inertia (joint 7 has 0.001 kg*m^2, joint 2 about 2.9).
# With the 1 ms zero-order hold this keeps Kd/M * dt = 2*zeta*wn*dt well below the
# discrete stability bound. Explicit Kp/Kd vectors can be passed instead.
DEFAULT_HOLD_FREQ_HZ = 2.0
DEFAULT_HOLD_ZETA = 1.0


def full_mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Dense joint-space mass matrix for the state in `data` (after mj_forward).

    The mj_fullM binding changed between MuJoCo releases: >= 3.4 takes (model, data, dst),
    older 3.x takes (model, dst, data.qM). Both are supported so the conda env and the venv work.
    """
    full_m = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, full_m)
    except TypeError:
        mujoco.mj_fullM(model, full_m, data.qM)
    return full_m


def inertia_scaled_gains(model: mujoco.MjModel, arm: ArmIndex, q: np.ndarray,
                         freq_hz: float = DEFAULT_HOLD_FREQ_HZ,
                         zeta: float = DEFAULT_HOLD_ZETA) -> tuple[np.ndarray, np.ndarray]:
    """(Kp, Kd) from the mass-matrix diagonal at configuration q."""
    data = mujoco.MjData(model)
    data.qpos[arm.qpos_adr] = q
    mujoco.mj_forward(model, data)
    inertia = np.diag(full_mass_matrix(model, data))[arm.dof_adr]
    wn = 2.0 * np.pi * freq_hz
    return inertia * wn ** 2, 2.0 * zeta * inertia * wn


class HoldingController:
    def __init__(self, arm: ArmIndex, kp: np.ndarray, kd: np.ndarray,
                 torque_limit: np.ndarray | None = None) -> None:
        self.arm = arm
        self.kp = np.asarray(kp, dtype=float).reshape(7)
        self.kd = np.asarray(kd, dtype=float).reshape(7)
        self.freq_hz: float | None = None   # set by inertia_scaled(); enables retune()
        self.zeta: float | None = None
        self.torque_limit = arm.torque_limit if torque_limit is None else np.asarray(torque_limit, float).reshape(7)
        self.q_target = np.zeros(7)
        self.saturated = np.zeros(7, dtype=bool)  # per-joint flag from the last compute()

    @classmethod
    def inertia_scaled(cls, model: mujoco.MjModel, arm: ArmIndex, q: np.ndarray,
                       freq_hz: float = DEFAULT_HOLD_FREQ_HZ, zeta: float = DEFAULT_HOLD_ZETA) -> "HoldingController":
        ctl = cls(arm, *inertia_scaled_gains(model, arm, q, freq_hz, zeta))
        ctl.freq_hz, ctl.zeta = freq_hz, zeta
        return ctl

    def retune(self, model: mujoco.MjModel, q_home: np.ndarray) -> None:
        """Recompute the inertia-scaled gains for a new home configuration (no-op for explicit gains)."""
        if self.freq_hz is not None:
            self.kp, self.kd = inertia_scaled_gains(model, self.arm, q_home, self.freq_hz, self.zeta)

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, q_home: np.ndarray) -> None:
        self.q_target = q_home.copy()
        self.saturated[:] = False

    def status(self) -> str:
        return "holding q_home"

    def compute(self, data: mujoco.MjData) -> np.ndarray:
        """Torque for the arm given consistent (q, qvel, qfrc_bias) in `data`."""
        q = data.qpos[self.arm.qpos_adr]
        qd = data.qvel[self.arm.dof_adr]
        bias = data.qfrc_bias[self.arm.dof_adr]
        tau = bias + self.kp * (self.q_target - q) - self.kd * qd
        self.saturated = np.abs(tau) >= self.torque_limit
        return np.clip(tau, -self.torque_limit, self.torque_limit)


class Simulation:
    """Owns model/data and advances physics with the holding torque held between updates."""

    def __init__(self, model: mujoco.MjModel, timing: TimingConfig | None = None,
                 controller: HoldingController | None = None) -> None:
        self.model = model
        self.timing = timing or TimingConfig()
        self.model.opt.timestep = self.timing.timestep
        self.arm = ArmIndex.from_model(model)
        self.q_home = home_qpos(model, self.arm)
        if controller is None:
            controller = HoldingController(self.arm, *inertia_scaled_gains(model, self.arm, self.q_home))
        self.controller = controller
        self.data = mujoco.MjData(model)
        self.n_steps = 0
        self.n_control_updates = 0
        self.reset()

    def reset(self) -> None:
        """Restore the home keyframe, zero velocities/time, and re-arm the controller."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(HOME_KEYFRAME).id)
        self.data.time = 0.0
        self.n_steps = 0
        self.n_control_updates = 0
        # Prime the state with pure gravity/Coriolis compensation so that the derived
        # quantities the controller sees at reset (accelerations, force sensors) describe
        # a held, static arm rather than a free-falling one (ctrl is 0 after mj_resetData).
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[self.arm.actuator_ids] = self.data.qfrc_bias[self.arm.dof_adr]
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset(self.model, self.data, self.q_home)
        self.data.ctrl[self.arm.actuator_ids] = self.controller.compute(self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_home(self, q: np.ndarray) -> None:
        """Make `q` the initial configuration: update the `home` keyframe, the controller, and reset."""
        q = np.asarray(q, dtype=float).reshape(7)
        lo, hi = self.model.jnt_range[self.arm.joint_ids].T
        if np.any(q < lo) or np.any(q > hi):
            raise ValueError(f"q is outside the joint limits: {q}")
        self.model.key_qpos[self.model.key(HOME_KEYFRAME).id, self.arm.qpos_adr] = q
        self.q_home = q.copy()
        if hasattr(self.controller, "retune"):
            self.controller.retune(self.model, self.q_home)
        self.reset()

    def set_initial_pose(self, site: str, pos, quat=None, *, rpy_deg=None, q_seed=None,
                         require_contact_free: bool = True, **ik_kwargs):
        """Set the initial state so that `site` is at `pos` (m, world) with orientation `quat`
        (wxyz) or `rpy_deg` (roll, pitch, yaw in degrees, R = Rz Ry Rx); None leaves the
        orientation free. Solves IK seeded at the current home, writes the solution as the new
        home (keyframe + controller target) and resets. Raises ValueError if IK fails or, with
        require_contact_free, if the pose is in contact. Returns the IKResult."""
        from .ik import quat_from_rpy, solve_site_ik
        if rpy_deg is not None:
            if quat is not None:
                raise ValueError("give either quat or rpy_deg, not both")
            quat = quat_from_rpy(*rpy_deg)
        seed = self.q_home if q_seed is None else np.asarray(q_seed, float)
        result = solve_site_ik(self.model, site, pos, quat, arm=self.arm, q_seed=seed, q_pref=seed, **ik_kwargs)
        if not result.converged:
            raise ValueError(f"no IK solution for site '{site}' at {np.round(pos, 4)}: {result.summary()}")
        previous = self.q_home.copy()
        self.set_home(result.q)
        if require_contact_free and self.data.ncon > 0:
            pairs = [(self.model.geom(c.geom1).name, self.model.geom(c.geom2).name) for c in self.data.contact[:self.data.ncon]]
            self.set_home(previous)
            raise ValueError(f"pose is in contact ({pairs}); initial state restored")
        return result

    def step(self) -> None:
        """Advance exactly one physics step; recompute torque every control_interval steps."""
        if self.n_steps % self.timing.control_interval == 0:
            # mj_step1 evaluates kinematics, qfrc_bias, contacts for the current state;
            # the torque is then computed from those same quantities before mj_step2 integrates.
            mujoco.mj_step1(self.model, self.data)
            self.data.ctrl[self.arm.actuator_ids] = self.controller.compute(self.data)
            self.n_control_updates += 1
            mujoco.mj_step2(self.model, self.data)
        else:
            mujoco.mj_step(self.model, self.data)
        self.n_steps += 1

    def advance(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    def steps_for(self, duration: float) -> int:
        """Integer number of physics steps covering `duration` seconds of simulated time."""
        return int(round(duration / self.timing.timestep))

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.arm.qpos_adr]

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel[self.arm.dof_adr]

    @property
    def tau(self) -> np.ndarray:
        return self.data.ctrl[self.arm.actuator_ids]
