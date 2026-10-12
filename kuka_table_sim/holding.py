"""Joint-space holding controller and the stepping loop that schedules it.

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


def inertia_scaled_gains(model: mujoco.MjModel, arm: ArmIndex, q: np.ndarray,
                         freq_hz: float = DEFAULT_HOLD_FREQ_HZ,
                         zeta: float = DEFAULT_HOLD_ZETA) -> tuple[np.ndarray, np.ndarray]:
    """(Kp, Kd) from the mass-matrix diagonal at configuration q."""
    data = mujoco.MjData(model)
    data.qpos[arm.qpos_adr] = q
    mujoco.mj_forward(model, data)
    full_m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full_m)
    inertia = np.diag(full_m)[arm.dof_adr]
    wn = 2.0 * np.pi * freq_hz
    return inertia * wn ** 2, 2.0 * zeta * inertia * wn


class HoldingController:
    def __init__(self, arm: ArmIndex, kp: np.ndarray, kd: np.ndarray,
                 torque_limit: np.ndarray | None = None) -> None:
        self.arm = arm
        self.kp = np.asarray(kp, dtype=float).reshape(7)
        self.kd = np.asarray(kd, dtype=float).reshape(7)
        self.torque_limit = arm.torque_limit if torque_limit is None else np.asarray(torque_limit, float).reshape(7)
        self.q_target = np.zeros(7)
        self.saturated = np.zeros(7, dtype=bool)  # per-joint flag from the last compute()

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
        """Restore the home keyframe, zero velocities/time, and re-arm the holding target."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(HOME_KEYFRAME).id)
        self.data.time = 0.0
        self.n_steps = 0
        self.n_control_updates = 0
        self.controller.q_target = self.q_home.copy()
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[self.arm.actuator_ids] = self.controller.compute(self.data)

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
