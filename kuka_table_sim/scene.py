"""Scene loading, timing configuration and arm index bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SCENE_XML = ASSETS_DIR / "kuka_table_scene.xml"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"joint{i}_torque" for i in range(1, 8))
ROBOT_BODIES = ("base", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
                "tool_mount", "ft_sensor", "sphere_tool")
HOME_KEYFRAME = "home"
FLANGE_SITE = "flange"   # centre of the iiwa14 mounting face, z out of the flange
TCP_SITE = "tcp"         # centre of the polishing tool's tip sphere
TOOL_TIP_GEOM = "tool_tip"
OVERVIEW_CAMERA = "overview"


@dataclass(frozen=True)
class TimingConfig:
    """All scheduling is done in integer physics steps derived from these values."""

    timestep: float = 0.00025      # physics step, s
    control_period: float = 0.001  # holding-torque update period, simulated s
    render_hz: float = 60.0        # viewer sync rate, approximately

    def __post_init__(self) -> None:
        if self.timestep <= 0 or self.control_period <= 0 or self.render_hz <= 0:
            raise ValueError("timestep, control_period and render_hz must be positive")
        ratio = self.control_period / self.timestep
        if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
            raise ValueError(f"control_period={self.control_period} must be a positive integer "
                             f"multiple of timestep={self.timestep}")

    @property
    def control_interval(self) -> int:
        """Physics steps per control update (4 for the defaults)."""
        return int(round(self.control_period / self.timestep))

    @property
    def render_interval(self) -> int:
        """Physics steps per viewer sync (67 for the defaults, i.e. ~59.7 Hz of simulated time)."""
        return max(1, int(round(1.0 / (self.render_hz * self.timestep))))


def load_model(timing: TimingConfig | None = None, xml_path: Path = SCENE_XML) -> mujoco.MjModel:
    """Load the scene and apply the configured physics timestep."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if timing is not None:
        model.opt.timestep = timing.timestep
    return model


@dataclass(frozen=True)
class ArmIndex:
    """Resolved indices for the seven arm joints and their torque actuators.

    Joint i (0-based) <-> actuator i; qpos_adr/dof_adr map joint i into qpos/qvel/qfrc_bias.
    """

    joint_ids: np.ndarray
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuator_ids: np.ndarray
    torque_limit: np.ndarray  # |ctrl| bound per actuator, N*m, from ctrlrange

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "ArmIndex":
        joint_ids = np.array([model.joint(n).id for n in ARM_JOINTS])
        actuator_ids = np.array([model.actuator(n).id for n in ARM_ACTUATORS])
        for j, a in zip(joint_ids, actuator_ids):
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"{model.joint(j).name} is not a hinge joint")
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or model.actuator_trnid[a, 0] != j:
                raise ValueError(f"{model.actuator(a).name} does not drive {model.joint(j).name}")
            if model.actuator_gear[a, 0] != 1.0 or model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED \
                    or model.actuator_gainprm[a, 0] != 1.0 or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_NONE:
                raise ValueError(f"{model.actuator(a).name} is not a unit-gain, bias-free torque motor")
            if not model.actuator_ctrllimited[a]:
                raise ValueError(f"{model.actuator(a).name} has no ctrlrange torque limit")
        limits = model.actuator_ctrlrange[actuator_ids]
        if np.any(limits[:, 0] != -limits[:, 1]):
            raise ValueError("torque limits are expected to be symmetric")
        return cls(
            joint_ids=joint_ids,
            qpos_adr=model.jnt_qposadr[joint_ids].copy(),
            dof_adr=model.jnt_dofadr[joint_ids].copy(),
            actuator_ids=actuator_ids,
            torque_limit=limits[:, 1].copy(),
        )


def home_qpos(model: mujoco.MjModel, arm: ArmIndex) -> np.ndarray:
    """Arm joint positions stored in the 'home' keyframe."""
    key = model.key(HOME_KEYFRAME).id
    return model.key_qpos[key, arm.qpos_adr].copy()
