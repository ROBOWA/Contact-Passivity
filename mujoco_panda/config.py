"""Configuration for the torque-controlled Panda table-contact PoC."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from mujoco_sliding.config import LedgerParams


MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "models", "panda_table.xml")
MODES = ("C0_nominal", "C1_whole_port_qp", "C2_residual_qp", "C3_dual_ledger_qp")


@dataclass
class PlantConfig:
    timestep: float = 0.001
    contact_layout: str = "multipoint"  # multipoint | single (ablation)
    contact_time_constant: float = 0.100
    contact_damping_ratio: float = 1.0
    contact_impedance: tuple[float, float, float] = (0.05, 0.95, 0.005)
    contact_margin: float = 0.001


@dataclass
class GovernorConfig:
    enabled: bool = False
    decel_time: float = 0.35
    resume_time: float = 0.75
    release_dwell: float = 0.15


@dataclass
class ControllerConfig:
    f_desired: float = 5.0
    approach_speed: float = 0.035
    contact_gap: float = 0.003
    ramp_duration: float = 0.8
    settle_duration: float = 0.4
    slide_speed: float = 0.03
    slide_ramp_duration: float = 0.60
    trajectory: str = "straight"       # straight | wipe
    straight_length: float = 0.12
    wipe_radii: tuple[float, float] = (0.055, 0.035)
    wipe_period: float = 8.0

    kp_approach: float = 350.0
    kd_translation: float = 35.0
    kp_tangent: float = 1200.0
    kd_tangent: float = 70.0
    k_force: float = 0.30
    ki_force: float = 3.0
    d_force: float = 24.0
    force_filter_time: float = 0.070
    integral_limit: float = 10.0
    kp_orientation: float = 55.0
    kd_orientation: float = 9.0
    kp_posture: float = 4.0
    kd_joint: float = 1.5

    torque_rate: tuple[float, ...] = (1000, 1000, 1000, 1000, 500, 500, 500)
    qvel_limit: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.4, 2.4, 2.4)
    joint_margin: float = 0.03
    governor: GovernorConfig = field(default_factory=GovernorConfig)


@dataclass
class HumanConfig:
    enabled: bool = False
    location: str = "tool"              # tool | forearm
    magnitude: float = 7.0
    direction: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    start: float = 3.8
    rise: float = 0.3
    hold: float = 1.2
    fall: float = 0.3


@dataclass
class PassivationConfig:
    mode: str = "C3_dual_ledger_qp"
    predictor: str = "oracle"           # oracle | friction
    mu_hat: float = 0.35
    normal_scale: float = 1.0            # controlled stiffness/force mismatch
    dynamics_mass_scale: float = 1.0     # controlled robot inertia-model error
    friction_v_s: float = 0.006
    residual: LedgerParams = LedgerParams(0.30, 0.02, 0.50)
    human: LedgerParams = LedgerParams(0.055, 0.005, 0.10)
    whole: LedgerParams = LedgerParams(0.30, 0.02, 0.50)
    eps_energy: float = 2e-4
    p_h_max: float = 0.18
    k_cbf: float = 8.0
    # Frozen after calibration; a bound on one-step arm-qvel prediction error.
    # The calibration includes the straight-path stopping transient.
    qvel_prediction_bound: float = 0.003
    w_tau: tuple[float, ...] = (1, 1, 1, 1, 2, 2, 2)
    w_task_tangent: float = 1.0e4
    w_task_normal: float = 5.0e4
    w_task_orientation: float = 1.0e4
    activation_tol: float = 1e-3
    emergency_damping: float = 12.0


@dataclass
class PandaConfig:
    duration: float = 7.0
    seed: int = 7
    plant: PlantConfig = field(default_factory=PlantConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    human: HumanConfig = field(default_factory=HumanConfig)
    passivation: PassivationConfig = field(default_factory=PassivationConfig)
