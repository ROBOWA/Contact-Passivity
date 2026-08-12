"""Configuration for the planar contact-sliding simulation.

All quantities are SI (m, s, N, N·m, rad). The world frame follows the MJCF
model: z up, surface top face at z = 0, motion restricted to the x-z plane.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import numpy as np

# ---------------------------------------------------------------------------
# Directions and projectors (fixed for a horizontal surface).
# ---------------------------------------------------------------------------

#: Tangential sliding direction t = [1, 0, 0]^T.
TANGENT = np.array([1.0, 0.0, 0.0])

#: Upward surface-normal direction n = [0, 0, 1]^T.
NORMAL = np.array([0.0, 0.0, 1.0])

#: Tangential Cartesian projector U = t t^T.
U_PROJ = np.outer(TANGENT, TANGENT)

#: Normal Cartesian projector C = n n^T.
C_PROJ = np.outer(NORMAL, NORMAL)

#: Path of the MJCF model shipped with this package.
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "models", "planar_two_link.xml")

# Names of the two geoms whose contacts constitute the task-contact channel.
SURFACE_GEOM = "surface"
PAD_GEOM = "end_effector_pad"
EE_SITE = "ee_site"
EE_BODY = "link2"  # body carrying the pad geom and the ee site


@dataclass
class ControllerConfig:
    """Gains and limits of the three-phase force/motion controller."""

    # --- Phase 1: approach (Cartesian PD toward the surface) ---
    approach_speed: float = 0.15        # downward descent rate of z target [m/s]
    kp_cart: float = 400.0              # Cartesian P gain [N/m] (x and z)
    kd_cart: float = 40.0               # Cartesian D gain [N·s/m]
    contact_gap_threshold: float = 0.002  # switch to RAMP below this gap [m]

    # --- Phase 2: normal-force ramp (PI force control in C-space) ---
    f_desired: float = 5.0              # desired steady normal force F_d [N]
    ramp_duration: float = 1.0          # cosine ramp 0 -> F_d [s]
    settle_duration: float = 0.5        # hold after ramp before sliding [s]
    k_f: float = 0.5                    # force P gain (dimensionless)
    k_i: float = 8.0                    # force I gain [1/s]
    d_f: float = 120.0                  # normal velocity damping [N·s/m]
    integral_limit: float = 8.0         # |k_i * integral| bound [N] (anti-windup)

    # --- Phase 3: sliding (tangential PD tracking in U-space) ---
    v_slide: float = 0.05               # desired tangential speed v_d [m/s]
    kx_slide: float = 400.0             # tangential position gain [N/m]
    dx_slide: float = 60.0              # tangential velocity gain [N·s/m]

    # --- Joint-space terms ---
    # Controller joint damping D_q. The MJCF model already has small joint
    # damping (0.1 N·m·s/rad, integrated implicitly); this term is the
    # *additional* controller-side damping, kept small on purpose so damping
    # is not double-counted.
    dq_joint: float = 0.5               # [N·m·s/rad]

    # --- Actuation limits ---
    tau_limit: float = 60.0             # |tau| bound per joint [N·m] (= ctrlrange)
    tau_rate_limit: float = 5000.0      # |dtau/dt| bound [N·m/s]


@dataclass
class HumanForceConfig:
    """A separately known Cartesian disturbance applied at the EE site."""

    enabled: bool = False
    magnitude: float = 3.0              # peak force magnitude [N]
    direction: tuple = (1.0, 0.0, 0.0)  # unit direction (normalized on use)
    t_start: float = 6.0                # profile start time [s]
    t_rise: float = 0.5                 # cosine rise duration [s]
    t_hold: float = 2.0                 # constant-magnitude duration [s]
    t_fall: float = 0.5                 # cosine fall duration [s]


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""

    duration: float = 10.0              # total simulated time [s]
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    human: HumanForceConfig = field(default_factory=HumanForceConfig)

    # Validation bound used by tests and reported in the summary.
    max_penetration: float = 0.005      # [m]

    # Output artifacts.
    output_dir: str = "results"

    def with_human_force(self) -> "SimulationConfig":
        return replace(self, human=replace(self.human, enabled=True))
