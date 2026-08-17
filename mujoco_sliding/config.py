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

#: Tangent-normal basis B_tn = [t n] (3 x 2). A planar command u = [u_t, u_n]
#: maps to the world frame as B_TN @ u; world vectors map down as B_TN.T @ v.
#: Power is invariant under this transform for in-plane vectors (verified in
#: tests; the mechanism has no y DOF, so v_y is structurally zero).
B_TN = np.column_stack([TANGENT, NORMAL])

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


@dataclass(frozen=True)
class LedgerParams:
    """Parameters of one energy ledger (initial charge, floor, cap) [J]."""

    e_init: float
    e_min: float
    e_max: float


@dataclass
class PassivationConfig:
    """Configuration of the selective-passivation layer (modes C0-C4).

    Modes:
      C0_nominal            existing nominal controller, ledgers only observed
      C1_whole_port_scalar  one conventional whole-port ledger on
                            F_meas^T v_ee, u = gamma * u_nom
      C2_residual_qp        QP with the residual-ledger constraint only
      C3_dual_ledger_qp     QP with residual + human ledgers + human
                            instantaneous-power constraint (proposed method)
      C4_dual_ledger_scalar same constraints as C3 but u = gamma * u_nom
                            (task-preservation ablation)
    """

    mode: str = "C3_dual_ledger_qp"

    # --- task-contact predictor ---
    predictor: str = "oracle"           # "oracle" | "friction"
    mu_hat: float = 0.30                # predictor friction coefficient
    v_s: float = 0.005                  # tanh regularization speed [m/s]

    # --- ledgers ---
    residual: LedgerParams = LedgerParams(0.30, 0.02, 0.50)
    human: LedgerParams = LedgerParams(0.05, 0.005, 0.08)
    # Conventional whole-port ledger (C1 constraint; observed elsewhere).
    whole: LedgerParams = LedgerParams(0.30, 0.02, 0.50)

    # --- QP constraint parameters ---
    eps_h: float = 1e-4                 # human-ledger constraint margin [J]
    eps_r: float = 1e-4                 # residual-ledger constraint margin [J]
    p_h_max: float = 0.10               # max instantaneous power extracted
                                        # from the human port [W]
    # CBF-style hard smoothing rows -p+ <= k_cbf (E - E_min). These make the
    # ledger approach its floor exponentially (time constant 1/k_cbf), so the
    # mandated one-step constraints never collide with the hard torque-rate
    # limit. Both mandated one-step rows stay in the QP as the certificate;
    # for k_cbf < 1/dt the CBF row is the tighter of the two near the floor.
    k_cbf: float = 50.0                 # [1/s]; 0 disables the smoothing rows
    # Feasibility power tolerance on all ledger lower-bound rows [W]. Without
    # it, a ledger sitting marginally below its floor (reachable through
    # accumulated one-step prediction error) turns the clamped constraint
    # p+ >= 0 into a hard velocity constraint enforced through a vanishing
    # force coefficient, which is numerically near-infeasible as the port
    # force ramps out. delta_p bounds the resulting extra extraction to
    # delta_p * t (5e-5 W here, i.e. eating into the eps_h/eps_r buffer, not
    # below E_min). Reported as part of the certificate tolerance.
    delta_p: float = 5e-5               # [W]

    # --- QP objective ---
    w_t: float = 1.0                    # tangential deviation weight
    w_n: float = 20.0                   # normal deviation weight
    eps_reg: float = 1e-6               # Tikhonov regularization on u

    # --- bookkeeping ---
    activation_tol: float = 1e-4        # |u - u_nom| > tol counts as active [N]

    # --- infeasibility fallback: bounded emergency damping ---
    emergency_damping: float = 40.0     # [N s/m] on v_tn
    emergency_u_max: float = 20.0       # bound on |u_emergency| per axis [N]


PASSIVATION_MODES = (
    "C0_nominal",
    "C1_whole_port_scalar",
    "C2_residual_qp",
    "C3_dual_ledger_qp",
    "C4_dual_ledger_scalar",
)


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""

    duration: float = 10.0              # total simulated time [s]
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    human: HumanForceConfig = field(default_factory=HumanForceConfig)
    # Additional human-force pulses (summed with `human`), e.g. the
    # helping-then-blocking scenario F.
    human_pulses: tuple = ()

    # Selective-passivation layer; None runs the original nominal pipeline
    # with no passivation bookkeeping at all (backward compatible).
    passivation: PassivationConfig | None = None

    # Validation bound used by tests and reported in the summary.
    max_penetration: float = 0.005      # [m]

    # Output artifacts.
    output_dir: str = "results"

    def with_human_force(self) -> "SimulationConfig":
        return replace(self, human=replace(self.human, enabled=True))
