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


#: Reference-governor variants (see ReferenceGovernor).
GOVERNOR_MODES = ("continuous_rebase", "stop_time_anchor")


@dataclass
class GovernorConfig:
    """Tangential moving-reference governor (task/recovery policy).

    Engaged by the human-energy CBF row (never by force detection alone),
    it shapes the NOMINAL task reference only — physical port powers
    p_H = F_H^T v_ee etc. are untouched, and the governor is NOT part of
    the passivity certificate.

    Two variants:

    ``continuous_rebase`` (original): v_g decays exponentially toward zero
        while x_g is continuously pulled onto the MOVING end-effector,
        x_g+ = x_g + dt (v_g + k_rebase (x - x_g)). This removes tangential
        position stiffness, so a sustained blocking force makes the robot
        retreat continuously (measured -33 mm over a 2-s block) and the
        human port does net positive work, recharging E_H to its cap.

    ``stop_time_anchor`` (proposed): on trigger the governed velocity
        follows a finite cosine deceleration to EXACTLY zero over
        ``t_decel`` while the rebase term is faded in with a smoothstep;
        the governed position at that stopping instant is captured once as
        a FIXED anchor x_a, held for the rest of the interaction. Keeping a
        fixed anchor retains position stiffness K_p (x_a - x), so the
        interaction settles at a bounded displacement instead of retreating.
    """

    enabled: bool = False
    mode: str = "stop_time_anchor"      # final C3; see GOVERNOR_MODES
    f_detect: float = 0.1               # human-force detection threshold [N]
    v_decay_tau: float = 0.1            # continuous_rebase: v_g decay time [s]
    k_rebase: float = 10.0              # x_g rebase rate [1/s]
    t_decel: float = 0.4                # stop_time_anchor: DECEL duration [s]
    t_resume: float = 0.75              # cosine ramp 0 -> v_d after release [s]
    clear_dwell: float = 0.1            # human-absent dwell before resuming [s]


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

    # --- Optional tangential reference governor (SLIDE phase) ---
    governor: GovernorConfig = field(default_factory=GovernorConfig)


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
    """Configuration of the selective-passivation layer.

    Primary (paper-grade) modes:
      C0_nominal                 existing nominal controller, ledgers observed
      C1_whole_port_qp           QP with ONE conventional whole-port ledger on
                                 F_meas^T v_ee (no contact-model subtraction,
                                 no human ledger) — clean accounting baseline
      C2_residual_qp             QP with the residual-ledger constraint only
      C3_dual_ledger_qp          QP with residual + human ledgers + human
                                 instantaneous-power constraint (proposed)
      C4_dual_ledger_safe_scalar C3's constraints, but the command is
                                 restricted to the line from a feasible
                                 minimum-norm safety anchor toward u_nom
                                 (task-preservation ablation)

    Legacy diagnostics (origin-ray scaling u = gamma u_nom; the ray need not
    intersect the feasible set, so these can go infeasible and fall back to
    emergency damping — kept for comparison only, EXCLUDED from paper-grade
    acceptance):
      C1_legacy_whole_port_origin_scalar
      C4_legacy_origin_scalar
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
    eps_h: float = 1e-4                 # human-ledger margin: E_safe - E_min [J]
    eps_r: float = 1e-4                 # residual/whole margin [J]
    p_h_max: float = 0.10               # max instantaneous power extracted
                                        # from the human port [W]
    # CBF-style hard smoothing rows p_lower+ >= -k_cbf (E - E_safe). These
    # make the ledger approach E_safe exponentially (time constant 1/k_cbf),
    # so the mandated one-step constraints never collide with the hard
    # torque-rate limit. Both one-step rows stay in the QP as the
    # certificate; the runtime asserts 0 < k_cbf * dt <= 1.
    #
    # k = 10 (vs 50 in iteration 1): the demanded deceleration while a
    # ledger transits into its floor-riding band scales with k; at k = 50
    # the per-step velocity change excites the soft-contact friction state
    # (the one-step-delayed force sample softens under braking, which
    # unbinds/rebinds the row in a 2-step limit cycle). At k = 10 the
    # residual-channel transit is measured chatter-free; the whole-port
    # channel (C1) chatters at ANY k because its port force is contact-
    # sample-dominated — documented as a structural finding.
    k_cbf: float = 10.0                 # [1/s]; 0 disables the smoothing rows

    # --- robust one-step prediction-error bound ---
    # Configured assumption ||v_ee,k+1_actual - v_ee,k+1_pred||_2 <= e_v_bound.
    # All ledger constraints use the robust lower power estimate
    #   p_lower+(u) = p_pred+(u) - ||F_i|| * e_v_bound,
    # which (given the bound holds) implies E_{k+1} >= E_safe >= E_min
    # exactly — no constant power slack anywhere. The bound is NEVER adapted
    # during a run; a measured violation is logged, counted, and fails the
    # acceptance check for certificate-bearing controllers.
    #
    # Value: measured across scenarios A-F and the mu_hat sweep, the
    # closed-loop error is median ~1.5e-4 m/s, p99 ~1e-3 m/s, with a maximum
    # of ~1.8e-2 m/s at the ungoverned post-release catch-up impact (the
    # spec's initial suggestion of 0.01 sits BELOW that measured maximum, so
    # per the measure-first rule the bound is set comfortably above it).
    # Governed runs stay below ~1.5e-3 m/s.
    e_v_bound: float = 0.025            # [m/s]

    # --- QP objective ---
    w_t: float = 1.0                    # tangential deviation weight
    w_n: float = 20.0                   # normal deviation weight
    eps_reg: float = 1e-6               # Tikhonov regularization on u

    # --- bookkeeping ---
    activation_tol: float = 1e-4        # |u - u_nom| > tol counts as active [N]

    # --- infeasibility fallback: bounded emergency damping ---
    emergency_damping: float = 40.0     # [N s/m] on v_tn
    emergency_u_max: float = 20.0       # bound on |u_emergency| per axis [N]

    @property
    def e_safe_h(self) -> float:
        return self.human.e_min + self.eps_h

    @property
    def e_safe_r(self) -> float:
        return self.residual.e_min + self.eps_r

    @property
    def e_safe_w(self) -> float:
        return self.whole.e_min + self.eps_r


PASSIVATION_MODES = (
    "C0_nominal",
    "C1_whole_port_qp",
    "C2_residual_qp",
    "C3_dual_ledger_qp",
    "C4_dual_ledger_safe_scalar",
    "C1_legacy_whole_port_origin_scalar",
    "C4_legacy_origin_scalar",
)

#: Paper-grade controllers: acceptance requires zero infeasible steps, zero
#: emergency steps, zero prediction-bound violations, and zero floor
#: violations on their certified ledgers.
PRIMARY_MODES = PASSIVATION_MODES[:5]


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
