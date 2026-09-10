"""Selective-passivation layer: one-step affine prediction, task-weighted QP,
scalar (gamma) baselines, and the per-step runtime for modes C0-C4.

Command parameterization
------------------------
The QP decision is the planar Cartesian command u = [u_t, u_n] mapped to
joint torques through

    tau(u) = tau0 + J_ee^T B_tn u,        tau0 = qfrc_bias - D_q qdot.

This is exactly the nominal controller's torque equation (tau_nom =
tau_bias + J^T f_cmd - D_q qdot) with u_nom = B_tn^T f_cmd, so u = u_nom
reproduces the unfiltered nominal command. MuJoCo's qfrc_bias contains
gravity + Coriolis only (NOT the MJCF joint damping, which lives in
qfrc_passive), so nothing is double-counted.

One-step affine prediction
--------------------------
Holding the current force samples constant over one 1 ms control step and
ignoring Jdot qdot:

    qacc(u)  = M^-1 (tau(u) + J^T F_meas + qfrc_passive - qfrc_bias)
    qdot+(u) = qdot + dt qacc(u)
    v_tn+(u) = B^T J qdot+(u) = A u + b

with A = dt (B^T J) M^-1 (J^T B). The bias term cancels against tau0, so
b = B^T J (qdot + dt M^-1 (-D_q qdot + qfrc_passive + J^T F_meas)).
Both the task contact and the human force are treated as acting at the EE
site (J_T = J_H = J_ee, first-POC assumption). The prediction is validated
against explicit dynamics and a real MuJoCo transition
(``validate_affine_prediction``), and the per-step prediction error is
logged.

Certified ledger updates use the SAME current-step force sample held over
the step, evaluated at the actual next velocity:
p_actual = F_k^T v_{ee,k+1}.

Scope of the certificate
------------------------
This is SELECTIVE PORT PASSIVATION: only the residual port (F_R = F_meas -
F_task_hat) and the nested human port (F_H) are constrained. The predicted
nominal task-contact power F_task_hat^T v_ee is deliberately excluded from
the residual ledger — the actively controlled robot is NOT claimed to be
globally passive.

Constraints (all hard; no hidden slack, no constant power relaxation):
with the configured one-step prediction-error bound ||e_v|| <= e_v_bound
and ep_i = ||F_i|| e_v_bound, the robust lower power estimate is
p_lower,i+(u) = p_pred,i+(u) - ep_i, and the rows are
  * human ledger      E_H + dt p_lower,H+(u)  >= E_safe,H = E_H_min + eps_H
  * residual ledger   E_R + dt p_lower,R+(u)  >= E_safe,R = E_R_min + eps_R
  * CBF smoothing     p_lower,i+(u) >= -k_cbf (E_i - E_safe,i)
    (0 < k_cbf dt <= 1 asserted). Extra hard rows that make depletion
    approach E_safe exponentially so the one-step rows never collide with
    the hard torque-rate limit; the one-step rows remain the certificate.
  * human power       -p_pred,H+(u) <= P_H_max   (prediction-based; measured
    compliance is within ||F_H|| times the realized error)
  * torque            -tau_max <= tau(u) <= tau_max
  * torque rate       |tau(u) - tau_prev| <= taudot_max dt

Given the bound holds at step k, the energy rows imply EXACTLY
E_{i,k+1} = E_{i,k} + dt p_i_actual >= E_{i,k} + dt p_lower,i+ >= E_safe,i
>= E_i_min: the certificate is conditional on the measured bound, which is
checked every step; violations are logged, counted, and fail acceptance —
never absorbed. The whole-port mode C1_whole_port_qp applies the same
treatment to the single ledger on F_meas^T v.

If the QP is infeasible the runtime logs the full state and constraint
margins, applies a bounded emergency damping command, and counts the event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
import scipy.sparse as sparse

import osqp

from .config import B_TN, ControllerConfig, PassivationConfig
from .contact_extraction import ContactResult, ModelHandles, site_jacobian
from .contact_model import TaskContactPredictor, decompose
from .controller import ControlOutput, SlidingForceController
from .energy_ledgers import EnergyLedger

# Constraint-row indices (fixed OSQP sparsity: dense 7 x 2 pattern).
ROW_E_H = 0        # human-ledger one-step row (whole-port ledger in C1)
ROW_E_R = 1        # residual-ledger one-step row
ROW_CBF_H = 2      # human CBF smoothing row (whole-port in C1)
ROW_CBF_R = 3      # residual CBF smoothing row
ROW_P_H = 4        # human instantaneous-power row
ROW_TAU_0 = 5      # torque box+rate, joint 0
ROW_TAU_1 = 6      # torque box+rate, joint 1
N_ROWS = 7
ROW_NAMES = ("E_H", "E_R", "CBF_H", "CBF_R", "P_H", "tau_0", "tau_1")

_INF = 1e30


# ---------------------------------------------------------------------------
# One-step affine prediction
# ---------------------------------------------------------------------------

@dataclass
class AffinePrediction:
    """v_tn+(u) = A u + b and the torque map tau(u) = tau0 + G u."""

    A: np.ndarray             # (2, 2) [m/s per N]
    b: np.ndarray             # (2,)   [m/s]
    G: np.ndarray             # (nv, 2) = J^T B
    tau0: np.ndarray          # (nv,)  = qfrc_bias - D_q qdot
    v_tn: np.ndarray          # (2,) current EE velocity in the t-n frame
    # Snapshot needed to predict from an arbitrary applied torque (used for
    # the prediction-error bound check on the ACTUALLY applied command).
    j_tn: np.ndarray = None   # (2, nv)
    minv: np.ndarray = None   # (nv, nv)
    qvel: np.ndarray = None   # (nv,)
    qfrc_rest: np.ndarray = None  # (nv,) J^T f_meas + qfrc_passive - qfrc_bias
    dt: float = 0.0

    def v_next(self, u: np.ndarray) -> np.ndarray:
        return self.A @ u + self.b

    def tau(self, u: np.ndarray) -> np.ndarray:
        return self.tau0 + self.G @ u

    def v_next_from_tau(self, tau: np.ndarray) -> np.ndarray:
        """Predicted v_tn+ for an arbitrary applied torque (no u needed)."""
        qacc = self.minv @ (tau + self.qfrc_rest)
        return self.j_tn @ (self.qvel + self.dt * qacc)


def mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, m, data.qM)
    return m


def compute_affine_prediction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    jp: np.ndarray,
    dq_joint: float,
    f_meas: np.ndarray,
) -> AffinePrediction:
    """Build A, b, G, tau0 from the current MjData state.

    ``f_meas`` is the current-step measured EE force sample (task contact,
    one-step delayed by the loop convention, plus the scripted human force),
    held constant over the step.
    """
    dt = model.opt.timestep
    minv = np.linalg.inv(mass_matrix(model, data))
    j_tn = B_TN.T @ jp                      # (2, nv)
    g = jp.T @ B_TN                         # (nv, 2)
    tau0 = data.qfrc_bias - dq_joint * data.qvel
    rhs0 = -dq_joint * data.qvel + data.qfrc_passive + jp.T @ f_meas
    qdot_free = data.qvel + dt * (minv @ rhs0)
    return AffinePrediction(
        A=dt * (j_tn @ minv @ g),
        b=j_tn @ qdot_free,
        G=g,
        tau0=tau0.copy(),
        v_tn=j_tn @ data.qvel,
        j_tn=j_tn,
        minv=minv,
        qvel=data.qvel.copy(),
        qfrc_rest=(jp.T @ f_meas + data.qfrc_passive
                   - data.qfrc_bias).copy(),
        dt=dt,
    )


def validate_affine_prediction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ModelHandles,
    dq_joint: float,
    f_h: np.ndarray,
    u_values: list[np.ndarray],
) -> list[dict]:
    """Compare A u + b against explicit dynamics AND a real MuJoCo step.

    Restores the MjData state afterwards. Returns one dict per tested u with
    the affine, explicit, and actual next EE velocities (t-n frame).
    """
    from .contact_extraction import extract_task_contact

    snap = (data.qpos.copy(), data.qvel.copy(), data.time,
            data.ctrl.copy(), data.qfrc_applied.copy(),
            data.qacc_warmstart.copy())

    contact = extract_task_contact(model, data, handles)
    jp = site_jacobian(model, data, handles.ee_site)
    f_meas = contact.f_task + f_h
    pred = compute_affine_prediction(model, data, jp, dq_joint, f_meas)

    dt = model.opt.timestep
    minv = np.linalg.inv(mass_matrix(model, data))
    # Snapshot everything the explicit computation needs BEFORE stepping, so
    # later iterations do not read a mutated MjData.
    qvel0 = data.qvel.copy()
    qfrc_passive0 = data.qfrc_passive.copy()
    qfrc_bias0 = data.qfrc_bias.copy()
    results = []
    for u in u_values:
        u = np.asarray(u, dtype=float)
        v_affine = pred.v_next(u)

        # Explicit dynamics computation (same formula, no factored A, b).
        tau = pred.tau(u)
        qacc = minv @ (tau + jp.T @ f_meas + qfrc_passive0 - qfrc_bias0)
        v_explicit = B_TN.T @ (jp @ (qvel0 + dt * qacc))

        # Actual MuJoCo transition.
        data.qpos[:] = snap[0]
        data.qvel[:] = snap[1]
        data.time = snap[2]
        data.qacc_warmstart[:] = snap[5]
        mujoco.mj_forward(model, data)
        data.ctrl[:] = tau
        data.qfrc_applied[:] = 0.0
        if np.any(f_h):
            point = data.site_xpos[handles.ee_site]
            mujoco.mj_applyFT(model, data, f_h, np.zeros(3), point,
                              handles.ee_body, data.qfrc_applied)
        mujoco.mj_step(model, data)
        jp1 = site_jacobian(model, data, handles.ee_site)
        v_actual = B_TN.T @ (jp1 @ data.qvel)

        results.append({
            "u": u,
            "v_affine": v_affine,
            "v_explicit": v_explicit,
            "v_actual": v_actual,
            "err_affine_vs_explicit": float(
                np.linalg.norm(v_affine - v_explicit)),
            "err_affine_vs_actual": float(
                np.linalg.norm(v_affine - v_actual)),
        })

    data.qpos[:] = snap[0]
    data.qvel[:] = snap[1]
    data.time = snap[2]
    data.ctrl[:] = snap[3]
    data.qfrc_applied[:] = snap[4]
    data.qacc_warmstart[:] = snap[5]
    mujoco.mj_forward(model, data)
    return results


# ---------------------------------------------------------------------------
# Constraint rows (shared by the QP and the scalar solvers)
# ---------------------------------------------------------------------------

@dataclass
class ConstraintRows:
    """l <= A_c u <= u_ for the fixed 7-row layout, in power/torque units."""

    a: np.ndarray             # (7, 2)
    lo: np.ndarray            # (7,)
    hi: np.ndarray            # (7,)

    def margins(self, u: np.ndarray) -> np.ndarray:
        """min(A_c u - lo, hi - A_c u) per row; +inf for disabled rows."""
        val = self.a @ u
        m_lo = np.where(self.lo <= -_INF, np.inf, val - self.lo)
        m_hi = np.where(self.hi >= _INF, np.inf, self.hi - val)
        return np.minimum(m_lo, m_hi)


MODES_HUMAN_ROWS = ("C3_dual_ledger_qp", "C4_dual_ledger_safe_scalar",
                    "C4_legacy_origin_scalar")
MODES_RESIDUAL_ROWS = ("C2_residual_qp",) + MODES_HUMAN_ROWS
MODES_WHOLE_ROWS = ("C1_whole_port_qp", "C1_legacy_whole_port_origin_scalar")


def build_constraint_rows(
    cfg: PassivationConfig,
    ctl_cfg: ControllerConfig,
    dt: float,
    pred: AffinePrediction,
    f_h_tn: np.ndarray,
    f_r_tn: np.ndarray,
    f_meas_tn: np.ndarray,
    e_h: float,
    e_r: float,
    e_w: float,
    tau_prev: np.ndarray | None,
    mode: str,
    ep_h: float = 0.0,
    ep_r: float = 0.0,
    ep_w: float = 0.0,
) -> ConstraintRows:
    """Build the 7 hard constraint rows for the given mode.

    Robust formulation: with the configured one-step velocity prediction-
    error bound ||e_v|| <= e_v_bound and a held port force F_i, the actual
    power satisfies p_actual >= p_pred - ||F_i|| e_v_bound. Each ledger row
    therefore constrains the ROBUST LOWER power estimate

        p_lower+(u) = p_pred+(u) - ep_i,      ep_i = ||F_i|| e_v_bound:

        energy:  E + dt p_lower+(u) >= E_safe        (E_safe = E_min + eps)
        CBF:     p_lower+(u) >= -k_cbf (E - E_safe)

    which, provided the bound holds, implies E_{k+1} >= E_safe >= E_min
    EXACTLY — there is no clamped right-hand side and no constant power
    slack. Rows are expressed in power units (the one-step inequality
    divided by dt) for uniform scaling:
        p_pred+(u) >= (E_safe - E) / dt + ep_i.

    The human instantaneous-power row stays on the predicted power (its
    measured compliance is within ||F_H|| times the realized error, reported
    in the logs).
    """
    a = np.zeros((N_ROWS, 2))
    lo = np.full(N_ROWS, -_INF)
    hi = np.full(N_ROWS, _INF)

    ah = f_h_tn @ pred.A          # p_H+(u) = ah @ u + bh
    bh = float(f_h_tn @ pred.b)
    ar = f_r_tn @ pred.A
    br = float(f_r_tn @ pred.b)
    aw = f_meas_tn @ pred.A
    bw = float(f_meas_tn @ pred.b)

    use_human = mode in MODES_HUMAN_ROWS
    use_residual = mode in MODES_RESIDUAL_ROWS
    use_whole = mode in MODES_WHOLE_ROWS

    if use_human:
        a[ROW_E_H] = ah
        lo[ROW_E_H] = (cfg.e_safe_h - e_h) / dt + ep_h - bh
        a[ROW_P_H] = ah
        lo[ROW_P_H] = -cfg.p_h_max - bh
        if cfg.k_cbf > 0.0:
            a[ROW_CBF_H] = ah
            lo[ROW_CBF_H] = -cfg.k_cbf * (e_h - cfg.e_safe_h) + ep_h - bh
    if use_residual:
        a[ROW_E_R] = ar
        lo[ROW_E_R] = (cfg.e_safe_r - e_r) / dt + ep_r - br
        if cfg.k_cbf > 0.0:
            a[ROW_CBF_R] = ar
            lo[ROW_CBF_R] = -cfg.k_cbf * (e_r - cfg.e_safe_r) + ep_r - br
    if use_whole:
        # C1 reuses the E_H / CBF_H slots for its single whole-port ledger.
        a[ROW_E_H] = aw
        lo[ROW_E_H] = (cfg.e_safe_w - e_w) / dt + ep_w - bw
        if cfg.k_cbf > 0.0:
            a[ROW_CBF_H] = aw
            lo[ROW_CBF_H] = -cfg.k_cbf * (e_w - cfg.e_safe_w) + ep_w - bw

    # Torque magnitude + rate, merged into one box per joint:
    #   max(-tau_max, tau_prev - dmax) <= tau0 + G u <= min(tau_max, ...).
    t_lo = np.full(2, -ctl_cfg.tau_limit)
    t_hi = np.full(2, ctl_cfg.tau_limit)
    if tau_prev is not None:
        dmax = ctl_cfg.tau_rate_limit * dt
        t_lo = np.maximum(t_lo, tau_prev - dmax)
        t_hi = np.minimum(t_hi, tau_prev + dmax)
    for j, row in enumerate((ROW_TAU_0, ROW_TAU_1)):
        a[row] = pred.G[j]
        lo[row] = t_lo[j] - pred.tau0[j]
        hi[row] = t_hi[j] - pred.tau0[j]

    return ConstraintRows(a=a, lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# OSQP wrapper (modes C2, C3)
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    u: np.ndarray
    ok: bool                      # solver returned a usable solution
    status: str
    solve_time: float             # wall time of update+solve [s]
    iters: int
    active: np.ndarray            # (7,) bool, |dual| > tol or margin ~ 0
    margins: np.ndarray           # (7,) post-solve margins
    gamma: float = np.nan         # scalar modes only
    infeasible: bool = False


class PassivityQP:
    """Warm-started sparse OSQP problem with the fixed 7 x 2 row layout.

    objective="track":    min 1/2 (u - u_nom)^T W (u - u_nom) + eps ||u||^2
    objective="min_norm": min 1/2 ||u||^2   (the C4 safety anchor)
    """

    def __init__(self, cfg: PassivationConfig, objective: str = "track"):
        self.cfg = cfg
        self.objective = objective
        if objective == "track":
            w = np.array([cfg.w_t, cfg.w_n])
        elif objective == "min_norm":
            w = np.zeros(2)
        else:
            raise ValueError(f"unknown objective {objective!r}")
        self._w = w
        diag = w + 2.0 * cfg.eps_reg if objective == "track" else np.ones(2)
        p_mat = sparse.csc_matrix(np.diag(diag))
        self._a_pattern = sparse.csc_matrix(np.ones((N_ROWS, 2)))
        self._prob = osqp.OSQP()
        self._prob.setup(
            p_mat, np.zeros(2), self._a_pattern,
            np.full(N_ROWS, -_INF), np.full(N_ROWS, _INF),
            # polish=False: OSQP 1.x prints a status line from C code even
            # with verbose=False; eps 1e-9 without polishing is accurate
            # enough for this 2-variable problem (verified in tests).
            verbose=False, eps_abs=1e-9, eps_rel=1e-9,
            max_iter=20000, polishing=False,
        )

    @staticmethod
    def _row_scaled(rows: ConstraintRows) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray]:
        """Exact per-row normalization to unit coefficient norm.

        The force-power rows have coefficients ~ |F| * dt / m_eff (1e-5
        scale), which stalls OSQP against tight tolerances; dividing a row
        and its bounds by ||a_i|| changes nothing mathematically. Rows whose
        coefficients vanish (port force -> 0) are dropped when they are
        vacuously satisfiable at the origin; a vanishing row with lo > 0
        is kept as-is so genuine infeasibility is still reported.
        """
        a = rows.a.copy()
        lo = rows.lo.copy()
        hi = rows.hi.copy()
        for i in range(N_ROWS):
            n = float(np.linalg.norm(a[i]))
            if n > 1e-9:
                a[i] /= n
                if lo[i] > -_INF:
                    lo[i] /= n
                if hi[i] < _INF:
                    hi[i] /= n
            elif lo[i] <= 1e-9 and hi[i] >= -1e-9:
                lo[i] = -_INF
                hi[i] = _INF
        return a, lo, hi

    def solve(self, u_nom: np.ndarray, rows: ConstraintRows) -> SolveResult:
        t0 = time.perf_counter()
        a_s, lo_s, hi_s = self._row_scaled(rows)
        self._prob.update(
            q=-(self._w * u_nom),
            Ax=a_s.flatten(order="F"),
            l=lo_s, u=hi_s,
        )
        res = self._prob.solve(raise_error=False)
        wall = time.perf_counter() - t0

        status = res.info.status
        ok = status in ("solved", "solved inaccurate")
        if ok and res.x is not None and np.all(np.isfinite(res.x)):
            u = np.asarray(res.x, dtype=float)
            duals = np.asarray(res.y, dtype=float)
        else:
            ok = False
            u = u_nom.copy()
            duals = np.zeros(N_ROWS)
        margins = rows.margins(u)
        active = (np.abs(duals) > 1e-6) | (margins < 1e-6)
        return SolveResult(
            u=u, ok=ok, status=status, solve_time=wall,
            iters=int(res.info.iter), active=active, margins=margins,
            infeasible=not ok,
        )


# ---------------------------------------------------------------------------
# Scalar (gamma) solver (modes C1, C4)
# ---------------------------------------------------------------------------

def _gamma_interval(anchor: np.ndarray, direction: np.ndarray,
                    rows: ConstraintRows, tol: float = 1e-9):
    """Feasible gamma interval for u = anchor + gamma * direction.

    Returns (g_lo, g_hi) intersected with [0, 1]; g_lo > g_hi means empty.
    ``tol`` absorbs solver-level constraint residuals at the anchor.
    """
    g_lo, g_hi = 0.0, 1.0
    coeff = rows.a @ direction        # (7,)
    offset = rows.a @ anchor
    for i in range(N_ROWS):
        c = coeff[i]
        lo_i = rows.lo[i] - offset[i] - tol
        hi_i = rows.hi[i] - offset[i] + tol
        if rows.lo[i] <= -_INF and rows.hi[i] >= _INF:
            continue
        if rows.lo[i] <= -_INF:
            lo_i = -np.inf
        if rows.hi[i] >= _INF:
            hi_i = np.inf
        if abs(c) < 1e-14:
            if lo_i > 0.0 or hi_i < 0.0:
                return 1.0, 0.0       # infeasible: 0 outside [lo, hi]
            continue
        if c > 0.0:
            i_lo = lo_i / c if np.isfinite(lo_i) else -np.inf
            i_hi = hi_i / c if np.isfinite(hi_i) else np.inf
        else:
            i_lo = hi_i / c if np.isfinite(hi_i) else -np.inf
            i_hi = lo_i / c if np.isfinite(lo_i) else np.inf
        g_lo = max(g_lo, i_lo)
        g_hi = min(g_hi, i_hi)
    return g_lo, g_hi


def solve_scalar_gamma(u_nom: np.ndarray, rows: ConstraintRows) -> SolveResult:
    """LEGACY origin-ray scaling: min (gamma - 1)^2 s.t. rows at
    u = gamma u_nom, gamma in [0, 1].

    The ray through the origin need not intersect the feasible set (e.g.
    under torque-rate or oblique human constraints), so this can be
    infeasible; it is kept only as a diagnostic baseline. Uses the same
    one-step power prediction as the QP.
    """
    t0 = time.perf_counter()
    g_lo, g_hi = _gamma_interval(np.zeros(2), u_nom, rows, tol=0.0)
    wall = time.perf_counter() - t0
    if g_lo > g_hi + 1e-12:
        margins = rows.margins(np.zeros(2))
        return SolveResult(
            u=np.zeros(2), ok=False, status="gamma infeasible",
            solve_time=wall, iters=0, active=np.zeros(N_ROWS, bool),
            margins=margins, gamma=np.nan, infeasible=True,
        )
    gamma = float(np.clip(1.0, g_lo, g_hi))
    u = gamma * u_nom
    margins = rows.margins(u)
    active = margins < 1e-8
    return SolveResult(
        u=u, ok=True, status="solved", solve_time=wall, iters=1,
        active=active, margins=margins, gamma=gamma,
    )


def solve_safe_anchor_scalar(u_nom: np.ndarray, rows: ConstraintRows,
                             anchor_qp: PassivityQP) -> SolveResult:
    """C4 safe-anchor scalar: 1-D movement from a feasible anchor to nominal.

    1. u_safe = argmin 1/2 ||u||^2 s.t. the same robust rows (OSQP).
    2. u(gamma) = u_safe + gamma (u_nom - u_safe), gamma in [0, 1].
    3. Pick the largest feasible gamma (the point on the line closest to
       nominal). gamma = 0 returns the anchor itself, so the mode is
       feasible whenever the safety-anchor QP is feasible.
    """
    t0 = time.perf_counter()
    sol_a = anchor_qp.solve(np.zeros(2), rows)
    if not sol_a.ok:
        wall = time.perf_counter() - t0
        return SolveResult(
            u=np.zeros(2), ok=False, status=f"anchor {sol_a.status}",
            solve_time=wall, iters=sol_a.iters,
            active=np.zeros(N_ROWS, bool), margins=rows.margins(np.zeros(2)),
            gamma=np.nan, infeasible=True,
        )
    u_safe = sol_a.u
    g_lo, g_hi = _gamma_interval(u_safe, u_nom - u_safe, rows, tol=1e-9)
    gamma = float(np.clip(g_hi, 0.0, 1.0))   # 0 is feasible by construction
    u = u_safe + gamma * (u_nom - u_safe)
    wall = time.perf_counter() - t0
    margins = rows.margins(u)
    active = margins < 1e-6
    return SolveResult(
        u=u, ok=True, status="solved", solve_time=wall,
        iters=sol_a.iters, active=active, margins=margins, gamma=gamma,
    )


# ---------------------------------------------------------------------------
# Per-step runtime for modes C0-C4
# ---------------------------------------------------------------------------

_SP_SCALAR_KEYS = [
    "sp_gamma", "sp_p_h_pred", "sp_p_r_pred", "sp_p_delta_pred",
    "sp_p_meas_pred", "sp_p_h_act", "sp_p_r_act", "sp_p_delta_act",
    "sp_p_meas_act", "sp_e_h", "sp_e_r", "sp_e_w", "sp_e_h_raw",
    "sp_e_r_raw", "sp_e_w_raw", "sp_e_h_below", "sp_e_r_below",
    "sp_e_w_below", "sp_solve_time", "sp_iters", "sp_status_ok",
    "sp_infeasible", "sp_emergency", "sp_active_any", "sp_pred_err",
    # Robust prediction-error-bound bookkeeping.
    "sp_bound_util",          # ||e_v|| / e_v_bound
    "sp_bound_violation",     # bool: measured error exceeded the bound
    "sp_ep_h", "sp_ep_r", "sp_ep_w",          # ||F_i|| e_v_bound [W]
    "sp_p_h_rlb", "sp_p_r_rlb", "sp_p_w_rlb",  # robust lower power at u
    "sp_margin_eh_rob", "sp_margin_er_rob", "sp_margin_ew_rob",
    "sp_safety_clipped",      # post-QP safety clip changed tau (emergency
                              # path only; must never fire on feasible sols)
    # Reference-governor bookkeeping.
    "sp_gov_active", "sp_gov_mode", "sp_gov_anchor",
    "sp_x_ref_original",
]
_SP_VEC_KEYS = {
    "sp_u_nom": 2, "sp_u": 2, "sp_tau_nom": 2, "sp_tau": 2,
    "sp_f_hat": 3, "sp_f_r": 3, "sp_f_delta": 3,
    "sp_margins": N_ROWS, "sp_active": N_ROWS,
    "sp_v_pred": 2, "sp_v_act": 2, "sp_v_pred_err": 2,
}

#: Ledgers whose floor each mode certifies (used for acceptance counting).
CERTIFIED_LEDGERS = {
    "C0_nominal": (),
    "C1_whole_port_qp": ("w",),
    "C1_legacy_whole_port_origin_scalar": ("w",),
    "C2_residual_qp": ("r",),
    "C3_dual_ledger_qp": ("h", "r"),
    "C4_dual_ledger_safe_scalar": ("h", "r"),
    "C4_legacy_origin_scalar": ("h", "r"),
}


class PassivationRuntime:
    """Wraps the nominal controller with the selective-passivation layer.

    Call ``control_step`` before ``mj_step`` (returns the applied torque)
    and ``post_step`` right after ``mj_step`` (certified ledger updates with
    the actual next velocity and the held force sample, plus logging).
    """

    def __init__(self, cfg: PassivationConfig, ctl_cfg: ControllerConfig,
                 handles: ModelHandles, dt: float):
        if cfg.mode not in CERTIFIED_LEDGERS:
            raise ValueError(f"unknown passivation mode {cfg.mode!r}")
        if cfg.k_cbf > 0.0:
            assert 0.0 < cfg.k_cbf * dt <= 1.0, (
                f"k_cbf * dt = {cfg.k_cbf * dt} must lie in (0, 1] for the "
                "discrete CBF guarantee")
        self.cfg = cfg
        self.ctl_cfg = ctl_cfg
        self.handles = handles
        self.dt = dt
        self.predictor = TaskContactPredictor(cfg.predictor, cfg.mu_hat,
                                              cfg.v_s)
        self.ledger_h = EnergyLedger(cfg.human, "human")
        self.ledger_r = EnergyLedger(cfg.residual, "residual")
        self.ledger_w = EnergyLedger(cfg.whole, "whole")
        self.qp = (PassivityQP(cfg, objective="track")
                   if cfg.mode in ("C1_whole_port_qp", "C2_residual_qp",
                                   "C3_dual_ledger_qp")
                   else None)
        self.anchor_qp = (PassivityQP(cfg, objective="min_norm")
                          if cfg.mode == "C4_dual_ledger_safe_scalar"
                          else None)
        self.tau_prev: np.ndarray | None = None
        self.n_infeasible = 0
        self.n_emergency = 0
        self.n_bound_violations = 0
        self.n_floor_violations = 0     # certified ledgers only
        self.n_safety_clipped = 0
        # Human-energy CBF row (ROW_CBF_H) active status from the PREVIOUS QP
        # step, fed to the reference governor's latch. Using the previous
        # step avoids an algebraic loop: the governor shapes u_nom, which the
        # QP then filters, so its own CBF status is only known afterwards.
        self._prev_cbf_h_active = False
        self.log: dict[str, list] = {k: [] for k in
                                     list(_SP_SCALAR_KEYS) + list(_SP_VEC_KEYS)}
        self._pending: dict | None = None

    # ------------------------------------------------------------------
    def control_step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        contact: ContactResult,
        f_h: np.ndarray,
        controller: SlidingForceController,
    ) -> tuple[ControlOutput, np.ndarray]:
        cfg = self.cfg
        jp = site_jacobian(model, data, self.handles.ee_site)
        v_ee = jp @ data.qvel

        out = controller.update(
            data, contact, integrate=False,
            f_h_norm=float(np.linalg.norm(f_h)),
            cbf_h_active_prev=self._prev_cbf_h_active,
        )
        u_nom = B_TN.T @ out.f_cmd

        f_hat = self.predictor.predict(contact.f_task, v_ee)
        dec = decompose(contact.f_task, f_hat, f_h)
        pred = compute_affine_prediction(model, data, jp,
                                         self.ctl_cfg.dq_joint, dec.f_meas)
        f_h_tn = B_TN.T @ dec.f_h
        f_r_tn = B_TN.T @ dec.f_r
        f_meas_tn = B_TN.T @ dec.f_meas

        # Robust power error bounds ep_i = ||F_i|| * e_v_bound.
        ep_h = float(np.linalg.norm(f_h_tn)) * cfg.e_v_bound
        ep_r = float(np.linalg.norm(f_r_tn)) * cfg.e_v_bound
        ep_w = float(np.linalg.norm(f_meas_tn)) * cfg.e_v_bound

        rows = build_constraint_rows(
            cfg, self.ctl_cfg, self.dt, pred, f_h_tn, f_r_tn, f_meas_tn,
            self.ledger_h.e, self.ledger_r.e, self.ledger_w.e,
            self.tau_prev, cfg.mode, ep_h=ep_h, ep_r=ep_r, ep_w=ep_w,
        )

        emergency = False
        safety_clipped = False
        if cfg.mode == "C0_nominal":
            u = u_nom.copy()
            sol = SolveResult(
                u=u, ok=True, status="nominal", solve_time=0.0, iters=0,
                active=np.zeros(N_ROWS, bool), margins=rows.margins(u),
            )
            tau = out.tau.copy()      # nominal path incl. its own limits
        else:
            if self.qp is not None:
                sol = self.qp.solve(u_nom, rows)
            elif self.anchor_qp is not None:
                sol = solve_safe_anchor_scalar(u_nom, rows, self.anchor_qp)
            else:
                sol = solve_scalar_gamma(u_nom, rows)
            if sol.infeasible:
                self.n_infeasible += 1
                self.n_emergency += 1
                emergency = True
                self._report_infeasible(data, rows, u_nom, sol)
                u = self._emergency_command(pred, u_nom)
            else:
                u = sol.u
            tau = pred.tau(u)
            # Non-certified safety clip: must be inactive for feasible QP
            # solutions (bounds are inside the QP); protects only the
            # emergency path and numerical corner cases. Any activation on
            # a feasible solution is counted and fails acceptance.
            tau_box = np.clip(tau, -self.ctl_cfg.tau_limit,
                              self.ctl_cfg.tau_limit)
            if self.tau_prev is not None:
                dmax = self.ctl_cfg.tau_rate_limit * self.dt
                tau_box = np.clip(tau_box, self.tau_prev - dmax,
                                  self.tau_prev + dmax)
            # 1e-7 N·m: solutions ON a binding torque/rate boundary get
            # trimmed by fp dust (~1e-15); only material clips count.
            safety_clipped = bool(np.any(np.abs(tau_box - tau) > 1e-7))
            if safety_clipped and not emergency:
                self.n_safety_clipped += 1
            tau = tau_box

        self.tau_prev = tau.copy()
        # Latch the human-energy CBF row's active status for the NEXT step's
        # governor decision (previous-step feedback, no algebraic loop).
        # Only meaningful for modes that carry the human rows.
        self._prev_cbf_h_active = bool(
            cfg.mode in MODES_HUMAN_ROWS and sol.active[ROW_CBF_H])
        # PI anti-windup by integrator freezing: skip the force-error
        # integration whenever the filtered normal command deviates from
        # the nominal one.
        normal_modified = abs(u[1] - u_nom[1]) > cfg.activation_tol
        controller.finish_step(out, frozen=normal_modified, tau_applied=tau)

        # The bound check compares against the prediction for the torque
        # ACTUALLY applied (identical to A u + b on unclipped solutions).
        v_pred = pred.v_next_from_tau(tau)
        self._pending = {
            "dec": dec, "pred": pred, "u": u, "u_nom": u_nom, "sol": sol,
            "tau": tau.copy(), "tau_nom": pred.tau(u_nom),
            "emergency": emergency, "safety_clipped": safety_clipped,
            "ep_h": ep_h, "ep_r": ep_r, "ep_w": ep_w,
            "p_h_pred": float(f_h_tn @ v_pred),
            "p_r_pred": float(f_r_tn @ v_pred),
            "p_meas_pred": float(f_meas_tn @ v_pred),
            "v_pred": v_pred,
            "out": out,
        }
        return out, tau

    # ------------------------------------------------------------------
    def post_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Certified ledger updates + logging, after mj_step.

        Uses the SAME force sample the QP saw (held over the step) with the
        actual next EE velocity: p_actual = F_k^T v_{ee,k+1}.
        """
        pend = self._pending
        assert pend is not None, "post_step called before control_step"
        self._pending = None
        dec, pred, sol = pend["dec"], pend["pred"], pend["sol"]

        jp1 = site_jacobian(model, data, self.handles.ee_site)
        v_next = jp1 @ data.qvel
        p_h = float(dec.f_h @ v_next)
        p_r = float(dec.f_r @ v_next)
        p_delta = float(dec.f_delta @ v_next)
        p_meas = float(dec.f_meas @ v_next)

        upd_h = self.ledger_h.update(p_h, self.dt)
        upd_r = self.ledger_r.update(p_r, self.dt)
        upd_w = self.ledger_w.update(p_meas, self.dt)

        v_act_tn = B_TN.T @ v_next
        log = self.log
        log["sp_u_nom"].append(pend["u_nom"].copy())
        log["sp_u"].append(pend["u"].copy())
        log["sp_tau_nom"].append(pend["tau_nom"].copy())
        log["sp_tau"].append(pend["tau"].copy())
        log["sp_f_hat"].append(dec.f_hat.copy())
        log["sp_f_r"].append(dec.f_r.copy())
        log["sp_f_delta"].append(dec.f_delta.copy())
        log["sp_gamma"].append(sol.gamma)
        log["sp_p_h_pred"].append(pend["p_h_pred"])
        log["sp_p_r_pred"].append(pend["p_r_pred"])
        log["sp_p_delta_pred"].append(pend["p_r_pred"] - pend["p_h_pred"])
        log["sp_p_meas_pred"].append(pend["p_meas_pred"])
        log["sp_p_h_act"].append(p_h)
        log["sp_p_r_act"].append(p_r)
        log["sp_p_delta_act"].append(p_delta)
        log["sp_p_meas_act"].append(p_meas)
        log["sp_e_h"].append(upd_h.e_next)
        log["sp_e_r"].append(upd_r.e_next)
        log["sp_e_w"].append(upd_w.e_next)
        log["sp_e_h_raw"].append(upd_h.e_raw)
        log["sp_e_r_raw"].append(upd_r.e_raw)
        log["sp_e_w_raw"].append(upd_w.e_raw)
        log["sp_e_h_below"].append(upd_h.below_min)
        log["sp_e_r_below"].append(upd_r.below_min)
        log["sp_e_w_below"].append(upd_w.below_min)
        log["sp_margins"].append(np.minimum(sol.margins, _INF))
        log["sp_active"].append(sol.active.astype(float))
        log["sp_solve_time"].append(sol.solve_time)
        log["sp_iters"].append(sol.iters)
        log["sp_status_ok"].append(sol.ok)
        log["sp_infeasible"].append(sol.infeasible)
        log["sp_emergency"].append(pend["emergency"])
        log["sp_active_any"].append(
            float(np.max(np.abs(pend["u"] - pend["u_nom"])))
            > self.cfg.activation_tol)
        log["sp_v_pred"].append(pend["v_pred"].copy())
        log["sp_v_act"].append(v_act_tn.copy())
        err_vec = v_act_tn - pend["v_pred"]
        err = float(np.linalg.norm(err_vec))
        log["sp_v_pred_err"].append(err_vec.copy())
        log["sp_pred_err"].append(err)

        # --- robust certificate bookkeeping ---
        cfg = self.cfg
        bound_violation = err > cfg.e_v_bound + 1e-12
        if bound_violation:
            self.n_bound_violations += 1
        log["sp_bound_util"].append(err / cfg.e_v_bound)
        log["sp_bound_violation"].append(bound_violation)
        log["sp_ep_h"].append(pend["ep_h"])
        log["sp_ep_r"].append(pend["ep_r"])
        log["sp_ep_w"].append(pend["ep_w"])
        p_h_rlb = pend["p_h_pred"] - pend["ep_h"]
        p_r_rlb = pend["p_r_pred"] - pend["ep_r"]
        p_w_rlb = pend["p_meas_pred"] - pend["ep_w"]
        log["sp_p_h_rlb"].append(p_h_rlb)
        log["sp_p_r_rlb"].append(p_r_rlb)
        log["sp_p_w_rlb"].append(p_w_rlb)
        # Robust energy margins E + dt p_lower - E_safe (pre-update ledgers).
        log["sp_margin_eh_rob"].append(
            upd_h.e_prev + self.dt * p_h_rlb - cfg.e_safe_h)
        log["sp_margin_er_rob"].append(
            upd_r.e_prev + self.dt * p_r_rlb - cfg.e_safe_r)
        log["sp_margin_ew_rob"].append(
            upd_w.e_prev + self.dt * p_w_rlb - cfg.e_safe_w)
        log["sp_safety_clipped"].append(pend["safety_clipped"])

        # Certified-ledger floor violations (exact, fp tolerance only).
        certified = CERTIFIED_LEDGERS[cfg.mode]
        for name, upd in (("h", upd_h), ("r", upd_r), ("w", upd_w)):
            if name in certified and upd.e_raw < {
                    "h": cfg.human, "r": cfg.residual,
                    "w": cfg.whole}[name].e_min - 1e-9:
                self.n_floor_violations += 1

        out = pend["out"]
        log["sp_gov_active"].append(out.governor_active)
        log["sp_gov_mode"].append(out.governor_mode)
        log["sp_gov_anchor"].append(out.governor_anchor)
        log["sp_x_ref_original"].append(out.x_ref_original)

    # ------------------------------------------------------------------
    def _emergency_command(self, pred: AffinePrediction,
                           u_nom: np.ndarray) -> np.ndarray:
        """Bounded infeasibility fallback: damp the tangential motion while
        PRESERVING the (bounded) nominal normal command.

        Dropping the normal push (the iteration-1 pure-damping fallback)
        unloaded the contact, which bounced the pad and made every
        subsequent constraint set infeasible; keeping the normal channel on
        the nominal command holds the contact so a failed controller
        degrades into 'stop sliding, keep pressing' instead of a bounce
        cascade. Still bounded, still counted, still not certified.
        """
        u_max = self.cfg.emergency_u_max
        return np.array([
            float(np.clip(-self.cfg.emergency_damping * pred.v_tn[0],
                          -u_max, u_max)),
            float(np.clip(u_nom[1], -u_max, u_max)),
        ])

    _MAX_INFEASIBLE_PRINTS = 10

    def _report_infeasible(self, data: mujoco.MjData, rows: ConstraintRows,
                           u_nom: np.ndarray, sol: SolveResult) -> None:
        # Full evidence (state, margins, ledgers) is preserved in the log
        # arrays for EVERY step; stdout reporting is capped to stay legible.
        if self.n_infeasible > self._MAX_INFEASIBLE_PRINTS:
            if self.n_infeasible == self._MAX_INFEASIBLE_PRINTS + 1:
                print("[passivation] further infeasible-step reports "
                      "suppressed (full evidence remains in the log)")
            return
        m_nom = rows.margins(u_nom)
        finite = [f"{ROW_NAMES[i]}={m_nom[i]:+.4g}" for i in range(N_ROWS)
                  if np.isfinite(m_nom[i])]
        print(f"[passivation] INFEASIBLE t={data.time:.3f}s mode={self.cfg.mode} "
              f"status={sol.status!r} E_H={self.ledger_h.e:.5f} "
              f"E_R={self.ledger_r.e:.5f} E_W={self.ledger_w.e:.5f} "
              f"qpos={np.array2string(data.qpos, precision=4)} "
              f"qvel={np.array2string(data.qvel, precision=4)} "
              f"margins@u_nom: {' '.join(finite)}")

    # ------------------------------------------------------------------
    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v) for k, v in self.log.items()}
        out["meta_e_v_bound"] = np.array(self.cfg.e_v_bound)
        out["meta_n_bound_violations"] = np.array(self.n_bound_violations)
        out["meta_n_floor_violations"] = np.array(self.n_floor_violations)
        out["meta_n_safety_clipped"] = np.array(self.n_safety_clipped)
        return out
