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

Constraints (all hard; no hidden slack):
  * human ledger      E_H + dt p_H+(u)  >= E_H_min + eps_H
  * residual ledger   E_R + dt p_R+(u)  >= E_R_min + eps_R
  * CBF smoothing     -p_H+(u) <= k_cbf (E_H - E_H_min)   (and residual /
    whole-port analogue). Documented extra hard rows that make depletion
    approach the floor exponentially so the mandated one-step rows never
    collide with the hard torque-rate limit; for k_cbf < 1/dt they are the
    tighter constraint near the floor, and the mandated rows remain in the
    QP as the certificate.
  * human power       -p_H+(u) <= P_H_max
  * torque            -tau_max <= tau(u) <= tau_max
  * torque rate       |tau(u) - tau_prev| <= taudot_max dt

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

    def v_next(self, u: np.ndarray) -> np.ndarray:
        return self.A @ u + self.b

    def tau(self, u: np.ndarray) -> np.ndarray:
        return self.tau0 + self.G @ u


#: MuJoCo renamed the packed mass matrix ``qM`` -> ``M`` and changed the
#: ``mj_fullM`` signature at the same time:
#:   <= 3.3.x   data.qM      mj_fullM(model, dst, qM_packed)
#:   >= 3.10    data.M       mj_fullM(model, data, dst)
#: Resolved once at import so the 1 kHz call path stays branch-cheap and a
#: fresh install on a current MuJoCo does not die with an AttributeError.
_MJ_FULLM_TAKES_DATA = not hasattr(mujoco.MjData, "qM")


def mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Dense joint-space mass matrix M(q), across MuJoCo binding revisions."""
    m = np.zeros((model.nv, model.nv))
    if _MJ_FULLM_TAKES_DATA:
        mujoco.mj_fullM(model, data, m)
    else:
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
) -> ConstraintRows:
    """Build the 7 hard constraint rows for the given mode.

    Energy rows are expressed in POWER units (the one-step inequality
    divided by dt) for uniform scaling:
        p+(u) >= (E_min + eps - E) / dt.
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

    use_human = mode in ("C3_dual_ledger_qp", "C4_dual_ledger_scalar")
    use_residual = mode in ("C2_residual_qp", "C3_dual_ledger_qp",
                            "C4_dual_ledger_scalar")
    use_whole = mode == "C1_whole_port_scalar"

    # Required-power right-hand sides are clamped at 0: whenever the ledger
    # sits AT or marginally BELOW its floor (which can happen by accumulated
    # one-step prediction error, since the certified update uses the actual
    # next velocity), the constraint demands "no further extraction"
    # (p+ >= 0) rather than an instantaneous recharge, which would be
    # physically unsatisfiable through the port's own force (e.g. F_H -> 0
    # during release would make ANY u infeasible forever). For E >= E_min +
    # eps the clamp is inactive and the mandated one-step certificate is
    # unchanged. Raw ledger violations remain logged and counted.
    dp = cfg.delta_p
    if use_human:
        a[ROW_E_H] = ah
        lo[ROW_E_H] = min(0.0, (cfg.human.e_min + cfg.eps_h - e_h) / dt) - dp - bh
        a[ROW_P_H] = ah
        lo[ROW_P_H] = -cfg.p_h_max - bh
        if cfg.k_cbf > 0.0:
            a[ROW_CBF_H] = ah
            lo[ROW_CBF_H] = (min(0.0, -cfg.k_cbf * (e_h - cfg.human.e_min))
                             - dp - bh)
    if use_residual:
        a[ROW_E_R] = ar
        lo[ROW_E_R] = (min(0.0, (cfg.residual.e_min + cfg.eps_r - e_r) / dt)
                       - dp - br)
        if cfg.k_cbf > 0.0:
            a[ROW_CBF_R] = ar
            lo[ROW_CBF_R] = (min(0.0, -cfg.k_cbf * (e_r - cfg.residual.e_min))
                             - dp - br)
    if use_whole:
        # C1 reuses the E_H / CBF_H slots for its single whole-port ledger.
        a[ROW_E_H] = aw
        lo[ROW_E_H] = (min(0.0, (cfg.whole.e_min + cfg.eps_r - e_w) / dt)
                       - dp - bw)
        if cfg.k_cbf > 0.0:
            a[ROW_CBF_H] = aw
            lo[ROW_CBF_H] = (min(0.0, -cfg.k_cbf * (e_w - cfg.whole.e_min))
                             - dp - bw)

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
    """Warm-started sparse OSQP problem with the fixed 7 x 2 row layout."""

    def __init__(self, cfg: PassivationConfig):
        self.cfg = cfg
        w = np.array([cfg.w_t, cfg.w_n])
        self._w = w
        p_mat = sparse.csc_matrix(np.diag(w + 2.0 * cfg.eps_reg))
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

def solve_scalar_gamma(u_nom: np.ndarray, rows: ConstraintRows) -> SolveResult:
    """Solve min (gamma - 1)^2 s.t. the same rows at u = gamma u_nom,
    gamma in [0, 1].

    Each row lo_i <= (a_i . u_nom) gamma <= hi_i is an interval in gamma;
    the feasible set is the intersection. The objective
    (gamma-1)^2 u_nom^T W u_nom is minimized by the feasible gamma closest
    to 1. Uses the same one-step power prediction as the QP.
    """
    t0 = time.perf_counter()
    g_lo, g_hi = 0.0, 1.0
    coeff = rows.a @ u_nom            # (7,)
    for i in range(N_ROWS):
        c = coeff[i]
        lo_i, hi_i = rows.lo[i], rows.hi[i]
        if lo_i <= -_INF and hi_i >= _INF:
            continue
        if abs(c) < 1e-14:
            if lo_i > 0.0 or hi_i < 0.0:
                g_lo, g_hi = 1.0, 0.0   # infeasible: 0 outside [lo, hi]
                break
            continue
        if c > 0.0:
            i_lo = -np.inf if lo_i <= -_INF else lo_i / c
            i_hi = np.inf if hi_i >= _INF else hi_i / c
        else:
            i_lo = -np.inf if hi_i >= _INF else hi_i / c
            i_hi = np.inf if lo_i <= -_INF else lo_i / c
        g_lo = max(g_lo, i_lo)
        g_hi = min(g_hi, i_hi)

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
]
_SP_VEC_KEYS = {
    "sp_u_nom": 2, "sp_u": 2, "sp_tau_nom": 2, "sp_tau": 2,
    "sp_f_hat": 3, "sp_f_r": 3, "sp_f_delta": 3,
    "sp_margins": N_ROWS, "sp_active": N_ROWS,
    "sp_v_pred": 2, "sp_v_act": 2,
}


class PassivationRuntime:
    """Wraps the nominal controller with the selective-passivation layer.

    Call ``control_step`` before ``mj_step`` (returns the applied torque)
    and ``post_step`` right after ``mj_step`` (certified ledger updates with
    the actual next velocity and the held force sample, plus logging).
    """

    def __init__(self, cfg: PassivationConfig, ctl_cfg: ControllerConfig,
                 handles: ModelHandles, dt: float):
        self.cfg = cfg
        self.ctl_cfg = ctl_cfg
        self.handles = handles
        self.dt = dt
        self.predictor = TaskContactPredictor(cfg.predictor, cfg.mu_hat,
                                              cfg.v_s)
        self.ledger_h = EnergyLedger(cfg.human, "human")
        self.ledger_r = EnergyLedger(cfg.residual, "residual")
        self.ledger_w = EnergyLedger(cfg.whole, "whole")
        self.qp = (PassivityQP(cfg)
                   if cfg.mode in ("C2_residual_qp", "C3_dual_ledger_qp")
                   else None)
        self.tau_prev: np.ndarray | None = None
        self.n_infeasible = 0
        self.n_emergency = 0
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

        out = controller.update(data, contact, integrate=False)
        u_nom = B_TN.T @ out.f_cmd

        f_hat = self.predictor.predict(contact.f_task, v_ee)
        dec = decompose(contact.f_task, f_hat, f_h)
        pred = compute_affine_prediction(model, data, jp,
                                         self.ctl_cfg.dq_joint, dec.f_meas)
        f_h_tn = B_TN.T @ dec.f_h
        f_r_tn = B_TN.T @ dec.f_r
        f_meas_tn = B_TN.T @ dec.f_meas

        rows = build_constraint_rows(
            cfg, self.ctl_cfg, self.dt, pred, f_h_tn, f_r_tn, f_meas_tn,
            self.ledger_h.e, self.ledger_r.e, self.ledger_w.e,
            self.tau_prev, cfg.mode,
        )

        emergency = False
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
            else:
                sol = solve_scalar_gamma(u_nom, rows)
            if sol.infeasible:
                self.n_infeasible += 1
                self.n_emergency += 1
                emergency = True
                self._report_infeasible(data, rows, u_nom, sol)
                u = self._emergency_command(pred)
            else:
                u = sol.u
            tau = pred.tau(u)
            # Non-certified safety clip: must be inactive for feasible QP
            # solutions (bounds are inside the QP); protects only the
            # emergency path and numerical corner cases.
            tau_box = np.clip(tau, -self.ctl_cfg.tau_limit,
                              self.ctl_cfg.tau_limit)
            if self.tau_prev is not None:
                dmax = self.ctl_cfg.tau_rate_limit * self.dt
                tau_box = np.clip(tau_box, self.tau_prev - dmax,
                                  self.tau_prev + dmax)
            tau = tau_box

        self.tau_prev = tau.copy()

        # PI anti-windup by integrator freezing: skip the force-error
        # integration whenever the filtered normal command deviates from
        # the nominal one.
        normal_modified = abs(u[1] - u_nom[1]) > cfg.activation_tol
        controller.finish_step(out, frozen=normal_modified, tau_applied=tau)

        self._pending = {
            "dec": dec, "pred": pred, "u": u, "u_nom": u_nom, "sol": sol,
            "tau": tau.copy(), "tau_nom": pred.tau(u_nom),
            "emergency": emergency,
            "p_h_pred": float(f_h_tn @ pred.v_next(u)),
            "p_r_pred": float(f_r_tn @ pred.v_next(u)),
            "p_meas_pred": float(f_meas_tn @ pred.v_next(u)),
            "v_pred": pred.v_next(u),
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
        log["sp_pred_err"].append(
            float(np.linalg.norm(pend["v_pred"] - v_act_tn)))

    # ------------------------------------------------------------------
    def _emergency_command(self, pred: AffinePrediction) -> np.ndarray:
        """Bounded damping command in the t-n plane (infeasibility fallback)."""
        u = -self.cfg.emergency_damping * pred.v_tn
        return np.clip(u, -self.cfg.emergency_u_max, self.cfg.emergency_u_max)

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
        return {k: np.asarray(v) for k, v in self.log.items()}
