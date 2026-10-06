"""Seven-variable absolute joint-torque selective-passivation QP."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp

from .config import MODES, ControllerConfig, PassivationConfig
from .ports import ContactPort, PandaHandles, SpatialPort, site_jacobian

# A solved candidate is rejected only beyond this explicitly logged numerical
# feasibility tolerance (in each row's original physical units).
FEASIBILITY_TOLERANCE = 5e-6


def mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    value = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, value, data.qM)
    return value


@dataclass
class DynamicsPrediction:
    b: np.ndarray                 # qdot+ = b + B tau_arm
    B: np.ndarray
    qvel_now: np.ndarray
    dt: float

    def qvel_next(self, tau: np.ndarray) -> np.ndarray:
        return self.b + self.B @ tau


def affine_dynamics(model: mujoco.MjModel, data: mujoco.MjData,
                    h: PandaHandles, tau_external: np.ndarray,
                    mass_scale: float = 1.0) -> DynamicsPrediction:
    """Explicit-Euler held-wrench prediction with all passive/bias terms."""
    dt = float(model.opt.timestep)
    if mass_scale <= 0:
        raise ValueError("mass_scale must be positive")
    # Values other than one deliberately perturb the controller's robot
    # inertia model while leaving the MuJoCo plant untouched.
    minv = np.linalg.inv(mass_scale * mass_matrix(model, data))
    selection = np.zeros((model.nv, 7))
    selection[h.dof_ids, np.arange(7)] = 1.0
    free = data.qvel + dt * minv @ (
        data.qfrc_passive - data.qfrc_bias + np.asarray(tau_external))
    return DynamicsPrediction(free, dt * minv @ selection,
                              data.qvel.copy(), dt)


@dataclass
class PortDecomposition:
    task_actual: np.ndarray
    task_predicted: np.ndarray
    human: np.ndarray
    external: np.ndarray
    residual: np.ndarray
    mismatch: np.ndarray
    predicted_force: np.ndarray


def predict_task_port(model: mujoco.MjModel, data: mujoco.MjData,
                      h: PandaHandles, contact: ContactPort,
                      human: SpatialPort, cfg: PassivationConfig) -> PortDecomposition:
    tau_task = contact.generalized_force
    if cfg.predictor == "oracle":
        tau_hat = tau_task.copy()
        f_hat = contact.net_force.copy()
    elif cfg.predictor == "friction":
        jp, _ = site_jacobian(model, data, h.tool_site)
        v = jp @ data.qvel
        fn = max(0.0, contact.normal_force)  # task channel only; never task+human
        f_hat = np.array([
            -cfg.mu_hat * fn * np.tanh(v[0] / cfg.friction_v_s),
            -cfg.mu_hat * fn * np.tanh(v[1] / cfg.friction_v_s),
            cfg.normal_scale * fn,
        ])
        tau_hat = jp.T @ f_hat
    else:
        raise ValueError(f"unknown predictor {cfg.predictor!r}")
    tau_ext = tau_task + human.generalized_force
    tau_res = tau_ext - tau_hat
    return PortDecomposition(tau_task.copy(), tau_hat,
                             human.generalized_force.copy(), tau_ext,
                             tau_res, tau_res - human.generalized_force,
                             f_hat)


@dataclass
class FilterResult:
    torque: np.ndarray
    torque_nominal: np.ndarray
    qvel_predicted: np.ndarray
    status: str
    fallback: bool
    active: bool
    solve_time: float
    minimum_margin: float
    energy_row_active: bool
    torque_limited: bool
    rate_limited: bool
    solver_status_val: int
    candidate_minimum_margin: float


def _bounded_nominal(tau: np.ndarray, tau_prev: np.ndarray,
                     lo: np.ndarray, hi: np.ndarray,
                     rate: np.ndarray, dt: float):
    lower = np.maximum(lo, tau_prev - rate * dt)
    upper = np.minimum(hi, tau_prev + rate * dt)
    return np.clip(tau, lower, upper), lower, upper


class TorquePassivityFilter:
    """QP decision is the complete 7-vector applied arm torque."""

    def __init__(self, model: mujoco.MjModel, h: PandaHandles,
                 controller_cfg: ControllerConfig, cfg: PassivationConfig,
                 initial_torque: np.ndarray):
        if cfg.mode not in MODES:
            raise ValueError(f"unsupported mode {cfg.mode!r}")
        self.model, self.h, self.ccfg, self.cfg = model, h, controller_cfg, cfg
        self.prev = np.asarray(initial_torque, dtype=float).copy()
        self.tau_lo = model.actuator_ctrlrange[h.actuator_ids, 0].copy()
        self.tau_hi = model.actuator_ctrlrange[h.actuator_ids, 1].copy()
        self.rate = np.asarray(controller_cfg.torque_rate, dtype=float)

    def solve(self, data: mujoco.MjData, tau_nom: np.ndarray,
              pred: DynamicsPrediction, ports: PortDecomposition,
              energies: dict[str, float]) -> FilterResult:
        dt, cfg = pred.dt, self.cfg
        bounded, tau_lower, tau_upper = _bounded_nominal(
            tau_nom, self.prev, self.tau_lo, self.tau_hi, self.rate, dt)
        if cfg.mode == "C0_nominal":
            qnext = pred.qvel_next(bounded)
            out = FilterResult(
                bounded, tau_nom.copy(), qnext, "nominal_bounded", False,
                bool(np.linalg.norm(bounded - tau_nom) > cfg.activation_tol),
                0.0, np.inf, False,
                bool(np.any((bounded <= self.tau_lo + 1e-8) | (bounded >= self.tau_hi - 1e-8))),
                bool(np.any(np.abs(bounded - self.prev) >= self.rate * dt - 1e-8)), 0,
                np.inf)
            self.prev = bounded.copy()
            return out

        rows, lower, upper, protected_rows = [], [], [], []

        def add_lower(a, value, protected=False):
            rows.append(np.asarray(a)); lower.append(float(value)); upper.append(np.inf)
            protected_rows.append(protected)

        def add_box(a, lo, hi):
            rows.append(np.asarray(a)); lower.append(float(lo)); upper.append(float(hi))
            protected_rows.append(False)

        def port_lower_row(tau_port: np.ndarray, energy: float, e_min: float):
            a = tau_port @ pred.B
            constant = float(tau_port @ pred.b)
            uncertainty = np.linalg.norm(tau_port) * cfg.qvel_prediction_bound
            e_safe = e_min + cfg.eps_energy
            needed = max((e_safe - energy) / dt,
                         -cfg.k_cbf * (energy - e_safe))
            add_lower(a, needed + uncertainty - constant, protected=True)

        if cfg.mode == "C1_whole_port_qp":
            port_lower_row(ports.external, energies["whole"], cfg.whole.e_min)
        else:
            port_lower_row(ports.residual, energies["residual"], cfg.residual.e_min)
            if cfg.mode == "C3_dual_ledger_qp":
                port_lower_row(ports.human, energies["human"], cfg.human.e_min)
                # -p_H <= Pmax robustly: p_H >= -Pmax + uncertainty.
                a = ports.human @ pred.B
                const = float(ports.human @ pred.b)
                unc = np.linalg.norm(ports.human) * cfg.qvel_prediction_bound
                add_lower(a, -cfg.p_h_max + unc - const, protected=True)

        eye = np.eye(7)
        for i in range(7):
            add_box(eye[i], tau_lower[i], tau_upper[i])

        # Predicted arm velocity and position limits.
        qlo = self.model.jnt_range[self.h.joint_ids, 0] + self.ccfg.joint_margin
        qhi = self.model.jnt_range[self.h.joint_ids, 1] - self.ccfg.joint_margin
        q = data.qpos[self.h.qpos_ids]
        vmax = np.asarray(self.ccfg.qvel_limit)
        for i, dof in enumerate(self.h.dof_ids):
            add_box(pred.B[dof], -vmax[i] - pred.b[dof], vmax[i] - pred.b[dof])
            add_box(dt * pred.B[dof], qlo[i] - q[i] - dt * pred.b[dof],
                    qhi[i] - q[i] - dt * pred.b[dof])

        A = np.vstack(rows)
        l, u = np.asarray(lower), np.asarray(upper)
        # Row normalization is numerical scaling only: multiplying each
        # complete inequality by a positive scalar leaves its feasible set
        # unchanged. Margins below are always recomputed in original units.
        row_scale = 1.0 / np.maximum(np.linalg.norm(A, axis=1), 1e-3)
        A_solver = row_scale[:, None] * A
        l_solver, u_solver = row_scale * l, row_scale * u
        weights = np.asarray(cfg.w_tau, dtype=float)
        jp, jr = site_jacobian(self.model, data, self.h.tool_site)
        a_translation = jp @ pred.B
        a_orientation = jr @ pred.B
        task_weights = np.diag([cfg.w_task_tangent, cfg.w_task_tangent,
                                cfg.w_task_normal])
        hessian = (np.diag(weights)
                   + a_translation.T @ task_weights @ a_translation
                   + cfg.w_task_orientation * a_orientation.T @ a_orientation)
        hessian = 0.5 * (hessian + hessian.T)
        P = sp.csc_matrix(2.0 * hessian)
        qvec = -2.0 * hessian @ tau_nom
        solver = osqp.OSQP()
        started = time.perf_counter()
        solver.setup(P=P, q=qvec, A=sp.csc_matrix(A_solver), l=l_solver, u=u_solver,
                     verbose=False, polishing=False, eps_abs=1e-7, eps_rel=1e-7,
                     max_iter=10000, adaptive_rho=True)
        result = solver.solve(raise_error=False)
        elapsed = time.perf_counter() - started
        ok = result.info.status_val in (1, 2) and result.x is not None
        tau = np.asarray(result.x) if ok else None
        margins = np.minimum((A @ tau - l) if ok else -np.inf,
                             (u - A @ tau) if ok else -np.inf)
        candidate_minimum_margin = float(np.min(margins)) if ok else -np.inf
        if ok and np.min(margins) < -FEASIBILITY_TOLERANCE:
            ok = False
        if not ok:
            emergency = data.qfrc_bias[self.h.dof_ids] - cfg.emergency_damping * data.qvel[self.h.dof_ids]
            tau, _, _ = _bounded_nominal(emergency, self.prev, self.tau_lo,
                                         self.tau_hi, self.rate, dt)
            margins = np.minimum(A @ tau - l, u - A @ tau)
            status = f"fallback:{result.info.status}"
        else:
            status = result.info.status
        qnext = pred.qvel_next(tau)
        protected_active = any(protected_rows[i] and margins[i] < 2e-4
                               for i in range(len(rows)))
        out = FilterResult(
            tau.copy(), tau_nom.copy(), qnext, status, not ok,
            bool(np.linalg.norm(tau - tau_nom) > cfg.activation_tol), elapsed,
            float(np.min(margins)), protected_active,
            bool(np.any((tau <= self.tau_lo + 1e-5) | (tau >= self.tau_hi - 1e-5))),
            bool(np.any(np.abs(tau - self.prev) >= self.rate * dt - 1e-5)),
            int(result.info.status_val), candidate_minimum_margin)
        self.prev = tau.copy()
        return out
