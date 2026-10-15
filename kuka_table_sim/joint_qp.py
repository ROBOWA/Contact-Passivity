"""Seven-variable hard-energy QP and nonlinear acceptance, all in joint torque.

The objective is ALWAYS relative to the original nominal command, including
shifted-center retries. Both oracle accounts are retained, with identical work.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from time import perf_counter

import numpy as np
import osqp
from scipy import sparse


@dataclass(frozen=True)
class QPConfig:
    eta: float = 0.02                 # central difference, N m
    rho: tuple = (80., 80., 60., 60., 40., 20., 20.)
    rate: tuple = (80000., 80000., 60000., 60000., 40000., 20000., 20000.)  # N m/s, simulation fixture
    torque_scale: tuple = (20., 20., 10., 10., 5., 2., 2.)
    epsilon_prediction: float = 1e-11 # deterministic held-force replay resolution, J
    epsilon_linearization: float = 2.5e-5  # J at reference force; calibrated scan envelope
    margin_force_reference: float = 5.0    # N; margin scales with measured force magnitude
    energy_tolerance: float = 1e-10   # J
    torque_tolerance: float = 1e-7    # N m
    gradient_zero: float = 1e-14      # J / N m (absolute)
    max_retries: int = 3
    solver_max_iter: int = 20000
    task_weight: float = 0.0
    task_scales: tuple = (1., .01, .01, .01, .1, .1, .1)  # N, m/s, rad/s

    def __post_init__(self):
        for name in ("rho", "rate", "torque_scale", "task_scales"):
            a = np.asarray(getattr(self, name))
            if a.shape != (7,) or np.any(a <= 0) or not np.all(np.isfinite(a)):
                raise ValueError(f"invalid {name}")
        if self.max_retries < 0 or self.solver_max_iter < 1 or self.eta <= 0 or self.task_weight < 0 or self.margin_force_reference <= 0:
            raise ValueError("invalid QP configuration")
        for name in ("epsilon_prediction", "epsilon_linearization", "energy_tolerance",
                     "torque_tolerance", "gradient_zero"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"invalid {name}")


class FilterFailure(RuntimeError):
    def __init__(self, reason, diagnostics=None):
        super().__init__(reason)
        self.reason, self.diagnostics = reason, diagnostics or {}


@dataclass
class QPSolution:
    torque: np.ndarray
    affine_work: np.ndarray
    energy_residual: np.ndarray
    box_lower: np.ndarray
    box_upper: np.ndarray
    status: str
    seconds: float


def active_set_recovery(H, g, lower, lo, hi, cfg):
    """Small deterministic fallback for one halfspace and a seven-dimensional
    box. Enumerate bound activity; solve free-variable KKT equations. No slack,
    changed bounds or accepted unconverged OSQP iterate. Used only on solver
    failure, then subjected to the same original-unit primal checks."""
    n = len(lo)
    for state in product((0,-1,1), repeat=n):
        state = np.asarray(state)
        free, bound = np.flatnonzero(state==0), np.flatnonzero(state!=0)
        x = np.where(state<0,lo,hi).copy()
        x[free] = 0.
        hff = H[np.ix_(free,free)]
        if len(free):
            x0 = np.linalg.solve(hff,-H[np.ix_(free,bound)] @ x[bound])
            v = np.linalg.solve(hff,g[free])
        for energy_active in (False,True):
            lam = 0.
            if len(free):
                if energy_active:
                    denom = float(g[free] @ v)
                    if denom <= 1e-30:
                        continue
                    lam = float((lower-g[bound]@x[bound]-g[free]@x0)/denom)
                x[free] = x0 + lam*v
            elif energy_active:
                # Degenerate feasible set can be a single box vertex. Find a
                # nonnegative energy multiplier satisfying ALL bound signs.
                if abs(float(g@x-lower)) > cfg.energy_tolerance:
                    continue
                lhs, coefficient = state*(H@x), state*g
                lmin, lmax = 0., np.inf
                impossible = False
                for a,b in zip(lhs,coefficient):
                    if b > 1e-30:
                        lmin=max(lmin,float(a/b))
                    elif b < -1e-30:
                        lmax=min(lmax,float(a/b))
                    elif a > 1e-10:
                        impossible=True
                if impossible or lmin > lmax:
                    continue
                lam=lmin
            if lam < -1e-10 or np.any(x<lo-cfg.torque_tolerance) or np.any(x>hi+cfg.torque_tolerance):
                continue
            if g@x < lower-cfg.energy_tolerance:
                continue
            residual = H@x-lam*g
            tol = 1e-8 * max(1.,float(np.max(np.abs(H@x))),float(np.max(np.abs(lam*g))))
            if np.any(residual[state<0] < -tol) or np.any(residual[state>0] > tol):
                continue
            if len(free) and np.max(np.abs(residual[free])) > tol:
                continue
            return x
    raise FilterFailure("solver_failure",dict(status="OSQP and bound-active KKT recovery failed"))


def torque_box(nominal, center, previous, limits, dt, cfg, rho=None):
    radius = np.asarray(cfg.rho if rho is None else rho)
    rate = np.asarray(cfg.rate) * dt
    lo = np.maximum.reduce([-limits - nominal, previous - rate - nominal, center - nominal - radius])
    hi = np.minimum.reduce([limits - nominal, previous + rate - nominal, center - nominal + radius])
    if np.any(lo > hi):
        raise FilterFailure("empty_actuator_bounds", dict(lower=lo.tolist(), upper=hi.tolist()))
    return lo, hi


def solve_qp(nominal, linear, energy, minimum, previous, limits, dt, cfg, H=None, rho=None,
             exact_zero=False):
    start = perf_counter()
    nominal, previous, limits = map(np.asarray, (nominal, previous, limits))
    energy, minimum = np.asarray(energy), np.asarray(minimum)
    g, w, center = linear.gradients, linear.prediction.work, linear.center
    if not np.array_equal(g[0], g[1]) or w[0] != w[1]:
        raise FilterFailure("oracle_identity")
    eps = 0.0 if exact_zero else cfg.epsilon_prediction + cfg.epsilon_linearization
    lo, hi = torque_box(nominal, center, previous, limits, dt, cfg, rho)
    lower = minimum - energy - w - g @ (nominal - center) + eps
    best_delta = np.where(g[0] >= 0,hi,lo)
    if np.max(np.abs(g)) >= cfg.gradient_zero and g[0] @ best_delta < np.max(lower) - cfg.energy_tolerance:
        raise FilterFailure("energy_infeasible",dict(best_margin=(g @ best_delta-lower).tolist(),
                            classification="affine halfspace cannot intersect actuator/rate/trust box; no global nonlinear claim"))
    scale = np.asarray(cfg.torque_scale)
    H = np.diag(1 / scale**2) if H is None else H
    if np.min(np.linalg.eigvalsh(H)) <= 0 or not np.all(np.isfinite(H)):
        raise ValueError("objective must be positive definite")
    if np.max(np.abs(g)) < cfg.gradient_zero:
        if np.max(lower) > cfg.energy_tolerance:
            raise FilterFailure("ineffective_gradient", dict(lower=lower.tolist(), work=w.tolist()))
        if np.all(lo <= 0) and np.all(hi >= 0):
            delta, status = np.zeros(7), "zero_gradient_nominal"
        else:
            delta = None
    else:
        delta = None
    if delta is None:
        # Whiten the SPD objective: delta=B z, B^T H B=I. The task cost can
        # otherwise be very anisotropic. Normalize every row, and consolidate
        # the two EXACTLY equal oracle rows to their tighter lower bound.
        # Both original energy rows are checked independently below.
        B = np.linalg.solve(np.linalg.cholesky(H).T, np.eye(7))
        es = max(float(np.linalg.norm(g[0] @ B)), 1e-12)
        bs = np.linalg.norm(B, axis=1)
        A = sparse.csc_matrix(np.vstack(((g[0] @ B / es)[None,:], B / bs[:,None])))
        P = sparse.eye(7, format="csc")
        solver = osqp.OSQP()
        solver.setup(P=P, q=np.zeros(7), A=A,
                     l=np.r_[np.max(lower) / es, lo / bs], u=np.r_[np.inf, hi / bs],
                     verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=cfg.solver_max_iter,
                     polishing=True, warm_starting=False)
        result = solver.solve(raise_error=False)
        status = result.info.status
        if result.info.status_val not in (1, 2) or result.x is None or not np.all(np.isfinite(result.x)):
            delta = active_set_recovery(H,g[0],float(np.max(lower)),lo,hi,cfg)
            status = "active_set_recovery(" + status + ")"
        else:
            delta = B @ result.x
    affine = w + g @ (nominal + delta - center)
    er = energy + affine - eps - minimum
    if np.any(delta < lo - cfg.torque_tolerance) or np.any(delta > hi + cfg.torque_tolerance):
        raise FilterFailure("physical_box_residual", dict(delta=delta.tolist(), lower=lo.tolist(), upper=hi.tolist()))
    if np.min(er) < -cfg.energy_tolerance:
        raise FilterFailure("physical_energy_residual", dict(residual=er.tolist()))
    return QPSolution(nominal + delta, affine, er, lo, hi, status, perf_counter() - start)


@dataclass
class Accepted:
    torque: np.ndarray
    prediction: object
    affine: np.ndarray
    linear: object
    attempts: list
    timing: dict


def accepted_command(predictor, state, nominal, force, energy, minimum, previous, cfg, base_qfrc=None):
    """Fresh nonlinear check after every QP, finite retries from ORIGINAL snapshot."""
    timing = dict(gradient=0., qp=0., nonlinear=0., retry=0.)
    # Work is identically zero for zero held force. A fixed affine margin would
    # spuriously demand recharge as a smooth pulse fades to zero. The measured
    # scan envelope is expressed per newton, with an explicit reference force.
    cfg = replace(cfg, epsilon_linearization=cfg.epsilon_linearization *
                  float(np.linalg.norm(force)) / cfg.margin_force_reference)
    exact_zero = bool(np.all(force == 0))  # ONLY the guaranteed held-input contract
    center, rho = nominal.copy(), np.asarray(cfg.rho).copy()
    H = None
    attempts = []
    for retry in range(cfg.max_retries + 1):
        retry_start = perf_counter()
        try:
            linear = predictor.linearize(state, center, force, cfg.eta, base_qfrc)
            timing["gradient"] += linear.seconds
            if H is None:  # Freeze nominal task sensitivity and cost through retries.
                sy = linear.task_sensitivity / np.asarray(cfg.task_scales)[:, None]
                H = np.diag(1 / np.asarray(cfg.torque_scale)**2) + cfg.task_weight * sy.T @ sy
            sol = solve_qp(nominal, linear, energy, minimum, previous, predictor.arm.torque_limit,
                           predictor.timing.control_period, cfg, H, rho, exact_zero)
            timing["qp"] += sol.seconds
            p = predictor.predict(state, sol.torque, force, base_qfrc)
            timing["nonlinear"] += p.seconds
            ep = 0. if exact_zero else cfg.epsilon_prediction
            residual = energy + p.work - ep - minimum
            accepted = bool(np.min(residual) >= -cfg.energy_tolerance)
            attempts.append(dict(retry=retry, status=sol.status, accepted=accepted,
                                 nonlinear_margin=residual.tolist(), affine=sol.affine_work.tolist(),
                                 affine_margin=sol.energy_residual.tolist(), gradient=linear.gradients.tolist(),
                                 center_work=linear.prediction.work.tolist(),
                                 energy=energy.tolist(), minimum=minimum.tolist(),
                                 epsilon_linearization=cfg.epsilon_linearization,
                                 objective_condition=float(np.linalg.cond(H)),
                                 torque=sol.torque.tolist(), center=center.tolist(), rho=rho.tolist()))
            if accepted:
                if retry:
                    timing["retry"] += perf_counter() - retry_start
                return Accepted(sol.torque, p, sol.affine_work, linear, attempts, timing)
            # A rejected nonlinear candidate can be a new center, but the QP
            # variable and objective remain tau - ORIGINAL nominal.
            attempts[-1]["reason"] = "nonlinear_rejection"
            center = np.clip(sol.torque, -predictor.arm.torque_limit + cfg.eta,
                             predictor.arm.torque_limit - cfg.eta)
            rho *= 0.5
        except ValueError as exc:
            raise FilterFailure("stencil_at_actuator_bound", dict(message=str(exc), attempts=attempts)) from exc
        except FilterFailure as exc:
            attempts.append(dict(retry=retry, status="failed", accepted=False, reason=exc.reason,
                                 center=center.tolist(), rho=rho.tolist(), details=dict(exc.diagnostics)))
            exc.diagnostics["attempts"] = attempts
            exc.diagnostics["timing_seconds"] = dict(timing)
            raise
        if retry:
            timing["retry"] += perf_counter() - retry_start
    raise FilterFailure("nonlinear_retries_exhausted", dict(attempts=attempts, timing=timing))
