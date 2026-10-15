"""Physical-unit algebra, original nominal objective, previous EXECUTED rate,
zero-gradient infeasibility and a nonlinear falsification/retry case."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from kuka_table_sim.joint_qp import QPConfig, FilterFailure, solve_qp, accepted_command


def lin(center=None, work=-1e-4, gradient=1e-5):
    center = np.zeros(7) if center is None else np.asarray(center)
    g = np.zeros((2,7))
    g[:,0] = gradient
    return NS(center=center, prediction=NS(work=np.full(2,work)), gradients=g,
              task_sensitivity=np.zeros((7,7)), seconds=0.)


def cfg(**kw):
    return replace(QPConfig(), epsilon_linearization=0., epsilon_prediction=0.,
                   torque_scale=(1.,)*7, **kw)


def test_shifted_center_objective_and_tighter_equal_oracle_account():
    center = np.zeros(7)
    center[0] = 3.
    nominal = np.zeros(7)
    nominal[0] = 1.
    # W(center)=-1e-4, g=1e-5; with E=9e-5 the required tau is 4.
    l = lin(center)
    r = solve_qp(nominal, l, np.array([9e-5, 2e-4]), np.zeros(2), nominal,
                 np.ones(7)*100, .001, cfg())
    assert r.torque[0] == pytest.approx(4., abs=1e-6)
    assert r.affine_work[0] == pytest.approx(-9e-5, abs=1e-11)
    assert np.max(np.abs(r.torque[1:])) < 1e-7
    assert np.min(r.energy_residual) >= -1e-10
    # A shifted center must not itself become the preferred command.
    r = solve_qp(nominal, l, np.ones(2), np.zeros(2), nominal, np.ones(7)*100, .001, cfg())
    np.testing.assert_allclose(r.torque, nominal, atol=1e-7, rtol=0)


def test_rate_reference_is_previous_executed_and_empty_intersection_is_error():
    nominal = np.ones(7)*5
    previous = np.ones(7)*2
    r = solve_qp(nominal, lin(gradient=0.,work=0.), np.ones(2), np.zeros(2), previous,
                 np.ones(7)*100, .001, cfg(rate=(100.,)*7))
    np.testing.assert_allclose(r.torque, np.ones(7)*2.1, atol=1e-7, rtol=0)
    with pytest.raises(FilterFailure, match="empty_actuator_bounds"):
        solve_qp(nominal, lin(), np.ones(2), np.zeros(2), previous,
                 np.ones(7)*100, .001, cfg(rate=(100.,)*7, rho=(.1,)*7))


def test_zero_nearzero_constant_rows_and_true_infeasibility():
    for g in (0., 1e-16):
        r = solve_qp(np.zeros(7), lin(work=0.,gradient=g), np.zeros(2), np.zeros(2), np.zeros(7),
                     np.ones(7), .001, cfg(), exact_zero=True)
        assert np.array_equal(r.torque, np.zeros(7))
        with pytest.raises(FilterFailure, match="ineffective_gradient"):
            solve_qp(np.zeros(7), lin(gradient=g), np.zeros(2), np.zeros(2), np.zeros(7),
                     np.ones(7), .001, cfg())
    with pytest.raises(FilterFailure, match="energy_infeasible"):
        solve_qp(np.zeros(7), lin(), np.zeros(2), np.zeros(2), np.zeros(7),
                 np.ones(7), .001, cfg())


def test_whitened_dense_objective_matches_independent_kkt_solution():
    rng = np.random.default_rng(91)
    q, _ = np.linalg.qr(rng.normal(size=(7,7)))
    H = q @ np.diag(np.geomspace(1.,1e6,7)) @ q.T
    l = lin(work=-.001, gradient=.0001)
    l.gradients[:] = rng.normal(size=(1,7)) * .0001
    need = .0005
    g = l.gradients[0]
    direction = np.linalg.solve(H,g)
    reference = direction * need / (g @ direction)
    r = solve_qp(np.zeros(7),l,np.ones(2)*.0005,np.zeros(2),np.zeros(7),
                 np.ones(7)*100,.001,cfg(),H=H)
    np.testing.assert_allclose(r.torque,reference,atol=1e-6,rtol=1e-6)
    assert np.min(r.energy_residual) >= -1e-10
    recovery = solve_qp(np.zeros(7),l,np.ones(2)*.0005,np.zeros(2),np.zeros(7),
                        np.ones(7)*100,.001,cfg(solver_max_iter=1),H=H)
    assert recovery.status.startswith("active_set_recovery")
    np.testing.assert_allclose(recovery.torque,reference,atol=1e-6,rtol=1e-6)
    # A tight affine energy row can leave exactly one actuator-box vertex.
    # The energy multiplier is still required when every variable is bound.
    vertex = lin(work=-7e-5)
    vertex.gradients[:] = 1e-5
    r = solve_qp(np.zeros(7),vertex,np.zeros(2),np.zeros(2),np.zeros(7),
                 np.ones(7),.001,cfg(solver_max_iter=1))
    np.testing.assert_allclose(r.torque,np.ones(7),atol=1e-7,rtol=0)


class CurvedPredictor:
    """Smooth known work response with a concave term: the first affine
    candidate is optimistic. It has a feasible nonlinear root inside the box."""
    arm = NS(torque_limit=np.ones(7)*100)
    timing = NS(control_period=.001)
    def predict(self, state, tau, force, base=None):
        x = tau[0]
        return NS(work=np.full(2, -1e-4 + .001*x - .001*x*x), seconds=0.)
    def linearize(self, state, center, force, eta, base=None):
        l = lin(center, self.predict(state,center,force).work[0], .001-.002*center[0])
        return l


def test_nonlinear_rejection_relinearization_and_finite_retry_limit():
    p = CurvedPredictor()
    args = (p, np.array([123.]), np.zeros(7), np.ones(3), np.ones(2)*5e-5,
            np.zeros(2), np.zeros(7))
    r = accepted_command(*args, cfg(max_retries=3))
    assert len(r.attempts) > 1
    assert not r.attempts[0]["accepted"] and r.attempts[-1]["accepted"]
    assert np.min(r.prediction.work + 5e-5) >= -1e-10
    assert r.attempts[1]["center"][0] > 0
    with pytest.raises(FilterFailure, match="nonlinear_retries_exhausted") as exc:
        accepted_command(*args, cfg(max_retries=0))
    assert len(exc.value.diagnostics["attempts"]) == 1
