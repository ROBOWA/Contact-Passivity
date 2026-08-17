"""Unit and integration tests for the selective-passivation layer.

Covers the 16 required items: decomposition identities, power identities and
signs, ledger charging cap and honest lower-bound bookkeeping, QP behavior
(inactive when nominal is feasible, energy / power / torque constraints),
the one-step affine prediction, torque delivery to MuJoCo, PI anti-windup,
cumulative passivity inequalities, C1 depletion, and no false activation.
"""

import numpy as np
import pytest

from mujoco_sliding import plotting
from mujoco_sliding.config import (B_TN, ControllerConfig, HumanForceConfig,
                                   LedgerParams, PassivationConfig,
                                   SimulationConfig, TANGENT, NORMAL)
from mujoco_sliding.contact_model import TaskContactPredictor, decompose
from mujoco_sliding.controller import Phase, SlidingForceController
from mujoco_sliding.energy_ledgers import EnergyLedger
from mujoco_sliding.passivity_qp import (AffinePrediction, PassivityQP,
                                         ROW_E_H, ROW_E_R, ROW_P_H,
                                         build_constraint_rows,
                                         solve_scalar_gamma,
                                         validate_affine_prediction)
from mujoco_sliding.simulation import run_simulation

from .conftest import press_pad_into_surface

DT = 1e-3

_BLOCKING = HumanForceConfig(enabled=True, magnitude=3.0,
                             direction=(-1.0, 0.0, 0.0),
                             t_start=6.0, t_rise=0.5, t_hold=2.0, t_fall=0.5)
_HELPING = HumanForceConfig(enabled=True, magnitude=3.0,
                            direction=(1.0, 0.0, 0.0),
                            t_start=5.0, t_rise=0.5, t_hold=2.0, t_fall=0.5)


@pytest.fixture(scope="module")
def c3_nohuman_log():
    return run_simulation(SimulationConfig(
        duration=8.0, passivation=PassivationConfig(mode="C3_dual_ledger_qp")))


@pytest.fixture(scope="module")
def c3_blocking_log():
    return run_simulation(SimulationConfig(
        duration=12.0, human=_BLOCKING,
        passivation=PassivationConfig(mode="C3_dual_ledger_qp")))


@pytest.fixture(scope="module")
def c3_helping_log():
    return run_simulation(SimulationConfig(
        duration=9.0, human=_HELPING,
        passivation=PassivationConfig(mode="C3_dual_ledger_qp")))


@pytest.fixture(scope="module")
def c1_nohuman_log():
    return run_simulation(SimulationConfig(
        duration=10.0,
        passivation=PassivationConfig(mode="C1_whole_port_scalar")))


# ---------------------------------------------------------------------------
# 1-3: decomposition and power identities
# ---------------------------------------------------------------------------

def test_oracle_decomposition_identities():
    f_task = np.array([-1.5, 0.0, 5.0])
    f_h = np.array([3.0, 0.0, -1.0])
    oracle = TaskContactPredictor("oracle")
    d = decompose(f_task, oracle.predict(f_task, np.zeros(3)), f_h)
    np.testing.assert_allclose(d.f_r, f_h, atol=1e-12)       # F_R = F_H
    np.testing.assert_allclose(d.f_delta, 0.0, atol=1e-12)   # F_Delta = 0

    d0 = decompose(f_task, oracle.predict(f_task, np.zeros(3)), np.zeros(3))
    np.testing.assert_allclose(d0.f_r, 0.0, atol=1e-12)      # no human: F_R = 0


def test_imperfect_no_human_residual_equals_mismatch():
    f_task = np.array([-1.5, 0.0, 5.0])
    v = np.array([0.05, 0.0, 0.0])
    pred = TaskContactPredictor("friction", mu_hat=0.20, v_s=0.005)
    f_hat = pred.predict(f_task, v)
    # Predictor never sees the human force: same f_hat with any f_h.
    np.testing.assert_allclose(
        f_hat, pred.predict(f_task, v), atol=1e-15)
    d = decompose(f_task, f_hat, np.zeros(3))
    np.testing.assert_allclose(d.f_r, d.f_delta, atol=1e-12)
    # mu mismatch: predicted friction magnitude is 0.20/0.30 of the true one.
    assert f_hat[0] == pytest.approx(-0.20 * 5.0 * np.tanh(0.05 / 0.005))
    assert f_hat[2] == pytest.approx(5.0)


def test_power_identity_pr_ph_pdelta(c3_blocking_log):
    lg = c3_blocking_log
    np.testing.assert_allclose(
        lg["sp_p_r_act"], lg["sp_p_h_act"] + lg["sp_p_delta_act"], atol=1e-12)


# ---------------------------------------------------------------------------
# 4: power signs + t-n power invariance
# ---------------------------------------------------------------------------

def test_power_signs_helping_and_blocking(c3_helping_log, c3_blocking_log):
    # Helping (+t) while sliding: the human injects energy (p_H > 0).
    lg = c3_helping_log
    hold = (lg["time"] > 5.6) & (lg["time"] < 7.4)
    assert lg["sp_p_h_act"][hold].mean() > 0.05
    # Blocking (-t) while sliding: the robot outputs energy (p_H < 0).
    lg = c3_blocking_log
    early_hold = (lg["time"] > 6.6) & (lg["time"] < 6.9)
    assert lg["sp_p_h_act"][early_hold].mean() < -0.01
    # Task friction dissipates during steady sliding.
    slide = lg["phase"] == int(Phase.SLIDE)
    steady = slide & (lg["time"] > 4.0) & (lg["time"] < 6.0)
    assert lg["p_task"][steady].mean() < -0.05


def test_power_invariant_under_tn_transform(c3_blocking_log):
    lg = c3_blocking_log
    i = 7000
    for f in (lg["f_task"][i], lg["f_h"][i], lg["sp_f_r"][i]):
        v = lg["ee_vel"][i]
        assert f @ v == pytest.approx((B_TN.T @ f) @ (B_TN.T @ v), abs=1e-15)


# ---------------------------------------------------------------------------
# 5-6: ledger behavior
# ---------------------------------------------------------------------------

def test_ledger_charging_cap(c3_helping_log):
    led = EnergyLedger(LedgerParams(0.05, 0.005, 0.08))
    upd = led.update(p=40.0, dt=DT)          # would exceed the cap
    assert upd.e_next == pytest.approx(0.08)
    assert upd.capped and upd.e_raw == pytest.approx(0.09)
    # Integration: the helping pulse (0.15 W into a 0.03 J headroom) must
    # cap E_H exactly at E_max and never exceed it.
    lg = c3_helping_log
    assert lg["sp_e_h"].max() == pytest.approx(0.08, abs=1e-9)
    assert np.all(lg["sp_e_h"] <= 0.08 + 1e-12)


def test_no_hidden_lower_bound_clipping():
    led = EnergyLedger(LedgerParams(0.05, 0.005, 0.08))
    led.e = 0.0051
    upd = led.update(p=-1.0, dt=DT)          # extracts 1 mJ
    assert upd.e_raw == pytest.approx(0.0041)
    assert upd.e_next == pytest.approx(0.0041)   # NOT clamped to e_min
    assert led.e == pytest.approx(0.0041)
    assert upd.below_min and upd.margin == pytest.approx(-0.0009)


# ---------------------------------------------------------------------------
# 7-10: QP constraint behavior (synthetic affine model)
# ---------------------------------------------------------------------------

def _synthetic_pred(v_drift=(0.05, 0.0)):
    # Effective mass ~3 kg per axis: A = dt/m * I; G = I (unit torque map).
    return AffinePrediction(
        A=DT / 3.0 * np.eye(2), b=np.asarray(v_drift, dtype=float),
        G=np.eye(2), tau0=np.zeros(2), v_tn=np.asarray(v_drift, dtype=float))


def _rows(cfg, pred, f_h_tn, f_r_tn, e_h, e_r, tau_prev=None,
          mode="C3_dual_ledger_qp", f_meas_tn=None, e_w=0.30):
    if f_meas_tn is None:
        f_meas_tn = f_h_tn
    return build_constraint_rows(cfg, ControllerConfig(), DT, pred,
                                 np.asarray(f_h_tn, float),
                                 np.asarray(f_r_tn, float),
                                 np.asarray(f_meas_tn, float),
                                 e_h, e_r, e_w, tau_prev, mode)


def test_qp_unchanged_when_nominal_feasible():
    cfg = PassivationConfig()
    qp = PassivityQP(cfg)
    pred = _synthetic_pred()
    u_nom = np.array([1.5, -5.0])
    rows = _rows(cfg, pred, [0.0, 0.0], [0.0, 0.0], e_h=0.05, e_r=0.30)
    sol = qp.solve(u_nom, rows)
    assert sol.ok and not sol.infeasible
    np.testing.assert_allclose(sol.u, u_nom, atol=1e-4)   # eps_reg shift only
    assert not sol.active[:5].any()


def test_qp_human_and_residual_energy_constraints():
    cfg = PassivationConfig()
    qp = PassivityQP(cfg)
    # Low drift velocity: a state consistent with the CBF having already
    # decelerated the robot as the ledger approached its floor.
    pred = _synthetic_pred(v_drift=(0.01, 0.0))
    u_nom = np.array([1.5, -5.0])
    f_h = np.array([-3.0, 0.0])              # blocking: p_H drift = -0.03 W
    for e_h, e_r in [(0.0055, 0.30), (0.05, 0.0205)]:
        rows = _rows(cfg, pred, f_h, f_h, e_h=e_h, e_r=e_r)
        sol = qp.solve(u_nom, rows)
        assert sol.ok
        p_h_next = float(f_h @ (pred.A @ sol.u + pred.b))
        # One-step certificates (delta_p feasibility tolerance included).
        assert e_h + DT * p_h_next >= 0.005 + cfg.eps_h - DT * cfg.delta_p - 1e-9
        assert e_r + DT * p_h_next >= 0.02 + cfg.eps_r - DT * cfg.delta_p - 1e-9
        assert not np.allclose(sol.u, u_nom)  # it had to intervene


def test_qp_instantaneous_human_power_constraint():
    cfg = PassivationConfig()
    qp = PassivityQP(cfg)
    pred = _synthetic_pred(v_drift=(0.04, 0.0))
    u_nom = np.array([5.0, -5.0])
    f_h = np.array([-3.0, 0.0])              # drift extraction 0.12 W > 0.10
    rows = _rows(cfg, pred, f_h, f_h, e_h=0.05, e_r=0.30)
    sol = qp.solve(u_nom, rows)
    assert sol.ok
    p_h_next = float(f_h @ (pred.A @ sol.u + pred.b))
    assert -p_h_next <= cfg.p_h_max + 1e-9
    assert sol.active[ROW_P_H]


def test_qp_torque_and_rate_constraints():
    cfg = PassivationConfig()
    ctl = ControllerConfig()
    qp = PassivityQP(cfg)
    pred = _synthetic_pred()
    u_nom = np.array([300.0, -300.0])        # violates the 60 Nm box via G=I
    tau_prev = np.array([50.0, -50.0])
    rows = _rows(cfg, pred, [0.0, 0.0], [0.0, 0.0], e_h=0.05, e_r=0.30,
                 tau_prev=tau_prev)
    sol = qp.solve(u_nom, rows)
    assert sol.ok
    tau = pred.tau(sol.u)
    dmax = ctl.tau_rate_limit * DT
    assert np.all(np.abs(tau) <= ctl.tau_limit + 1e-6)
    assert np.all(np.abs(tau - tau_prev) <= dmax + 1e-6)
    # Scalar solver honors the same rows.
    sol_g = solve_scalar_gamma(u_nom, rows)
    assert sol_g.ok
    tau_g = pred.tau(sol_g.u)
    assert np.all(np.abs(tau_g) <= ctl.tau_limit + 1e-6)
    assert np.all(np.abs(tau_g - tau_prev) <= dmax + 1e-6)


# ---------------------------------------------------------------------------
# 11: affine one-step prediction
# ---------------------------------------------------------------------------

def test_affine_prediction_matches_dynamics_and_mujoco(model, data, handles):
    press_pad_into_surface(model, data, handles, depth=0.0013,
                           qvel=np.array([0.02, -0.05]))
    f_h = np.array([2.0, 0.0, -1.0])
    res = validate_affine_prediction(
        model, data, handles, dq_joint=0.5, f_h=f_h,
        u_values=[np.zeros(2), np.array([-1.5, -5.0]), np.array([8.0, -12.0])])
    for r in res:
        # Affine factorization is exact vs the explicit computation.
        assert r["err_affine_vs_explicit"] < 1e-12
        # vs the real MuJoCo step: held-contact-force + implicit-integrator
        # error. At this artificially posed static-contact state the
        # constraint force is transient (see the README note on static
        # solves), so the bound is loose here; the tight in-run bound is
        # asserted in test_prediction_error_logged_and_small.
        assert r["err_affine_vs_actual"] < 2e-2


def test_prediction_error_logged_and_small(c3_nohuman_log):
    err = c3_nohuman_log["sp_pred_err"]
    assert np.median(err) < 5e-4
    assert np.percentile(err, 99) < 5e-3


# ---------------------------------------------------------------------------
# 12: the filtered torque reaches MuJoCo
# ---------------------------------------------------------------------------

def test_filtered_torque_reaches_mujoco(c3_nohuman_log):
    lg = c3_nohuman_log
    # The logged applied command IS the passivation-layer torque...
    np.testing.assert_allclose(lg["ctrl"], lg["sp_tau"], atol=1e-12)
    # ...and MuJoCo's actuator force (read one step later) equals it.
    np.testing.assert_allclose(lg["tau_applied"][1:], lg["ctrl"][:-1],
                               atol=1e-9)


# ---------------------------------------------------------------------------
# 13: PI anti-windup (integrator freezing)
# ---------------------------------------------------------------------------

def test_anti_windup_integrator_freezing(model, data, handles):
    from mujoco_sliding.contact_extraction import extract_task_contact

    press_pad_into_surface(model, data, handles, depth=0.002)
    cfg = ControllerConfig()
    ctl = SlidingForceController(model, handles, cfg)
    ctl.phase = Phase.RAMP
    ctl._t_ramp_start = data.time - cfg.ramp_duration
    contact = extract_task_contact(model, data, handles)
    contact.f_n = 0.0                        # persistent force error

    out = ctl.update(data, contact, integrate=False)
    assert ctl.integral == 0.0               # not yet integrated
    ctl.finish_step(out, frozen=True)        # QP modified normal command
    assert ctl.integral == 0.0               # frozen
    out = ctl.update(data, contact, integrate=False)
    ctl.finish_step(out, frozen=False)       # QP left the command nominal
    assert ctl.integral == pytest.approx(out.e_f * ctl.dt)


# ---------------------------------------------------------------------------
# 14: cumulative-energy inequalities (blocking scenario, C3)
# ---------------------------------------------------------------------------

def test_cumulative_energy_inequalities(c3_blocking_log):
    lg = c3_blocking_log
    dt = float(lg["meta_timestep"])
    w_h = -np.cumsum(lg["sp_p_h_act"]) * dt
    w_r = -np.cumsum(lg["sp_p_r_act"]) * dt
    # W_i(t) <= E_i(0) - E_i_min + numerical tolerance, for ALL t.
    assert w_h.max() <= (0.05 - 0.005) + 1e-3
    assert w_r.max() <= (0.30 - 0.02) + 1e-3
    # Raw ledger floors within tolerance; peak extraction within 5%.
    assert lg["sp_e_h_raw"].min() >= 0.005 - 1e-4
    assert np.max(-lg["sp_p_h_act"]) <= 0.10 * 1.05
    assert int(lg["meta_n_infeasible"]) == 0


# ---------------------------------------------------------------------------
# 15: C1 depletes from ordinary task friction
# ---------------------------------------------------------------------------

def test_c1_depletes_during_nominal_sliding(c1_nohuman_log):
    lg = c1_nohuman_log
    # The whole-port ledger drains on ordinary task friction alone and hits
    # its floor (raw violations stay visible; nothing is clamped away).
    assert lg["sp_e_w"].min() < 0.021
    assert lg["sp_e_w_below"].any()
    # The scalar filter throttles the task (gamma < 1) and repeatedly falls
    # back to emergency damping...
    assert lg["sp_active_any"].any()
    assert np.nanmin(lg["sp_gamma"]) < 0.9
    assert int(lg["meta_n_emergency"]) > 0
    # ...which destroys the normal-force task (nominal holds 5.00 +/- 0.01 N;
    # here the regulation collapses into large oscillations).
    late = lg["time"] > 8.0
    assert lg["f_n"][late].std() > 1.0


# ---------------------------------------------------------------------------
# 16: no false activation with exact model / no human
# ---------------------------------------------------------------------------

def test_c3_no_false_activation_exact_model(c3_nohuman_log):
    lg = c3_nohuman_log
    contact_phase = lg["phase"] >= int(Phase.RAMP)
    assert lg["sp_active_any"][contact_phase].mean() < 1e-3
    assert np.abs(lg["sp_e_h"] - 0.05).max() < 1e-4
    assert np.abs(lg["sp_e_r"] - 0.30).max() < 1e-4
    assert int(lg["meta_n_infeasible"]) == 0


# ---------------------------------------------------------------------------
# housekeeping: plotting reference lines stay in sync with the config
# ---------------------------------------------------------------------------

def test_plotting_ledger_lines_match_config():
    cfg = PassivationConfig()
    assert plotting._E_H_LINES == (cfg.human.e_min, cfg.human.e_max)
    assert plotting._E_R_LINES == (cfg.residual.e_min, cfg.residual.e_max)
    assert plotting._P_H_MAX == cfg.p_h_max
