import mujoco
import numpy as np

from mujoco_panda.config import PandaConfig
from mujoco_panda.passivity import affine_dynamics, mass_matrix
from mujoco_panda.simulation import load_model, run


def test_affine_dynamics_includes_bias_passive_and_external_terms():
    model, data, h = load_model()
    external = np.linspace(-1., 1., model.nv)
    pred = affine_dynamics(model, data, h, external)
    torque = np.linspace(-3., 3., 7)
    selection = np.zeros(model.nv); selection[h.dof_ids] = torque
    explicit = data.qvel + model.opt.timestep * np.linalg.solve(
        mass_matrix(model, data),
        selection + external + data.qfrc_passive - data.qfrc_bias)
    assert np.allclose(pred.qvel_next(torque), explicit, atol=1e-12)


def test_affine_prediction_is_checked_against_real_mujoco_transition():
    model, data, h = load_model()
    torque = data.qfrc_bias[h.dof_ids] + np.array([.2, -.1, .15, -.2, .05, 0, -.03])
    pred = affine_dynamics(model, data, h, np.zeros(model.nv))
    predicted = pred.qvel_next(torque)
    data.ctrl[h.actuator_ids] = torque
    mujoco.mj_step(model, data)
    error = np.linalg.norm(data.qvel - predicted)
    assert error < 1e-3


def test_c3_ee_blocking_preserves_protected_floors_without_fallback():
    cfg = PandaConfig(duration=6.0); cfg.human.enabled = True
    cfg.human.start = 3.0; cfg.human.hold = 1.0
    cfg.passivation.mode = "C3_dual_ledger_qp"
    result = run(cfg)
    assert result.metrics.qp_fallback_steps == 0
    assert result.metrics.ledger_floor_violations == 0
    assert result.metrics.prediction_bound_violations == 0
    assert result.metrics.human_output_J <= (
        cfg.passivation.human.e_init - cfg.passivation.human.e_min + 2e-3)


def test_sampled_ledger_work_and_physical_contact_work_are_both_auditable():
    cfg = PandaConfig(duration=3.0); cfg.passivation.mode = "C2_residual_qp"
    result = run(cfg); d = result.log; dt = .001
    assert np.isclose(d["w_task_sample"][-1], np.sum(d["p_task_sample"]) * dt)
    assert np.isclose(d["w_task_physical"][-1], np.sum(d["p_task_physical"]) * dt)
    assert np.isclose(d["w_human_physical"][-1], np.sum(d["p_human_physical"]) * dt)
    assert abs(d["w_task_physical"][-1] - d["w_task_sample"][-1]) > 1e-5
    assert np.max(d["qvel_prediction_error"]) <= cfg.passivation.qvel_prediction_bound
