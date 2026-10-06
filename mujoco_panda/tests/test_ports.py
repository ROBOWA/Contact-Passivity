import mujoco
import numpy as np

from mujoco_panda.config import PandaConfig
from mujoco_panda.passivity import predict_task_port
from mujoco_panda.ports import (assert_power_identity, extract_task_contact,
                                spatial_site_port)
from mujoco_panda.simulation import load_model, run


def _contact_state():
    result = run(PandaConfig(duration=2.5))
    candidates = np.flatnonzero(result.log["contact_solver"])
    assert len(candidates)
    k = int(candidates[-1])
    model, data, h = load_model()
    data.qpos[h.qpos_ids] = [result.log[f"q{i}"][k] for i in range(1, 8)]
    data.qvel[h.dof_ids] = [result.log[f"qd{i}"][k] for i in range(1, 8)]
    data.ctrl[h.actuator_ids] = [result.log[f"tau{i}"][k] for i in range(1, 8)]
    mujoco.mj_forward(model, data)
    return model, data, h


def test_tool_and_forearm_use_their_own_jacobians_and_power_identity():
    model, data, h = load_model()
    data.qvel[:] = [.2, -.1, .15, .05, -.08, .11, -.04]
    mujoco.mj_forward(model, data)
    wrench = np.array([3., -2., 1., .2, -.1, .3])
    tool = spatial_site_port(model, data, h.tool_site, wrench)
    arm = spatial_site_port(model, data, h.forearm_site, wrench)
    assert_power_identity(tool, data.qvel); assert_power_identity(arm, data.qvel)
    assert not np.allclose(tool.jacobian, arm.jacobian)
    assert not np.isclose(tool.power, arm.power)


def test_contact_power_is_sum_of_per_point_wrench_twist_products():
    model, data, h = _contact_state()
    contact = extract_task_contact(model, data, h)
    assert contact.active
    assert np.isclose(contact.power, sum(p.power for p in contact.points), atol=1e-10)
    assert np.allclose(contact.generalized_force,
                       sum((p.generalized_force for p in contact.points),
                           start=np.zeros(model.nv)))
    assert np.isclose(contact.power, contact.generalized_force @ data.qvel,
                      atol=1e-9)


def test_oracle_residual_is_human_and_oracle_no_human_is_zero():
    model, data, h = _contact_state(); contact = extract_task_contact(model, data, h)
    human = spatial_site_port(model, data, h.forearm_site,
                              np.array([2., -1., .5, 0, 0, 0]))
    cfg = PandaConfig().passivation; cfg.predictor = "oracle"
    p = predict_task_port(model, data, h, contact, human, cfg)
    assert np.allclose(p.residual, p.human)
    zero = spatial_site_port(model, data, h.tool_site, np.zeros(6))
    p0 = predict_task_port(model, data, h, contact, zero, cfg)
    assert np.allclose(p0.residual, 0, atol=1e-12)


def test_friction_predictor_uses_task_normal_channel_not_human_force():
    model, data, h = _contact_state(); contact = extract_task_contact(model, data, h)
    cfg = PandaConfig().passivation; cfg.predictor = "friction"
    zero = spatial_site_port(model, data, h.tool_site, np.zeros(6))
    loaded = spatial_site_port(model, data, h.tool_site,
                               np.array([0., 0., 100., 0, 0, 0]))
    a = predict_task_port(model, data, h, contact, zero, cfg)
    b = predict_task_port(model, data, h, contact, loaded, cfg)
    assert np.allclose(a.predicted_force, b.predicted_force)
    assert a.predicted_force.shape == (3,)  # two tangents plus surface normal
