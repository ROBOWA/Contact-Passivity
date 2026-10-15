"""Physical tests: independent force mapping/work, full-state replay, isolation."""
import copy

import mujoco
import numpy as np
import pytest

from kuka_table_sim.scenarios import make_force_sim
from kuka_table_sim.oracle_physics import (
    HumanPort, OraclePredictor, TaskMeter, integration_state, restore_state,
)


@pytest.fixture(scope="module")
def sliding():
    s = make_force_sim(record=False)
    s.advance(s.steps_for(12.0))
    mujoco.mj_step1(s.model, s.data)
    tau = s.controller.compute(s.data)
    return s, tau


def independent_interval(model, timing, state, tau, force, body="link5", local=(0., .04, .12)):
    """Independent mj_step execution. Force mapping uses J^T; task power uses
    generalized constraint power, rather than either production meter."""
    d = mujoco.MjData(model)
    restore_state(model, d, state)
    base = d.qfrc_applied.copy()
    bid = model.body(body).id
    jp = np.zeros((3, model.nv))
    d.ctrl[:] = tau
    wh = wt = 0.
    for _ in range(timing.control_interval):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        point = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local
        mujoco.mj_jac(model, d, jp, None, point, bid)
        qv = d.qvel.copy()
        wh += float(force @ jp @ qv) * timing.timestep
        d.qfrc_applied[:] = base + jp.T @ force
        mujoco.mj_step(model, d)
        wt += float(qv @ d.qfrc_constraint) * timing.timestep
    return d, np.array([wh, wh]), wt


def test_port_point_mapping_power_sign_and_independent_wrench(sliding):
    s, _ = sliding
    m, d = s.model, s.data
    port = HumanPort(m)
    point, v, _, _, _ = port.sample(d, np.zeros(3))
    assert np.linalg.norm(v) > 1e-3
    for sign in (-1, 1):
        force = sign * 5 * v / np.linalg.norm(v)
        point, v, tau, power, defect = port.sample(d, force)
        assert sign * power > 0 and abs(defect) < 1e-14
        mapped = np.zeros(m.nv)
        mujoco.mj_applyFT(m, d, force, np.zeros(3), point, port.body, mapped)
        np.testing.assert_allclose(tau, mapped, atol=1e-14, rtol=0)
        # Body force at its COM + moment r x f is an independent representation.
        other = mujoco.MjData(m)
        restore_state(m, other, integration_state(m, d))
        other.qfrc_applied[:] = 0
        other.xfrc_applied[:] = 0
        mujoco.mj_step1(m, other)
        other.xfrc_applied[port.body, :3] = force
        other.xfrc_applied[port.body, 3:] = np.cross(point - other.xipos[port.body], force)
        mujoco.mj_step2(m, other)
        ref = mujoco.MjData(m)
        restore_state(m, ref, integration_state(m, d))
        mujoco.mj_step1(m, ref)
        port.apply(ref, force, np.zeros(m.nv))
        mujoco.mj_step2(m, ref)
        np.testing.assert_allclose(other.qvel, ref.qvel, atol=2e-14, rtol=0)


def test_snapshot_replay_isolation_order_and_independent_execution(sliding):
    s, tau = sliding
    before = integration_state(s.model, s.data)
    ctl = copy.deepcopy(s.controller.diagnostics())
    pred = OraclePredictor(s.model, s.timing, HumanPort(s.model))
    force = np.array([5., 0., 0.])
    p = pred.predict(before, tau, force)
    other = pred.predict(before, tau + np.array([1., -2., .5, 0., 1., 0., 0.]), force)
    repeat = pred.predict(before, tau, force)
    assert np.array_equal(p.final_state, repeat.final_state)
    assert np.array_equal(p.work, repeat.work)
    assert not np.array_equal(p.final_state, other.final_state)
    assert np.array_equal(before, integration_state(s.model, s.data))
    for key, val in ctl.items():
        assert np.array_equal(val, s.controller.diagnostics()[key], equal_nan=True) if not isinstance(val,str) else val == s.controller.diagnostics()[key]
    independent, wh, wt = independent_interval(s.model, s.timing, before, tau, force)
    np.testing.assert_allclose(p.work, wh, atol=1e-12, rtol=0)
    assert abs(p.task_work - wt) < 1e-12
    qn = mujoco.MjData(s.model)
    restore_state(s.model, qn, p.final_state)
    np.testing.assert_allclose(qn.qpos, independent.qpos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(qn.qvel, independent.qvel, atol=1e-11, rtol=0)
    zero = pred.predict(before, tau, np.zeros(3))
    assert np.array_equal(zero.work, np.zeros(2))
    assert p.work[0] == p.work[1]


def test_gradients_against_execution_directional_response_and_eta(sliding):
    s, tau = sliding
    state = integration_state(s.model, s.data)
    pred = OraclePredictor(s.model, s.timing, HumanPort(s.model))
    force = np.array([5., 0., 0.])
    lins = [pred.linearize(state, tau, force, eta) for eta in (.01, .02, .04)]
    for lin in lins:
        assert np.array_equal(lin.gradients[0], lin.gradients[1])
        np.testing.assert_allclose(lin.gradients, lins[1].gradients, atol=1e-11, rtol=1e-3)
    lin = lins[1]
    for delta in (np.array([0., 2., 0., -1., 0., .2, 0.]), np.ones(7) * .5):
        _, actual, _ = independent_interval(s.model, s.timing, state, tau + delta, force)
        nonlinear = pred.predict(state, tau + delta, force)
        np.testing.assert_allclose(nonlinear.work, actual, atol=1e-12, rtol=0)
        affine = lin.prediction.work + lin.gradients @ delta
        assert np.max(affine - actual) < 2e-7
    with pytest.raises(ValueError, match="stencil"):
        pred.linearize(state, s.arm.torque_limit, force)


def test_shared_task_meter_includes_contact_moments_and_frame():
    # Independent model with spinning/torsional contact (the KUKA baseline uses condim=3).
    xml = '''<mujoco><option gravity="0 0 -9.81" cone="elliptic"/>
    <worldbody><geom name="tabletop" type="plane" size="1 1 .1"/>
    <body pos="0 0 .099"><freejoint/><geom name="tool_tip" type="sphere"
    size=".1" mass="1" condim="6" friction="1 .1 .1"/></body></worldbody></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qvel[:] = [.1, -.05, -.02, .3, .2, 1.]
    qv = d.qvel.copy()
    mujoco.mj_forward(m, d)
    sample = TaskMeter(m).sample(d)
    assert np.linalg.norm(sample["M"]) > 1e-6
    assert abs(sample["power"] - float(qv @ d.qfrc_constraint)) < 1e-10
