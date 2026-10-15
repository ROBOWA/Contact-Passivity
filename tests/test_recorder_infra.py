"""Controller-agnostic checks of the stage-coherent stepping loop and the recorder (priority 1)."""

import mujoco
import numpy as np
import pytest

from kuka_table_sim import load_model, Simulation, TimingConfig
from kuka_table_sim.holding import HoldingController
from kuka_table_sim.ik import quat_from_rpy, solve_site_ik
from kuka_table_sim.recorder import Recorder
from kuka_table_sim.scene import ArmIndex

TIP_RADIUS = 0.009
SURFACE = 0.15


def _hold_sim(record: float | None = None, legacy: bool = False):
    model = load_model(TimingConfig())
    arm = ArmIndex.from_model(model)
    sim = Simulation(model, TimingConfig(), HoldingController.inertia_scaled(model, arm, np.zeros(7)))
    if record:
        sim.recorder = Recorder(sim, record, legacy_mixed=legacy)
        sim.reset()
    return sim


def _evaluate(model, q, v, t, ctrl):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:], d.time, d.ctrl[:] = q, v, t, ctrl
    mujoco.mj_forward(model, d)
    return d


def _press_and_slide_targets(sim):
    """Joint targets that press the tip 2 mm below the surface and drag it across 8 cm (joint-space PD hold)."""
    m = sim.model
    qs = []
    for x in (0.50, 0.58):
        r = solve_site_ik(m, "tcp", [x, 0.0, SURFACE + TIP_RADIUS - 0.002], quat_from_rpy(0, 180, 0), arm=sim.arm,
                          q_seed=sim.q_home, q_pref=sim.q_home)
        assert r.converged
        qs.append(r.q)
    return qs


def test_mujoco_stage_semantics_on_the_installed_version():
    """After mj_step the acceleration-stage and pre-integration quantities are the evaluation at x_k, while
    time/qpos/qvel are x_{k+1}; after mj_step1 at a control instant the sensor still holds the previous step."""
    sim = _hold_sim()
    m, d = sim.model, sim.data
    tcp = m.site("tcp").id
    q0 = sim.q_home.copy()
    amp, freq = np.array([0, 0.1, 0, -0.12, 0, 0.25, 0.8]), np.array([0, 1.7, 0, 2.3, 0, 3.1, 2.2])
    while d.time < 0.7:
        if sim.n_steps % 4 == 0:
            sim.controller.q_target = q0 + amp * np.sin(2 * np.pi * freq * d.time)
        sim.step()
    while sim.n_steps % 4 != 1:                         # the next step is a non-control substep: ctrl is constant
        sim.step()
    q_k, v_k, t_k = d.qpos.copy(), d.qvel.copy(), d.time
    sim.step()
    ctrl = d.ctrl.copy()
    e_k = _evaluate(m, q_k, v_k, t_k, ctrl)
    e_k1 = _evaluate(m, d.qpos, d.qvel, d.time, ctrl)
    assert d.time == pytest.approx(t_k + m.opt.timestep)
    assert np.array_equal(d.sensordata[:3], e_k.sensordata[:3])        # the FT sensor: evaluation at x_k, exactly
    assert np.array_equal(d.qacc, e_k.qacc)
    assert np.array_equal(d.site_xpos[tcp], e_k.site_xpos[tcp])        # pose: FK at x_k, not at the integrated state
    assert not np.array_equal(d.site_xpos[tcp], e_k1.site_xpos[tcp])
    assert not np.array_equal(d.qvel, v_k)                             # qvel already advanced
    while sim.n_steps % 4 != 0:
        sim.step()
    mujoco.mj_step1(m, d)                                              # at the next control instant ...
    now = _evaluate(m, d.qpos, d.qvel, d.time, d.ctrl)
    assert np.linalg.norm(d.sensordata[:3] - now.sensordata[:3]) > 1e-7   # ... sensordata is the PREVIOUS step's value
    mujoco.mj_step2(m, d)


def test_two_stage_step_equals_mj_step_bitwise():
    sim = _hold_sim()
    model, arm = sim.model, sim.arm
    sim.controller.q_target = sim.q_home + 0.05             # make it move
    d_ref = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d_ref, 0)
    d_ref.qpos[:] = sim.data.qpos
    n, ci = sim.steps_for(0.5), sim.timing.control_interval
    tmp = HoldingController(arm, sim.controller.kp, sim.controller.kd)       # same gains as the simulation's controller
    tmp.q_target = sim.q_home + 0.05
    for k in range(n):
        if k % ci == 0:
            mujoco.mj_forward(model, d_ref)
            d_ref.ctrl[arm.actuator_ids] = tmp.compute(d_ref)
        mujoco.mj_step(model, d_ref)
    sim.advance(n)
    assert np.array_equal(sim.data.qpos, d_ref.qpos) and np.array_equal(sim.data.qvel, d_ref.qvel)


def test_enabling_the_recorder_does_not_change_the_trajectory():
    out = []
    for rec in (False, True):
        sim = _hold_sim(record=3.0 if rec else None, legacy=rec)
        qa, qb = _press_and_slide_targets(sim)
        for _ in range(sim.steps_for(3.0)):
            if sim.n_steps % 4 == 0:
                s = 0.5 - 0.5 * np.cos(2 * np.pi * sim.data.time / 3.0)
                sim.controller.q_target = qa + s * (qb - qa)
            sim.step()
        out.append((sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.ctrl.copy(), sim.data.time, sim.data.ncon))
    a, b = out
    assert all(np.array_equal(x, y) for x, y in zip(a[:3], b[:3])) and a[3] == b[3] and a[4] == b[4] > 0   # in contact at the end


def test_task_contact_power_equals_the_constraint_power_of_the_same_evaluation():
    """P = sum F.v_p + M.omega at the contact point equals qvel . qfrc_constraint exactly (only the tip contact
    is active), while the mixed-stage product a post-step logger would form is measurably wrong."""
    sim = _hold_sim(record=6.0, legacy=True)
    qa, qb = _press_and_slide_targets(sim)
    for _ in range(sim.steps_for(6.0)):
        if sim.n_steps % 4 == 0:
            s = 0.5 - 0.5 * np.cos(2 * np.pi * sim.data.time / 3.0)
            sim.controller.q_target = qa + s * (qb - qa)
        sim.step()
    p = sim.recorder.finish().phys
    sel = (p["n_task"] > 0) & (p["n_unexpected"] == 0)
    assert sel.sum() > 8000                                        # pressing and sliding for most of the run
    assert np.abs(p["F_task"][sel, 2]).max() > 5.0                 # a real contact force
    assert np.abs(p["P_task"][sel] - p["P_constraint"][sel]).max() < 1e-9
    assert np.abs(p["P_mixed"][sel] - p["P_constraint"][sel]).max() > 1e3 * np.abs(p["P_task"][sel] - p["P_constraint"][sel]).max()
    assert (p["P_task"] > 0).any() and (p["P_task"] < 0).any()    # signed, not clipped or absolute-valued
    assert np.array_equal(p["dW"], p["P_task"] * 0.00025)         # documented rectangle rule


def test_unexpected_contacts_are_reported_separately_and_not_counted_as_task_work():
    sim = _hold_sim(record=1.0)
    m, d = sim.model, sim.data
    d.qpos[sim.arm.qpos_adr] = sim.q_home + np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])   # forearm into the table
    mujoco.mj_forward(m, d)
    sim.advance(10)
    log = sim.recorder.finish()
    assert log.phys["n_unexpected"].max() > 0 and log.meta["unexpected_pairs"]
    assert np.all(log.phys["n_task"] == 0) and np.all(log.phys["P_task"] == 0)


def test_run_metadata_records_configuration_versions_and_periods():
    sim = _hold_sim(record=0.1)
    sim.advance(100)
    meta = sim.recorder.finish().meta
    assert meta["mujoco"] == mujoco.__version__ and meta["numpy"] == np.__version__
    assert meta["timestep"] == 0.00025 and meta["control_period"] == 0.001 and meta["control_interval"] == 4
    assert len(meta["commit"]) in (7, 40) or meta["commit"] == "unknown"
    assert meta["integrator"] == "mjINT_IMPLICITFAST" and meta["cone"] == "mjCONE_ELLIPTIC"
    assert meta["model_file_sha256"] and meta["conventions"]["work"]
