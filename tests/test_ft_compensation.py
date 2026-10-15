"""Priority 2: inertial loads must not be mistaken for contact."""

import numpy as np
import mujoco
import pytest

from kuka_table_sim import load_model, Simulation, TimingConfig
from kuka_table_sim import metrics as mx
from kuka_table_sim import scenarios as sc
from kuka_table_sim.force_control import ForceControlConfig
from kuka_table_sim.ft_sensor import FTSensor
from kuka_table_sim.holding import HoldingController
from kuka_table_sim.scene import ArmIndex


def test_sensor_audit_matches_the_model():
    model = load_model(TimingConfig())
    sim = Simulation(model, TimingConfig(), HoldingController.inertia_scaled(model, ArmIndex.from_model(model), np.zeros(7)))
    ft = FTSensor(model)
    a = ft.audit(sim.data)
    assert a["sensor_body"] == "ft_sensor" and a["supported_bodies"] == ["ft_sensor", "sphere_tool"]
    assert a["mass"] == pytest.approx(0.5) and a["mass"] == pytest.approx(model.body_subtreemass[model.body("ft_sensor").id])
    assert a["com_in_site_frame"][2] == pytest.approx(0.0175, abs=1e-9)           # on the tool axis, 17.5 mm from the site
    assert abs(a["com_in_site_frame"][0]) < 1e-9 and abs(a["com_in_site_frame"][1]) < 1e-9
    assert np.allclose(ft.gravity, [0, 0, -9.81])


def test_static_free_air_reads_weight_and_corrects_to_zero(free_air_static):
    p = free_air_static.log.phys
    assert np.abs(p["ft_corr"]).max() < 1e-9 and np.abs(p["ft_grav"]).max() < 1e-9
    assert np.abs(p["a_com"]).max() < 1e-9
    # the sensor sign convention: the parent supports the weight, so the world-frame reading is +m g upward
    raw_world = np.einsum("ij,j->i", np.eye(3), np.array([0, 0, 0.5 * 9.81]))
    sim = free_air_static.sim
    s = FTSensor(sim.model).sample(sim.data, sim.data.time)
    assert np.allclose(s.raw_world, raw_world, atol=1e-6)


def test_coordinate_acceleration_matches_finite_differences_of_the_com_velocity():
    model = load_model(TimingConfig())
    arm = ArmIndex.from_model(model)
    sim = Simulation(model, TimingConfig(), HoldingController.inertia_scaled(model, arm, np.zeros(7)))
    q0 = sim.q_home.copy()
    amp, freq = np.array([0, 0.1, 0, -0.12, 0, 0.25, 0.8]), np.array([0, 1.7, 0, 2.3, 0, 3.1, 2.2])
    ft = FTSensor(model)
    v6, vel, acc = np.zeros(6), [], []
    d = sim.data
    for _ in range(sim.steps_for(2.0)):
        if sim.n_steps % 4 == 0:
            sim.controller.q_target = q0 + amp * np.sin(2 * np.pi * freq * d.time)
        mujoco.mj_step1(model, d)
        if sim.n_steps % 4 == 0:
            d.ctrl[arm.actuator_ids] = sim.controller.compute(d)
        m_v = sum(ft.masses[i] * (mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0) or v6[3:6]) for i, b in enumerate(ft.bodies))
        vel.append(np.asarray(m_v) / ft.mass)
        mujoco.mj_step2(model, d)
        sim.n_steps += 1
        acc.append(ft.sample(d, 0.0).a_com)
    fd = (np.array(vel[1:]) - np.array(vel[:-1])) / model.opt.timestep
    err = np.abs(np.array(acc[:-1]) - fd).max()
    assert np.abs(fd).max() > 5.0 and err < 0.02               # coordinate acceleration (gravity NOT folded in)


def test_dynamic_free_air_old_failure_reproduced_and_corrected(free_air_dynamic):
    p = free_air_dynamic.log.phys
    assert int(p["present"].sum()) == 0                                    # genuinely free air
    assert np.linalg.norm(p["a_com"], axis=1).max() > 8.0
    assert p["ft_grav"][:, 2].max() > 3.0 and p["ft_grav"][:, 2].min() < -3.0   # the old estimate: several N of false force
    assert np.abs(p["ft_corr"]).max() < 1e-6                                    # the corrected estimate: nothing
    ci = free_air_dynamic.log.meta["control_interval"]
    old_false = [r for r in mx.runs(p["ft_grav"][:, 2] > 1.0) if r[1] - r[0] >= 3 * ci]
    assert len(old_false) >= 5                                                  # old logic would have declared CONTACT
    # TCP acceleration alone is not enough: compensate with the tcp site acceleration instead of the COM's
    assert p["ft_grav"][:, 2].std() > 1.0


def test_tcp_only_acceleration_leaves_a_residual_that_the_com_does_not():
    model = load_model(TimingConfig())
    arm = ArmIndex.from_model(model)
    sim = Simulation(model, TimingConfig(), HoldingController.inertia_scaled(model, arm, np.zeros(7)))
    q0, d, ft = sim.q_home.copy(), sim.data, FTSensor(model)
    amp, freq = np.array([0, 0.1, 0, -0.12, 0, 0.25, 0.8]), np.array([0, 1.7, 0, 2.3, 0, 3.1, 2.2])
    tcp = model.site("tcp").id
    r6 = np.zeros(6)
    res_com, res_tcp = 0.0, 0.0
    for _ in range(sim.steps_for(3.0)):
        if sim.n_steps % 4 == 0:
            sim.controller.q_target = q0 + amp * np.sin(2 * np.pi * freq * d.time)
        mujoco.mj_step1(model, d)
        if sim.n_steps % 4 == 0:
            d.ctrl[arm.actuator_ids] = sim.controller.compute(d)
        mujoco.mj_step2(model, d)
        sim.n_steps += 1
        s = ft.sample(d, 0.0)
        mujoco.mj_objectAcceleration(model, d, mujoco.mjtObj.mjOBJ_SITE, tcp, r6, 0)
        res_com = max(res_com, np.linalg.norm(s.corrected))
        res_tcp = max(res_tcp, np.linalg.norm(s.gravity_only + s.mass * (r6[3:6] + s.gravity)))
    assert res_com < 1e-6 and res_tcp > 1.0


def test_pressing_force_is_recovered_without_using_contact_truth(nominal):
    log = nominal.log
    p = log.phys
    pressing = p["present"] & (np.abs(p["F_task"][:, 2]) > 1.0)
    assert pressing.sum() > 1000
    assert np.abs(p["ft_corr"][pressing, 2] - p["F_task"][pressing, 2]).max() < 0.3
    assert np.abs(p["ft_corr"][:, :2][pressing] - p["F_task"][:, :2][pressing]).max() < 0.3     # tangential too
    # the controller never reads contact truth: no reference to the recorder's contact fields or mj_contactForce
    import inspect
    from kuka_table_sim import force_control
    src = inspect.getsource(force_control)
    assert "mj_contactForce" not in src and ".ncon" not in src and "qfrc_constraint" not in src and "efc_" not in src


def test_fast_free_air_descent_makes_no_false_contact_and_detects_the_touchdown(fast_approach):
    log = fast_approach.log
    p = log.phys
    first_true = float(p["t"][np.argmax(p["present"])])
    trans = mx.transitions(log)
    t_contact = next(t for t, _, b in trans if b == "contact")
    assert t_contact >= first_true                                          # no CONTACT before genuine contact
    assert t_contact - first_true < 0.01                                    # detected within 10 ms of touching
    free = p["t"] < first_true
    assert np.linalg.norm(p["a_com"][free], axis=1).max() > 10.0            # the descent really accelerates the tool
    assert p["ft_grav"][free, 2].max() > 3.0 and np.abs(p["ft_corr"][free, 2]).max() < 1e-6
    old_false = [r for r in mx.runs((p["ft_grav"][:, 2] > 1.0) & free) if r[1] - r[0] >= 3 * log.meta["control_interval"]]
    assert old_false                                                        # the old gravity-only logic would have latched CONTACT


@pytest.mark.parametrize("creep", [0.003, 0.005, 0.01])
def test_genuine_touchdown_is_detected_reliably(creep):
    res = sc.run_nominal(9.0, config=ForceControlConfig(creep_speed=creep, slide_distance=0.0))
    p = res.log.phys
    first_true = float(p["t"][np.argmax(p["present"])])
    t_contact = next(t for t, _, b in mx.transitions(res.log) if b == "contact")
    assert 0.0 <= t_contact - first_true < 0.01
