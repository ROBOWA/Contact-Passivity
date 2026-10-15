"""Priority 1: stage/timestamp coherence of samples, logging parity, and the contact-work bookkeeping."""

import numpy as np
import mujoco
import pytest

from kuka_table_sim import load_model, Simulation, TimingConfig
from kuka_table_sim import scenarios as sc
from kuka_table_sim.force_control import ForceApproachController, ForceControlConfig
from kuka_table_sim.scene import ArmIndex


def _fresh_force_sim(**cfg):
    model = load_model(TimingConfig())
    arm = ArmIndex.from_model(model)
    ctl = ForceApproachController(model, arm, TimingConfig.control_period, ForceControlConfig(**cfg))
    return Simulation(model, TimingConfig(), ctl)


def _evaluate(model, q, v, t, ctrl):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:], d.time, d.ctrl[:] = q, v, t, ctrl
    mujoco.mj_forward(model, d)
    return d





def test_control_update_consumes_the_sample_of_the_previous_physics_step(nominal):
    log = nominal.log
    ci, dt = log.meta["control_interval"], log.meta["timestep"]
    c, p = log.ctl, log.phys
    t_ctl, t_ft = c["t"], c["t_ft"]
    assert t_ft[0] == pytest.approx(0.0)                                   # the reset evaluation
    assert np.allclose(t_ctl[1:] - t_ft[1:], dt, atol=1e-12)               # exactly one physics step old
    j = np.arange(1, len(t_ctl))
    assert np.array_equal(c["f_n"][j], p["ft_corr"][ci * j - 1, 2])        # same evaluation as physics row 4j-1
    assert np.array_equal(c["ft_corr"][j], p["ft_corr"][ci * j - 1])
    assert np.array_equal(p["t"][ci * j - 1], t_ft[j])


def test_physics_rows_describe_one_evaluation(nominal):
    """P_task from the tool-point velocity equals qvel . qfrc_constraint of the same evaluation exactly."""
    p = nominal.log.phys
    sel = (p["n_task"] > 0) & (p["n_unexpected"] == 0)
    assert sel.sum() > 50000
    assert np.abs(p["P_task"][sel] - p["P_constraint"][sel]).max() < 1e-9
    # the mixed-stage power (force from x_k with velocity of x_k+1) is measurably wrong where it matters
    err_mixed = np.abs(p["P_mixed"][sel] - p["P_constraint"][sel]).max()
    assert err_mixed > 1e3 * np.abs(p["P_task"][sel] - p["P_constraint"][sel]).max()


def test_work_is_signed_and_integrated_by_the_documented_rectangle_rule(nominal):
    log = nominal.log
    p, c = log.phys, log.ctl
    dt, ci = log.meta["timestep"], log.meta["control_interval"]
    assert np.array_equal(p["dW"], p["P_task"] * dt)
    assert p["W_cum"][-1] == pytest.approx(p["dW"].sum(), rel=1e-12, abs=1e-15)
    assert c["W_interval"].sum() == pytest.approx(p["dW"].sum(), rel=1e-9, abs=1e-12)
    j = 1234
    assert c["W_interval"][j] == pytest.approx(p["dW"][ci * j:ci * (j + 1)].sum(), rel=1e-12, abs=1e-18)
    assert (p["P_task"] > 0).any() and (p["P_task"] < 0).any()              # signed: nothing clipped or absolute-valued
    from kuka_table_sim import metrics as mx
    w = mx.work_stats(log, (9.0, 25.0))
    assert w["negative_J"] < -1.0 and w["positive_J"] > 0                   # sliding friction dissipates, no smoothing
    # sliding with viscous friction: work per full out-stroke is about -c * (path integral of v^2); check sign and size
    assert -1.5 < w["rect_J"] < -1.2



def test_halving_the_physics_timestep_at_a_fixed_control_period_changes_contact_work_little():
    out = {}
    for dt in (TimingConfig.timestep, TimingConfig.timestep / 2):
        res = sc.run_nominal(12.0, timestep=dt, control_period=TimingConfig.control_period)
        t0 = res.log.meta and float(res.log.ctl["t"][np.argmax(res.log.ctl["phase"] >= 1)])
        out[dt] = sc.mx.work_stats(res.log, (t0, 11.99))
        assert res.log.meta["control_interval"] == round(TimingConfig.control_period / dt)
    (a, b) = list(out.values())
    for key in ("rect_J", "trap_J"):
        assert abs(a[key] - b[key]) < 0.02 * abs(a[key]) + 1e-4
