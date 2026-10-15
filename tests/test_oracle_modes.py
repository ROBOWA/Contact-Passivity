from dataclasses import replace

import mujoco
import numpy as np

from kuka_table_sim.oracle import OracleFilter
from kuka_table_sim.oracle_physics import HumanPulse
from kuka_table_sim.scenarios import make_force_sim


def test_no_human_modes_are_transparent_to_plant_controller_and_accounts():
    output = []
    for mode in (None, "off", "shadow", "active"):
        s = make_force_sim(record=False)
        if mode:
            s.oracle = OracleFilter(s, mode, HumanPulse(magnitude=0.), energy_init=(0.,0.))
        s.advance(s.steps_for(.15))
        output.append((s.q.copy(), s.qvel.copy(), s.tau.copy()))
        if mode:
            c, _ = s.oracle.arrays()
            assert np.all(c["actual"] == 0) and np.all(c["energy_after"] == 0)
            assert np.all(c["intervention"] == 0)
    for out in output[1:]:
        assert all(np.array_equal(x,y) for x,y in zip(out,output[0]))


def test_shadow_accounts_use_executed_work_without_lower_clipping_and_reset():
    s = make_force_sim(record=False)
    # Seed measurable motion without a long approach, then apply resistance.
    s.data.qvel[:] = [.2, .1, 0., 0., 0., 0., 0.]
    mujoco.mj_step1(s.model,s.data)
    pulse = HumanPulse(start=0., ramp=.001, hold=.1)
    s.oracle = OracleFilter(s, "shadow", pulse, energy_init=(0.,0.))
    s.advance(s.steps_for(.05))
    c, p = s.oracle.arrays()
    actual = np.sum(c["actual"],axis=0)
    np.testing.assert_allclose(s.oracle.energy, actual, atol=1e-15,rtol=0)
    assert np.min(c["energy_after"]) < 0  # diagnostic violations retained
    assert np.max(np.abs(c["predicted"] - c["actual"])) < 1e-11
    assert np.array_equal(c["actual"][:,0], c["actual"][:,1])
    assert np.max(np.abs(p["power_identity"])) < 1e-12
    s.reset()
    assert np.array_equal(s.oracle.energy, np.zeros(2)) and not s.oracle.rows
    assert not s.controller.filter_intervened
