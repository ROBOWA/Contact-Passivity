from dataclasses import replace
import copy
import numpy as np
import pytest
from kuka_table_sim.selective_ports import SelectiveConfig,capped_update,account_powers,TaskModel
from kuka_table_sim.selective_prediction import SelectivePredictor
from kuka_table_sim.selective_qp import constraints
from kuka_table_sim.selective_execution import SelectiveGuard
from kuka_table_sim.selective_experiments import Profile,scenarios
from kuka_table_sim.energy_state import GovernedController,HybridSnapshot,controller_bytes
from kuka_table_sim.scenarios import make_force_sim
from kuka_table_sim.oracle_physics import integration_state


def test_cap_discards_only_charge_and_preserves_negative_account():
    e=np.array([.12,.30,.30]);work=np.array([.1,-.4,.01])
    after,discard=capped_update(e,work,(.15,.30,.30))
    np.testing.assert_allclose(after,[.15,-.1,.30],atol=1e-16)
    np.testing.assert_allclose(e+work-discard,after,atol=1e-16)
    np.testing.assert_allclose(discard,[.07,0,.01],atol=1e-16)


def test_power_ports_have_correct_sign_and_independent_residual():
    p,d=account_powers(-.1,-.3,-.2)
    np.testing.assert_allclose(p,[-.1,-.2,-.4]);assert d==pytest.approx(-.1)


@pytest.fixture(scope='module')
def contact_snapshot():
    sim=make_force_sim(record=False)
    sim.advance(sim.steps_for(9.8))
    sim.controller=GovernedController.from_controller(sim.controller)
    snap=HybridSnapshot.capture(sim,np.array([.12,.3,.3]),sim.data.ctrl[sim.arm.actuator_ids],sim.data.qfrc_applied)
    return sim,snap


def test_independent_pair_model_and_exact_account_identity(contact_snapshot):
    sim,s=contact_snapshot;original=sim.model.pair_friction.copy();cfg=SelectiveConfig()
    exact=SelectivePredictor(sim,'sphere_tool',[0,0,.137],1.,cfg)
    mismatch=SelectivePredictor(sim,'sphere_tool',[0,0,.137],.2,cfg)
    assert exact.task_model.model is not sim.model
    before=integration_state(sim.model,sim.data).copy();controller=controller_bytes(sim.controller)
    a=exact.interval(s,np.zeros(3));b=mismatch.interval(s,np.zeros(3))
    np.testing.assert_array_equal(sim.model.pair_friction,original)
    np.testing.assert_array_equal(integration_state(sim.model,sim.data),before)
    assert controller_bytes(sim.controller)==controller
    np.testing.assert_allclose(a.trace['delta'],0,atol=1e-13,rtol=0)
    assert np.max(np.abs(b.trace['delta']))>1e-3
    np.testing.assert_array_equal(a.next.plant,b.next.plant)
    assert np.all(b.trace['power'][:,0]==0)
    np.testing.assert_allclose(b.trace['power'][:,1],b.trace['delta'],atol=0)


def test_governor_modes_use_identical_physical_constraints(contact_snapshot):
    sim,s=contact_snapshot;cfg=SelectiveConfig();pred=SelectivePredictor(sim,'sphere_tool',[0,0,.137],.2,cfg)
    force=np.array([-2.,0,0]);p=pred.interval(s,force)
    n3,v3,t3=constraints(p,s,force,'K3',cfg);n4,v4,t4=constraints(p,s,force,'K4',cfg)
    assert n3==n4;np.testing.assert_array_equal(v3,v4);np.testing.assert_array_equal(t3,t4)
    i=n3.index('H_terminal_peak_force_rate')
    v=p.terminal_human_velocity
    assert v3[i]==pytest.approx(cfg.peak_human_W+force@v-.06*np.linalg.norm(v)-cfg.power_guard_W)


def test_prediction_execution_full_account_replay(contact_snapshot):
    template,s=contact_snapshot;sim=s.fork(template)
    class Held:
        def force(self,t):return np.array([-2.,0,0])
    g=SelectiveGuard(sim,'K0',Held(),'sphere_tool',[0,0,.137],.2,SelectiveConfig())
    g.last_force=np.array([-2.,0,0]);sim.advance(4)
    c,p=g.arrays();assert len(p['t'])==4
    assert np.max(p['prediction_error'])<1e-12
    np.testing.assert_allclose(p['power'][:,0],np.einsum('ij,ij->i',p['force'],p['human_velocity']),atol=1e-15)
    np.testing.assert_allclose(p['power'][:,1],p['power'][:,0]+p['task_power']-p['task_hat_power'],atol=1e-15)
    np.testing.assert_allclose(p['power'][:,0],np.einsum('ij,ij->i',p['human_tau'],p['qd']),atol=1e-14)


def test_script_satisfies_declared_causal_force_class():
    for case in scenarios().values():
        f=np.array([Profile(case).force(t) for t in np.arange(0,18,.001)])
        assert np.max(np.linalg.norm(np.diff(f,axis=0),axis=1))<=.06+1e-12


def test_invalid_account_initialization_rejected():
    with pytest.raises(ValueError):SelectiveConfig(initial=(.2,.3,.3))
