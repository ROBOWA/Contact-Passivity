from dataclasses import replace
import numpy as np
import pytest
import mujoco

from kuka_table_sim.scenarios import make_force_sim
from kuka_table_sim.energy_state import HybridSnapshot,GovernedController,controller_bytes
from kuka_table_sim.energy_prediction import GuardConfig,BranchPredictor,backup_rollout
from kuka_table_sim.energy_qp import Row,solve_multirow,filtered_interval
from kuka_table_sim.energy_guard import EnergyGuard,EnergySimulation
from kuka_table_sim.joint_qp import FilterFailure
from kuka_table_sim.oracle_physics import HumanPulse


def test_coupled_active_rows_analytic_and_infeasible():
    e=np.eye(7);H=np.eye(7);nom=np.zeros(7);lim=np.ones(7)*10
    rows=[Row('sum',0.,e[0]+e[1],1.,np.inf,1e-8),Row('x',0.,e[0],.6,np.inf,1e-8)]
    tau,d=solve_multirow(nom,nom,H,rows,nom,lim,lim,lim,1.)
    assert np.allclose(tau[:2],[.6,.4],atol=1e-8)
    assert np.max(d['violation'])<1e-8
    rows.append(Row('opposite',0.,e[0]+e[1],-np.inf,.9,1e-8))
    with pytest.raises(FilterFailure,match='affine_constraints_infeasible'):
        solve_multirow(nom,nom,H,rows,nom,lim,lim,lim,1.)


def test_independent_execution_prefix_and_state_order():
    s=make_force_sim(duration=1.)
    s.recorder=None
    s.controller=GovernedController.from_controller(s.controller)
    snap=HybridSnapshot.capture(s,[.1,.2],s.tau,np.zeros(s.model.nv))
    pred=BranchPredictor(s,'link5',(0,.04,.12),GuardConfig())
    force=np.array([-5.,0,0]);a=pred.interval(snap,force)
    pred.interval(snap,force,torque=a.nominal+np.arange(7)*.01)
    again=pred.interval(snap,force)
    assert np.array_equal(a.next.plant,again.next.plant)
    assert a.next.controller==again.next.controller
    assert HybridSnapshot.capture(s,[.1,.2],s.tau,np.zeros(s.model.nv)).signature()==snap.signature()
    # Independently execute using mj_applyFT and a fresh controller compute,
    # not BranchPredictor.advance/interval. Work from generalized force power.
    b=snap.fork(s);m,d=b.model,b.data
    mujoco.mj_step1(m,d);tau=b.controller.compute(d);d.ctrl[b.arm.actuator_ids]=tau
    body=m.body('link5').id;prefix=[];work=0.
    for j in range(4):
        t=float(d.time);mujoco.mj_step1(m,d)
        point=d.xpos[body]+d.xmat[body].reshape(3,3)@np.array([0,.04,.12])
        qforce=np.zeros(m.nv);mujoco.mj_applyFT(m,d,force,np.zeros(3),point,body,qforce)
        work+=float(qforce@d.qvel)*m.opt.timestep;prefix.append(work)
        d.qfrc_applied[:]=qforce;mujoco.mj_step2(m,d);b.controller.observe(m,d,t)
        assert np.allclose(d.qfrc_actuator[b.arm.dof_adr],tau,atol=1e-12,rtol=0)
    assert np.allclose(prefix,a.trace['prefix'],atol=1e-16,rtol=0)
    assert np.array_equal(HybridSnapshot.capture(b,[.1,.2],tau,np.zeros(m.nv)).plant,a.next.plant)
    assert controller_bytes(b.controller)==a.next.controller


def test_disabled_and_nohuman_transparency_and_one_commit():
    a=make_force_sim(duration=.05);b=make_force_sim(duration=.05)
    a.recorder=b.recorder=None
    b.__class__=EnergySimulation;b.guard=None
    for _ in range(30):a.advance(4);b.advance(4)
    assert np.array_equal(a.data.qpos,b.data.qpos)
    g=EnergyGuard(b,'D',HumanPulse(magnitude=0.),'link5',(0,.04,.12),(0.,0.))
    for _ in range(10):a.advance(4);b.advance(4)
    assert np.array_equal(a.data.qpos,b.data.qpos)
    assert a.n_control_updates==b.n_control_updates==40
    assert np.array_equal(g.energy,[0.,0.])
    assert np.max(np.abs(np.array([r['predicted']-r['actual'] for r in g.rows])))==0.


def test_different_account_minima_and_unresolved_is_not_zero():
    s=make_force_sim(duration=.05);s.recorder=None
    snap=HybridSnapshot.capture(s,[.2,.3],s.tau,np.zeros(s.model.nv))
    pred=BranchPredictor(s,'link5',(0,.04,.12),replace(GuardConfig(),horizons=(.001,)))
    # Bound after signed work, not after comparing raw identical work accounts.
    p,_,_=filtered_interval(pred,snap,np.zeros(3),minimum=[.2,.1])
    assert np.array_equal(p.next.energy,[.2,.3])
    with pytest.raises(FilterFailure):filtered_interval(pred,snap,np.zeros(3),minimum=[.2001,.1])
    b=backup_rollout(pred,snap,np.array([-5.,0,0]))
    assert b.status!='SETTLED'
    assert b.status in ('UNRESOLVED','INVALID')


def test_contact_backup_prefix_reserve_cache_and_actual_execution(monkeypatch):
    s=make_force_sim(duration=10.,record=False);s.advance(s.steps_for(9.6))
    # At this point real force is newly measured; future policy has no pulse.
    class MeasuredForce:
        def force(self,t):
            assert abs(t-s.data.time)<1e-10 # predictions may never query this future fixture
            return np.array([-5.,0,0])
    g=EnergyGuard(s,'D',MeasuredForce(),'link5',(0,.04,.12),(1.,.9),minimum=(.1,.2))
    before=HybridSnapshot.capture(s,g.energy,g.previous,g.base)
    p=g.predictor.interval(before,np.array([-5.,0,0]),1.)
    perturbed=g.predictor.interval(before,np.array([-5.,0,0]),1.,p.nominal+np.arange(7)*.1)
    assert p.trace['prefix'][0]==perturbed.trace['prefix'][0]
    assert p.work!=perturbed.work
    backup=backup_rollout(g.predictor,before,np.array([-5.,0,0]))
    assert backup.status=='SETTLED'
    cumulative=np.cumsum(np.concatenate([p.trace['power'] for p in backup.intervals])*.00025)
    assert abs(backup.reserve-max(0.,-np.min(cumulative)))<1e-14
    assert len(backup.horizons)>=3
    assert all(g.predictor.task_ok(p.trace) for p in backup.intervals)
    for _ in range(4):s.advance(4)
    assert sum(r['cache_hit'] for r in g.rows)==3
    assert max(abs(r['predicted']-r['actual']) for r in g.rows)<1e-12
    assert all(r['alpha']==1 for r in g.rows) # ample budget remains transparent
    assert np.min(g.energy-g.minimum)>0
    def reject(*args,**kwargs):raise FilterFailure('test_rejected_proposal')
    monkeypatch.setattr(g,'plan',reject);g.cache_steps=0
    s.advance(4) # verified suffix remains available after a rejected replan
    assert g.rows[-1]['cache_hit']
    g.cache_force=np.array([-4.,0,0]) # exact force invalidation, no reuse
    snapshot=HybridSnapshot.capture(s,g.energy,g.previous,g.base)
    with pytest.raises(FilterFailure,match='no_verified_reserve_candidate'):s.advance(4)
    assert HybridSnapshot.capture(s,g.energy,g.previous,g.base).signature()==snapshot.signature()
    s.reset();assert np.array_equal(g.energy,g.initial)
    assert not g.cache and g.cache_steps==0 and g.snapshot is None
    assert s.controller.stroke.clock.alpha==1
