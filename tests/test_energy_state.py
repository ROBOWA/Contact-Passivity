import numpy as np
import mujoco

from kuka_table_sim.energy_state import ProgressClock,GovernedController,HybridSnapshot,clone_controller,controller_bytes
from kuka_table_sim.scenarios import make_force_sim


def test_virtual_clock_chain_rule_integral_and_smooth_restart():
    c=ProgressClock(.1);c.advance(10.);c.set_target(0.)
    sigma=[];alpha=[];speed=[]
    for t in np.linspace(10,10.1,1001):
        c.advance(t);sigma.append(c.sigma);alpha.append(c.alpha);speed.append(c.alpha_dot)
    assert abs(c.sigma-10.05)<1e-12
    assert c.alpha==0 and c.alpha_dot==0
    assert max(np.abs(speed))<40 and min(alpha)>=0 and max(alpha)<=1
    # X(sigma)=sigma²: independently finite difference physical position.
    x=np.array(sigma)**2
    v=2*np.array(sigma)*alpha
    assert np.max(np.abs(np.gradient(x,1e-4)[2:-2]-v[2:-2]))<3e-5
    c.advance(10.5);assert abs(c.sigma-10.05)<1e-12
    c.set_target(1.);c.advance(10.6)
    assert c.alpha==1 and c.alpha_dot==0 and abs(c.sigma-10.1)<1e-12


def test_governed_enabled_unit_speed_is_bitwise_nominal_and_reset():
    sims=[make_force_sim(record=False) for _ in range(2)]
    sims[1].controller=GovernedController.from_controller(sims[1].controller)
    for s in sims:s.advance(s.steps_for(.2))
    assert np.array_equal(sims[0].q,sims[1].q)
    assert np.array_equal(sims[0].qvel,sims[1].qvel)
    s=sims[1];s.controller.stroke.clock.set_target(0.);s.advance(400);s.reset()
    assert s.controller.stroke.clock.alpha==1 and s.controller.stroke.clock.sigma==0
    assert s.n_steps==0 and s.controller.f_int==0


def test_hybrid_snapshot_deep_isolation_serialization_and_fork_replay(tmp_path):
    s=make_force_sim(record=False)
    s.controller=GovernedController.from_controller(s.controller)
    s.advance(400)
    snap=HybridSnapshot.capture(s,[.1,.2],s.tau,np.zeros(7));before=controller_bytes(s.controller)
    a=snap.fork(s);b=snap.fork(s)
    assert a.controller.model is s.model and a.controller.ft.model is s.model
    assert not np.shares_memory(a.controller.tan_int,s.controller.tan_int)
    a.controller.tan_int[0]=.01
    assert before==controller_bytes(s.controller)
    a=snap.fork(s)
    for sim in (a,b):sim.advance(40)
    assert np.array_equal(a.q,b.q) and np.array_equal(a.qvel,b.qvel)
    assert controller_bytes(a.controller)==controller_bytes(b.controller)
    assert before==controller_bytes(s.controller)
    snap.save(tmp_path/'s.npz');loaded=HybridSnapshot.load(tmp_path/'s.npz')
    assert loaded.signature()==snap.signature()
    c=loaded.fork(s);c.advance(40)
    assert np.array_equal(c.q,a.q)
