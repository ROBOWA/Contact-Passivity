"""Physics witness for ACTIVE Stage 2 task rows; not controller tuning.

An explicit additional 20 N downward nominal-command fault creates a real force
band violation. Default [7,13] N rows must correct it, then an independent split
plant execution checks the returned torque and all four work/force samples.
"""
import mujoco
import numpy as np
from kuka_table_sim.scenarios import make_force_sim
from kuka_table_sim.energy_state import GovernedController,HybridSnapshot
from kuka_table_sim.energy_prediction import BranchPredictor,GuardConfig
from kuka_table_sim.energy_qp import filtered_interval
from kuka_table_sim.oracle_physics import TaskMeter,integration_state


def test_active_task_qp_with_nominal_command_fault_and_independent_execution():
    sim=make_force_sim(record=False)
    sim.controller=GovernedController.from_controller(sim.controller)
    sim.advance(sim.steps_for(9.6))
    snap=HybridSnapshot.capture(sim,[1.,1.],sim.tau,np.zeros(7))
    class FaultedNominal(BranchPredictor):
        def prepare(self,state,target=None):
            branch,nominal=super().prepare(state,target)
            J=np.zeros((3,self.model.nv))
            mujoco.mj_jacSite(self.model,branch.data,J,None,self.tcp)
            return branch,nominal-J.T@np.array([0.,0.,20.])
    pred=FaultedNominal(sim,'link5',(0,.04,.12),GuardConfig());force=np.array([-5.,0,0])
    bad=pred.interval(snap,force,1.)
    assert np.max(bad.trace['normal_force'])>13.
    good,attempts,_=filtered_interval(pred,snap,force,1.,True)
    assert attempts and attempts[-1]['accepted'] and len(attempts[-1]['names'])==32
    assert pred.task_ok(good.trace)
    assert snap.signature()==HybridSnapshot.capture(sim,[1.,1.],sim.tau,np.zeros(7)).signature()
    # Independently apply the accepted complete command. No predictor execution
    # helper, no predicted work copied into the observed ledger.
    actual=snap.fork(sim);m,d=actual.model,actual.data;body=m.body('link5').id
    meter=TaskMeter(m);work=0.;prefix=[];forces=[]
    d.ctrl[actual.arm.actuator_ids]=good.torque
    for _ in range(4):
        t=float(d.time);mujoco.mj_step1(m,d)
        point=d.xpos[body]+d.xmat[body].reshape(3,3)@np.array([0.,.04,.12])
        generalized=np.zeros(m.nv);mujoco.mj_applyFT(m,d,force,np.zeros(3),point,body,generalized)
        work+=float(generalized@d.qvel)*m.opt.timestep;prefix.append(work)
        d.qfrc_applied[:]=snap.base_qfrc+generalized
        mujoco.mj_step2(m,d);actual.controller.observe(m,d,t)
        forces.append(meter.sample(d)['F'][2])
        assert np.allclose(d.qfrc_actuator[actual.arm.dof_adr],good.torque,rtol=0,atol=1e-10)
    assert np.min(forces)>=7.-1e-7 and np.max(forces)<=13.+1e-7
    assert np.allclose(prefix,good.trace['prefix'],rtol=0,atol=1e-15)
    assert np.allclose(forces,good.trace['normal_force'],rtol=0,atol=1e-10)
    assert np.array_equal(integration_state(m,d),good.next.plant)
