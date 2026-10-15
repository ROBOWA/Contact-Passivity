"""Shadow-only independent execution of a rejected backup from its full snapshot.

Uses the original Simulation.step, own mj_applyFT force callback and a controller
subclass. Production BranchPredictor.advance is NOT used for execution. Negative
shadow energy is retained to verify that the online guard correctly rejected it.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kuka_table_sim.scenarios import make_force_sim
from kuka_table_sim.energy_state import HybridSnapshot,GovernedController
from kuka_table_sim.energy_prediction import BranchPredictor,GuardConfig,backup_rollout
from kuka_table_sim.oracle_physics import integration_state,TaskMeter
from kuka_table_sim.oracle_experiments import source_hash


class IndependentBackup(GovernedController):
    def compute(self,data):
        self.stroke.clock.advance(float(data.time));self.stroke.clock.set_target(0.)
        nominal=super().compute(data)
        point=data.xpos[self.audit_body]+data.xmat[self.audit_body].reshape(3,3)@self.audit_local
        J=np.zeros((3,self.model.nv));mujoco.mj_jac(self.model,data,J,None,point,self.audit_body)
        torque=nominal-J.T@(self.audit_damping*(J@data.qvel))
        self.filter_intervened=bool(np.max(np.abs(torque-nominal))>1e-7)
        return torque

    def observe(self,model,data,t):
        super().observe(model,data,t)
        if self.audit_pending is not None:
            record=self.audit_pending
            record['normal_force']=self.audit_meter.sample(data)['F'][2]
            record['torque']=data.qfrc_actuator[self.arm.dof_adr].copy()
            self.audit_rows.append(record);self.audit_pending=None


def run(snapshot_path,config_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    snap=HybridSnapshot.load(snapshot_path);cfg=GuardConfig(**json.loads(Path(config_path).read_text()))
    with np.load(snapshot_path) as z:force=z['force'].copy()
    template=make_force_sim(record=False);pred=BranchPredictor(template,'link5',(0,.04,.12),cfg)
    plan=backup_rollout(pred,snap,force)
    if plan.status!='SETTLED':raise ValueError('audit requires a settled but unaffordable backup')
    actual=snap.fork(template);ctl=actual.controller;ctl.__class__=IndependentBackup
    ctl.audit_body=actual.model.body('link5').id;ctl.audit_local=np.array([0.,.04,.12]);ctl.audit_damping=cfg.damping
    ctl.audit_pending=None;ctl.audit_rows=[];ctl.audit_meter=TaskMeter(actual.model)
    def apply_force(t,data):
        point=data.xpos[ctl.audit_body]+data.xmat[ctl.audit_body].reshape(3,3)@ctl.audit_local
        generalized=np.zeros(actual.model.nv)
        mujoco.mj_applyFT(actual.model,data,force,np.zeros(3),point,ctl.audit_body,generalized)
        power=float(generalized@data.qvel)
        data.qfrc_applied[:]=snap.base_qfrc+generalized
        ctl.audit_pending=dict(t=t,power=power)
    actual.disturbance=apply_force
    state_errors=[]
    for interval in plan.intervals:
        actual.advance(4)
        state_errors.append(np.max(np.abs(integration_state(actual.model,actual.data)-interval.next.plant)))
    power=np.array([r['power'] for r in ctl.audit_rows]);prefix=np.cumsum(power*.00025)
    predicted_prefix=np.cumsum(np.concatenate([p.trace['power'] for p in plan.intervals])*.00025)
    result=dict(scope='ISOLATED SHADOW AUDIT ONLY; rejected action was never executed by online trial',
                source_sha256=source_hash(),snapshot=str(Path(snapshot_path).resolve()),held_force=force.tolist(),
                terminal_status=plan.status,terminal_time_s=len(plan.intervals)*.001,
                reserve_from_independent_generalized_work_J=max(0.,-float(np.min(prefix))),
                reserve_from_predictor_J=plan.reserve,max_prefix_work_difference_J=float(np.max(np.abs(prefix-predicted_prefix))),
                max_complete_plant_state_difference=float(max(state_errors)),
                initial_accounts_J=snap.energy.tolist(),minimum_shadow_energy_J=(snap.energy+np.min(prefix)).tolist(),
                conclusion='An unaffordable backup is rejected; negative shadow energy is evidence, not an accepted energy result.')
    np.savez_compressed(out/'shadow.npz',power=power,work_prefix=prefix,predicted_prefix=predicted_prefix,
                        normal_force=np.array([r['normal_force'] for r in ctl.audit_rows]),
                        torque=np.array([r['torque'] for r in ctl.audit_rows]))
    with (out/'audit.json').open('x') as f:json.dump(result,f,indent=2)
    assert result['max_complete_plant_state_difference']<1e-10
    assert result['max_prefix_work_difference_J']<1e-11
    print(json.dumps(result))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--snapshot',required=True,type=Path);a.add_argument('--config',required=True,type=Path);a.add_argument('--out',required=True,type=Path)
    args=a.parse_args();run(args.snapshot,args.config,args.out)
