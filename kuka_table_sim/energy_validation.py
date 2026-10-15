"""Independent Stage 2 replay, stencil and forecast-mismatch measurements."""
from pathlib import Path
import argparse
import numpy as np
from .scenarios import make_force_sim
from .oracle import OracleFilter
from .oracle_physics import HumanPulse
from .energy_state import HybridSnapshot,GovernedController
from .energy_prediction import BranchPredictor,GuardConfig
from .oracle_experiments import save_json,source_hash
from .recorder import git_info
from .joint_qp import QPConfig


class Measured:
    varying=False
    def __init__(self,force,epoch,vary=False):self.value=force;self.epoch=epoch;self.varying=vary
    def force(self,t):
        return self.value*(1.+.5*np.sin(2*np.pi*100*(t-self.epoch))) if self.varying else self.value.copy()


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(7342)
    common=make_force_sim(duration=11.,record=False)
    common.controller=GovernedController.from_controller(common.controller)
    f=OracleFilter(common,'off',HumanPulse(start=9.5),energy_init=(1.,1.));common.oracle=f
    records=[];arrays={};snapshots=[]
    for t in (6.700,6.701,6.702,6.703,6.704,8.71,9.6,10.18):
        common.advance(common.steps_for(t-common.data.time))
        s=HybridSnapshot.capture(common,f.energy,f.previous,f.base_qfrc);snapshots.append(s)
        branch=s.fork(common);pred=BranchPredictor(branch,'link5',(0,.04,.12),GuardConfig());force=f.pulse.force(t)
        base=pred.interval(s,force,1.);fields=('prefix','normal_force','normal_velocity','cross_error','orientation')
        derivatives={};etas=(.01,.02,.04)
        for eta in etas:
            grad={k:np.empty(base.trace[k].shape+(7,)) for k in fields}
            for j in range(7):
                delta=np.eye(7)[j]*eta
                plus=pred.interval(s,force,1.,base.nominal+delta);minus=pred.interval(s,force,1.,base.nominal-delta)
                for key in fields:grad[key][...,j]=(plus.trace[key]-minus.trace[key])/(2*eta)
            derivatives[eta]=grad
        errors={k:[] for k in fields};mode_crossings=0
        deltas=rng.uniform(-1,1,size=(24,7))*np.asarray(QPConfig().rho)*.2
        for delta in deltas:
            actual=pred.interval(s,force,1.,base.nominal+delta)
            mode_crossings+=int(not np.array_equal(actual.trace['present'],base.trace['present']))
            for k in fields:
                error=base.trace[k]+derivatives[.02][k]@delta-actual.trace[k]
                errors[k].append(error)
        arrays[str(t)+'_deltas']=deltas
        for k,v in errors.items():arrays[str(t)+'_'+k+'_errors']=np.asarray(v)
        records.append(dict(t=t,force=force,contact_mode_crossings=mode_crossings,samples=len(deltas),
                            max_abs_affine_errors={k:float(np.max(np.abs(v))) for k,v in errors.items()},
                            stencil_spread={k:float(max(np.max(np.abs(derivatives[e][k]-derivatives[.02][k])) for e in etas)) for k in fields}))
        assert s.signature()==HybridSnapshot.capture(common,f.energy,f.previous,f.base_qfrc).signature()
    # Independent production Simulation loop, own OracleFilter off execution and
    # generalized dynamics; no BranchPredictor advance helper on execution side.
    snap=snapshots[-1];force=np.array([-5.,0,0]);private=snap.fork(common)
    exact=OracleFilter(private,'off',Measured(force,float(private.data.time)),energy_init=snap.energy)
    exact.previous=snap.previous.copy();exact.base_qfrc=snap.base_qfrc.copy();private.oracle=exact
    template=snap.fork(common);pred=BranchPredictor(template,'link5',(0,.04,.12),GuardConfig());current=snap
    errors=[]
    for _ in range(20):
        p=pred.interval(current,force,1.);private.advance(4)
        actual=HybridSnapshot.capture(private,exact.energy,exact.previous,exact.base_qfrc)
        errors.append(np.max(np.abs(actual.plant-p.next.plant)))
        assert np.allclose(p.next.plant,actual.plant,atol=1e-11,rtol=0)
        current=p.next
    mismatch=snap.fork(common);off=OracleFilter(mismatch,'off',Measured(force,float(mismatch.data.time),True),energy_init=snap.energy)
    off.previous=snap.previous.copy();off.base_qfrc=snap.base_qfrc.copy();mismatch.oracle=off
    held=pred.interval(snap,force,1.);mismatch.advance(4)
    defect=held.work-float(off.rows[-1]['actual'][0])
    result=dict(**git_info(),source_sha256=source_hash(),seed=7342,snapshots=records,
                closed_loop_20ms_max_state_error=float(max(errors)),
                forecast_mismatch=dict(mode='shadow only; 100 Hz within-tick force modulation; no held-force guarantee',
                                       held_work_J=held.work,actual_work_J=float(off.rows[-1]['actual'][0]),prediction_minus_actual_J=defect),
                interpretation='Stencil errors are measurements, not uniform envelopes. All candidates require fresh nonlinear validation.')
    np.savez_compressed(out/'validation.npz',**arrays)
    save_json(out/'validation.json',result);print(result)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path);run(ap.parse_args().out)
