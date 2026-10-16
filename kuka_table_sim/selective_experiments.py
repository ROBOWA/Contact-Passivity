"""Run preregistered selective comparisons in new, exclusive output directories."""
import argparse
from dataclasses import asdict,replace
import gzip
import json
from pathlib import Path
import platform
import time
import numpy as np
import mujoco,scipy,osqp
from .scenarios import make_force_sim
from .oracle_physics import HumanPulse
from .selective_ports import SelectiveConfig
from .selective_execution import SelectiveGuard
from .energy_state import HybridSnapshot
from .energy_diagnostic import progress
from .joint_qp import FilterFailure
from .oracle_experiments import source_hash,save_json,percentiles
from .recorder import git_info,_json_default


def scenarios():
    common=dict(body='sphere_tool',local=[0.,0.,.137],nominal_friction=1.,duration=18.)
    blocking=dict(start=9.5,ramp=.2,hold=1.,magnitude=5.,direction=[-1.,0.,0.])
    help_=dict(start=9.5,ramp=.2,hold=.2,magnitude=5.,direction=[1.,0.,0.])
    changes={
        'A_exact_nohuman':dict(pulses=[]),
        'B_mismatch_nohuman':dict(pulses=[],nominal_friction=.2),
        'C_blocking':dict(pulses=[blocking]),
        'D_combined':dict(pulses=[blocking],nominal_friction=.2),
        'E_forearm_block':dict(pulses=[blocking],body='link5',local=[0.,.04,.12]),
        'F_help_block':dict(pulses=[help_,dict(blocking,start=10.3)]),
        'S_starved':dict(pulses=[blocking],initial_H=.001),
    }
    return {k:dict(common,**v,name=k) for k,v in changes.items()}


class Profile:
    def __init__(self,case):self.pulses=[HumanPulse(**p) for p in case['pulses']]
    def force(self,t):return sum((p.force(t) for p in self.pulses),np.zeros(3))


def metrics(log,c,p,case,mode,wall,ticks,failure,events,cfg):
    h=.00025;n=len(p.get('t',[]));active=p['t']>=8.71-1e-10 if n else np.array([],bool)
    ref,measured,count=progress(log)
    attempts=sum(len(e.get('attempts',[])) for e in events)
    return dict(case=case['name'],mode=mode,completed_s=float(log.phys['t'][-1]+h) if len(log.phys['t']) else 0.,
        completion='STOP' if failure else 'COMPLETE',failure=failure,wall_seconds=wall,tick_seconds=percentiles(ticks),
        initial_J=list(cfg.initial),caps_J=list(cfg.caps),
        final_J=p['energy'][-1].tolist() if n else list(cfg.initial),
        minimum_J=np.minimum(np.min(p['energy'],axis=0),cfg.initial).tolist() if n else list(cfg.initial),
        net_outgoing_J=(-np.sum(p['power'],axis=0)*h).tolist() if n else [0.]*3,
        gross_outgoing_J=(np.sum(np.maximum(-p['power'],0),axis=0)*h).tolist() if n else [0.]*3,
        discarded_J=np.sum(p['discarded'],axis=0).tolist() if n else [0.]*3,
        peak_human_outgoing_W=float(max(0.,-np.min(p['power'][:,0]))) if n else 0.,
        residual_human_max_difference_W=float(np.max(np.abs(p['delta']))) if n else 0.,
        residual_human_difference_work_J=float(np.sum(p['delta'])*h) if n else 0.,
        force_max_error_N=float(np.max(np.abs(p['normal_force'][active]-10))) if np.any(active) else None,
        force_mean_error_N=float(np.mean(np.abs(p['normal_force'][active]-10))) if np.any(active) else None,
        contact_absent_s=float(np.sum(~p['present'][active])*h) if np.any(active) else None,
        measured_progress_m=float(measured[-1]) if len(measured) else 0.,reference_progress_m=float(ref[-1]) if len(ref) else 0.,
        completed_strokes=int(count),intervention_ticks=int(np.sum(c.get('intervention',[]))),qp_attempts=attempts,
        governor_min_alpha=float(np.min(c['alpha'])) if len(c.get('alpha',[])) else 1.,
        max_prediction_error=np.max(p['prediction_error'],axis=0).tolist() if n else None)


def run(case,mode,out,duration=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=SelectiveConfig();duration=duration or case['duration']
    if 'initial_H' in case:cfg=replace(cfg,initial=(case['initial_H'],cfg.initial[1],cfg.initial[2]))
    sim=make_force_sim(duration=duration);g=SelectiveGuard(sim,mode,Profile(case),case['body'],case['local'],case['nominal_friction'],cfg)
    metadata=dict(**git_info(),source_sha256=source_hash(),case=case,mode=mode,config=asdict(cfg),duration=duration,
        versions=dict(python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,scipy=scipy.__version__,osqp=osqp.__version__),
        platform=platform.platform(),plant_friction=g.task_model.plant_friction,nominal_friction=g.task_model.nominal_friction,
        nominal_task_parameter_hash=g.task_model.parameter_hash,forecast='current measured force only; true-plant dynamics oracle',
        account_order=['H','R','W'])
    save_json(out/'metadata.json',metadata)
    ticks=[];failure=None;start=time.perf_counter()
    try:
        while sim.data.time<duration-.000125:
            tick=time.perf_counter();sim.advance(4);ticks.append(time.perf_counter()-tick)
    except (FilterFailure,ValueError) as exc:
        failure=dict(t=float(sim.data.time),reason=getattr(exc,'reason',str(exc)),diagnostics=getattr(exc,'diagnostics',{}))
        if g.snapshot is not None:g.snapshot.save(out/'failure_pre.npz',force=g.force)
        HybridSnapshot.capture(sim,g.energy,g.previous,g.base).save(out/'failure_complete.npz',force=g.force)
        save_json(out/'failure.json',failure)
    wall=time.perf_counter()-start;log=sim.recorder.finish();c,p=g.arrays()
    log.save(out/'baseline_log.npz')
    np.savez_compressed(out/'selective.npz',**{'ctl__'+k:v for k,v in c.items()},**{'phys__'+k:v for k,v in p.items()},tick_seconds=ticks)
    with gzip.open(out/'events.json.gz','wt') as fp:json.dump(g.events,fp,default=_json_default)
    HybridSnapshot.capture(sim,g.energy,g.previous,g.base).save(out/'final_state.npz')
    result=metrics(log,c,p,case,mode,wall,ticks,failure,g.events,cfg);save_json(out/'metrics.json',result)
    metadata['source_end_sha256']=source_hash();metadata['asset_hashes']=log.meta['model_file_sha256'];save_json(out/'metadata.json',metadata)
    print(json.dumps({k:result[k] for k in ('case','mode','completion','completed_s','minimum_J','peak_human_outgoing_W','qp_attempts')},default=_json_default),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['freeze','case'])
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--case',choices=list(scenarios()),default='C_blocking')
    ap.add_argument('--mode',choices=['K0','K1','K2','K3','K4'],default='K4');ap.add_argument('--duration',type=float)
    a=ap.parse_args()
    if a.command=='freeze':
        if a.out.exists():raise FileExistsError(a.out)
        save_json(a.out,dict(config=asdict(SelectiveConfig()),scenarios=scenarios(),starting_revision='263136e22d721e1e72ec7325c71647ce4167a6d4'));return 0
    r=run(scenarios()[a.case],a.mode,a.out,a.duration);return 0 if r['completion']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
