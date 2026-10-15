"""Exclusive Stage 2 A/B/C/D/UF experiments; original CLI defaults are untouched.

python -m kuka_table_sim.energy_experiments case --case main --variant D --out NEW
python -m kuka_table_sim.energy_experiments matrix --out NEW --heldout MANIFEST
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time
import subprocess
import numpy as np
import mujoco,osqp,scipy

from .scenarios import make_force_sim
from .oracle import OracleFilter
from .oracle_physics import HumanPulse
from .joint_qp import QPConfig,FilterFailure
from .energy_state import HybridSnapshot
from .energy_prediction import GuardConfig
from .energy_guard import EnergyGuard
from .energy_diagnostic import progress
from .oracle_experiments import save_json,source_hash,percentiles
from .recorder import git_info,_json_default
from . import metrics as mx


def development_cases():
    base=dict(magnitude=5.,start=9.5,ramp=.2,hold=1.,direction=[-1.,0.,0.],body='link5',local=[0.,.04,.12],budget=.12,rate_factor=1.)
    changes={'nohuman-ample':dict(magnitude=0.,budget=1.),'nohuman-zero':dict(magnitude=0.,budget=0.),
             'ample':dict(budget=1.),'main':{},'historical':dict(magnitude=4.5,start=10.1,ramp=.15,hold=.8),
             'elbow-ample':dict(body='link3',local=[.02,.04,.16],budget=1.),
             'elbow-tight':dict(body='link3',local=[.02,.04,.16]),'assist':dict(magnitude=-5.),
             'stress':dict(budget=.001),'reduced-rate':dict(rate_factor=.02)}
    return {k:dict(base,**v,name=k) for k,v in changes.items()}


def metrics(log,c,p,case,variant,wall,ticks,failed,config):
    """Recompute from raw EXECUTED powers/contact arrays; never use predictions."""
    dt=log.meta['timestep'];t=p['t'];window=t>=config.task_start-1e-10
    w=p['power']*dt;E=case['budget']+np.cumsum(w)
    fn=log.phys['F_task'][:,2];present=log.phys['present'];mask=log.phys['t']>=config.task_start-1e-10
    err=np.abs(fn[mask]-10.)
    ref,measured,completed=progress(log)
    tracking=mx.tracking_stats(log,(config.task_start,float(log.phys['t'][-1])))
    cross=float(np.max(np.abs(p['cross_error'][window]))*1000) if 'cross_error' in p else tracking['cross_max_mm']
    ori=float(np.max(np.linalg.norm(p['orientation'][window],axis=1))*180/np.pi) if 'orientation' in p else tracking['ori_max_deg']
    alpha=c.get('alpha',np.ones(len(c['t'])))
    release=case['start']+2*case['ramp']+case['hold']
    recover=np.flatnonzero((c['t']>=release-1e-10)&(alpha>=.95))
    alpha_recovery=float(c['t'][recover[0]]-release) if len(recover) else None
    after=np.flatnonzero(log.phys['t']>=release-1e-10);force_recovery=None
    good=np.abs(fn-10.)<=.5;required=round(.5/dt)
    if len(after)>=required:
        consecutive=np.convolve((~good[after]).astype(int),np.ones(required,dtype=int),mode='valid')
        at=np.flatnonzero(consecutive==0)
        if len(at):force_recovery=float(log.phys['t'][after[at[0]]]-release)
    torque=c['tau_exec'];rate=np.asarray(QPConfig().rate)*case.get('rate_factor',1.)*.001
    rate_residual=float(np.max(np.abs(torque-c['tau_previous'])-rate))
    limits=np.array([320,320,176,176,110,40,40.])
    bound_residual=float(np.max(np.abs(torque)-limits))
    prediction=c['predicted']
    actual=c['actual']
    if prediction.ndim==2:prediction=prediction[:,0];actual=actual[:,0]
    finite=np.isfinite(prediction)
    prediction_error=float(np.max(np.abs(prediction[finite]-actual[finite]))) if np.any(finite) else None
    absence=float(np.sum(~present[mask])*dt)
    contact_ok=absence==0 and float(np.max(err))<=3.+1e-7 and cross<=3.+1e-4 and ori<=2.+1e-4
    complete=failed is None and float(t[-1]+dt)>=18.-1e-8
    return dict(variant=variant,case=case['name'],completed_s=float(t[-1]+dt),failure=failed,
                completion='PASS' if complete else 'FAIL',energy='PASS' if np.min(E)>=-1e-10 else 'FAIL',
                contact='PASS' if contact_ok else 'FAIL',net_outgoing_J=-float(np.sum(w)),gross_outgoing_J=float(np.sum(np.maximum(-w,0))),
                min_substep_energy_J=float(np.min(E)),final_energy_J=float(E[-1]),force_mae_N=float(np.mean(err)),
                force_max_error_N=float(np.max(err)),force_peak_N=float(np.max(fn[mask])),contact_absence_s=absence,
                cross_max_mm=cross,orientation_max_deg=ori,reference_progress_m=float(ref[-1]),measured_progress_m=float(measured[-1]),
                completed_strokes=completed,pause_s=float(np.sum((alpha<=.01)&(c['t']>=config.task_start))*.001),
                slowed_s=float(np.sum((alpha<.95)&(c['t']>=config.task_start))*.001),alpha_recovery_s=alpha_recovery,
                force_recovery_s=force_recovery,minimum_alpha=float(np.min(alpha)),failure_count=int(failed is not None),
                torque_bound_residual_Nm=bound_residual,rate_bound_residual_Nm=rate_residual,
                prediction_max_abs_J=prediction_error,power_identity_max_W=float(np.max(np.abs(p['power_identity']))),
                account_independent_recompute_max_J=float(np.max(np.abs((p.get('energy',p.get('prefix_energy'))[:,0])-E))),
                wall_seconds=wall,realtime_factor=float((t[-1]+dt)/wall),tick_seconds=percentiles(ticks),
                deadline_misses=int(np.sum(np.asarray(ticks)>.001)),
                timings={k:percentiles(c[k][c[k]>0]) for k in c if k.startswith('seconds_')},
                interventions=int(np.sum(np.max(np.abs(torque-c['tau_nom']),axis=1)>1e-7)),
                governor_ticks=int(np.sum(alpha<1.-1e-10)),
                task_window_start_s=config.task_start,progress='PENDING matched UF reference',
                recovery='PASS' if complete and alpha_recovery is not None and alpha_recovery<=2.+1e-8 and force_recovery is not None else 'FAIL')


def run_case(case,variant,out,config=None,duration=18.):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    start_hash=source_hash();start_git=git_info()
    config=replace(config or GuardConfig(),rate_factor=case.get('rate_factor',1.))
    pulse=HumanPulse(**{k:case[k] for k in ('magnitude','start','ramp','hold','direction')})
    sim=make_force_sim(duration=duration)
    if variant in ('A','UF'):
        f=OracleFilter(sim,'active' if variant=='A' else 'off',pulse,case['body'],case['local'],(case['budget'],)*2,
                       config=QPConfig(task_weight=1.,rate=tuple(np.asarray(QPConfig().rate)*config.rate_factor)))
        sim.oracle=f
    else:f=EnergyGuard(sim,variant,pulse,case['body'],case['local'],(case['budget'],)*2,config=config)
    ticks=[];failed=None;start=time.perf_counter();pre=None
    try:
        while sim.data.time<duration-.000125:
            # Full pre-compute snapshot also covers the frozen A path, which
            # otherwise mutates nominal state before its endpoint filter runs.
            if variant in ('A','UF'):
                pre=HybridSnapshot.capture(sim,f.energy,f.previous,f.base_qfrc)
            ts=time.perf_counter();sim.advance(4);ticks.append(time.perf_counter()-ts)
    except (FilterFailure,ValueError) as exc:
        failed=dict(t=float(sim.data.time),reason=getattr(exc,'reason',str(exc)),diagnostics=getattr(exc,'diagnostics',{}))
        if variant in ('A','UF'):
            pre.save(out/'failure_pre.npz',force=f.force)
            HybridSnapshot.capture(sim,f.energy,f.previous,f.base_qfrc).save(out/'failure_complete.npz',force=f.force)
        else:f.save_failure(out,sim)
        save_json(out/'failure.json',failed)
    wall=time.perf_counter()-start
    log=sim.recorder.finish();c,p=f.arrays()
    log.save(out/'baseline_log.npz')
    np.savez_compressed(out/'guard.npz',**{'ctl__'+k:v for k,v in c.items()},**{'phys__'+k:v for k,v in p.items()},tick_seconds=np.asarray(ticks))
    with gzip.open(out/'events.json.gz','wt') as fp:json.dump(f.events,fp,default=_json_default,allow_nan=True)
    meta=dict(**start_git,source_sha256=start_hash,source_end_sha256=source_hash(),case=case,variant=variant,config=asdict(config),duration=duration,
              forecast='current measured world force vector held throughout each complete rollout',
              versions=dict(python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,scipy=scipy.__version__,osqp=osqp.__version__),
              platform=platform.platform(),asset_hashes=log.meta['model_file_sha256'])
    save_json(out/'metadata.json',meta)
    result=metrics(log,c,p,case,variant,wall,ticks,failed,config)
    save_json(out/'metrics.json',result)
    print(json.dumps({k:result[k] for k in ('case','variant','completion','energy','contact','completed_s','net_outgoing_J','force_max_error_N','failure')},default=_json_default),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['case','matrix'])
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--case',default='main')
    ap.add_argument('--variant',choices=['A','B','C','D','UF'],default='D')
    ap.add_argument('--variants',default='UF,A,B,C,D');ap.add_argument('--cases',default='all')
    ap.add_argument('--heldout',type=Path);ap.add_argument('--config',type=Path)
    args=ap.parse_args();cases=development_cases()
    if args.heldout:
        for c in json.loads(args.heldout.read_text())['cases']:cases[c['name']]=c
    cfg=GuardConfig(**json.loads(args.config.read_text())) if args.config else GuardConfig()
    if args.command=='case':
        r=run_case(cases[args.case],args.variant,args.out,cfg)
        return 0 if r['completion']=='PASS' else 2
    args.out.mkdir(parents=True,exist_ok=False);results={}
    selected=cases if args.cases=='all' else {k:cases[k] for k in args.cases.split(',')}
    for name,case in selected.items():
        results[name]={}
        for variant in args.variants.split(','):
            result=run_case(case,variant,args.out/name/variant,cfg)
            results[name][variant]=result
        uf=results[name].get('UF')
        if uf:
            for variant,r in results[name].items():
                ratio=r['measured_progress_m']/max(uf['measured_progress_m'],1e-15)
                r['progress_ratio_to_UF']=ratio
                r['reference_progress_ratio_to_UF']=r['reference_progress_m']/max(uf['reference_progress_m'],1e-15)
                r['progress']='PASS' if r['completion']=='PASS' and ratio>=.8 else 'FAIL'
                save_json(args.out/name/variant/'metrics.json',r)
        save_json(args.out/'summary.json',results)
    return 0 if all(r['completion']=='PASS' for c in results.values() for r in c.values()) else 2


if __name__=='__main__':raise SystemExit(main())
