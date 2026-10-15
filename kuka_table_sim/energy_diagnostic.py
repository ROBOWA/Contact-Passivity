"""Hindsight-only timing experiment. NOT an online trigger.

Each 0/20/50/100 ms case forks a complete pre-onset plant/controller/account
snapshot. Only slowdown timing varies; policy, task weight, force and bounds do not.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from .energy_state import GovernedController,HybridSnapshot
from .joint_qp import QPConfig,FilterFailure
from .oracle import OracleFilter
from .oracle_experiments import fixture,save_json,summarize,source_hash
from .recorder import Recorder,RunLog
from .scenarios import make_force_sim


def progress(log):
    c=log.ctl
    n=c['strokes'].astype(int)
    ref=n*.1+np.where(n%2==0,c['s_d'],.1-c['s_d'])
    measured=(c['x'][:,0]-c['x'][0,0])
    count=0;frontier=0.;series=[]
    for x in measured:
        advance=x if count%2==0 else .1-x
        frontier=max(frontier,float(np.clip(advance,0,.1)))
        if advance>=.0995:
            count+=1;frontier=0.
        series.append(count*.1+frontier)
    return ref,np.asarray(series),count


def joined(a,b):
    return RunLog({k:np.concatenate((a.phys[k],b.phys[k])) for k in a.phys},
                  {k:np.concatenate((a.ctl[k],b.ctl[k])) for k in a.ctl},b.meta)


def run(out,transition=.1):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    result={}
    for case,event in [('resistance',10.285),('held-out',10.753)]:
        pulse,body,local=fixture(case)
        common=make_force_sim(duration=18.)
        common.controller=GovernedController.from_controller(common.controller,transition)
        common.oracle=OracleFilter(common,'active',pulse,body,local,(.12,.12),config=QPConfig(task_weight=1.))
        pre=event-.105
        common.advance(common.steps_for(pre))
        prefix=common.recorder.finish()
        snap=HybridSnapshot.capture(common,common.oracle.energy,common.oracle.previous,common.oracle.base_qfrc)
        (out/case).mkdir()
        snap.save(out/case/'pre_onset_snapshot.npz')
        for early in (0,.02,.05,.1):
            folder=out/case/f'early_{round(early*1000):03d}ms';folder.mkdir(exist_ok=False)
            sim=snap.fork(common)
            sim.recorder=Recorder(sim,18.-sim.data.time+.01)
            f=OracleFilter(sim,'active',pulse,body,local,(.12,.12),config=QPConfig(task_weight=1.))
            f.energy=snap.energy.copy();f.previous=snap.previous.copy();f.base_qfrc=snap.base_qfrc.copy()
            f.rows=list(common.oracle.rows);f.phys=list(common.oracle.phys)
            sim.oracle=f
            onset=event-early;govern=[];failure=None;step_times=[]
            onset_energy=None;onset_work=None;min_decel=None;final_pre=None
            start=time.perf_counter()
            try:
                while sim.data.time<18.-sim.timing.timestep/2:
                    clock=sim.controller.stroke.clock
                    clock.advance(sim.data.time)
                    force=pulse.force(sim.data.time)
                    if sim.data.time>=onset-1e-9:
                        if onset_energy is None:
                            onset_energy=f.energy.copy();onset_work=float(np.sum([r['actual'][0] for r in f.rows]))
                        clock.set_target(0. if np.linalg.norm(force)>1e-12 else 1.)
                    final_pre=HybridSnapshot.capture(sim,f.energy,f.previous,f.base_qfrc)
                    tick=time.perf_counter();sim.advance(sim.timing.control_interval);step_times.append(time.perf_counter()-tick)
                    govern.append([f.rows[-1]['t'],clock.alpha,clock.alpha_dot,clock.sigma,clock.target])
                    if onset_energy is not None and sim.data.time<=onset+transition+.1:
                        e=float(np.min(f.energy));min_decel=e if min_decel is None else min(min_decel,e)
            except (FilterFailure,ValueError) as exc:
                failure=dict(reason=getattr(exc,'reason',str(exc)),t=float(sim.data.time),diagnostics=getattr(exc,'diagnostics',{}))
                final_pre.save(folder/'failure_complete_pre.npz')
                HybridSnapshot.capture(sim,f.energy,f.previous,f.base_qfrc).save(folder/'failure_complete_post.npz')
                save_json(folder/'failure.json',failure)
            wall=time.perf_counter()-start
            log=joined(prefix,sim.recorder.finish())
            c,p=f.arrays()
            log.save(folder/'baseline_log.npz')
            np.savez_compressed(folder/'oracle.npz',**{'ctl__'+k:v for k,v in c.items()},**{'phys__'+k:v for k,v in p.items()},governor=np.asarray(govern))
            metrics=summarize(log,c,p,f,wall,step_times,failure)
            ref,meas,strokes=progress(log)
            metrics.update(offline_hindsight=True,onset=onset,early_seconds=early,transition=transition,
                           energy_at_onset_J=None if onset_energy is None else onset_energy.tolist(),
                           max_net_deceleration_cost_J=None if min_decel is None else float(np.min(onset_energy)-min_decel),
                           reference_progress_m=float(ref[-1]),measured_progress_m=float(meas[-1]),completed_strokes=strokes,
                           source_sha256=source_hash())
            save_json(folder/'metrics.json',metrics)
            result[f'{case}_{round(early*1000)}ms']=metrics
            print(case,early,metrics['status'],metrics.get('net_outgoing_J'),metrics['contact']['geometric_absence']['total_s'],metrics['force']['original_10N_max_abs'],flush=True)
    save_json(out/'diagnosis.json',result)
    plot(out,result)
    return result


def plot(out,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    for col,case in enumerate(['resistance','held-out']):
        for early in (0,20,50,100):
            z=np.load(out/case/f'early_{early:03d}ms'/'oracle.npz')
            axs[0,col].plot(z['ctl__t'],z['ctl__energy_after'][:,0],label=f'{early} ms early')
            axs[1,col].plot(z['phys__t'],z['phys__normal_force'])
            g=z['governor'];axs[2,col].plot(g[:,0],g[:,1])
        axs[0,col].legend();axs[0,col].set_title(case)
        for row,y in enumerate(['Energy J','Normal force N','alpha']):
            axs[row,col].set(xlim=(9.5,13.),xlabel='physical time s',ylabel=y)
        axs[1,col].axhline(10,color='k',ls='--')
    fig.savefig(out/'early_intervention.png',dpi=140);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.out)

if __name__=='__main__':main()
