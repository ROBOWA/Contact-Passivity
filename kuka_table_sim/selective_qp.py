"""Shared filter: the governor is NOT a constraint or reserve-policy change.

K1 protects whole monitored work; K2 protects residual work; K3/K4 protect
H/R plus actual human peak power. K4 differs ONLY in reference-clock target.
All rows are checked by a fresh nonlinear rollout before an action executes.
"""
from time import perf_counter
import numpy as np
from .joint_qp import QPConfig,FilterFailure
from .energy_qp import Row,solve_multirow
from .selective_ports import ACCOUNT_NAMES

PROTECTED={'K0':(), 'K1':(2,), 'K2':(1,), 'K3':(0,1), 'K4':(0,1)}


def constraints(p,snapshot,force,mode,cfg):
    names=[];values=[];tols=[]
    for account in PROTECTED[mode]:
        for j in range(4):
            names.append(f'{ACCOUNT_NAMES[account]}_energy_{j}')
            values.append(p.trace['energy'][j,account]-cfg.floors[account]-cfg.energy_guard_J);tols.append(cfg.energy_tolerance_J)
            names.append(f'{ACCOUNT_NAMES[account]}_barrier_{j}')
            values.append(p.trace['power'][j,account]+cfg.barrier_rate*(p.trace['energy_before'][j,account]-cfg.floors[account]-cfg.safe_energy_J)-cfg.power_guard_W);tols.append(cfg.power_tolerance_W)
    if mode in ('K3','K4'):
        for j in range(4):
            names.append(f'H_peak_{j}');values.append(cfg.peak_human_W+p.trace['power'][j,0]-cfg.power_guard_W);tols.append(cfg.power_tolerance_W)
        lower=float(force@p.terminal_human_velocity)-cfg.force_rate_N_s*.001*np.linalg.norm(p.terminal_human_velocity)
        names.extend(('H_terminal_peak_force_rate','H_terminal_barrier_force_rate'))
        values.extend((cfg.peak_human_W+lower-cfg.power_guard_W,
                       lower+cfg.barrier_rate*(p.next.energy[0]-cfg.floors[0]-cfg.safe_energy_J)-cfg.power_guard_W))
        tols.extend((cfg.power_tolerance_W,cfg.power_tolerance_W))
    return names,np.asarray(values),np.asarray(tols)


def filtered(predictor,snapshot,force,mode,target):
    cfg=predictor.cfg;qp=QPConfig(task_weight=cfg.task_weight)
    first=predictor.interval(snapshot,force,target);nominal=first.nominal
    if np.any(first.trace['unexpected']):raise FilterFailure('unexpected_contact')
    names,values,tols=constraints(first,snapshot,force,mode,cfg)
    bounds=np.all(np.abs(nominal-snapshot.previous)<=np.asarray(qp.rate)*.001+cfg.torque_tolerance_Nm)
    human_trigger=any(n.startswith('H_') and v < -tol for n,v,tol in zip(names,values,tols))
    if mode=='K0' or (np.all(values>=-tols) and bounds):return first,[],dict(rollout=first.seconds,gradient=0.,qp=0.,human_trigger=False)
    center=nominal.copy();rho=np.asarray(qp.rho).copy();attempts=[];H=None
    timing=dict(rollout=first.seconds,gradient=0.,qp=0.,human_trigger=human_trigger)
    for retry in range(cfg.retries+1):
        if np.any(np.abs(center)+cfg.eta_Nm>predictor.arm.torque_limit):raise FilterFailure('stencil_at_bound',dict(attempts=attempts))
        tick=perf_counter();base=predictor.interval(snapshot,force,target,center)
        names,v,tols=constraints(base,snapshot,force,mode,cfg)
        G=np.empty((len(v),7));sy=np.empty((7,7))
        for j in range(7):
            step=np.eye(7)[j]*cfg.eta_Nm
            plus=predictor.interval(snapshot,force,target,center+step)
            minus=predictor.interval(snapshot,force,target,center-step)
            G[:,j]=(constraints(plus,snapshot,force,mode,cfg)[1]-constraints(minus,snapshot,force,mode,cfg)[1])/(2*cfg.eta_Nm)
            sy[:,j]=(plus.y-minus.y)/(2*cfg.eta_Nm)
        timing['gradient']+=perf_counter()-tick
        if H is None:
            normalized=sy/np.asarray(qp.task_scales)[:,None]
            H=np.diag(1/np.asarray(qp.torque_scale)**2)+cfg.task_weight*normalized.T@normalized
        rows=[Row(n,value,g,0.,np.inf,tol) for n,value,g,tol in zip(names,v,G,tols)]
        tick=perf_counter()
        try:
            torque,diagnostics=solve_multirow(nominal,center,H,rows,snapshot.previous,predictor.arm.torque_limit,rho,np.asarray(qp.rate),.001,cfg.torque_tolerance_Nm)
        except FilterFailure as exc:
            timing['qp']+=perf_counter()-tick
            attempts.append(dict(retry=retry,accepted=False,reason=exc.reason,diagnostics=exc.diagnostics,names=names))
            raise FilterFailure(exc.reason,dict(attempts=attempts,timing=timing)) from exc
        timing['qp']+=perf_counter()-tick
        check=predictor.interval(snapshot,force,target,torque);timing['rollout']+=check.seconds
        _,actual,tol=constraints(check,snapshot,force,mode,cfg)
        ok=bool(np.all(actual>=-tol) and not np.any(check.trace['unexpected']));diagnostics.update(retry=retry,accepted=ok,nonlinear_values=actual.tolist())
        attempts.append(diagnostics)
        if ok:return check,attempts,timing
        center=torque.copy();rho*=.5
    raise FilterFailure('nonlinear_retries_exhausted',dict(attempts=attempts,timing=timing))
