"""General seven-variable, MULTI-ROW QP. Never calls Stage 1's one-row fallback.
Affine rows are proposals; fresh nonlinear per-physics checks decide acceptance.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import osqp
from scipy import sparse

from .joint_qp import QPConfig,FilterFailure


@dataclass
class Row:
    name: str
    value: float
    gradient: np.ndarray
    lower: float
    upper: float
    tolerance: float


def solve_multirow(nominal,center,H,rows,previous,limits,rho,rate,dt,torque_tolerance=1e-7):
    lo=np.maximum.reduce([-limits-nominal,previous-rate*dt-nominal,center-nominal-rho])
    hi=np.minimum.reduce([limits-nominal,previous+rate*dt-nominal,center-nominal+rho])
    if np.any(lo>hi):raise FilterFailure('empty_actuator_bounds')
    G=np.asarray([r.gradient for r in rows]);value=np.asarray([r.value for r in rows])
    lower=np.asarray([r.lower for r in rows])-value-G@(nominal-center)
    upper=np.asarray([r.upper for r in rows])-value-G@(nominal-center)
    tolerances=np.asarray([r.tolerance for r in rows])
    zero=np.max(np.abs(G),axis=1)<1e-14
    if np.any(zero & ((lower>tolerances)|(upper<-tolerances))):
        raise FilterFailure('constant_row_infeasible',dict(rows=[rows[i].name for i in np.flatnonzero(zero & ((lower>tolerances)|(upper<-tolerances)))]))
    B=np.linalg.solve(np.linalg.cholesky(H).T,np.eye(7))
    raw=np.vstack((G@B,B));scale=np.maximum(np.linalg.norm(raw,axis=1),1e-12)
    lower_all=np.r_[lower,lo];upper_all=np.r_[upper,hi]
    solver=osqp.OSQP()
    solver.setup(P=sparse.eye(7,format='csc'),q=np.zeros(7),A=sparse.csc_matrix(raw/scale[:,None]),
                 l=lower_all/scale,u=upper_all/scale,eps_abs=1e-9,eps_rel=1e-9,
                 max_iter=20000,verbose=False,polishing=True,warm_starting=False)
    result=solver.solve(raise_error=False)
    if result.info.status_val not in (1,2) or result.x is None or not np.all(np.isfinite(result.x)):
        reason='affine_constraints_infeasible' if result.info.status_val in (3,4) else 'solver_failure'
        raise FilterFailure(reason,dict(status=result.info.status,rows=[r.name for r in rows],gradients=G.tolist(),
                            center=center.tolist(),nominal=nominal.tolist(),H=H.tolist(),center_values=value.tolist(),
                            physical_lower=[r.lower for r in rows],physical_upper=[r.upper for r in rows],
                            box_lower=lo.tolist(),box_upper=hi.tolist()))
    delta=B@result.x;physical=value+G@(nominal+delta-center)
    violation=np.maximum(np.asarray([r.lower for r in rows])-physical,physical-np.asarray([r.upper for r in rows]))
    if np.any(violation>tolerances):
        raise FilterFailure('physical_row_residual',dict(rows=[rows[i].name for i in np.flatnonzero(violation>tolerances)],violation=violation.tolist()))
    if np.any(delta<lo-torque_tolerance) or np.any(delta>hi+torque_tolerance):raise FilterFailure('physical_box_residual')
    return nominal+delta,dict(status=result.info.status,names=[r.name for r in rows],values=physical.tolist(),
                              lower=[r.lower for r in rows],upper=[r.upper for r in rows],
                              gradients=G.tolist(),center_values=value.tolist(),center=center.tolist(),
                              torque=(nominal+delta).tolist(),violation=violation.tolist(),box_lower=lo.tolist(),box_upper=hi.tolist())


def nonlinear_ok(p,energy,minimum,predictor,protect):
    cfg=predictor.cfg
    if np.min(energy[:,None]+p.trace['prefix'][None,:]-minimum[:,None]) < -cfg.energy_tolerance:return False,'energy_prefix'
    if protect and not predictor.task_ok(p.trace):return False,'contact_task'
    if np.max(p.trace['unexpected'])>0:return False,'unexpected_contact'
    return True,None


def filtered_interval(predictor,snapshot,force,target=None,protect=False,minimum=None):
    cfg=predictor.cfg;qp=QPConfig(task_weight=cfg.task_weight)
    minimum=np.zeros(2) if minimum is None else np.asarray(minimum)
    first=predictor.interval(snapshot,force,target)
    nominal=first.nominal;center=nominal.copy();rho=np.asarray(qp.rho).copy()
    rate=np.asarray(qp.rate)*cfg.rate_factor
    ok,why=nonlinear_ok(first,snapshot.energy,minimum,predictor,protect)
    bounds=np.all(np.abs(nominal-snapshot.previous)<=rate*.001+cfg.torque_tolerance)
    if ok and bounds:return first,[],dict(gradient=0.,qp=0.,nonlinear=first.seconds)
    H=None;attempts=[];timing=dict(gradient=0.,qp=0.,nonlinear=first.seconds)
    for retry in range(cfg.retries+1):
        if np.any(np.abs(center)+cfg.eta>predictor.arm.torque_limit):raise FilterFailure('stencil_at_actuator_bound')
        start=perf_counter();base=predictor.interval(snapshot,force,target,center)
        fields=('prefix','normal_force','normal_velocity','cross_error','orientation')
        gradients={k:np.zeros(np.shape(base.trace[k])+(7,)) for k in fields}
        sy=np.empty((7,7))
        for j in range(7):
            step=np.eye(7)[j]*cfg.eta
            plus=predictor.interval(snapshot,force,target,center+step)
            minus=predictor.interval(snapshot,force,target,center-step)
            for k in fields:gradients[k][...,j]=(plus.trace[k]-minus.trace[k])/(2*cfg.eta)
            sy[:,j]=(plus.y-minus.y)/(2*cfg.eta)
        timing['gradient']+=perf_counter()-start
        if H is None:
            normalized=sy/np.asarray(qp.task_scales)[:,None]
            H=np.diag(1/np.asarray(qp.torque_scale)**2)+cfg.task_weight*normalized.T@normalized
        rows=[]
        for account in range(2):
            for j in range(4):
                # Four independently checked work prefixes. Deterministic
                # nonlinear replay removes affine approximation at acceptance.
                rows.append(Row(f'energy_{account}_prefix_{j}',base.trace['prefix'][j],gradients['prefix'][j],
                                minimum[account]-snapshot.energy[account]+(cfg.energy_margin if np.any(force) else 0.),np.inf,cfg.energy_tolerance))
        if protect:
            for j in range(4):
                if base.trace['t'][j]<cfg.task_start-1e-10:continue
                for key,lower,upper in [('normal_force',cfg.force_lower,cfg.force_upper),
                                         ('normal_velocity',-cfg.normal_speed,cfg.normal_speed),
                                         ('cross_error',-cfg.cross_error,cfg.cross_error)]:
                    rows.append(Row(f'{key}_{j}',base.trace[key][j],gradients[key][j],lower,upper,cfg.task_tolerance))
                for axis in range(3):
                    lim=cfg.orientation/np.sqrt(3)
                    rows.append(Row(f'orientation_{j}_{axis}',base.trace['orientation'][j,axis],gradients['orientation'][j,axis],-lim,lim,cfg.task_tolerance))
        start=perf_counter()
        torque,diagnostics=solve_multirow(nominal,center,H,rows,snapshot.previous,predictor.arm.torque_limit,
                                         rho,rate,.001,cfg.torque_tolerance)
        timing['qp']+=perf_counter()-start
        checked=predictor.interval(snapshot,force,target,torque)
        timing['nonlinear']+=checked.seconds
        ok,reason=nonlinear_ok(checked,snapshot.energy,minimum,predictor,protect)
        diagnostics.update(retry=retry,accepted=ok,rejection_reason=reason)
        attempts.append(diagnostics)
        if ok:return checked,attempts,timing
        center=torque.copy();rho*=.5
    raise FilterFailure('nonlinear_retries_exhausted',dict(attempts=attempts))
