"""Lossless, named executed-signal adapter and results-only metrics.

No simulation/controller modules are imported. Raw observations retain every
physics and control sample; missing capabilities are absent, never zero filled.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


UNITS = {
    't_phys':'s; start of physics interval', 'h':'s', 't_control':'s; command boundary',
    'power_H':'W; into robot', 'work_H':'J; signed left rectangle',
    'energy_H':'J; after executed physics increment', 'energy_R':'J; after executed physics increment',
    'force_H':'N; world; actual application point', 'velocity_H':'m/s; world; actual application point',
    'point_H':'m; world', 'normal_force':'N; world +z force ON tool', 'present':'bool',
    'task_power':'W; actual tool/table material-point wrench power', 'task_work':'J; signed left rectangle',
    'tcp_position':'m; world; pre-integration', 'tcp_velocity':'m/s; world; pre-integration',
    'x':'m; world; control-boundary TCP', 'v':'m/s; world; control-boundary TCP',
    'x_des':'m; online governed reference', 'v_des':'m/s; online governed reference',
    'force_des':'N; internal online target; original task target remains 10 N',
    'phase':'controller code', 'strokes':'completed reference strokes', 's_des':'m',
    'alpha':'dimensionless', 'alpha_target':'dimensionless', 'sigma':'s; virtual time',
    'reserve':'J; unexecuted current-held-force backup prediction',
    'reserve_margin':'J', 'tau_exec':'N m; complete held joint torque',
    'tau_nom':'N m; governed nominal joint torque', 'tau_previous':'N m',
    'tau_ungoverned':'N m; nominal counterfactual on actual state; diagnostic',
    'x_ungoverned':'m; counterfactual reference on actual state; diagnostic',
    'tick_seconds':'wall s; whole successful control tick',
    'orientation':'rad', 'cross_error':'m',
}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*1024*1024): h.update(b)
    return h.hexdigest()


def write_json(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def archive_case(folder, dest, kind='archived executed simulation'):
    """Preserve sufficient COMPLETE raw samples for every plotted curve/check."""
    folder, dest = Path(folder),Path(dest)
    dest.mkdir(parents=True,exist_ok=False)
    meta=json.loads((folder/'metadata.json').read_text())
    old=json.loads((folder/'metrics.json').read_text())
    with np.load(folder/'guard.npz',allow_pickle=False) as z:
        c={k[5:]:z[k] for k in z.files if k.startswith('ctl__')}
        p={k[6:]:z[k] for k in z.files if k.startswith('phys__')}
        ticks=z['tick_seconds'] if 'tick_seconds' in z.files else None
    with np.load(folder/'baseline_log.npz',allow_pickle=False) as z:
        bc={k[5:]:z[k] for k in z.files if k.startswith('ctl__')}
        bp={k[6:]:z[k] for k in z.files if k.startswith('phys__')}
        bm=json.loads(str(z['meta_json']))
    h=float(bm['timestep'])
    E=p.get('energy',p.get('prefix_energy'))
    data=dict(t_phys=p['t'], h=np.array(h), t_control=c['t'],power_H=p['power'],
              work_H=p.get('work',p['power']*h),energy_H=E[:,0],energy_R=E[:,1],
              force_H=p['force'],velocity_H=p['velocity'],point_H=p['point'],
              normal_force=p['normal_force'],present=bp['present'],
              task_power=bp['P_task'],task_work=bp['dW'],tcp_position=bp['tcp'],tcp_velocity=bp['tcp_v'],
              x=bc['x'],v=bc['v'],x_des=bc['x_d'],v_des=bc['v_d'],force_des=bc['f_d'],
              phase=bc['phase'],strokes=bc['strokes'],s_des=bc['s_d'],
              tau_exec=c['tau_exec'],tau_nom=c['tau_nom'],tau_previous=c['tau_previous'])
    for name in ('alpha','alpha_target','sigma','reserve','reserve_margin','tau_ungoverned','x_ungoverned'):
        if name in c:data[name]=c[name]
    for name in ('orientation','cross_error'):
        if name in p:data[name]=p[name]
    if ticks is not None:data['tick_seconds']=ticks
    np.savez_compressed(dest/'executed.npz',**data)
    with gzip.open(folder/'events.json.gz','rt') as f:events=json.load(f)
    # Preserve events losslessly (including rejected predictions); these never
    # enter executed arrays or plotted trajectories.
    (dest/'events.json.gz').write_bytes((folder/'events.json.gz').read_bytes())
    for name in ('initial_state.npz','final_state.npz','failure_pre.npz','failure_complete.npz','failure.json','process.json'):
        if (folder/name).is_file():(dest/name).write_bytes((folder/name).read_bytes())
    source_files={f.name:dict(path=str(f.resolve()),sha256=sha(f),bytes=f.stat().st_size)
                  for f in sorted(folder.iterdir()) if f.is_file()}
    for suffix,tag in (('.process.json','process.json'),('.stdout.txt','process.stdout.txt')):
        f=folder.parent/(folder.name+suffix)
        if f.is_file():
            (dest/tag).write_bytes(f.read_bytes())
            source_files[tag]=dict(path=str(f.resolve()),sha256=sha(f),bytes=f.stat().st_size)
    provenance=dict(source_kind=kind,original_directory=str(folder.resolve()),source_files=source_files,
                    metadata=meta,baseline_metadata=bm,archived_metrics=old,
                    sample_policy='all executed physics and control samples; no decimation or smoothing',
                    units={k:UNITS[k] for k in data},
                    stage='t_phys is x_k; powers evaluated coherently at x_k; energy is AFTER P_k*h; '
                          'energy curves plotted at t_phys+h; references and command at t_control',
                    raw_signal_map=dict(power_H='guard.phys__power',work_H='guard.phys__work or power*h',
                        energy_H='guard.phys__energy[:,0] or prefix_energy[:,0]',
                        energy_R='guard.phys__energy[:,1] or prefix_energy[:,1]',
                        force_H='guard.phys__force',velocity_H='guard.phys__velocity',
                        task_power='baseline.phys__P_task',task_work='baseline.phys__dW',
                        normal_force='guard.phys__normal_force',x='baseline.ctl__x',v='baseline.ctl__v',
                        x_des='baseline.ctl__x_d',v_des='baseline.ctl__v_d'),
                    output_sha256=sha(dest/'executed.npz'))
    write_json(dest/'provenance.json',provenance)
    return provenance


def load_case(folder):
    folder=Path(folder)
    prov=json.loads((folder/'provenance.json').read_text())
    if sha(folder/'executed.npz')!=prov['output_sha256']:raise ValueError('preserved signal hash mismatch')
    with np.load(folder/'executed.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with gzip.open(folder/'events.json.gz','rt') as f:events=json.load(f)
    return data,prov,events


def ordered_progress(x,strokes,s):
    reference=strokes*.1+np.where(strokes.astype(int)%2==0,s,.1-s)
    measured=[];count=0;frontier=0.
    for val in x-x[0]:
        advance=val if count%2==0 else .1-val
        frontier=max(frontier,float(np.clip(advance,0,.1)))
        if advance>=.0995:count+=1;frontier=0.
        measured.append(count*.1+frontier)
    return reference,np.asarray(measured),count


def sustained(t,good,start,duration,h):
    """First actual sample starting a complete all-good window, never pad a stop."""
    idx=np.flatnonzero(t>=start-1e-9);n=round(duration/h)
    if len(idx)<n:return None
    bad=np.r_[0,np.cumsum(~np.asarray(good[idx],bool))]
    at=np.flatnonzero(bad[n:]-bad[:-n]==0)
    # Binary floating timestamps can land a few picoseconds before a boundary.
    # Clamp elapsed duration to zero, never the raw ledger or signed work.
    return max(0.,float(t[idx[at[0]]]-start)) if len(at) else None


def compute(data,provenance,events,window_start=8.71,window_end=None):
    d=data;m=provenance['metadata'];case=m['case'];h=float(d['h']);t=d['t_phys'];tc=d['t_control']
    if window_end is None:window_end=float(t[-1]+h)
    mask=(t>=window_start-1e-9)&(t<window_end-1e-9)
    cm=(tc>=window_start-1e-9)&(tc<window_end-1e-9)
    if not np.any(mask):raise ValueError('no executed samples in metric window')
    # Integration always starts with the full physics trace, never the plot view.
    work=np.cumsum(d['work_H']);E=case['budget']+work
    err=d['normal_force'][mask]-10.
    ref,progress,count=ordered_progress(d['x'][:,0],d['strokes'],d['s_des'])
    alpha=d.get('alpha')
    intervention=np.max(np.abs(d['tau_exec']-d['tau_nom']),axis=1)>1e-7
    if 'alpha_target' in d:intervention|=d['alpha_target']<1.-1e-10
    first=np.flatnonzero(intervention)
    release=case['start']+2*case['ramp']+case['hold']
    rampout=case['start']+case['ramp']+case['hold']
    if case.get('profile'):
        release=max(p['start']+2*p['ramp']+p['hold'] for p in case['profile'])
        last=max(case['profile'],key=lambda p:p['start']);rampout=last['start']+last['ramp']+last['hold']
    # Explicitly sustained recovery; alpha eligibility uses actual command rate.
    force_rec=sustained(t,np.abs(d['normal_force']-10.)<=.5,release,.5,h)
    alpha_rec=sustained(tc,alpha>=.95,release,.5,.001) if alpha is not None else None
    qp_records=[]
    for e in events:
        qp_records.extend(a for a in e.get('attempts',[]) if 'names' in a)
    activity={}
    for a in qp_records:
        for i,name in enumerate(a['names']):
            value=float(a['values'][i]);lo=float(a['lower'][i]);hi=float(a['upper'][i])
            tol=1e-10 if name.startswith('energy') else 1e-7
            if abs(value-lo)<=tol or abs(value-hi)<=tol:activity[name]=activity.get(name,0)+1
    failure=provenance['archived_metrics'].get('failure')
    result=dict(case=case['name'],variant=m['variant'],window_start_s=window_start,window_end_s=window_end,
        completed_s=float(t[-1]+h),completed=bool(failure is None and t[-1]+h>=m.get('duration',18)-1e-8),
        stop_reason=None if failure is None else failure['reason'],
        human_work_in_J=float(work[-1]),net_outgoing_J=-float(work[-1]),
        initial_energy_J=float(case['budget']),max_raw_energy_J=float(max(case['budget'],E.max())),
        gross_outgoing_J=float(np.maximum(-d['work_H'],0).sum()),
        min_substep_energy_J=float(min(case['budget'],E.min())),final_energy_J=float(E[-1]),
        peak_outgoing_power_W=float(np.max(-d['power_H'])),
        force_rmse_N=float(np.sqrt(np.mean(err**2))),force_mae_N=float(np.mean(np.abs(err))),
        force_max_error_N=float(np.max(np.abs(err))),force_peak_N=float(np.max(d['normal_force'][mask])),
        contact_absence_s=float(np.count_nonzero(~d['present'][mask])*h),
        tangent_rmse_m_s=float(np.sqrt(np.mean((d['v'][cm,0]-d['v_des'][cm,0])**2))),
        tangent_peak_speed_m_s=float(np.max(np.abs(d['tcp_velocity'][mask,0]))),
        measured_progress_m=float(progress[-1]),reference_progress_m=float(ref[-1]),completed_strokes=count,
        peak_tracking_debt_m=float(np.max(np.abs(d['x_des'][cm,0]-d['x'][cm,0]))),
        first_intervention_s=float(tc[first[0]]) if len(first) else None,
        intervention_ticks=int(np.count_nonzero(intervention)),
        torque_correction_ticks=int(np.count_nonzero(np.max(np.abs(d['tau_exec']-d['tau_nom']),axis=1)>1e-7)),
        max_held_command_change_Nm=float(np.max(np.abs(d['tau_exec']-d['tau_previous']))),
        max_governed_nominal_correction_Nm=float(np.max(np.abs(d['tau_exec']-d['tau_nom']))),
        alpha_recovery_from_pulse_end_s=alpha_rec,force_recovery_from_pulse_end_s=force_rec,
        force_recovery_from_rampout_s=sustained(t,np.abs(d['normal_force']-10.)<=.5,rampout,.5,h),
        rampout_s=rampout,pulse_end_s=release,qp_attempt_records=len(qp_records),qp_active_rows=activity,
        # seconds_qp can undercount relinearizations; use the independent audit
        # for exact solved/rejected/frozen-A solver activity, never infer from delta.
        task_work_in_J=float(np.sum(d['task_work'])),
        work_energy_identity_max_J=float(np.max(np.abs(d['energy_H']-E))),
        ledger_equality_max_J=float(np.max(np.abs(d['energy_H']-d['energy_R']))),
        force_velocity_power_identity_max_W=float(np.max(np.abs(np.einsum('ij,ij->i',d['force_H'],d['velocity_H'])-d['power_H']))),
        wall_seconds=provenance['archived_metrics'].get('wall_seconds'))
    if case.get('profile'):
        result['phase_work']={}
        for i,pulse in enumerate(case['profile']):
            end=pulse['start']+2*pulse['ramp']+pulse['hold']
            ix=(t>=pulse['start']-1e-10)&(t<end-1e-10);w=d['work_H'][ix]
            result['phase_work'][pulse.get('name',str(i))]=dict(
                signed_incoming_J=float(w.sum()),net_outgoing_J=-float(w.sum()),
                gross_outgoing_J=float(np.maximum(-w,0).sum()),
                positive_power_samples=int(np.count_nonzero(d['power_H'][ix]>0)),
                negative_power_samples=int(np.count_nonzero(d['power_H'][ix]<0)))
    if alpha is not None:
        result.update(pause_s=float(np.count_nonzero((alpha<=.01)&cm)*.001),
                      slowed_s=float(np.count_nonzero((alpha<.95)&cm)*.001),
                      virtual_time_delay_s=float(np.max(tc-d['sigma'])))
    if 'tau_ungoverned' in d:
        result['max_total_change_from_ungoverned_Nm']=float(np.max(np.abs(d['tau_exec']-d['tau_ungoverned'])))
    elif m['variant'] in ('A','UF'):
        result['max_total_change_from_ungoverned_Nm']=result['max_governed_nominal_correction_Nm']
    if 'tick_seconds' in d:
        a=d['tick_seconds'];result['tick_seconds']={k:float(np.quantile(a,q)) for k,q in [('median',.5),('p95',.95),('p99',.99),('max',1)]}
        result['deadline_misses']=int(np.count_nonzero(a>.001))
        result['timed_ticks']=len(a)
    return result


def validate(d,p):
    t=d['t_phys'];tc=d['t_control'];h=float(d['h'])
    errors=[]
    undefined={}
    for name,a in d.items():
        if name in ('x_des','v_des'):
            # Normal-axis references are undefined (NaN) under force control.
            # Only the finite along-path x component is used in this report.
            undefined[name+'_normal_axis']=int(np.count_nonzero(~np.isfinite(a[:,2])))
            a=a[:,:2]
        if np.issubdtype(a.dtype,np.number) and not np.all(np.isfinite(a)):errors.append(name+': nonfinite')
    if not np.allclose(np.diff(t),h,rtol=0,atol=2e-11):errors.append('physics sampling/dropped samples')
    if not np.allclose(np.diff(tc),.001,rtol=0,atol=2e-11):errors.append('control sampling/dropped samples')
    if len(t)!=4*len(tc):errors.append('partial control interval')
    if not np.allclose(t[::4],tc,rtol=0,atol=2e-11):errors.append('clock alignment')
    if len(d['energy_H'])!=len(t) or len(d['x_des'])!=len(tc):errors.append('array length')
    r=compute(d,p,[])
    if r['work_energy_identity_max_J']>1e-11:errors.append('account identity')
    if r['force_velocity_power_identity_max_W']>1e-10:errors.append('human point power identity')
    return dict(status='PASS' if not errors else 'FAIL',errors=errors,
                physics_samples=len(t),control_samples=len(tc),physics_dt_s=h,control_dt_s=.001,
                retained_all_samples=True,undefined_normal_reference_samples=undefined,
                minimum_raw_energy_J=r['min_substep_energy_J'])
