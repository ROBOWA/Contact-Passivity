"""Independent executed-data audit for the KUKA PoC report.

This file deliberately imports neither plotting code nor controller/metric helpers.
Full guard/baseline logs are preferred. Compact archives retain every physics
sample from the declared task window onward, plus every nonzero human power;
their omitted prefix is identified explicitly rather than integrated at a new dt.

Example: python scripts/poc_figures_audit.py --root PATH/final_matrix_01 --out NEW.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


class IntegrityError(ValueError):
    """The supplied data cannot support the requested numerical calculation."""


@dataclass
class RawCase:
    folder: Path
    metadata: dict
    physics: dict
    control: dict
    reference: dict
    baseline: dict
    dt: float
    indices: np.ndarray
    total_physics_count: int
    source_kind: str
    tick_seconds: np.ndarray | None = None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _arrays(path, prefix):
    with np.load(path, allow_pickle=False) as z:
        return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}


def load_case(folder):
    """Read raw arrays without importing any implementation or plotting helper."""
    folder = Path(folder)
    m = json.loads((folder / 'metadata.json').read_text())
    if (folder / 'guard.npz').is_file():
        p = _arrays(folder / 'guard.npz', 'phys__')
        c = _arrays(folder / 'guard.npz', 'ctl__')
        b = _arrays(folder / 'baseline_log.npz', 'phys__')
        r = _arrays(folder / 'baseline_log.npz', 'ctl__')
        with np.load(folder / 'baseline_log.npz', allow_pickle=False) as z:
            logmeta = json.loads(str(z['meta_json']))
        if logmeta.get('saved_stride', 1) != 1 or logmeta.get('saved_float32', False):
            raise IntegrityError('full raw baseline was decimated or cast to float32')
        with np.load(folder / 'guard.npz', allow_pickle=False) as z:
            ticks = z['tick_seconds'] if 'tick_seconds' in z.files else None
        return RawCase(folder, m, p, c, r, b, float(logmeta['timestep']),
                       np.arange(len(p['t'])), len(p['t']), 'full_executed_raw', ticks)
    with np.load(folder / 'trace.npz', allow_pickle=False) as z:
        p = {k: z[k] for k in z.files if not k.startswith(('ctl__', 'reference__'))
             and k not in ('physics_indices', 'total_physics_count')}
        c = {k[5:]: z[k] for k in z.files if k.startswith('ctl__')}
        r = {k[11:]: z[k] for k in z.files if k.startswith('reference__')}
        indices, total = z['physics_indices'], int(z['total_physics_count'])
    if len(indices) < 2:
        raise IntegrityError('cannot infer physics dt from fewer than two compact samples')
    dindex, dtime = np.diff(indices), np.diff(p['t'])
    if np.any(dindex <= 0):
        raise IntegrityError('compact physics indices are not strictly increasing')
    dt = float(np.median(dtime / dindex))
    return RawCase(folder, m, p, c, r, {}, dt, indices, total,
                   'compact_executed_raw_declared_zero_human_prefix')


def validate_time(t, dt, indices=None, name='physics'):
    t = np.asarray(t, dtype=float)
    if len(t) == 0 or not np.all(np.isfinite(t)) or not np.isfinite(dt) or dt <= 0:
        raise IntegrityError(f'{name}: empty/nonfinite time or invalid dt')
    ix = np.arange(len(t)) if indices is None else np.asarray(indices)
    if len(ix) != len(t) or np.any(np.diff(ix) <= 0) or np.any(np.diff(t) <= 0):
        raise IntegrityError(f'{name}: length mismatch or nonmonotone timestamps/indices')
    defect = float(np.max(np.abs(np.diff(t) - np.diff(ix) * dt))) if len(t) > 1 else 0.
    if defect > max(1e-10, dt * 1e-6):
        raise IntegrityError(f'{name}: dropped or misaligned samples ({defect:g} s)')
    return defect


def signed_work(power, dt, initial):
    """Left rectangles, signed on-robot work and uncapped raw prefix energy."""
    power = np.asarray(power, dtype=float)
    if not len(power) or not np.all(np.isfinite(power)):
        raise IntegrityError('empty or nonfinite executed power')
    work = power * dt
    energy = initial + np.cumsum(work, dtype=np.float64)
    return dict(human_work_in_J=float(np.sum(work, dtype=np.float64)),
                net_outgoing_J=float(-np.sum(work, dtype=np.float64)),
                gross_outgoing_J=float(np.sum(np.maximum(-work, 0), dtype=np.float64)),
                gross_incoming_J=float(np.sum(np.maximum(work, 0), dtype=np.float64)),
                min_substep_energy_J=float(np.min(energy)),
                final_energy_J=float(energy[-1]),
                peak_outgoing_power_W=float(np.max(np.maximum(-power, 0)))), energy


def ordered_progress(reference):
    """Independent transcription of the declared 0.1 m ordered-stroke metric.

    Reference progress uses the controller's completed strokes and local target.
    Measured progress advances a 0.1 m frontier, changing direction only within
    0.5 mm of an endpoint. Neither path length nor absolute displacement is used.
    """
    needed = ('strokes', 's_d', 'x')
    if any(k not in reference for k in needed):
        return None, None, None
    count = reference['strokes'].astype(int)
    desired = .1 * count + np.where(count % 2 == 0, reference['s_d'], .1-reference['s_d'])
    displacement = reference['x'][:, 0] - reference['x'][0, 0]
    current, frontier, out = 0, 0., []
    for x in displacement:
        distance = x if current % 2 == 0 else .1 - x
        frontier = max(frontier, min(.1, max(0., float(distance))))
        if distance >= .0995:
            current += 1
            frontier = 0.
        out.append(.1 * current + frontier)
    return desired, np.asarray(out), current


def sustained_recovery(t, good, start, dt, hold=.5):
    """First fully observed good interval after start; never extend a stopped run."""
    t, good = np.asarray(t), np.asarray(good, bool)
    if start is None:
        return None
    keep = np.flatnonzero(t >= start - 1e-10)
    required = int(round(hold / dt))
    if not len(keep) or len(keep) < required:
        return None
    flags = good[keep].astype(np.int64)
    prefix = np.r_[0, np.cumsum(flags)]
    matches = np.flatnonzero(prefix[required:] - prefix[:-required] == required)
    return max(0., float(t[keep[matches[0]]] - start)) if len(matches) else None


def _endpoints(metadata):
    case = metadata['case']
    profile=metadata.get('profile', {})
    pulses=case.get('pulses',case.get('profile'))
    if pulses is None:
        pulses=profile if isinstance(profile,list) else profile.get('pulses')
    if pulses:
        return max(float(p['start']+p['ramp']+p['hold']) for p in pulses), max(
            float(p['start']+2*p['ramp']+p['hold']) for p in pulses)
    if all(k in case for k in ('start', 'ramp', 'hold')):
        return float(case['start']+case['ramp']+case['hold']), float(case['start']+2*case['ramp']+case['hold'])
    return None, None


def audit_events(folder, failure_override=None):
    """Count actual solver records separately from attempts and reserve proposals.

    Stage 1 pre-solver feasibility failures are not OSQP solves. Stage 2 failed
    solver diagnostics can be saved in rejections/failure.json instead of the
    main attempts array. Deduplicate repeated copies of the same diagnostic.
    Returned counts describe recorded calls; no zero QP activity is inferred
    merely from zero intervention or missing timing arrays.
    """
    path = Path(folder) / 'events.json.gz'
    events = json.load(gzip.open(path, 'rt')) if path.is_file() else []
    failure_path = Path(folder) / 'failure.json'
    failure = failure_override if failure_override is not None else (
        json.loads(failure_path.read_text()) if failure_path.is_file() else None)
    seen, attempts, statuses = set(), [], Counter()
    rejections, fallback, active = Counter(), 0, Counter()
    qp_success, qp_failed, pre_solver = 0, 0, 0
    max_normalized_violation, max_logged_defect = 0., 0.

    def visit(node, tick=None):
        nonlocal fallback, qp_success, qp_failed, pre_solver
        nonlocal max_normalized_violation, max_logged_defect
        if isinstance(node, list):
            for value in node:
                visit(value, tick)
            return
        if not isinstance(node, dict):
            return
        tick = node.get('t', tick)
        if 'target' in node and 'reason' in node:
            rejections[str(node['reason'])] += 1
        diagnostic = ('status' in node and any(k in node for k in ('names', 'rows', 'retry', 'affine_margin')))
        if diagnostic:
            key = (tick, json.dumps(node, sort_keys=True, allow_nan=True))
            if key not in seen:
                seen.add(key)
                attempts.append(node)
                status = str(node.get('status'))
                statuses[status] += 1
                if status == 'failed':
                    pre_solver += 1
                elif status == 'zero_gradient_nominal':
                    pre_solver += 1
                elif status.startswith('solved') or status.startswith('active_set_recovery('):
                    qp_success += 1
                    fallback += status.startswith('active_set_recovery(')
                else:
                    qp_failed += 1
                if 'names' in node:
                    names = node['names']
                    values = np.asarray(node['center_values']) + np.asarray(node['gradients']) @ (
                        np.asarray(node['torque'])-np.asarray(node['center']))
                    lower, upper = np.asarray(node['lower'], float), np.asarray(node['upper'], float)
                    tolerance = np.array([1e-10 if s.startswith('energy_') else 1e-7 for s in names])
                    distance = np.minimum(np.abs(values-lower), np.abs(values-upper))
                    for name in np.asarray(names)[distance <= tolerance]:
                        active[str(name)] += 1
                    max_normalized_violation = max(max_normalized_violation, float(np.max(
                        np.maximum(lower-values, values-upper) / tolerance)))
                    max_logged_defect = max(max_logged_defect, float(np.max(np.abs(values-node['values']))))
                elif 'affine_margin' in node:
                    for j, margin in enumerate(node['affine_margin']):
                        if abs(margin) <= 1e-10:
                            active[f'energy_account_{j}'] += 1
        for k, value in node.items():
            if isinstance(value, (dict, list)):
                visit(value, tick)

    visit(events)
    visit(failure)
    return dict(event_records=len(events), qp_attempt_records=len(attempts),
                qp_solves=qp_success+qp_failed, qp_successful_solves=qp_success,
                qp_failed_solver_calls=qp_failed, pre_solver_failure_or_analytic_records=pre_solver,
                qp_fallbacks=fallback, qp_status_counts=dict(statuses),
                qp_active_rows=dict(active), qp_active_row_count=sum(active.values()),
                reserve_candidate_rejections=dict(rejections),
                qp_max_normalized_violation=max_normalized_violation,
                qp_logged_row_defect=max_logged_defect,
                stop_reason=None if failure is None else failure.get('reason'),
                failure=failure,
                count_scope='Recorded solver diagnostics, including saved failures. Unlogged discarded candidate QP calls cannot be reconstructed.')


def audit_arrays(raw, task_start=None, force_target=10., recovery_hold=.5):
    """Compute all headline values solely from executed raw arrays."""
    p, c, r, b, m = raw.physics, raw.control, raw.reference, raw.baseline, raw.metadata
    t, dt, ix = p['t'], raw.dt, raw.indices
    case = m['case']
    start = float(m.get('config', {}).get('task_start', 8.71) if task_start is None else task_start)
    check = dict(physics_time_defect_s=validate_time(t, dt, ix),
                 omitted_physics_samples=int(raw.total_physics_count-len(t)),
                 physics_samples=len(t), physics_dt_s=dt,
                 initial_unrecorded_samples=int(ix[0]),
                 sampling_stage='x_k kinematics before integration; solved contact at x_k after step2; signed left rectangles',
                 moment_at_human_port='none: source applies a force at body-local point, with zero free moment')
    if np.any(ix < 0) or int(ix[-1]) != raw.total_physics_count-1:
        raise IntegrityError('truncated or invalid physics indices at end of run')
    for label, arrays in [('physics',p),('control',c),('reference',r),('baseline',b)]:
        count = len(arrays.get('t', []))
        for name, arr in arrays.items():
            if np.ndim(arr) and len(arr) != count:
                raise IntegrityError(f'{label}.{name}: array length mismatch')
            # NaN placeholder predictions in UF are not executed quantities.
            if name.startswith(('predicted', 'gradient', 'affine')):
                continue
            if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
                # Normal-axis position/velocity/acceleration are deliberately
                # undefined during force control. They are not plotted or used
                # in tangential metrics; preserve the NaNs and record their scope.
                if label == 'reference' and name in ('x_d','v_d','a_d','z_d'):
                    if name != 'z_d' and not np.all(np.isfinite(arr[:,:2])):
                        raise IntegrityError(f'{label}.{name}: nonfinite tangential reference')
                    check.setdefault('undefined_normal_reference_values',{})[name]=int(np.count_nonzero(~np.isfinite(arr)))
                else:
                    raise IntegrityError(f'{label}.{name}: nonfinite raw value')
    task = t >= start-1e-10
    if not np.any(task):
        raise IntegrityError('no samples in the declared task window')
    if np.any(np.diff(ix[task]) != 1):
        raise IntegrityError('task window is decimated; force/contact metrics would be invalid')
    if ix[0] > 0 and float(t[0]) > start+dt:
        raise IntegrityError('missing beginning of declared task window')
    metrics, E = signed_work(p['power'], dt, float(case['budget']))
    metrics['min_raw_ledger_including_initial_J']=min(float(case['budget']),metrics['min_substep_energy_J'])
    metrics['peak_human_force_N']=float(np.max(np.linalg.norm(p['force'],axis=1)))
    metrics['peak_human_point_speed_m_s']=float(np.max(np.linalg.norm(p['velocity'],axis=1)))
    recorded = p.get('energy', p.get('prefix_energy'))
    if recorded is None:
        raise IntegrityError('raw ledger samples are missing')
    check['account_recompute_defect_J'] = float(np.max(np.abs(recorded-E[:,None])))
    check['account_difference_J'] = float(np.max(np.abs(recorded[:,0]-recorded[:,1])))
    check['force_velocity_power_defect_W'] = float(np.max(np.abs(
        np.sum(p['force']*p['velocity'],axis=1)-p['power'])))
    if 'qvel' in p and 'human_tau' in p:
        check['generalized_human_power_defect_W'] = float(np.max(np.abs(
            np.sum(p['human_tau']*p['qvel'],axis=1)-p['power'])))
    if 'work' in p:
        check['recorded_work_defect_J'] = float(np.max(np.abs(p['work']-p['power']*dt)))
    if b:
        check['baseline_time_defect_s'] = float(np.max(np.abs(b['t']-t)))
        check['force_meter_difference_N'] = float(np.max(np.abs(p['normal_force']-b['F_task'][:,2])))
        check['task_work_quadrature_defect_J'] = float(np.max(np.abs(b['dW']-b['P_task']*dt)))
        single = b['n_task'] <= 1
        task_power = np.sum(b['F_task']*b['v_c'],axis=1)+np.sum(b['M_task']*b['w_tool'],axis=1)
        check['single_contact_force_moment_power_defect_W'] = float(np.max(np.abs(
            b['P_task'][single]-task_power[single]))) if np.any(single) else None
        check['task_vs_constraint_power_defect_W'] = float(np.max(np.abs(b['P_task']-b['P_constraint'])))
        metrics['task_work_in_J']=float(np.sum(b['dW'],dtype=np.float64))
        metrics['task_work_out_net_J']=-metrics['task_work_in_J']
        fn, present = b['F_task'][:,2], b['present']
    else:
        fn, present = p['normal_force'], p.get('present')
        if 'task_power' in p:
            metrics['retained_task_window_work_in_J']=float(np.sum(p['task_power']*dt,dtype=np.float64))
        if 'task_work' in p:
            metrics['task_work_in_J']=float(np.sum(p['task_work'],dtype=np.float64))
            check['task_work_quadrature_defect_J']=float(np.max(np.abs(p['task_work']-p['task_power']*dt)))
    err = fn[task]-force_target
    completed_s = float(t[-1]+dt)
    duration = float(m.get('duration', 18.))
    ramp_out, release = _endpoints(m)
    failure_path = raw.folder/'failure.json'
    failure = m.get('_recorded_failure')
    if failure is None and failure_path.is_file():
        failure=json.loads(failure_path.read_text())
    done = completed_s >= duration-1e-8 and failure is None
    ct = c.get('t')
    if ct is not None:
        cdt = float(np.median(np.diff(ct))) if len(ct)>1 else dt*4
        check['control_time_defect_s'] = validate_time(ct,cdt,name='control')
        check['control_dt_s'] = cdt
        check['physics_per_control_interval'] = int(round(cdt/dt))
        if len(ct)*int(round(cdt/dt)) != raw.total_physics_count:
            check['incomplete_control_interval_physics_samples'] = raw.total_physics_count-len(ct)*int(round(cdt/dt))
        # Governor is explicitly absent in UF/A; this is implementation semantics,
        # not a substitution for a missing signal on a governor-capable run.
        alpha = c.get('alpha')
        if alpha is None and m.get('variant') in ('UF','A'):
            alpha = np.ones(len(ct))
        if alpha is not None:
            active = ct >= start-1e-10
            metrics.update(minimum_alpha=float(np.min(alpha)),
                           pause_s=float(np.count_nonzero((alpha<=.01)&active)*cdt),
                           slowed_s=float(np.count_nonzero((alpha<.95)&active)*cdt),
                           governor_ticks=int(np.count_nonzero(alpha<1.-1e-10)),
                           alpha_recovery_s=sustained_recovery(ct,alpha>=.95,release,cdt,recovery_hold))
            first = np.flatnonzero((ct>=release-1e-10)&(alpha>=.95)) if release is not None else []
            metrics['alpha_first_crossing_recovery_s'] = max(0.,float(ct[first[0]]-release)) if len(first) else None
        if all(k in c for k in ('tau_exec','tau_nom')):
            changed=np.max(np.abs(c['tau_exec']-c['tau_nom']),axis=1)>1e-7
            target_changed=c.get('alpha_target',np.ones(len(ct)))<1.
            intervened=np.flatnonzero(changed|target_changed)
            metrics.update(interventions=int(np.count_nonzero(changed)),
                           first_intervention_s=float(ct[intervened[0]]) if len(intervened) else None,
                           max_change_from_governed_Nm=float(np.max(np.abs(c['tau_exec']-c['tau_nom']))))
            if 'tau_ungoverned' in c:
                metrics['max_total_change_from_ungoverned_Nm']=float(np.max(np.abs(c['tau_exec']-c['tau_ungoverned'])))
            if 'tau_previous' in c:
                metrics['max_command_step_Nm']=float(np.max(np.abs(c['tau_exec']-c['tau_previous'])))
        if 'sigma' in c:
            metrics['virtual_time_delay_s']=float(np.max(ct-c['sigma']))
        if 'predicted_prefix' in c and raw.source_kind=='full_executed_raw':
            parts=int(round(cdt/dt))
            if len(t)==len(ct)*parts:
                check['all_prefix_prediction_defect_J']=float(np.max(np.abs(
                    np.cumsum(p['power'].reshape(-1,parts)*dt,axis=1)-c['predicted_prefix'])))
        if 'predicted_normal_force' in c and len(t)==len(ct)*4:
            check['stage_force_prediction_defect_N']=float(np.max(np.abs(
                p['normal_force'].reshape(-1,4)-c['predicted_normal_force'])))
    metrics.update(case=case['name'],variant=m['variant'],completed_s=completed_s,
                   completion='PASS' if done else 'STOPPED' if failure else 'INCOMPLETE',
                   stop_reason=None if failure is None else failure.get('reason'),
                   energy='PASS' if metrics['min_substep_energy_J']>=-1e-10 else 'FAIL',
                   task_window_start_s=start, task_window_end_s=completed_s,
                   force_target_N=force_target,force_mae_N=float(np.mean(np.abs(err))),
                   force_rmse_N=float(np.sqrt(np.mean(err**2))),force_max_error_N=float(np.max(np.abs(err))),
                   force_peak_N=float(np.max(fn[task])),
                   contact_absence_s=float(np.count_nonzero(~present[task])*dt) if present is not None else None,
                   ramp_out_s=ramp_out,release_s=release,
                   interaction_complete=bool(release is not None and completed_s>=release-1e-8),
                   force_recovery_s=sustained_recovery(t,np.abs(fn-force_target)<=.5,release,dt,recovery_hold),
                   force_recovery_from_rampout_s=sustained_recovery(t,np.abs(fn-force_target)<=.5,ramp_out,dt,recovery_hold),
                   recovery_hold_s=recovery_hold,
                   interaction_point_body=case.get('body'),interaction_point_local_m=case.get('local'))
    desired, measured, count = ordered_progress(r)
    if desired is not None:
        rt=r['t'];mask=rt>=start-1e-10
        debt=desired-measured
        metrics.update(reference_progress_m=float(desired[-1]),measured_progress_m=float(measured[-1]),
                       completed_strokes=count,peak_ref_debt_m=float(np.max(debt[mask])),
                       terminal_ref_debt_m=float(debt[-1]),
                       peak_abs_reference_error_m=float(np.max(np.abs(r['x_d'][mask,0]-r['x'][mask,0]))) if 'x_d' in r else None)
        if all(k in r for k in ('v','v_d')):
            ve=r['v'][mask,0]-r['v_d'][mask,0]
            metrics.update(tangent_rmse_m_s=float(np.sqrt(np.mean(ve**2))),
                           tangent_mae_m_s=float(np.mean(np.abs(ve))),
                           tangent_peak_abs_error_m_s=float(np.max(np.abs(ve))),
                           tangent_peak_speed_m_s=float(np.max(np.abs(r['v'][mask,0]))))
    if 'tcp_velocity' in p:
        metrics['tangent_peak_physics_speed_m_s']=float(np.max(np.abs(p['tcp_velocity'][task,0])))
    if raw.tick_seconds is not None:
        tick=np.asarray(raw.tick_seconds)
        if len(tick)!=len(c['t']) or not np.all(np.isfinite(tick)):
            raise IntegrityError('whole-controller timing length/nonfinite defect')
        metrics.update(deadline_misses=int(np.count_nonzero(tick>.001)),
                       tick_seconds=dict(zip(('median','p95','p99','max'),map(float,
                            (*np.quantile(tick,[.5,.95,.99]),np.max(tick))))))
    elif raw.source_kind.startswith('compact'):
        metrics['tick_seconds']=None
        check['whole_controller_timing']='not retained in compact archive; use full guard.npz'
    return metrics,check


def compare_metrics(metrics, existing):
    """Compare shared headline values; tolerance excludes display rounding."""
    fields=('completed_s','net_outgoing_J','gross_outgoing_J','min_substep_energy_J','final_energy_J',
            'force_mae_N','force_max_error_N','force_peak_N','contact_absence_s',
            'reference_progress_m','measured_progress_m','completed_strokes','pause_s','slowed_s',
            'force_recovery_s','minimum_alpha','interventions','governor_ticks','virtual_time_delay_s','deadline_misses')
    out={}
    for field in fields:
        if field not in metrics or field not in existing:
            continue
        a,b=metrics[field],existing[field]
        if a is None or b is None:
            ok=a is None and b is None;difference=None
        else:
            difference=abs(a-b);ok=difference<=1e-8
        out[field]=dict(status='PASS' if ok else 'FAIL',difference=difference,tolerance=1e-8)
    return out


def identity_errors(checks):
    defects=[v for k,v in checks.items() if k.endswith(('_defect_J','_difference_J','_defect_W',
               '_difference_N','_defect_N')) and v is not None]
    return ['physical/account identity exceeds 1e-9 in its declared SI units'] if any(v>1e-9 for v in defects) else []


def point_kinematics_spot_check(raw, scene=None):
    """Rebuild J(q) at the declared body point independently of port helpers.

    This read-only check reconstructs kinematics from archived q/qdot. It never
    integrates the plant or invokes the controller, and also checks the saved
    actual application point. Samples include power extremes and the first,
    middle and last nonzero-force evaluations.
    """
    if 'q' not in raw.baseline or 'qd' not in raw.baseline:
        return dict(status='NOT_RETAINED',reason='compact archive has no joint state')
    import mujoco
    scene=Path(scene or Path(__file__).resolve().parents[1]/'assets'/'kuka_table_scene.xml')
    model=mujoco.MjModel.from_xml_path(str(scene))
    data=mujoco.MjData(model)
    body=model.body(raw.metadata['case']['body']).id
    local=np.asarray(raw.metadata['case']['local'])
    p,b=raw.physics,raw.baseline
    nonzero=np.flatnonzero(np.any(p['force']!=0.,axis=1))
    selected=nonzero if len(nonzero) else np.flatnonzero(p['t']>=raw.metadata.get('config',{}).get('task_start',8.71))
    if not len(selected):
        selected=np.arange(len(p['t']))
    indices=sorted(set([int(selected[0]),int(selected[len(selected)//2]),int(selected[-1]),
                        int(np.argmin(p['power'])),int(np.argmax(p['power']))]))
    position,velocity,power=0.,0.,0.
    jac=np.zeros((3,model.nv))
    for i in indices:
        if model.nq!=len(b['q'][i]) or model.nv!=len(b['qd'][i]):
            raise IntegrityError('independent point spot check joint-state dimension mismatch')
        data.qpos[:]=b['q'][i];data.qvel[:]=b['qd'][i]
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        world=data.xpos[body]+data.xmat[body].reshape(3,3)@local
        mujoco.mj_jac(model,data,jac,None,world,body)
        calculated=jac@data.qvel
        velocity=max(velocity,float(np.max(np.abs(calculated-p['velocity'][i]))))
        power=max(power,abs(float(p['force'][i]@calculated)-float(p['power'][i])))
        if 'point' in p:
            position=max(position,float(np.max(np.abs(world-p['point'][i]))))
    return dict(status='PASS' if max(position,velocity,power)<1e-10 else 'FAIL',
                samples=indices,scene_path=str(scene.resolve()),scene_sha256=sha256(scene),
                body=raw.metadata['case']['body'],local_m=local.tolist(),
                max_point_difference_m=position,max_velocity_difference_m_s=velocity,
                max_power_difference_W=power,tolerance=1e-10)


def audit_case(folder, task_start=None, force_target=10., recovery_hold=.5):
    raw=load_case(folder)
    metrics,checks=audit_arrays(raw,task_start,force_target,recovery_hold)
    events=audit_events(folder)
    metrics.update({k:v for k,v in events.items() if k not in ('failure','count_scope')})
    checks['qp_count_scope']=events['count_scope']
    checks['independent_point_kinematics']=point_kinematics_spot_check(raw)
    comparison={}
    old=Path(folder)/'metrics.json'
    if old.is_file():
        existing=json.loads(old.read_text());comparison=compare_metrics(metrics,existing)
        # Wall seconds are a measured scalar in saved run metadata, not inferred
        # by summing partial stage timings or excluding rejected candidate work.
        metrics['wall_seconds']=existing.get('wall_seconds')
        if metrics['wall_seconds']:
            metrics['realtime_factor']=metrics['completed_s']/metrics['wall_seconds']
    errors=identity_errors(checks)
    if any(v['status']=='FAIL' for v in comparison.values()):
        errors.append('existing metric disagrees with independent calculation')
    if checks['independent_point_kinematics']['status']=='FAIL':
        errors.append('independent actual-point Jacobian reconstruction disagrees')
    return dict(case=metrics['case'],variant=metrics['variant'],source_kind=raw.source_kind,
                metrics=metrics,checks=checks,comparison=comparison,status='FAIL' if errors else 'PASS',errors=errors,
                metadata=raw.metadata,
                inputs={p.name:dict(path=str(p.resolve()),sha256=sha256(p),bytes=p.stat().st_size)
                        for p in sorted(Path(folder).iterdir()) if p.is_file() and p.suffix in ('.json','.npz','.gz')})


def paired_identity(folders, left='C', right='D'):
    """Compare actual executed traces, never infer equality from rounded tables."""
    grouping={}
    for folder in folders:
        m=json.loads((folder/'metadata.json').read_text())
        grouping.setdefault(m['case']['name'],{})[m['variant']]=folder
    out={}
    for name, variants in grouping.items():
        if left not in variants or right not in variants:
            continue
        a,b=load_case(variants[left]),load_case(variants[right])
        fields={}
        for group in ('physics','control','reference'):
            aa,bb=getattr(a,group),getattr(b,group)
            names={'physics':('t','power','normal_force','energy','force','velocity'),
                   'control':('t','tau_exec','alpha','sigma'),
                   'reference':('t','x','v','x_d','v_d')}[group]
            for key in names:
                if key in aa and key in bb:
                    fields[group+'.'+key]=bool(aa[key].dtype==bb[key].dtype and aa[key].shape==bb[key].shape
                                               and aa[key].tobytes()==bb[key].tobytes())
        out[name]=dict(bitwise_executed_identity=all(fields.values()),fields=fields)
    return out


def load_normalized_case(folder):
    """Decode the portable bundle independently of the adapter being checked."""
    folder=Path(folder)
    provenance=json.loads((folder/'provenance.json').read_text())
    if sha256(folder/'executed.npz')!=provenance['output_sha256']:
        raise IntegrityError('normalized executed.npz hash mismatch')
    with np.load(folder/'executed.npz',allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    p={key:d[value] for key,value in dict(t='t_phys',power='power_H',work='work_H',
        velocity='velocity_H',force='force_H',point='point_H',normal_force='normal_force',
        present='present',task_power='task_power',task_work='task_work',tcp_velocity='tcp_velocity').items()}
    p['energy']=np.column_stack((d['energy_H'],d['energy_R']))
    c={'t':d['t_control']}
    for key in ('tau_exec','tau_nom','tau_previous','tau_ungoverned','alpha','alpha_target','sigma','reserve','reserve_margin'):
        if key in d:
            c[key]=d[key]
    r={'t':d['t_control']}
    for key,value in dict(x='x',v='v',x_d='x_des',v_d='v_des',f_d='force_des',phase='phase',
                          strokes='strokes',s_d='s_des').items():
        r[key]=d[value]
    m=dict(provenance['metadata'],_recorded_failure=provenance.get('archived_metrics',{}).get('failure'))
    n=len(p['t'])
    raw=RawCase(folder,m,p,c,r,{},float(d['h']),np.arange(n),n,'full_executed_normalized',d.get('tick_seconds'))
    return raw,provenance,d


def normalized_raw_parity(raw, provenance, data):
    """Check each archived signal against the exact raw file when available.

    Portable rechecks do not require the original absolute paths. In that case
    the full matrix audit and preservation hashes retain the one-time raw check;
    this function reports that direct source comparison is unavailable.
    """
    source=Path(provenance['original_directory'])
    if not (source/'guard.npz').is_file() or not (source/'baseline_log.npz').is_file():
        return dict(status='SOURCE_NOT_LOCAL',reason='original paths unavailable; normalized hash verified')
    original=load_case(source)
    p,c,r,b=original.physics,original.control,original.reference,original.baseline
    energy=p.get('energy',p.get('prefix_energy'))
    expected=dict(t_phys=p['t'],h=np.array(original.dt),t_control=c['t'],power_H=p['power'],
        work_H=p.get('work',p['power']*original.dt),energy_H=energy[:,0],energy_R=energy[:,1],
        force_H=p['force'],velocity_H=p['velocity'],point_H=p['point'],normal_force=p['normal_force'],
        present=b['present'],task_power=b['P_task'],task_work=b['dW'],tcp_position=b['tcp'],tcp_velocity=b['tcp_v'],
        x=r['x'],v=r['v'],x_des=r['x_d'],v_des=r['v_d'],force_des=r['f_d'],phase=r['phase'],
        strokes=r['strokes'],s_des=r['s_d'],tau_exec=c['tau_exec'],tau_nom=c['tau_nom'],tau_previous=c['tau_previous'])
    for key in ('alpha','alpha_target','sigma','reserve','reserve_margin','tau_ungoverned','x_ungoverned'):
        if key in c:
            expected[key]=c[key]
    for key in ('orientation','cross_error'):
        if key in p:
            expected[key]=p[key]
    if original.tick_seconds is not None:
        expected['tick_seconds']=original.tick_seconds
    equality={k:bool(k in data and v.dtype==data[k].dtype and v.shape==data[k].shape
                      and v.tobytes()==data[k].tobytes()) for k,v in expected.items()}
    missing=sorted(set(data)-set(expected))
    return dict(status='PASS' if all(equality.values()) and not missing else 'FAIL',
                exact_signal_parity=equality,unexpected_fields=missing,
                source_directory=str(source),source_files={k:sha256(source/k) for k in ('guard.npz','baseline_log.npz')})


REPORT_ALIASES={
    'completed_s':'completed_s','stop_reason':'stop_reason','human_work_in_J':'human_work_in_J',
    'net_outgoing_J':'net_outgoing_J','gross_outgoing_J':'gross_outgoing_J',
    'min_substep_energy_J':'min_raw_ledger_including_initial_J','final_energy_J':'final_energy_J',
    'peak_outgoing_power_W':'peak_outgoing_power_W','force_rmse_N':'force_rmse_N','force_mae_N':'force_mae_N',
    'force_max_error_N':'force_max_error_N','force_peak_N':'force_peak_N','contact_absence_s':'contact_absence_s',
    'tangent_rmse_m_s':'tangent_rmse_m_s','tangent_peak_speed_m_s':'tangent_peak_physics_speed_m_s',
    'reference_progress_m':'reference_progress_m','measured_progress_m':'measured_progress_m',
    'completed_strokes':'completed_strokes','peak_tracking_debt_m':'peak_abs_reference_error_m',
    'first_intervention_s':'first_intervention_s','torque_correction_ticks':'interventions',
    'max_held_command_change_Nm':'max_command_step_Nm',
    'max_governed_nominal_correction_Nm':'max_change_from_governed_Nm',
    'max_total_change_from_ungoverned_Nm':'max_total_change_from_ungoverned_Nm',
    'alpha_recovery_from_pulse_end_s':'alpha_recovery_s',
    'force_recovery_from_pulse_end_s':'force_recovery_s','force_recovery_from_rampout_s':'force_recovery_from_rampout_s',
    'rampout_s':'ramp_out_s','pulse_end_s':'release_s','task_work_in_J':'task_work_in_J',
    'pause_s':'pause_s','slowed_s':'slowed_s','virtual_time_delay_s':'virtual_time_delay_s',
    'deadline_misses':'deadline_misses','qp_solves':'qp_solves','qp_successful_solves':'qp_successful_solves',
    'qp_failed_solver_calls':'qp_failed_solver_calls','qp_fallbacks':'qp_fallbacks',
    'qp_active_row_count':'qp_active_row_count',
}


def compare_reported(metrics, reported, aliases=REPORT_ALIASES):
    comparisons={}
    for output,independent in aliases.items():
        if output not in reported or independent not in metrics:
            continue
        a,b=metrics[independent],reported[output]
        if a is None or b is None:
            ok=a is None and b is None;defect=None
        elif isinstance(a,str) or isinstance(b,str):
            ok=a==b;defect=None
        else:
            defect=abs(float(a)-float(b));ok=defect<=1e-8
        comparisons[output]=dict(status='PASS' if ok else 'FAIL',independent=a,reported=b,
                                 independent_definition=independent,absolute_difference=defect,tolerance=1e-8)
    # A and UF have no governor, so do not pretend their absent alpha metric is
    # an observed signal merely to match the implementation's None display.
    if 'completed' in reported and 'completion' in metrics:
        comparisons['completed']=dict(status='PASS' if reported['completed']==(metrics['completion']=='PASS') else 'FAIL',
                                      independent=metrics['completion']=='PASS',reported=reported['completed'])
    if 'tick_seconds' in reported and 'tick_seconds' in metrics:
        for q,value in (metrics['tick_seconds'] or {}).items():
            other=reported['tick_seconds'].get(q)
            comparisons['tick_seconds.'+q]=dict(status='PASS' if other is not None and abs(value-other)<=1e-12 else 'FAIL',
                                                independent=value,reported=other,tolerance=1e-12)
    return comparisons


def _common_window_metrics(raw,start,end):
    p,r=raw.physics,raw.reference
    mask=(p['t']>=start-1e-10)&(p['t']<end-1e-10)
    ctl=(r['t']>=start-1e-10)&(r['t']<end-1e-10)
    err=p['normal_force'][mask]-10.
    ve=r['v'][ctl,0]-r['v_d'][ctl,0]
    return dict(force_rmse_N=float(np.sqrt(np.mean(err**2))),force_mae_N=float(np.mean(np.abs(err))),
                force_max_error_N=float(np.max(np.abs(err))),force_peak_N=float(np.max(p['normal_force'][mask])),
                contact_absence_s=float(np.count_nonzero(~p['present'][mask])*raw.dt),
                tangent_rmse_m_s=float(np.sqrt(np.mean(ve**2))))


def audit_bundle(bundle, reported_metrics=None):
    """Independent complete-signal adapter and all displayed metric recheck."""
    bundle=Path(bundle)
    catalog=json.loads((bundle/'catalog.json').read_text())
    reported=json.loads(Path(reported_metrics).read_text()) if reported_metrics else {}
    cases,raws=[],{}
    for entry in catalog['cases']:
        key=entry['key'];raw,provenance,data=load_normalized_case(bundle/entry['folder']);raws[key]=raw
        metrics,checks=audit_arrays(raw)
        if 'alpha' not in raw.control:
            metrics['alpha_recovery_s']=None
        event=audit_events(raw.folder,raw.metadata.get('_recorded_failure'))
        metrics.update({k:v for k,v in event.items() if k not in ('failure','count_scope')})
        metrics['intervention_ticks']=int(np.count_nonzero(
            (np.max(np.abs(raw.control['tau_exec']-raw.control['tau_nom']),axis=1)>1e-7)|
            (raw.control.get('alpha_target',np.ones(len(raw.control['t'])))<1.-1e-10)))
        if 'tau_ungoverned' not in raw.control and raw.metadata['variant'] in ('A','UF'):
            metrics['max_total_change_from_ungoverned_Nm']=metrics['max_change_from_governed_Nm']
        checks['raw_signal_parity']=normalized_raw_parity(raw,provenance,data)
        errors=identity_errors(checks)
        if checks['raw_signal_parity']['status']=='FAIL':
            errors.append('normalized signals disagree with source raw arrays')
        cases.append(dict(key=key,case=metrics['case'],variant=metrics['variant'],metrics=metrics,
                          checks=checks,status='FAIL' if errors else 'PASS',errors=errors))
    for name in ('main','historical'):
        ukey=name+'__UF'
        if ukey not in raws:
            continue
        uref,up,_=ordered_progress(raws[ukey].reference)
        for record in cases:
            key=record['key']
            if not key.startswith(name+'__'):
                continue
            xref,xp,_=ordered_progress(raws[key].reference);n=min(len(uref),len(xref))
            record['metrics'].update(peak_original_reference_debt_m=float(np.max(np.abs(uref[:n]-xref[:n]))),
                terminal_measured_progress_debt_to_UF_m=float(up[n-1]-xp[n-1]),
                progress_ratio_to_UF_full_run=float(xp[-1]/up[-1]))
    additional={name:name for name in ('intervention_ticks','peak_original_reference_debt_m',
                                      'terminal_measured_progress_debt_to_UF_m','progress_ratio_to_UF_full_run')}
    for record in cases:
        key=record['key']
        record['comparison']=compare_reported(record['metrics'],reported.get(key,{}),dict(REPORT_ALIASES,**additional))
        if any(r['status']=='FAIL' for r in record['comparison'].values()):
            record['status']='FAIL';record['errors'].append('reported headline differs from independent calculation')
    common=[]
    common_path=Path(reported_metrics).parent/'common_window_metrics.json' if reported_metrics else None
    if common_path and common_path.exists():
        for key,values in json.loads(common_path.read_text()).items():
            m=_common_window_metrics(raws[key],values['window_start_s'],values['window_end_s'])
            common.append(dict(key=key,comparison=compare_reported(m,values,{k:k for k in m})))
    extensions={}
    for name in ('main','historical'):
        folder=bundle/('extension_'+name+'_D_01')
        if not (folder/'extension_log.npz').exists() or name+'__D' not in raws:
            continue
        original=raws[name+'__D'];segment=_arrays(folder/'extension_log.npz','ctl__')
        joined={k:np.concatenate((original.reference[k],segment[k])) for k in ('t','x','strokes','s_d')}
        _,measured,_=ordered_progress(joined)
        _,target,_=ordered_progress(raws[name+'__UF'].reference)
        first=np.flatnonzero(measured>=target[-1])
        extra=float(joined['t'][first[0]]-original.physics['t'][-1]-original.dt) if len(first) else None
        declared=json.loads((folder/'extension.json').read_text())
        defect=abs(extra-declared['extra_physical_time_s']) if extra is not None else None
        extensions[name]=dict(status='PASS' if defect is not None and defect<=1e-8 else 'FAIL',
                              extra_physical_time_s=extra,reference_amount_m=float(target[-1]),
                              declared_extra_time_s=declared['extra_physical_time_s'],absolute_difference_s=defect,
                              source_segment_sha256=sha256(folder/'extension_log.npz'))
    common_ok=all(c['status']=='PASS' for r in common for c in r['comparison'].values())
    return dict(schema='kuka-poc-independent-normalized-audit-v1',auditor_sha256=sha256(__file__),
                status='PASS' if all(c['status']=='PASS' for c in cases) and common_ok and all(
                    c['status']=='PASS' for c in extensions.values()) else 'FAIL',cases=cases,
                common_window_comparisons=common,extensions=extensions,
                inputs_root=str(bundle.resolve()),reported_metrics=str(reported_metrics),
                counts=dict(cases=len(cases),headline_comparisons=sum(len(c['comparison']) for c in cases),
                    common_window_comparisons=sum(len(c['comparison']) for c in common)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',action='append',type=Path,default=[])
    parser.add_argument('--case',action='append',type=Path,default=[])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--task-start',type=float)
    parser.add_argument('--bundle',type=Path,help='portable normalized inputs containing catalog.json')
    parser.add_argument('--reported-metrics',type=Path,help='independently check generated metrics.json')
    args=parser.parse_args(argv)
    if args.bundle:
        result=audit_bundle(args.bundle,args.reported_metrics)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False)
        print(json.dumps(dict(status=result['status'],counts=result['counts'])))
        return 0 if result['status']=='PASS' else 2
    folders=set(args.case)
    for root in args.root:
        folders.update(p.parent for p in root.rglob('metadata.json')
                       if (p.parent/'trace.npz').is_file() or (p.parent/'guard.npz').is_file())
    if not folders:
        parser.error('provide raw --case directory or --root containing cases')
    result=dict(schema='kuka-poc-independent-audit-v1',auditor_sha256=sha256(__file__),
                calculation_scope='Executed samples only; uncapped raw ledgers; no smoothing, extension or lower clipping',
                tolerances=dict(account_and_power_identity=1e-9,old_metric_comparison=1e-8,time_s=1e-10),cases=[],errors=[])
    for folder in sorted(folders):
        try:
            item=audit_case(folder,args.task_start)
            result['cases'].append(item)
            print(item['case'],item['variant'],item['status'],flush=True)
        except (ValueError,KeyError,OSError) as exc:
            result['errors'].append(dict(folder=str(folder),error=f'{type(exc).__name__}: {exc}'))
            print(folder,exc,file=sys.stderr,flush=True)
    result['paired_C_D']=paired_identity(sorted(folders)) if not result['errors'] else {}
    result['status']='PASS' if not result['errors'] and all(c['status']=='PASS' for c in result['cases']) else 'FAIL'
    result['counts']=dict(cases=len(result['cases']),errors=len(result['errors']),
                          passes=sum(c['status']=='PASS' for c in result['cases']))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    return 0 if result['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
