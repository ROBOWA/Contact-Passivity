"""Recompute Stage 2 acceptance and figures from executed raw logs, exclusively.

This reporting script is outside the frozen numerical package. It may be improved
without changing the numerical source hash. Failed trials remain in every table.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kuka_table_sim.recorder import RunLog,_json_default
from kuka_table_sim.energy_prediction import GuardConfig
from kuka_table_sim.energy_experiments import metrics,development_cases


def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,default=_json_default,allow_nan=False)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def frozen_tree_hash(commit):
    root=Path(__file__).resolve().parents[1]
    names=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only',commit,'--','kuka_table_sim'],text=True).splitlines()
    h=hashlib.sha256()
    for name in sorted(n for n in names if Path(n).parent==Path('kuka_table_sim') and n.endswith('.py')):
        h.update(Path(name).name.encode());h.update(subprocess.check_output(['git','-C',str(root),'show',commit+':'+name]))
    return h.hexdigest()


def read_case(folder):
    meta=json.loads((folder/'metadata.json').read_text());z=np.load(folder/'guard.npz',allow_pickle=False)
    c={k[5:]:z[k] for k in z.files if k.startswith('ctl__')};p={k[6:]:z[k] for k in z.files if k.startswith('phys__')}
    log=RunLog.load(folder/'baseline_log.npz');old=json.loads((folder/'metrics.json').read_text())
    # Implementation helper independently recomputes from saved actual data;
    # additional dot-product/cumsum checks below use different formulations.
    r=metrics(log,c,p,meta['case'],meta['variant'],old['wall_seconds'],z['tick_seconds'],old['failure'],GuardConfig(**meta['config']))
    changed=(np.max(np.abs(c['tau_exec']-c['tau_nom']),axis=1)>1e-7)|(c.get('alpha_target',np.ones(len(c['t'])))<1.)
    at=np.flatnonzero(changed)
    r['first_intervention_s']=float(c['t'][at[0]]) if len(at) else None
    r['first_intervention_energy_J']=c['energy_before'][at[0]].tolist() if len(at) else None
    r['max_total_change_from_ungoverned_Nm']=float(np.max(np.abs(c['tau_exec']-c.get('tau_ungoverned',c['tau_nom']))))
    r['max_change_from_governed_Nm']=float(np.max(np.abs(c['tau_exec']-c['tau_nom'])))
    h=log.meta['timestep'];work=float(np.dot(p['power'],np.full(len(p['power']),h)))
    velocity_identity=float(np.max(np.abs(np.einsum('ij,ij->i',p['force'],p['velocity'])-p['power'])))
    budget=meta['case']['budget'];E=budget+np.add.accumulate(p['power']*h)
    recorded=p.get('energy',p.get('prefix_energy'))
    independent=dict(work_difference_J=abs(work+r['net_outgoing_J']),
                     force_velocity_power_defect_W=velocity_identity,
                     account_recompute_defect_J=float(np.max(np.abs(recorded-E[:,None]))),
                     account_difference_J=float(np.max(np.abs(recorded[:,0]-recorded[:,1]))),
                     force_meter_difference_N=float(np.max(np.abs(p['normal_force']-log.phys['F_task'][:,2]))),
                     torque_logged_difference_Nm=float(np.max(np.abs(log.phys['tau_applied']-np.repeat(c['tau_exec'],4,axis=0)[:len(p['t'])]))))
    if 'predicted_prefix' in c:
        actual=p['power'].reshape(-1,4)*h
        independent['all_prefix_prediction_defect_J']=float(np.max(np.abs(np.cumsum(actual,axis=1)-c['predicted_prefix'])))
        if 'predicted_normal_force' in c:
            independent['stage_force_prediction_defect_N']=float(np.max(np.abs(p['normal_force'].reshape(-1,4)-c['predicted_normal_force'])))
            independent['stage_orientation_prediction_defect_rad']=float(np.max(np.abs(p['orientation'].reshape(-1,4,3)-c['predicted_orientation'])))
    if meta['variant'] in ('C','D'):
        active=np.any(p['force'][::4]!=0,axis=1)
        independent['reserve_min_gap_J']=float(np.min(c['energy_before'][active]-c['reserve'][active,None]-c['reserve_margin'][active,None])) if np.any(active) else 0.
    qp_count=0;qp_violation=0.;logged_defect=0.
    with gzip.open(folder/'events.json.gz','rt') as f:events=json.load(f)
    for event in events:
        for trial in event.get('attempts',[]):
            if 'names' not in trial:continue # frozen A has a separate tested solver
            qp_count+=1;G=np.asarray(trial['gradients']);base=np.asarray(trial['center_values'])
            value=base+G@(np.asarray(trial['torque'])-np.asarray(trial['center']))
            lo=np.asarray(trial['lower'],float);hi=np.asarray(trial['upper'],float)
            tolerance=np.asarray([1e-10 if name.startswith('energy_') else 1e-7 for name in trial['names']])
            qp_violation=max(qp_violation,float(np.max(np.maximum(lo-value,value-hi)/tolerance)))
            logged_defect=max(logged_defect,float(np.max(np.abs(value-trial['values']))))
    independent.update(qp_rows_recomputed=qp_count,qp_max_normalized_violation=qp_violation,qp_logged_row_defect=logged_defect)
    return r,meta,c,p,log,independent


def compact(folder,dest,c,p,log):
    # Retain EVERY task-evaluation physics sample and EVERY nonzero human-power
    # sample. Omitted powers are exactly zero; no float32, extrema sampling or
    # striding. Full pre-contact raw logs remain at the manifest path and hash.
    keep=(p['t']>=8.71-1e-10)|(p['power']!=0)
    i=np.flatnonzero(keep);a={k:p[k][i] for k in ('t','power','velocity','force','normal_force')}
    a['energy']=p.get('energy',p.get('prefix_energy'))[i]
    a['present']=log.phys['present'][i];a['task_power']=log.phys['P_task'][i]
    a['physics_indices']=i;a['total_physics_count']=np.array(len(p['t']))
    for k in ('orientation','cross_error'):
        if k in p:a[k]=p[k][i]
    for k in ('t','alpha','alpha_dot','alpha_target','sigma','reserve','tau_exec','tau_nom','tau_ungoverned','tau_previous','predicted_prefix','min_prefix'):
        if k in c:a['ctl__'+k]=c[k]
    for k in ('t','phase','f_d','f_n','f_int','tan_int','rot_int','x_d','v_d','a_d','s_d','e_o','strokes','stroke_state','x','v'):
        if k in log.ctl:a['reference__'+k]=log.ctl[k]
    np.savez_compressed(dest/'trace.npz',**a)
    for name in ('metadata.json','failure.json','failure_pre.npz','failure_complete.npz','final_state.npz','events.json.gz'):
        if (folder/name).exists():shutil.copy2(folder/name,dest/name)


def figures(data,out):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['path.simplify']=False
    import matplotlib.pyplot as plt
    for name in ('main','historical'):
        if name not in data:continue
        fig,axes=plt.subplots(4,2,figsize=(13,13),sharex=True)
        colors=dict(UF='tab:blue',A='tab:orange',B='tab:green',C='tab:red',D='tab:purple')
        for variant,(r,m,c,p,l,check) in data[name].items():
            label=f'{variant} ({r["completion"]})';t=p['t'];E=m['case']['budget']+np.cumsum(p['power']*.00025)
            style=dict(color=colors[variant],lw=.8,label=label)
            axes[0,0].plot(t,E,**style)
            if 'reserve' in c:axes[0,0].plot(c['t'],c['reserve'],ls=':',lw=.6,color=colors[variant])
            axes[0,1].plot(t,p['power'],**style)
            axes[1,0].plot(t,l.phys['F_task'][:,2],**style)
            axes[1,1].plot(t,(~l.phys['present']).astype(int),**style)
            axes[2,0].plot(c['t'],c.get('alpha',np.ones(len(c['t']))),**style)
            from kuka_table_sim.energy_diagnostic import progress
            ref,measured,_=progress(l)
            axes[2,1].plot(l.ctl['t'],measured,**style)
            axes[2,1].plot(l.ctl['t'],ref,ls=':',lw=.5,color=colors[variant])
            axes[3,0].plot(c['t'],np.max(np.abs(c['tau_exec']-c['tau_previous']),axis=1),**style)
            axes[3,1].plot(t,np.linalg.norm(p['velocity'],axis=1),**style)
        for ax,title in zip(axes.flat,['Both identical accounts (solid); backup reserve (dotted), J','Human power, W',
                                     'Normal force, N (fixed target 10)','Geometric absence',
                                     'Actual alpha','Measured progress; reference dotted, m',
                                     'Maximum held-command change, N m','Human-point speed, m/s']):
            ax.set_title(title);ax.grid(alpha=.2);ax.set_xlim(8.71,18.)
        axes[0,0].axhline(0,color='k',lw=.5);axes[0,0].legend(fontsize=7)
        axes[1,0].axhline(10,color='k',lw=.5)
        axes[1,0].axhspan(7,13,color='green',alpha=.1)
        axes[3,0].set_xlabel('Physical simulation time, s');axes[3,1].set_xlabel('Physical simulation time, s')
        fig.suptitle(name+' — all variants; stopped trials end at failure');fig.tight_layout();fig.savefig(out/(name+'_comparison.png'),dpi=150);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
        for variant,(r,m,c,p,l,check) in data[name].items():
            release=m['case']['start']+2*m['case']['ramp']+m['case']['hold']
            axes[0].plot(c['t']-release,c.get('alpha',np.ones(len(c['t']))),label=variant,lw=.8,color=colors[variant])
            axes[1].plot(p['t']-release,l.phys['F_task'][:,2]-10,label=variant,lw=.8,color=colors[variant])
        axes[0].set_ylabel('alpha');axes[0].axhline(.95,color='k',lw=.5);axes[0].legend()
        axes[1].set_ylabel('Force error, N');axes[1].axhspan(-.5,.5,color='green',alpha=.1)
        axes[1].set_xlabel('Physical time since force release, s')
        for ax in axes:ax.set_xlim(-.2,2.5);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/(name+'_recovery.png'),dpi=150);plt.close(fig)
    names=list(data);variants=['A','B','C','D'];scores=np.zeros((len(names),4))
    for i,n in enumerate(names):
        for j,v in enumerate(variants):
            if v in data[n]:
                r=data[n][v][0];scores[i,j]=1 if r['completion']=='FAIL' else 2 if r['contact']=='FAIL' or r['progress']=='FAIL' or r['energy']=='FAIL' else 3
    fig,ax=plt.subplots(figsize=(8,max(5,len(names)*.32)))
    ax.imshow(scores,vmin=0,vmax=3,cmap=plt.matplotlib.colors.ListedColormap(['white','#c45a5a','#e8bd69','#69b391']),aspect='auto')
    ax.set_xticks(range(4),variants);ax.set_yticks(range(len(names)),names)
    ax.set_title('Red: stopped; yellow: completed with unmet gate; green: completion/energy/contact/progress')
    fig.tight_layout();fig.savefig(out/'feasibility.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--roots',nargs='+',type=Path,default=[])
    ap.add_argument('--matrix-parent',type=Path);ap.add_argument('--validation',type=Path)
    ap.add_argument('--backup-audit',type=Path);ap.add_argument('--extensions',nargs='*',type=Path,default=[])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--preservation',type=Path,required=True)
    ap.add_argument('--tests',type=Path,required=True);ap.add_argument('--freeze',type=Path,required=True)
    ap.add_argument('--repeats',nargs='*',type=Path,default=[]);args=ap.parse_args()
    if args.matrix_parent:args.roots.extend(sorted(p for p in args.matrix_parent.iterdir() if p.is_dir()))
    if not args.roots:ap.error('provide --roots or --matrix-parent')
    args.out.mkdir(parents=True,exist_ok=False);raw=args.out/'raw';raw.mkdir();data={};manifest=[];fingerprints={}
    freeze=json.loads(args.freeze.read_text())
    for root in args.roots:
        for path in sorted(root.glob('*/*/metrics.json')):
            folder=path.parent;r,m,c,p,l,check=read_case(folder);name=m['case']['name'];v=m['variant']
            if v in data.get(name,{}):raise ValueError('duplicate case/variant '+name+'/'+v)
            dest=raw/(name+'__'+v);dest.mkdir();compact(folder,dest,c,p,l)
            fingerprints.setdefault(name,{})[v]=dict(q=hashlib.sha256(l.phys['q'].tobytes()).hexdigest(),
                                                     qd=hashlib.sha256(l.phys['qd'].tobytes()).hexdigest(),
                                                     tau=hashlib.sha256(l.phys['tau_applied'].tobytes()).hexdigest(),
                                                     alpha_one=bool(np.all(c.get('alpha',np.ones(len(c['t'])))==1)))
            data.setdefault(name,{})[v]=(r,m,c,p,l,check) if name in ('main','historical') else (r,m,None,None,None,check)
            manifest.append(dict(case=name,variant=v,path=str(folder.resolve()),files={f.name:dict(size=f.stat().st_size,sha256=sha(f)) for f in folder.iterdir() if f.is_file()}))
    heldout=json.loads((args.freeze.parent/'heldout_manifest.json').read_text())
    required=set(development_cases())|{c['name'] for c in heldout['cases']}
    if set(data)!=required or any(set(vs)!=set(('UF','A','B','C','D')) for vs in data.values()):raise ValueError('incomplete matrix: every required case must have UF/A/B/C/D')
    for name,variants in data.items():
        uf=variants['UF'][0]
        for v,item in variants.items():
            r=item[0];r['progress_ratio_to_UF']=r['measured_progress_m']/max(uf['measured_progress_m'],1e-15)
            r['reference_progress_ratio_to_UF']=r['reference_progress_m']/max(uf['reference_progress_m'],1e-15)
            r['progress']='PASS' if r['completion']=='PASS' and r['progress_ratio_to_UF']>=.8 else 'FAIL'
    checks={n:{v:item[5] for v,item in vs.items()} for n,vs in data.items()}
    results={n:{v:item[0] for v,item in vs.items()} for n,vs in data.items()}
    def passfail(value):return 'PASS' if value else 'FAIL'
    preservation=json.loads(args.preservation.read_text())
    transparency={}
    for name in ('nohuman-ample','nohuman-zero','ample'):
        if name not in data:continue
        uf=fingerprints[name]['UF'];transparency[name]={}
        for v in ('A','B','C','D'):
            item=fingerprints[name][v]
            transparency[name][v]=dict(bitwise_q=uf['q']==item['q'],bitwise_qvel=uf['qd']==item['qd'],
                                         bitwise_torque=uf['tau']==item['tau'],alpha_one=item['alpha_one'])
    final=[data[n]['D'][0] for n in data]
    primary=[data[n]['D'][0] for n in ('main','historical')]
    held=[data[n]['D'][0] for n in data if n.startswith('new-')]
    tree_hash=frozen_tree_hash(freeze['commit'])
    numerical_provenance=tree_hash==freeze['source_sha256'] and all(m['source_sha256']==tree_hash and m.get('source_end_sha256')==tree_hash for vs in data.values() for r,m,*rest in vs.values())
    frozen_config=json.loads((args.freeze.parent/'frozen_config.json').read_text())
    declared_cases=development_cases()|{c['name']:c for c in heldout['cases']}
    settings_preserved=sha(args.freeze.parent/'frozen_config.json')==freeze['config_sha256'] and sha(args.freeze.parent/'heldout_manifest.json')==freeze['heldout_sha256']
    for name,vs in data.items():
        expected=dict(frozen_config,rate_factor=declared_cases[name].get('rate_factor',1.))
        settings_preserved=settings_preserved and all(m['config']==expected and m['case']==declared_cases[name] for r,m,*rest in vs.values())
    original_manifest=json.loads((args.freeze.parent/'preservation_manifest.json').read_text())
    original_hashes={item['path']:item['sha256'] for item in original_manifest['tracked']}
    asset_preserved=all(original_hashes.get(p)==h for p,h in freeze['asset_sha256'].items()) and all(
        h==freeze['asset_sha256']['assets/'+name][:16] for vs in data.values() for r,m,*rest in vs.values() for name,h in m['asset_hashes'].items())
    physical_checks=all(max(check[k] for k in ('work_difference_J','account_recompute_defect_J','account_difference_J','force_velocity_power_defect_W'))<1e-10 for vs in data.values() for *_,check in vs.values())
    reserve_checks=all(check.get('reserve_min_gap_J',0.)>=-1e-10 for vs in data.values() for *_,check in vs.values())
    qp_checks=all(check['qp_max_normalized_violation']<=1.+1e-6 and check['qp_logged_row_defect']<=1e-10 for vs in data.values() for *_,check in vs.values())
    repeated={}
    for folder in args.repeats:
        rr,meta,*_=read_case(folder);original=results[rr['case']][rr['variant']]
        fields=('net_outgoing_J','gross_outgoing_J','min_substep_energy_J','force_max_error_N','contact_absence_s','measured_progress_m','completed_s')
        repeated[str(folder)]=dict(case=rr['case'],variant=rr['variant'],max_absolute_difference=max(abs(rr[k]-original[k]) for k in fields),
                                  same_failure_reason=(rr['failure'] or {}).get('reason')==(original['failure'] or {}).get('reason'),
                                  raw_files={f.name:dict(size=f.stat().st_size,sha256=sha(f)) for f in folder.iterdir() if f.is_file()})
    backup_audit=json.loads((args.backup_audit/'audit.json').read_text()) if args.backup_audit else None
    extensions={str(p):json.loads((p/'extension.json').read_text()) for p in args.extensions}
    gates=dict(preservation=preservation['status'],frozen_numerical_provenance=passfail(numerical_provenance),unchanged_plant_assets=passfail(asset_preserved),frozen_settings_and_holdouts=passfail(settings_preserved),
               existing_and_new_tests='PENDING see test transcript',independent_power_work_accounts=passfail(physical_checks),
               independent_reserve_accounts=passfail(reserve_checks),logged_multirow_physical_residuals=passfail(qp_checks),
               independent_closed_loop_backup=passfail(backup_audit is not None and backup_audit['source_sha256']==tree_hash and backup_audit['max_complete_plant_state_difference']<1e-10 and backup_audit['max_prefix_work_difference_J']<1e-11),
               transparency=passfail(all(all(all(x.values()) for x in vs.values()) for vs in transparency.values())),
               final_executed_torque_rate=passfail(all(r['torque_bound_residual_Nm']<=1e-7 and r['rate_bound_residual_Nm']<=1e-7 for r in final)),
               final_energy_all_physics=passfail(all(r['energy']=='PASS' for r in final)),
               final_original_force_target=passfail(all(min(r['internal_force_target_range_N'])>=10.-1e-10 and max(r['internal_force_target_range_N'])<=10.+1e-10 for r in final)),
               primary_completion=passfail(all(r['completion']=='PASS' for r in primary)),
               primary_contact=passfail(all(r['contact']=='PASS' and r['force_mae_N']<=.5 for r in primary)),
               primary_progress=passfail(all(r['progress']=='PASS' for r in primary)),
               primary_recovery=passfail(all(r['recovery']=='PASS' for r in primary)),
               generalization=passfail(bool(held) and all(all(r[k]=='PASS' for k in ('completion','energy','contact','progress','recovery')) and r['force_mae_N']<=.5 for r in held)),
               fresh_process_repeats=passfail(bool(repeated) and all(r['max_absolute_difference']<1e-10 and r['same_failure_reason'] for r in repeated.values())))
    text=args.tests.read_text();gates['existing_and_new_tests']='PASS' if 'passed' in text and 'failed' not in text.lower() else 'FAIL'
    status='PASS' if all(x=='PASS' for x in gates.values()) else 'PARTIAL'
    acceptance=dict(status=status,gates=gates,numerical_freeze=freeze,targets=dict(force_mae_N=.5,force_max_error_N=3.,cross_mm=3.,orientation_deg=2.,progress_fraction=.8,alpha_recovery_s=2.,energy_tolerance_J=1e-10,window_start_s=8.71),
                    independent_checks=checks,transparency=transparency,repeats=repeated,independent_backup=backup_audit,extensions=extensions,
                    known_regression='historical is development, not held out',
                    evidence_limit='finite held-current-force policy rollouts and empirical terminal persistence; no recursive feasibility or invariant-set proof')
    acceptance['categories']=dict(correctness=passfail(all(gates[k]=='PASS' for k in ('frozen_numerical_provenance','existing_and_new_tests','independent_power_work_accounts','final_executed_torque_rate'))),
                                  completion=gates['primary_completion'],energy=gates['final_energy_all_physics'],contact=gates['primary_contact'],
                                  progress=gates['primary_progress'],preservation=gates['preservation'],generalization=gates['generalization'])
    write(args.out/'acceptance.json',acceptance);write(args.out/'metrics.json',results);write(args.out/'raw_manifest.json',manifest)
    if args.validation:
        shutil.copy2(args.validation/'validation.json',args.out/'validation.json')
        shutil.copy2(args.validation/'validation.npz',args.out/'validation.npz')
    if args.backup_audit:shutil.copytree(args.backup_audit,args.out/'independent_backup_audit')
    for p in args.extensions:shutil.copytree(p,args.out/p.name)
    figures(data,args.out)
    rows=[]
    for name,variants in results.items():
        for v,r in variants.items():
            rows.append(f'| {name} | {v} | {r["completion"]} | {r["completed_s"]:.3f} | {r["net_outgoing_J"]:.9f} | {r["gross_outgoing_J"]:.9f} | {r["min_substep_energy_J"]:.3g} | {r["force_mae_N"]:.3f} / {r["force_max_error_N"]:.3f} | {r["contact_absence_s"]:.5f} | {r["progress_ratio_to_UF"]:.3f} | {r["pause_s"]:.3f} |')
    gate_rows='\n'.join(f'| {k} | {v} |' for k,v in gates.items())
    report=f'''# KUKA energy guard Stage 2 — {status}

This opt-in implementation adds four-physics-prefix energy enforcement, a causal
closed-loop backup reserve, smooth path timing and explicit task constraints.
Original Stage 1 code, branch tips and results are preserved. No hardware was used.
Unmet gates: {', '.join(k for k,v in gates.items() if v!='PASS') or 'none'}.
Completion, energy, contact, progress and generalization are separate outcomes.

## Provenance and preservation

Numerical source commit `{freeze['commit']}`; package SHA256
`{freeze['source_sha256']}`. Configuration and new untouched fixtures were frozen
before final evaluation. Platform/config/assets are recorded per case. Reporting
commits may be later. Original worktree and branch remain at
`32c4866183b020c50724160749b588f140228db5`. The external verified backup contains
540 ignored/untracked files (5,238,703,332 bytes); 428 tracked files are hash checked.
See preservation manifest, final verification, reviews and raw manifest.

## Algorithm and information

The unchanged plant uses 1 ms control and four 0.25 ms split physics steps.
The only exogenous predictor input is the current measured WORLD force vector,
held throughout every rollout. The human point and Jacobian move every substep.
The real fixture is sampled once per control boundary. Future pulse onset,
release, phase and magnitude are never passed to candidate/backup prediction.
This exact held-force claim does not cover arbitrary within-tick force variation.

Each branch restores full `mjSTATE_INTEGRATION` plus serialized controller, FT
observer, integrators, FSM, stroke, clock, energy, prior executed torque and base
external force. Controller compute runs once per future control interval; observe
runs at every future physics step. Live compute state is installed once after
acceptance and real physics/observer updates run independently. Rejected branches
cannot change live state. Complete trusted-local failure snapshots retain this
hybrid state and counterfactual/cache configuration.

Signed work is `dW = force dot (J_point qdot) * h`; both accounts use the same
actual human work exactly. Every executed physics sample updates `E += dW`.
There is no clipping, recharge other than positive physical work, budget change,
or reserve addition. Gross outgoing is `sum(max(-dW,0))` and remains separate.

## QP

Decision is seven joint torques, with objective
`0.5*(tau-tau_nom)^T H*(tau-tau_nom)` anchored to the ORIGINAL nominal command
of that proposal. `H = diag(1/torque_scale^2) + S_nom^T W_y S_nom`, weight 1 in
every matched variant. Scales and original torque/rate/trust bounds are unchanged;
the reduced-rate challenge uses an explicitly labeled factor 0.02.
Torque scales are (20,20,10,10,5,2,2) N m; task output scales are
(1 N,0.01 m/s x3,0.1 rad/s x3). Motor limits are
(320,320,176,176,110,40,40) N m; trust halfwidths are
(80,80,60,60,40,20,20) N m, with rates equal to these halfwidths /0.001 s.
The central torque stencil is 0.02 N m and never clips an override. Four work
prefix rows per account replace the endpoint row. D also uses four-stage normal
force [7,13] N, normal speed +/-0.005 m/s, cross error +/-3 mm, and conservative
orientation component bounds +/-2/sqrt(3) degrees. Fresh nonlinear norm checking
uses the original 2 degree target. Contact presence and unexpected contacts are
checked nonlinearly. The fixed established-contact window is 8.71 s onward,
including all subsequent SEARCH/recovery and force reramping.

The new multi-row OSQP solver whitens the objective and scales rows; all original
N m/J/task-unit residuals decide acceptance, independently of solver status.
The old one-halfspace fallback is never called. Four bounded retries use shrinking
trust regions while retaining the original objective. Affine infeasibility is
reported as local affine infeasibility, not global physical impossibility.
Every accepted action receives a fresh nonlinear interval check. Stage-zero
power can be torque insensitive: a depleted account may require stopping before
the next interval even though an endpoint-only QP previously returned a solution.

## Backup and governor

Backup target alpha is zero with a 0.1 s quintic transition and human-point
damping `tau = tau_governed_nom - J_H^T (40*v_H)`. Nominal force/posture/FSM and
observers continue evolving. The reserve is the worst cumulative outgoing
PHYSICS prefix of that specific backup, not an optimum or a new energy source.
A continued-progress proposal permits a declared 20 ms reaction delay before
the same brake; the entire delay is included in reserve, task and rate checks.
The bounded search favors targets (1,0.5,0), checking both account margins.
An ample continued-progress plan is reconsidered after at most 20 ms.

Horizon checkpoints are 20/50/100/250/500/1000/2000 ms, at the same physical
integration resolution. Unsettled tails are UNRESOLVED, never zero reserve.
Terminal checks require human speed <=0.001 m/s, abs human power <=0.002 W,
tool speed <=0.001 m/s, angular speed <=0.01 rad/s, alpha <=0.01 and alpha rate
<=0.1/s, plus contact/task and actuator/rate admissibility throughout. These
conditions persist for an additional 20 ms (21 consecutive control intervals).
They provide empirical evidence only, not an invariant terminal set.

Acceptance requires available energy to cover the whole plan's worst prefix plus
1e-5 J reserve guard, and every first-interval physics prefix >= -1e-10 J.
The one-interval QP uses a 1e-8 J numerical guard for nonzero force. Independent
held-force replay errors are reported in acceptance.json; these guards are
conservative resolutions for deterministic replay, not bounds on unknown future
force. The old affine Stage 1 margin is not reused for multi-step forecasting.
With current measured force exactly zero, the held-zero forecast has identically
zero human work for any motion, so a zero-reserve shortcut skips the long backup.
The logged held_zero status describes that forecast assumption; it does not claim
the person will keep applying zero force. This shortcut does not establish a
settled backup or recursive task feasibility. D still freshly checks every first
interval's task and energy prefixes; changed measurements are reconsidered next tick.
Cached suffixes require exact full dynamic-value state and exact force equality;
both account margins are rechecked. Each cached first command is replayed fresh.

The clock obeys `sigma_dot=alpha`, with `v=X'(sigma)*alpha` and
`a=X''(sigma)*alpha^2+X'(sigma)*alpha_dot`. Physical sensor/integrator/FSM timers
remain physical. Energy pause does not trigger SEARCH or reduce the 10 N target.
Reference progress accumulates only virtual time, with unchanged spatial strokes
and no catch-up. Measured progress advances ordered endpoint frontiers, not
absolute jitter. Original and governed torques/references are logged via an
isolated ungoverned controller counterfactual, including its actual observations.

## Matched results

UF is the matched unfiltered reference. A is frozen Stage 1; B changes only
prefix enforcement and the validated multi-row solver; C adds reserve/governor;
D adds explicit per-stage task protection. Historical is a known regression.
New cases were untouched until the frozen-source evaluation. Stopped trials
retain their full prefix and failure snapshots; their energy status does not
imply task completion.

| case | variant | completion | end s | net out J | gross out J | min physics E J | MAE / max error N | absence s | measured progress / UF | pause s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

All case details, orientation/cross errors, reference progress, completed strokes,
release recovery, QP/candidate rejection history and runtime quantiles are in
metrics.json and raw artifacts. Figures use physical time and retain every peak.
Both accounts coincide by construction; they do not test account mismatch.
Main D first intervenes at {primary[0]['first_intervention_s']:.3f} s with
{primary[0]['first_intervention_energy_J'][0]:.9f} J remaining, versus
{results['main']['A']['first_intervention_s']:.3f} s and
{results['main']['A']['first_intervention_energy_J'][0]:.9f} J for A.
Its maximum total joint-torque difference from the isolated ungoverned controller
is {primary[0]['max_total_change_from_ungoverned_Nm']:.3f} N m; the maximum correction
from its governed nominal is {primary[0]['max_change_from_governed_Nm']:.3f} N m.
Small per-tick command changes therefore do not mean a small total governor effect.
Continuation experiments through 20 s measure the extra physical time needed to
reach the matched UF run's 18 s measured amount, without resetting the clock,
controller or accounts: {json.dumps({p:dict(extra_physical_time_s=v['extra_physical_time_s'],achieved=v['achieved_18s_UF_amount']) for p,v in extensions.items()})}.

## Engineering gates

| gate | status |
|---|---|
{gate_rows}

Runtime includes Python allocation, serialization, isolated prediction and
buffered logging. Whole tick and component quantiles, deadline misses and
wall/simulation factors are logged. Tick quantiles describe accepted/completed
ticks; a stopped tick's duration is separately retained as seconds_failed_tick,
and total wall time includes it. Final matrix jobs may share the CPU; fresh
repeats provide an additional execution check. This implementation does not
establish 1 kHz execution; reserve forecasts can take orders of magnitude longer.

## Evidence, reproduction and limits

The preliminary same-policy 0/20/50/100 ms hindsight diagnostic is frozen in
../early_diagnosis.md. Earlier action improved contact but did not alone meet the
force band; outcomes were not monotonic. It supports a limited braking-reserve
hypothesis, not a universal cause. Online policy uses no hindsight timing.

Full logs remain at paths and SHA256 hashes in raw_manifest.json. Committed
compact traces retain EVERY physics sample from 8.71 s onward and every nonzero
human-power sample, all control/constraint events and complete failure states.
Omitted human power is exactly zero; no float32 conversion or extrema striding.
Regenerate results in NEW output directories; commands are in ../reproduction.md.
Final tests are copied beside this report. Three review/fix/retest rounds are in
../reviews.md. Unmet gates remain explicit. Finite held-current-force reserve and
terminal persistence do not prove safety against a changing future human force,
recursive feasibility, global viability or hardware performance.

The next technical question is whether a task-preserving backup with a robust
force-change envelope and useful progress can remain feasible across the tested
budgets, rather than spending additional effort on an unvalidated local margin.
'''
    (args.out/'report.md').write_text(report)
    failed=', '.join(k for k,v in gates.items() if v!='PASS')
    (args.out/'morning_summary_zh.md').write_text(f'''# Stage 2 早间摘要 — {status}

已实现可选的四个物理子步能量约束、因果制动备用能量估计、平滑虚拟时间进度调节和接触保护。
原始代码、分支和旧结果通过哈希检查，保存在原工作区；新实验全部位于独立工作区。

提前减速实验支持“需要留出制动能量”的有限假设，但提前减速本身不能保证 ±3 N 力误差。
主场景最终 D：完成状态 {primary[0]['completion']}，净输出 {primary[0]['net_outgoing_J']:.9f} J，
最大力误差 {primary[0]['force_max_error_N']:.3f} N，接触缺失 {primary[0]['contact_absence_s']:.5f} s，
进度为未过滤对照的 {primary[0]['progress_ratio_to_UF']*100:.1f}%。
历史失败场景最终 D：完成状态 {primary[1]['completion']}，最大力误差 {primary[1]['force_max_error_N']:.3f} N。

仍未通过：{failed}。能量满足、试验完成、任务质量和泛化是独立结论。
所有停止试验和失败快照均保留；不能把明确停止当作完成控制任务。
备用轨迹只假定当前测得的人力保持不变，并不证明未来变力下的递归可行性或硬件安全。
运行时间不能证明 1 kHz 实时控制。详见 report.md、acceptance.json 和 reviews.md。
''')
    shutil.copy2(args.tests,args.out/'tests.txt')
    print(json.dumps(dict(status=status,gates=gates)))


if __name__=='__main__':main()
