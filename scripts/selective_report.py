"""Recalculate executed evidence, export plot data, and build the comparison report.

The audit below deliberately reconstructs accounts/constraints from raw execution
arrays without importing the controller's account or constraint functions.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES=['K0','K1','K2','K3','K4']
CASES=['A_exact_nohuman','B_mismatch_nohuman','C_blocking','D_combined','E_forearm_block','F_help_block','S_starved']
COLORS=dict(zip(MODES,['#333333','#D55E00','#0072B2','#CC79A7','#009E73']))
PROTECTED={'K0':[],'K1':[2],'K2':[1],'K3':[0,1],'K4':[0,1]}


def dump(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def load(root,case,mode):
    folder=root/case/mode
    with np.load(folder/'selective.npz') as z:
        p={k[6:]:z[k] for k in z.files if k.startswith('phys__')}
        c={k[5:]:z[k] for k in z.files if k.startswith('ctl__')}
    with np.load(folder/'baseline_log.npz') as z:
        b={k[5:]:z[k] for k in ('ctl__t','ctl__x','ctl__x_d','ctl__v','ctl__v_d','ctl__e_o')}
        bm=json.loads(str(z['meta_json']))
    meta=json.loads((folder/'metadata.json').read_text());m=json.loads((folder/'metrics.json').read_text())
    with gzip.open(folder/'events.json.gz','rt') as f:events=json.load(f)
    return dict(p=p,c=c,b=b,bm=bm,meta=meta,m=m,events=events,folder=folder)


def audit(run):
    p,c,m,meta=run['p'],run['c'],run['m'],run['meta'];cfg=meta['config'];h=.00025
    E=np.asarray(cfg['initial']).copy();maxledger=0.;maxdiscard=0.
    for power,stored,discard in zip(p['power'],p['energy'],p['discarded']):
        raw=E+power*h;expected_discard=np.maximum(0.,raw-np.array(cfg['caps']))
        E=raw-expected_discard
        maxledger=max(maxledger,float(np.max(np.abs(E-stored))))
        maxdiscard=max(maxdiscard,float(np.max(np.abs(discard-expected_discard))))
    ph=np.einsum('ij,ij->i',p['force'],p['human_velocity'])
    pt=np.einsum('ij,ij->i',p['task_force'],p['task_point_velocity'])+np.einsum('ij,ij->i',p['task_moment'],p['task_omega'])
    hat=np.einsum('ij,ij->i',p['task_hat_force'],p['task_hat_point_velocity'])+np.einsum('ij,ij->i',p['task_hat_moment'],p['task_hat_omega'])
    reconstructed=np.column_stack([ph,ph+pt-hat,ph+pt])
    power_error=float(np.max(np.abs(reconstructed-p['power'])))
    jacobian_error=float(np.max(np.abs(ph-np.einsum('ij,ij->i',p['human_tau'],p['qd']))))
    protected=PROTECTED[meta['mode']]
    energy_ok=all(np.min(p['energy'][:,i])>=cfg['floors'][i]-1e-10 for i in protected)
    barrier_min={str(i):float(np.min(p['power'][:,i]+cfg['barrier_rate']*(p['energy_before'][:,i]-cfg['floors'][i]-cfg['safe_energy_J']))) for i in protected}
    barrier_ok=all(v>=-1e-7 for v in barrier_min.values())
    peak=float(max(0.,-np.min(ph)));peak_ok=meta['mode'] not in ('K3','K4') or peak<=cfg['peak_human_W']+1e-7
    command_error=float(np.max(np.abs(p['torque'][:4*len(c['t'])]-np.repeat(c['tau_exec'],4,axis=0))))
    limits=np.array(run['bm']['torque_limit']);torque_ok=bool(np.all(np.abs(p['torque'])<=limits+1e-7))
    rates=np.array([80,80,60,60,40,20,20])
    bounds_ok=meta['mode']=='K0' or bool(np.all(np.abs(c['tau_exec']-c['tau_previous'])<=rates+1e-7) and np.all(np.abs(c['tau_exec']-c['tau_nom'])<=rates+1e-7))
    attempts=[a for e in run['events'] for a in e.get('attempts',[])]
    affine_max=0.;nonlinear_min=0.;accepted=0
    for a in attempts:
        if not a.get('accepted'):continue
        accepted+=1
        vals=np.asarray(a['values']);tols=np.array([1e-10 if '_energy_' in n else 1e-7 for n in a['names']])
        affine_max=max(affine_max,float(np.max(-vals-tols)))
        nonlinear_min=min(nonlinear_min,float(np.min(np.array(a['nonlinear_values'])+tols)))
    # A solver invocation that fails is retained in the attempts list too.
    count_ok=len(attempts)==m['qp_attempts']
    held_force_ok=bool(np.all(p['force']==np.repeat(p['force'][::4],4,axis=0)[:len(p['force'])]))
    increments=np.linalg.norm(np.diff(p['force'][::4],axis=0),axis=1)
    force_class_ok=not len(increments) or bool(np.max(increments)<=cfg.get('force_rate_N_s',60.)*.001+1e-12)
    identity=np.array(cfg['initial'])+np.sum(p['power'],axis=0)*h-np.sum(p['discarded'],axis=0)
    net=(-np.sum(p['power'],axis=0)*h).tolist();gross=(np.maximum(-p['power'],0).sum(axis=0)*h).tolist()
    metrics_ok=bool(np.max(np.abs(np.asarray(m['net_outgoing_J'])-net))<1e-12 and np.max(np.abs(np.asarray(m['gross_outgoing_J'])-gross))<1e-12)
    passed=(maxledger<1e-12 and maxdiscard<1e-12 and power_error<1e-12 and jacobian_error<1e-12 and energy_ok and barrier_ok and peak_ok and command_error<1e-10 and torque_ok and bounds_ok and affine_max<=0 and nonlinear_min>=-1e-12 and count_ok and metrics_ok and held_force_ok and force_class_ok and np.max(np.abs(identity-p['energy'][-1]))<1e-10 and np.max(p['prediction_error'][:,0])<=1e-7 and np.max(p['prediction_error'][:,1])<=1e-10 and not np.any(p['unexpected']) and meta['source_sha256']==meta['source_end_sha256'])
    return dict(status='PASS' if passed else 'FAIL',max_ledger_error_J=maxledger,max_discard_error_J=maxdiscard,
        power_reconstruction_error_W=power_error,human_generalized_power_error_W=jacobian_error,
        protected_energy_ok=energy_ok,protected_barrier_ok=barrier_ok,barrier_min_W=barrier_min,
        protected_peak_ok=peak_ok,command_error_Nm=command_error,torque_bounds_ok=torque_ok,rate_trust_ok=bounds_ok,
        qp_attempt_count=len(attempts),accepted_qp_count=accepted,affine_excess=affine_max,nonlinear_excess=-nonlinear_min,
        accounting_identity_error_J=float(np.max(np.abs(identity-p['energy'][-1]))),metrics_ok=metrics_ok,
        measured_force_held=held_force_ok,measured_force_class_ok=force_class_ok)


def enrich(run):
    p,c,b,m=run['p'],run['c'],run['b'],run['m'];case=run['meta']['case']
    release=max((x['start']+2*x['ramp']+x['hold'] for x in case['pulses']),default=8.71)
    post=p['t']>=release-1e-10;last=p['t']>=17.-1e-10;cb=b['t']>=release-1e-10
    m.update(release_s=release,post_release_available=bool(np.any(post)),
        release_peak_speed_m_s=float(np.max(np.abs(b['v'][cb,0]))) if np.any(cb) else None,
        release_force_peak_error_N=float(np.max(np.abs(p['normal_force'][post]-10))) if np.any(post) else None,
        release_reference_debt_mm=float(np.abs(b['x_d'][np.flatnonzero(cb)[0],0]-b['x'][np.flatnonzero(cb)[0],0])*1000) if np.any(cb) else None,
        post_release_max_reference_debt_mm=float(np.max(np.abs(b['x_d'][cb,0]-b['x'][cb,0]))*1000) if np.any(cb) else None,
        last_second_force_max_error_N=float(np.max(np.abs(p['normal_force'][last]-10))) if np.any(last) else None,
        last_second_force_mean_error_N=float(np.mean(np.abs(p['normal_force'][last]-10))) if np.any(last) else None,
        maximum_reference_debt_mm=float(np.max(np.abs(b['x_d'][:,0]-b['x'][:,0]))*1000),
        first_intervention_s=float(c['t'][np.flatnonzero(c['intervention'])[0]]) if np.any(c['intervention']) else None,
        max_torque_correction_Nm=float(np.max(np.abs(c['tau_exec']-c['tau_nom']))))


def plot_case(data,case,fields,title,out,window,selected=MODES,filename=None):
    fig,axes=plt.subplots(2,2,figsize=(10.2,6.6),layout='constrained')
    for ax,(key,label) in zip(axes.flat,fields):
        for mode in selected:
            r=data.get((case,mode))
            if r is None:continue
            p,c,b=r['p'],r['c'],r['b'];t=p['t']
            values={
                'EH':p['energy'][:,0],'ER':p['energy'][:,1],'EW':p['energy'][:,2],
                'Pout':-p['power'][:,0],'Pdelta':p['delta'],'Fn':p['normal_force'],
                'vx':p['human_velocity'][:,0],
                'x':b['x'][:,0]-b['x'][0,0],
                'debt':1000*(b['x_d'][:,0]-b['x'][:,0]),'alpha':c['alpha'],
                'correction':np.linalg.norm(c['tau_exec']-c['tau_nom'],axis=1),
                'discard':np.cumsum(p['discarded'][:,0]),
            }[key]
            if key in ('x','debt'):t=b['t']
            elif key in ('alpha','correction'):t=c['t']
            mask=(t>=window[0])&(t<=window[1]);ii=np.flatnonzero(mask)
            # Rendering reduction retains each bucket's extrema, plus last sample.
            if len(ii)>6000:
                buckets=np.array_split(ii,2500);ii=np.unique(np.r_[[j for bucket in buckets for j in (bucket[0],bucket[np.argmin(values[bucket])],bucket[np.argmax(values[bucket])],bucket[-1])]])
            ax.plot(t[ii],values[ii],color=COLORS[mode],label=mode,lw=1.15)
            if r['m']['completion']=='STOP':
                end=r['m']['completed_s'];ax.axvline(end,color=COLORS[mode],ls=':',lw=.8,alpha=.65)
                if len(ii):ax.plot(t[ii[-1]],values[ii[-1]],'x',color=COLORS[mode],ms=5)
        if key in ('EH','ER','EW'):ax.axhline(0,color='black',ls='--',lw=.7)
        if key=='EH':ax.axhline(.15,color='black',ls=':',lw=.7)
        if key=='Pout':ax.axhline(.15,color='black',ls='--',lw=.7)
        if key=='Fn':ax.axhline(10,color='black',ls='--',lw=.7)
        for pulse in data[(case,selected[-1])]['meta']['case']['pulses']:
            ax.axvspan(pulse['start'],pulse['start']+2*pulse['ramp']+pulse['hold'],color='#777777',alpha=.07)
        ax.set(xlim=window,xlabel='Time [s]',ylabel=label);ax.grid(alpha=.2)
    axes.flat[0].legend(ncol=len(selected),fontsize=8,loc='best');fig.suptitle(title,fontsize=13)
    path=out/(filename or case);fig.savefig(path.with_suffix('.pdf'));fig.savefig(path.with_suffix('.png'),dpi=150);plt.close(fig)


def texescape(value):
    return str(value).replace('\\','/').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--refresh',action='store_true',help='Regenerate reporting artifacts only; retain numerical runs')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=a.refresh);figdir=a.out/'figures';figdir.mkdir(exist_ok=a.refresh);raw=a.out/'raw';raw.mkdir(exist_ok=a.refresh)
    data={};audits={};source=set();configs=set();assets=set();exits={}
    for case in CASES:
        for mode in (['K3','K4'] if case=='S_starved' else MODES):
            r=load(a.root,case,mode);enrich(r);key=(case,mode);data[key]=r
            audits[case+'/'+mode]=audit(r);source.add(r['meta']['source_sha256'])
            configs.add(json.dumps(r['bm']['config'],sort_keys=True));assets.add(json.dumps(r['meta']['asset_hashes'],sort_keys=True))
            # Full-precision, full-rate sufficient data for account/power plots;
            # full simulator state and QP histories remain in the run directory.
            compact={k:r['p'][k] for k in ('t','power','energy','discarded','delta','normal_force','present','force','human_velocity')}
            compact.update({'ctl_'+k:r['c'][k] for k in ('t','alpha','sigma','tau_nom','tau_exec','intervention','qp_attempts')})
            compact.update({'baseline_'+k:r['b'][k] for k in ('t','x','x_d','v')})
            np.savez_compressed(raw/f'{case}_{mode}.npz',**compact)
            (raw/f'{case}_{mode}_metadata.json').write_text((r['folder']/'metadata.json').read_text())
            shutil.copyfile(r['folder']/'events.json.gz',raw/f'{case}_{mode}_events.json.gz')
            for name in ('failure.json','failure_pre.npz','failure_complete.npz','final_state.npz'):
                if (r['folder']/name).exists():shutil.copyfile(r['folder']/name,raw/f'{case}_{mode}_{name}')
    # Confirm exact-model no-human transparency directly, without rounded metrics.
    base=data[('A_exact_nohuman','K0')];transparency={}
    for mode in ('K2','K3','K4'):
        r=data[('A_exact_nohuman',mode)]
        transparency[mode]=bool(np.array_equal(base['b']['x'],r['b']['x']) and np.array_equal(base['p']['torque'],r['p']['torque']))
    governor={}
    for case in CASES:
        r3,r4=data[(case,'K3')],data[(case,'K4')]
        n=min(len(r3['c']['t']),len(r4['c']['t']));ix=np.flatnonzero(r4['c']['alpha_target'][:n]!=1.)
        n=int(ix[0]) if len(ix) else n
        governor[case]=dict(identical_before_first_governor_action=bool(np.array_equal(r3['p']['q'][:4*n],r4['p']['q'][:4*n]) and np.array_equal(r3['c']['tau_exec'][:n],r4['c']['tau_exec'][:n])),
            shared_prefix_ticks=n,identical_config=r3['meta']['config']==r4['meta']['config'])
    suites=[]
    for name in ('tests_full.xml','tests_report_audit.xml'):
        suite=ET.parse(a.out.parent/name).getroot();suites.extend(list(suite) if suite.tag=='testsuites' else [suite])
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    matrix=json.loads((a.root/'matrix.json').read_text());exit_ok=all(r['exit_code'] in (0,2) and ((r['exit_code']==0)==(r['metrics']['completion']=='COMPLETE')) for r in matrix['results'])
    revisions={r['meta']['commit'] for r in data.values()};git_hashes={}
    for revision in revisions:
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'kuka_table_sim/'],text=True).splitlines()
        sha=hashlib.sha256()
        for path in sorted(p for p in paths if p.endswith('.py') and p.count('/')==1):
            sha.update(Path(path).name.encode());sha.update(subprocess.check_output(['git','show',revision+':'+path]))
        git_hashes[revision]=sha.hexdigest()
    source_gate=len(revisions)==1 and set(git_hashes.values())==source
    nominal_regression=json.loads((a.out.parent/'nominal_regression.json').read_text())
    preservation=json.loads((a.out.parent/'preservation_after.json').read_text())
    summary=dict(status='PASS' if all(v['status']=='PASS' for v in audits.values()) and all(transparency.values()) and all(v['identical_before_first_governor_action'] and v['identical_config'] for v in governor.values()) and len(source)==len(configs)==len(assets)==1 and exit_ok and source_gate and nominal_regression['status']==preservation['status']=='PASS' and tests['failures']==tests['errors']==0 else 'FAIL',
        runs=audits,exact_nohuman_transparency=transparency,matched_governor=governor,
        numerical_source_sha256=list(source),recorded_git_source_hashes=git_hashes,source_provenance_ok=source_gate,
        nominal_controller_configurations=len(configs),plant_asset_sets=len(assets),process_exits_ok=exit_ok,tests=tests,
        frozen_baseline_regression=nominal_regression['status'],earlier_worktree_preservation=preservation['status'])
    dump(a.out/'audit.json',summary)
    common={}
    for case in CASES:
        modes=['K3','K4'] if case=='S_starved' else MODES
        end=min(data[(case,mode)]['m']['completed_s'] for mode in modes)
        rows={}
        for mode in modes:
            r=data[(case,mode)];p=r['p'];mask=p['t']<end-1e-10
            steady=mask&(p['t']>=8.71-1e-10)
            rows[mode]=dict(net_H_outgoing_J=float(-np.sum(p['power'][mask,0])*.00025),
                gross_H_outgoing_J=float(np.maximum(-p['power'][mask,0],0).sum()*.00025),
                peak_H_outgoing_W=float(max(0.,-np.min(p['power'][mask,0]))),
                force_max_error_N=float(np.max(np.abs(p['normal_force'][steady]-10))),
                contact_absent_s=float(np.sum(~p['present'][steady])*.00025))
        common[case]=dict(common_end_s=end,modes=rows)
    dump(a.out/'common_window_metrics.json',common)
    allmetrics=[data[k]['m'] for k in data];dump(a.out/'metrics.json',allmetrics)
    with (a.out/'metrics.csv').open('w') as f:
        columns=['case','mode','completion','completed_s','peak_human_outgoing_W','force_max_error_N','contact_absent_s','measured_progress_m','intervention_ticks','qp_attempts','governor_min_alpha','first_intervention_s','last_second_force_max_error_N']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(allmetrics)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    figures=[
        ('fig3_exact','A_exact_nohuman',[('EW','Whole-port account [J]'),('ER','Residual account [J]'),('x','TCP x progress [m]'),('Fn','Normal force [N]')],
         'F3 adaptation: predictable task work and whole-port depletion',(8.5,18),MODES,
         'Exact model, no human. K2/K3/K4 should be transparent; K1 accounts for ordinary task work. Crosses and dotted lines mark explicit stops, never zero-padded continuations.'),
        ('fig4_mismatch','B_mismatch_nohuman',[('EH','Human account [J]'),('ER','Residual account [J]'),('Pdelta','Task-model mismatch power [W]'),('x','TCP x progress [m]')],
         'F4 adaptation: task-model mismatch without human input',(8.5,18),MODES,
         'Actual friction remains 1; nominal pair friction is .20. Nonzero residual power with zero human power establishes independent accounting. It does not establish robustness to unknown plant dynamics.'),
        ('fig5_human','C_blocking',[('Pout','Human outgoing power [W]'),('EH','Human account [J]'),('ER','Residual account [J]'),('Fn','Normal force [N]')],
         'F5 adaptation: dedicated human energy and peak-power protection',(9.2,12.7),MODES,
         'All modes use the same 5 N TCP block. K3/K4 enforce the .15 W human peak bound and H/R rows. K2 has the larger residual budget and no separate H constraint. Compare completion as well as safe prefixes.'),
        ('fig6_combined','D_combined',[('Pout','Human outgoing power [W]'),('Pdelta','Mismatch power [W]'),('EH','Human account [J]'),('ER','Residual account [J]')],
         'F6 adaptation: simultaneous human input and task-model mismatch',(9.2,18),['K0','K2','K3','K4'],
         'At every executed sample, pR=pH+(pT-pT_hat). H and R have independently initialized budgets. The cap is applied separately to each account; the raw reconstruction checks both identities.'),
        ('fig7_charge','F_help_block',[('Pout','Human outgoing power [W]'),('EH','Human account [J]'),('discard','Discarded H charging work [J]'),('Fn','Normal force [N]')],
         'F7 adaptation: helping, upper-cap saturation, then blocking',(9.2,13.2),['K0','K2','K3','K4'],
         'One continuous trial: helping starts at 9.5 s, blocking at 10.3 s. H charges only to .15 J; discarded charging is explicitly logged. Net outgoing work and gross outgoing work are distinct.'),
        ('fig8_governor','C_blocking',[('alpha','Reference clock speed alpha'),('debt','Reference minus TCP x [mm]'),('vx','Human-point x speed [m/s]'),('correction','Joint correction norm [Nm]')],
         'F8 adaptation: matched K3/K4, governor OFF/ON',(9.2,18),['K3','K4'],
         'K3/K4 share the QP, bounds, budgets and model; only the reference-clock governor changes. A stopped K3 trace cannot establish a post-release catch-up peak. Joint correction is not a reconstructed Cartesian command.'),
        ('fig9_help_recovery','F_help_block',[('alpha','Reference clock speed alpha'),('debt','Reference minus TCP x [mm]'),('vx','Human-point x speed [m/s]'),('Fn','Normal force [N]')],
         'Matched governor comparison through helping and blocking',(9.2,18),['K3','K4'],
         'Both trials complete. After the 11.7 s release, K3/K4 peak x speeds are .056434/.064006 m/s and force errors .113279/1.416587 N. The governor reduces human output and reference debt but worsens these recovery peaks.'),
        ('fig10_forearm','E_forearm_block',[('Pout','Human outgoing power [W]'),('EH','Human account [J]'),('Fn','Normal force [N]'),('x','TCP x progress [m]')],
         'Second human port: forearm rather than TCP',(9.2,18),MODES,
         'Same world-fixed blocking pulse at link5 local (0,.04,.12) m. Human power uses that actual moving point Jacobian, not TCP velocity.'),
        ('fig11_starved','S_starved',[('Pout','Human outgoing power [W]'),('EH','Human account [J]'),('alpha','Reference clock speed alpha'),('Fn','Normal force [N]')],
         'Small-budget challenge: H initialized at .001 J',(9.2,18),['K3','K4'],
         'Only the initial H budget changes. Explicit failures are retained with complete pre-action and failure snapshots; no safe-fallback or viability guarantee is inferred.'),
    ]
    dump(a.out/'figure_spec.json',[dict(filename=f,case=c,fields=fields,title=t,window=w,modes=m,caption=caption) for f,c,fields,t,w,m,caption in figures])
    for filename,case,fields,title,window,modes,caption in figures:
        plot_case(data,case,fields,title,figdir,window,modes,filename)
    completed=sum(m['completion']=='COMPLETE' for m in allmetrics);c0=data[('C_blocking','K0')]['m'];c4=data[('C_blocking','K4')]['m']
    reduction=100*(1-c4['net_outgoing_J'][0]/c0['net_outgoing_J'][0])
    progress_ratio=c4['measured_progress_m']/c0['measured_progress_m']
    findings=f"{completed}/32 trials completed; stopped trials remain in the figures. In the primary TCP block, K4 completed 18 s, reduced net human output by {reduction:.1f}% relative to K0, and reached {c4['peak_human_outgoing_W']:.6f} W peak human output. Its maximum 10 N force error was {c4['force_max_error_N']:.3f} N and contact absence was {c4['contact_absent_s']:.5f} s. It achieved {100*progress_ratio:.1f}% of K0 measured progress in the same 18 s. K3 stopped at {data[('C_blocking','K3')]['m']['completed_s']:.3f} s; this supports a completion benefit for the governor on this fixture, not a measured K3 post-release recovery peak."
    methods="These are new closed-loop KUKA comparisons inspired by the previous 2-DOF reference. K0 nominal; K1 whole monitored port; K2 residual only; K3 H/R plus human peak power, governor OFF; K4 identical K3 QP, governor ON. The actual plant and nominal 10 N controller are unchanged. A separate contact model evaluates predicted task work at each actual state. Only its explicit tip/table tangential friction changes (1 versus .20). Seven joint torques and a virtual reference clock adapt the reference architecture; they are not its original two-variable Cartesian controller or spatial-anchor governor."
    limitations="This is deterministic simulation evidence, not a mathematical proof of general safety or reproduction of the reference paper's numerical results. Prediction uses true-plant dynamics with current measured force held for 1 ms; the task model alone is mismatched. The terminal human guard assumes force increments no greater than 60 N/s times 1 ms. Finite-difference rows are checked by fresh nonlinear rollouts; stopped cases expose solver, nonlinear acceptance, or local feasibility limits. No verified fallback, hardware execution, continuous-time certificate, or universal task-preservation claim is made. Gross outgoing energy is not bounded by a net-work account. K3/K4 have no Stage 2 reserve/backup or hard contact-quality constraints."
    f3=data[('F_help_block','K3')]['m'];f4=data[('F_help_block','K4')]['m']
    recovery=f"Both K3 and K4 complete the helping/blocking case. From its 11.7 s release through 18 s, K3/K4 peak x speed is {f3['release_peak_speed_m_s']:.6f}/{f4['release_peak_speed_m_s']:.6f} m/s and maximum 10 N force error is {f3['release_force_peak_error_N']:.6f}/{f4['release_force_peak_error_N']:.6f} N. K4 reduces post-release reference debt from {f3['post_release_max_reference_debt_mm']:.3f} to {f4['post_release_max_reference_debt_mm']:.3f} mm and gross human output from {f3['gross_outgoing_J'][0]:.6f} to {f4['gross_outgoing_J'][0]:.6f} J, but increases those recovery peaks. The reference paper's smoother-recovery claim is therefore not reproduced by this governor adaptation."
    lines=['# KUKA selective comparisons — PARTIAL','',findings,'',recovery,'',methods,'',
        '| Controller | Protected accounts | Human peak | Governor |','|---|---|---|---|',
        '| K0 | monitored only | no | off |','| K1 | W=H+actual task work | no | off |',
        '| K2 | R=H+task-model mismatch | no | off |','| K3 | H and R | .15 W | off |','| K4 | same as K3 | .15 W | on |','',
        '| Case | Mode | End s | Result | H net out J | H gross out J | min H J | min R J | peak H W | max force error N | lost contact s | QP attempts |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in allmetrics:
        lines.append(f"| {m['case']} | {m['mode']} | {m['completed_s']:.3f} | {m['completion']} | {m['net_outgoing_J'][0]:.6f} | {m['gross_outgoing_J'][0]:.6f} | {m['minimum_J'][0]:.6g} | {m['minimum_J'][1]:.6g} | {m['peak_human_outgoing_W']:.6f} | {m['force_max_error_N']:.3f} | {m['contact_absent_s']:.5f} | {m['qp_attempts']} |")
    k1=data[('A_exact_nohuman','K1')]['m'];mis=data[('B_mismatch_nohuman','K3')]['m'];human2=data[('C_blocking','K2')]['m'];charge=data[('F_help_block','K4')]['m']
    evidence=[
        ('F3: avoid charging predictable task work',f"K1 first filters at {k1['first_intervention_s']:.3f} s and stops at {k1['completed_s']:.3f} s ({k1['failure']['reason']}); K2/K3/K4 match nominal bitwise. This is accounting/filter evidence, not proof of unavoidable physical infeasibility."),
        ('F4: model error appears separately',f"No-human mismatch gives maximum |pR-pH|={mis['residual_human_max_difference_W']:.6f} W and minimum R={mis['minimum_J'][1]:.6f} J; H remains .12 J. No residual intervention is required in this finite window."),
        ('F5: dedicated human protection',f"Residual-only K2 reaches H={human2['minimum_J'][0]:.6f} J and {human2['peak_human_outgoing_W']:.6f} W peak output. K4 completes with H={c4['minimum_J'][0]:.6f} J minimum and peak {c4['peak_human_outgoing_W']:.6f} W. K3 only demonstrates a protected prefix."),
        ('F6: simultaneous human and mismatch',"Executed power decomposition is independently checked for every mode. Completion and task quality must be read separately from that accounting identity."),
        ('F7: bounded bidirectional charging',f"K4 helping/blocking discards {charge['discarded_J'][0]:.6f} J of H charging at the .15 J cap; final H={charge['final_J'][0]:.6f} J. Net H output={charge['net_outgoing_J'][0]:.6f} J, gross={charge['gross_outgoing_J'][0]:.6f} J."),
        ('F8: governor changes recovery',"K3/K4 have identical pre-governor trajectories and constraints. Primary K3 stops before release. "+recovery),
    ]
    dump(a.out/'reference_claims.json',dict(reference='paper/main.pdf and paper/results_section.tex',comparisons=[dict(claim=k,evidence=v) for k,v in evidence]))
    lines+=['','| Reference comparison | New KUKA evidence |','|---|---|']+[f'| {k} | {v} |' for k,v in evidence]
    lines+=['',limitations,'',f"Numerical source: `{next(iter(source))}` at `{data[('A_exact_nohuman','K0')]['meta']['commit']}`. Audit: **{summary['status']}**. Regression plus report-audit suite: {tests['tests']} tests, {tests['failures']} failures, {tests['errors']} errors. All executed H/R/W ledgers, cap discard, physical power duals, actuator bounds, peak/CBF rows and accepted QP residuals were recalculated from raw arrays. Audit PASS denotes consistent implementation evidence; it does not convert stopped trials into successful controllers.",'',
        'The new no-human nominal run matches the earlier frozen baseline bitwise for all 72,000 physics samples (q, velocity, applied torque, task force and power) and 18,000 control samples (TCP, reference and measured force). Reference NaNs before trajectory initialization have identical bytes and masks. The preservation audit passes for all 4,417 files across the three earlier worktrees.','',
        'Full logs: `results/selective/final/` (ignored by Git). Full-rate, float64 account/force/progress plot data, complete final/failure states, and lossless QP histories for all 32 cases: `raw/`. Do not compare a stopped prefix with an 18 s total without the completion column. Exact-model K2/K3/K4 trajectories match K0 bitwise. Human and residual powers coincide only when the nominal task model is exact; budgets remain distinct.','']
    for filename,case,fields,title,window,modes,caption in figures:lines.extend([f'![{title}](figures/{filename}.png)','',caption,''])
    (a.out/'report.md').write_text('\n'.join(lines))
    tex=[r'\documentclass[10pt,letterpaper]{article}',r'\usepackage[margin=.65in]{geometry}',r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern,graphicx,booktabs,array}',r'\usepackage[hidelinks]{hyperref}',r'\usepackage{fancyhdr}',r'\pagestyle{fancy}\fancyhf{}\fancyhead[L]{KUKA selective comparisons --- PARTIAL}\fancyhead[R]{16 October 2026}\fancyfoot[C]{\thepage}',r'\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\emergencystretch}{2em}',r'\begin{document}',r'\section*{Selective accounting and matched governor comparisons}',texescape(findings)+r'\par',texescape(recovery)+r'\par',texescape(methods)+r'\par',r'\begin{center}\begin{tabular}{llll}\toprule Mode & Protected accounts & Human peak & Governor\\\midrule K0 & monitored only & none & off\\ K1 & whole W & none & off\\ K2 & residual R & none & off\\ K3 & H and R & .15 W & off\\ K4 & same K3 QP & .15 W & on\\\bottomrule\end{tabular}\end{center}',texescape(limitations)+r'\par',
        r'\textbf{Parameters and audit.} H/R/W initial=(.12,.30,.30) J; caps=(.15,.30,.30) J; floors=0; CBF rate=5/s; safe-energy offset=25 microjoules. Four .25 ms physics steps per held 1 ms torque. Central FD=.02 Nm, task weight=1, hard rows and nonlinear acceptance. K4 pauses after an H-triggered intervention and resumes below .02 N measured force, with .1 s clock transitions.',
        texescape(f"Independent raw-array audit: {summary['status']}; regression suite: {tests['tests']} tests, {tests['failures']} failures, {tests['errors']} errors. An audit PASS checks implementation consistency; stopped trials remain failures to complete.")+r'\par',
        r'\textbf{Provenance.} Source commit \texttt{'+data[('A_exact_nohuman','K0')]['meta']['commit'][:12]+r'}. Source SHA256: \texttt{'+next(iter(source))[:32]+r'}\\\texttt{'+next(iter(source))[32:]+r'}. Frozen design/configuration, raw data, audit and reproduction commands accompany this PDF.']
    for start,title in [(0,'All primary comparisons'),(15,'Remaining comparisons and challenge')]:
        tex.extend([r'\clearpage\section*{'+title+'}',r'\begingroup\small\setlength{\tabcolsep}{3pt}\begin{center}\begin{tabular}{llrrrrrrr}\toprule Case & Mode/end s & H net J & min H J & min R J & Peak W & Force err N & Loss s & QPs\\\midrule'])
        end=15 if start==0 else len(allmetrics)
        for m in allmetrics[start:end]:
            case=m['case'].split('_')[0];mode=m['mode']+(' STOP' if m['completion']=='STOP' else '')
            tex.append(f"{case} & {mode}/{m['completed_s']:.3f} & {m['net_outgoing_J'][0]:.5f} & {m['minimum_J'][0]:.4g} & {m['minimum_J'][1]:.4g} & {m['peak_human_outgoing_W']:.4f} & {m['force_max_error_N']:.2f} & {m['contact_absent_s']:.4f} & {m['qp_attempts']} "+r'\\')
        tex.extend([r'\bottomrule\end{tabular}\end{center}\endgroup\par',r'A: exact/no human; B: mismatch/no human; C: TCP block; D: mismatch plus TCP block; E: forearm block; F: help then block; S: starved H. Force error is measured relative to the original 10 N target over the available 8.71 s onward window. Totals from stopped prefixes must not be read as complete-run savings. Negative unprotected account values remain visible.\par'])
        for m in allmetrics[start:end]:
            if m['failure']:tex.append(texescape(f"{m['case']}/{m['mode']}: stopped at {m['completed_s']:.3f} s: {m['failure']['reason']}.")+r'\\')
    for filename,case,fields,title,window,modes,caption in figures:
        tex.extend([r'\clearpage\section*{'+texescape(title)+'}',r'\includegraphics[width=\textwidth]{figures/'+filename+'.pdf}',texescape(caption)+r'\par'])
        for mode in modes:
            m=data[(case,mode)]['m']
            text=f"{mode}: {m['completion']}, end {m['completed_s']:.3f} s; net/gross H output {m['net_outgoing_J'][0]:.6f}/{m['gross_outgoing_J'][0]:.6f} J; min H/R {m['minimum_J'][0]:.6g}/{m['minimum_J'][1]:.6g} J; peak H {m['peak_human_outgoing_W']:.6f} W; measured progress {m['measured_progress_m']:.4f} m; QP attempts {m['qp_attempts']}."
            tex.append(texescape(text)+r'\par')
    tex.extend([r'\clearpage\section*{Which reference claims are supported?}'])
    for k,v in evidence:tex.extend([r'\textbf{'+texescape(k)+r'.} '+texescape(v)+r'\par'])
    tex.append(texescape(limitations))
    tex.append(r'\end{document}');(a.out/'report.tex').write_text('\n'.join(tex)+'\n')
    cmd=['/opt/homebrew/bin/tectonic','-X','compile','--keep-logs',str(a.out/'report.tex')]
    build=subprocess.run(cmd,capture_output=True,text=True);(a.out/'pdf_build.stdout.txt').write_text(build.stdout);(a.out/'pdf_build.stderr.txt').write_text(build.stderr)
    dump(a.out/'manifest.json',dict(audit_status=summary['status'],runs=len(data),completed=completed,source_commit=data[('A_exact_nohuman','K0')]['meta']['commit'],source_sha256=list(source),pdf_exit_code=build.returncode,root=str(a.root.resolve()),raw_account_sampling='full 0.25 ms, float64; control arrays full 1 ms',figures=[f[0] for f in figures]))
    print(json.dumps(dict(audit=summary['status'],runs=len(data),completed=completed,pdf_exit_code=build.returncode)))
    return 0 if summary['status']=='PASS' and build.returncode==0 else 2


if __name__=='__main__':raise SystemExit(main())
