"""Single entry point: preserved executed logs -> metrics, figures, editable PDF.

Simulation generation is separate: scripts/poc_figures_simulate.py.
Output must be a clean NEW directory. No controller or simulation is imported.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import platform
import importlib.metadata
import numpy as np
from poc_figures_data import load_case,compute,validate,sha,write_json
from poc_figures_plots import Figures
from poc_figures_crosswalk import write_crosswalk


ROOT=Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def build(a):
    a.out.mkdir(parents=True,exist_ok=False)
    cat=json.loads((a.inputs/'catalog.json').read_text());data={};metrics={};validation={}
    for c in cat['cases']:
        key=c['key'];path=a.inputs/c['folder'];d,p,e=load_case(path)
        data[key]=(d,p,e,path);metrics[key]=compute(d,p,e)
        validation[key]=validate(d,p)
        if validation[key]['status']!='PASS':raise ValueError((key,validation[key]))
    # Matched common windows use only samples executed by every policy, while
    # the complete-trace metric table keeps each real outcome and termination.
    common={};end=min(metrics['main__'+v]['completed_s'] for v in ('UF','A','B','C','D'))
    for v in ('UF','A','B','C','D'):
        key='main__'+v;d,p,e,_=data[key];common[key]=compute(d,p,e,window_end=end)
    # Independent raw-data audit is an input artifact, not the implementation
    # under test. Only overlapping exact definitions are compared numerically.
    audit=json.loads((a.inputs/'matrix_audit.json').read_text()) if (a.inputs/'matrix_audit.json').exists() else None
    discrepancies=[]
    compare_keys=('net_outgoing_J','gross_outgoing_J','min_substep_energy_J','force_mae_N',
                  'force_max_error_N','measured_progress_m','reference_progress_m','contact_absence_s')
    for key,r in metrics.items():
        old=data[key][1]['archived_metrics']
        for k in compare_keys:
            if k in old and abs(r[k]-old[k])>1e-8:discrepancies.append(dict(case=key,metric=k,computed=r[k],archived=old[k]))
    independent_comparisons=[]
    audit_records=audit['cases'][:] if audit is not None else []
    if (a.inputs/'fresh_audit.json').exists():
        audit_records+=json.loads((a.inputs/'fresh_audit.json').read_text())['cases']
    if audit_records:
        for record in audit_records:
            candidates=[k for k in data if data[k][1]['metadata']['case']['name']==record['case'] and data[k][1]['metadata']['variant']==record['variant']
                        and str(Path(record['inputs']['guard.npz']['path']).parent)==data[k][1]['original_directory']]
            if not candidates:continue
            if len(candidates)!=1:raise ValueError('ambiguous independent raw case')
            key=candidates[0]
            r=metrics[key];other=record['metrics']
            for k in compare_keys+('force_rmse_N','tangent_rmse_m_s','peak_outgoing_power_W','task_work_in_J'):
                if k in other:
                    defect=abs(r[k]-other[k])
                    independent_comparisons.append(dict(case=key,metric=k,absolute_difference=defect,tolerance=1e-8))
                    if defect>1e-8:discrepancies.append(dict(case=key,metric=k,computed=r[k],independent=other[k]))
            for k in ('qp_solves','qp_successful_solves','qp_failed_solver_calls','qp_fallbacks','qp_status_counts',
                      'reserve_candidate_rejections','qp_active_rows','qp_active_row_count'):
                if k in other:r[k]=other[k]
    if discrepancies:raise ValueError(('independent archived-metric mismatch',discrepancies))
    # Exact C=D physics/commands are checked independently over all13 cases.
    # This local selected-trace check also checks undefined reference NaN bytes.
    parity={}
    c,d=data['main__C'][0],data['main__D'][0]
    for k in c:
        if k=='tick_seconds':continue
        parity[k]=(c[k].dtype==d[k].dtype and c[k].shape==d[k].shape and c[k].tobytes()==d[k].tobytes())
    if not all(parity.values()):raise ValueError('C/D parity mismatch')
    # Original (UF) time-indexed reference debt is diagnostic and only defined
    # within the shared actual control window; no held-last-sample extrapolation.
    from poc_figures_data import ordered_progress
    for name in ('main','historical'):
        u=data[name+'__UF'][0];ur,up,_=ordered_progress(u['x'][:,0],u['strokes'],u['s_des'])
        for key in [k for k in data if k.startswith(name+'__')]:
            x=data[key][0];xr,xp,_=ordered_progress(x['x'][:,0],x['strokes'],x['s_des'])
            n=min(len(xr),len(ur));metrics[key]['peak_original_reference_debt_m']=float(np.max(np.abs(ur[:n]-xr[:n])))
            metrics[key]['terminal_measured_progress_debt_to_UF_m']=float(up[n-1]-xp[n-1])
            metrics[key]['progress_ratio_to_UF_full_run']=float(xp[-1]/up[-1])
    # Preserve independently audited activity totals in output without treating
    # unlogged rejected candidates as zero computation.
    write_json(a.out/'metrics.json',metrics);write_json(a.out/'common_window_metrics.json',common)
    fields=sorted({k for r in metrics.values() for k,v in r.items() if not isinstance(v,(dict,list))})
    with (a.out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['key']+fields);w.writeheader()
        for key,r in metrics.items():w.writerow(dict(key=key,**{k:r.get(k) for k in fields}))
    write_json(a.out/'data_validation.json',dict(status='PASS',cases=validation,
                metric_comparisons=independent_comparisons,discrepancies=discrepancies,
                selected_C_D_bytewise_parity=parity))
    write_crosswalk(a.out)
    Figures(data,metrics,a.out).make()
    reviews=[]
    if a.reviews and a.reviews.exists():
        shutil.copytree(a.reviews,a.out/'reviews')
        for p in sorted(a.reviews.glob('round_*.json')):reviews.append(json.loads(p.read_text()))
    review_summary=dict(completed_rounds=len(reviews),rounds=reviews,
                        status='VERIFIED' if len(reviews)==4 and all(r['verdict']=='PASS' for r in reviews) else 'IN REVIEW')
    write_json(a.out/'review_summary.json',review_summary)
    command='PYTHONDONTWRITEBYTECODE=1 /Users/carl/code/contactSimulation/.venv/bin/python scripts/kuka_poc_figures.py --inputs '+str(a.inputs)+' --out NEW_OUTPUT --reviews '+str(a.reviews)
    context=dict(source_commit=cat['source_stage2_head'],current_commit=git('rev-parse','HEAD'),
        branch=git('branch','--show-current'),build_command=command,review_rounds=reviews,
        matrix_counts=audit,common_window_metrics=common,common_window_end_s=end,
        equal_progress=dict(main_extra_s=cat['extensions']['main']['extra_physical_time_s'],
                            historical_extra_s=cat['extensions']['historical']['extra_physical_time_s']),
        output_label=str(a.out),help_complete='help-then-block__D' in metrics)
    from poc_figures_report import write_report
    write_report(a.out,metrics,context)
    result=subprocess.run(['tectonic','--keep-logs','--outdir',str(a.out),str(a.out/'report.tex')],capture_output=True,text=True)
    (a.out/'pdf_build.stdout.txt').write_text(result.stdout)
    (a.out/'pdf_build.stderr.txt').write_text(result.stderr)
    if result.returncode:raise RuntimeError('PDF build exit '+str(result.returncode)+': '+result.stderr[-3000:])
    pages=a.out/'rendered_pages';pages.mkdir()
    render_command=['pdftoppm','-r','110','-png',str(a.out/'report.pdf'),str(pages/'page')]
    render=subprocess.run(render_command,capture_output=True,text=True)
    (a.out/'pdf_render.stderr.txt').write_text(render.stderr)
    if render.returncode:raise RuntimeError('PDF rendering failed: '+render.stderr)
    reproduction='''# Reproduce the results-only build\n\nRun from the repository root with the compatible Python environment and tectonic installed.\nInputs are preserved full-resolution executed signals, config/metadata, events, and independently audited all-matrix evidence. No simulation is run by the build. Output must not exist.\n\n```sh\n'''+command+'\n```\n\nNew simulation commands and frozen profiles are in `../support/new_runs_config.json` and `scripts/poc_figures_simulate.py --help`.\n\nAll power curves use actual point velocity; work uses signed physics increments. No missing prediction is replaced by zero. Reserve is an unexecuted forecast.\n'
    (a.out/'reproduction.md').write_text(reproduction)
    code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scripts').glob('*poc_figures*.py'))}
    inputs={str(p.relative_to(a.inputs)):sha(p) for p in sorted(a.inputs.rglob('*')) if p.is_file()}
    versions={name:importlib.metadata.version(name) for name in ('numpy','matplotlib','mujoco','osqp','scipy','pytest')}
    manifest=dict(source_stage2_head=cat['source_stage2_head'],difference_from_prompt_reference='none',
        difference_scope='Resolved starting synchronized Stage2 head relative to user reference; this analysis branch adds files',
        report_source_commit=git('rev-parse','HEAD'),branch=context['branch'],analysis_code_hashes=code,
        inputs_root=str(a.inputs),inputs=inputs,versions=dict(versions,python=platform.python_version()),
        platform=platform.platform(),command=command,analysis=cat['analysis'],
        pdf_command=['tectonic','--keep-logs','--outdir',str(a.out),str(a.out/'report.tex')],pdf_build_exit=result.returncode,
        pdf_render_command=render_command,pdf_render_exit=render.returncode,rendered_pages=len(list(pages.glob('*.png'))),
        source_boundaries='Frozen numerical/controller package unchanged; new scripts only; physical/control dt .00025/.001 s',
        figure_curves='curve_manifest.json',reviews='review_summary.json',
        outputs={str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()})
    write_json(a.out/'manifest.json',manifest)
    print(json.dumps(dict(status='BUILT',cases=len(data),figures=len(list((a.out/'figures').glob('*.pdf'))),
                         out=str(a.out),review_status=review_summary['status'])))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--reviews',type=Path)
    build(ap.parse_args())
