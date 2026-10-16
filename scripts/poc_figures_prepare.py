"""Preserve full-resolution executed figure inputs into a NEW portable bundle."""
import argparse
import json
from pathlib import Path
import shutil
from poc_figures_data import archive_case,write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage2-runs',required=True,type=Path)
    ap.add_argument('--new-runs',type=Path)
    ap.add_argument('--support',required=True,type=Path)
    ap.add_argument('--matrix-audit',type=Path)
    ap.add_argument('--reproduction-proof',type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    selected={'main':['UF','A','B','C','D'],'historical':['UF','A','D'],
              'stress':['D'],'nohuman-zero':['D'],'nohuman-ample':['D']}
    catalog=[]
    for name,variants in selected.items():
        for v in variants:
            root=a.stage2_runs/'final_matrix_01'/name/name/v
            key=name+'__'+v
            p=archive_case(root,a.out/key)
            catalog.append(dict(key=key,folder=key,source_kind=p['source_kind']))
    if a.new_runs:
        for folder,key in [('help_then_block_01','help-then-block__D'),
                           ('help_then_block_02','help-then-block-repeat__D')]:
            root=a.new_runs/folder
            p=archive_case(root,a.out/key,'fresh continuous help-then-block executed simulation')
            catalog.append(dict(key=key,folder=key,source_kind=p['source_kind']))
        # Preserve original new-run logs as well as the named signal adapters.
        # This supports independent kinematic/state audits on another checkout.
        newraw=a.out/'new_run_raw';newraw.mkdir()
        for name in ('help_then_block_01','help_then_block_02','fresh_nohuman_D_01','fresh_stress_D_01'):
            source=a.new_runs/name
            shutil.copytree(source,newraw/name)
            for suffix in ('.process.json','.stdout.txt'):
                f=a.new_runs/(name+suffix)
                if f.exists():shutil.copy2(f,newraw/f.name)
    # Preserve independent all-matrix raw recomputation and freeze/replay proof.
    for name in ('matrix_audit.json','new_runs_config.json','new_runs_reuse.json','new_runs_reproduction.json'):
        if name=='matrix_audit.json' and a.matrix_audit:continue
        if name=='new_runs_reproduction.json' and a.reproduction_proof:continue
        if (a.support/name).exists():shutil.copy2(a.support/name,a.out/name)
    if a.matrix_audit:
        # Canonical name in the NEW bundle; prior support/audit artifacts stay intact.
        shutil.copy2(a.matrix_audit,a.out/'matrix_audit.json')
    if a.reproduction_proof:
        json.loads(a.reproduction_proof.read_text()) # reject a truncated prior attempt
        shutil.copy2(a.reproduction_proof,a.out/'new_runs_reproduction.json')
    if (a.support/'round03_fresh_audit.json').exists():
        shutil.copy2(a.support/'round03_fresh_audit.json',a.out/'fresh_audit.json')
    extension={}
    for name in ('main','historical'):
        root=a.stage2_runs/('extension_'+name+'_D_01')
        extension[name]=json.loads((root/'extension.json').read_text())
        # Extension is actual continuation; preserve its raw segment and metadata.
        dest=a.out/('extension_'+name+'_D_01');dest.mkdir()
        for filename in ('extension.json','extension_log.npz','guard.npz'):
            shutil.copy2(root/filename,dest/filename)
    write_json(a.out/'catalog.json',dict(cases=catalog,extensions=extension,
        source_stage2_head='41359f20430c930eec5afaebffa2cfbc482f07da',
        analysis=dict(task_start_s=8.71,nominal_end_s=18.,original_force_target_N=10.,
            recovery_force_band_N=.5,recovery_alpha=.95,recovery_duration_s=.5,
            common_window_policy='[8.71, earliest actual executed end) within each matched case',
            archived_metric_tolerance=1e-8,independent_metric_tolerance=1e-8),
        note='No controller imports or modifications. Every retained executed trace is complete.' ))
    print(json.dumps(dict(cases=len(catalog),out=str(a.out))))


if __name__=='__main__':main()
