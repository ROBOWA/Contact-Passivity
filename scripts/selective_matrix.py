"""Independent processes, exclusive run paths, explicit expected stop exit codes."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--jobs',type=int,default=4)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cases=['A_exact_nohuman','B_mismatch_nohuman','C_blocking','D_combined','E_forearm_block','F_help_block']
    tasks=[(c,f'K{k}') for c in cases for k in range(5)]+[('S_starved',k) for k in ('K3','K4')]
    start=time();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def run(task):
        case,mode=task;out=a.out/case/mode;stdout=a.out/f'{case}_{mode}.stdout.txt'
        command=[sys.executable,'-m','kuka_table_sim.selective_experiments','case','--case',case,'--mode',mode,'--out',str(out)]
        with stdout.open('x') as fp:r=subprocess.run(command,stdout=fp,stderr=subprocess.STDOUT,env=env)
        result=dict(case=case,mode=mode,exit_code=r.returncode,command=command)
        if r.returncode in (0,2) and (out/'metrics.json').exists():
            result['metrics']=json.loads((out/'metrics.json').read_text())
        return result
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for result in pool.map(run,tasks):
            results.append(result)
            (a.out/'matrix.json').write_text(json.dumps(dict(results=results,wall_seconds=time()-start),indent=2)+'\n')
            m=result.get('metrics',{});print(result['case'],result['mode'],result['exit_code'],m.get('completed_s'),flush=True)
    errors=[r for r in results if 'metrics' not in r or r['exit_code'] not in (0,2)]
    return 1 if errors else (2 if any(r['exit_code']==2 for r in results) else 0)


if __name__=='__main__':raise SystemExit(main())
