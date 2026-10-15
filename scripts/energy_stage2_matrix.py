"""Schedule independent fresh-process physical trials; no agent delegation."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kuka_table_sim.energy_experiments import development_cases


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',required=True,type=Path);a.add_argument('--heldout',required=True,type=Path)
    a.add_argument('--config',required=True,type=Path);a.add_argument('--jobs',type=int,default=4);args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    names=list(development_cases())+[c['name'] for c in json.loads(args.heldout.read_text())['cases']]
    # Start long primary/regression jobs first; untouched cases retain independent
    # fresh processes and fixed settings, irrespective of scheduling order.
    names=sorted(names,key=lambda n:0 if n in ('main','historical') else 1 if n.startswith('new-') else 2)
    def run(name):
        command=[sys.executable,'-m','kuka_table_sim.energy_experiments','matrix','--cases',name,'--out',str(args.out/name),
                 '--heldout',str(args.heldout),'--config',str(args.config)]
        start=time.perf_counter()
        with (args.out/(name+'.stdout.txt')).open('x') as f:
            p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        result=dict(case=name,returncode=p.returncode,wall_seconds=time.perf_counter()-start,command=command)
        print(json.dumps(result),flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:results=list(pool.map(run,names))
    with (args.out/'processes.json').open('x') as f:json.dump(dict(jobs=args.jobs,runs=results),f,indent=2)
    raise SystemExit(1 if any(r['returncode'] not in (0,2) for r in results) else 0)


if __name__=='__main__':main()
