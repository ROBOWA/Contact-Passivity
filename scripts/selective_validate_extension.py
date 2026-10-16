"""Verify that longer recordings preserve the original experiment prefix."""
import argparse
import json
from pathlib import Path
import numpy as np


def validate(original,extended):
    oldmeta=json.loads((original/'metadata.json').read_text());newmeta=json.loads((extended/'metadata.json').read_text())
    metrics=json.loads((extended/'metrics.json').read_text());checks={}
    with np.load(original/'baseline_log.npz') as a,np.load(extended/'baseline_log.npz') as b:
        for key in ('phys__t','phys__q','phys__qd','phys__tau_applied','phys__F_task','ctl__t','ctl__x_d'):
            x,y=a[key],b[key];checks[key]=bool(len(y)>=len(x) and x.dtype==y.dtype and x.tobytes()==y[:len(x)].tobytes())
        t=b['phys__t'];after=t>=18.-1e-10
        x=b['phys__tcp'][:,0];v=b['phys__tcp_v'][:,0]
        tail=dict(samples=int(np.sum(after)),x_span_m=float(np.ptp(x[after])) if np.any(after) else 0.,
                  moves_positive_x=bool(np.any(v[after]>.001)),moves_negative_x=bool(np.any(v[after]<-.001)))
    with np.load(original/'selective.npz') as a,np.load(extended/'selective.npz') as b:
        for key in ('phys__t','phys__power','phys__energy','phys__force','ctl__alpha','ctl__tau_exec'):
            x,y=a[key],b[key];checks[key]=bool(len(y)>=len(x) and x.dtype==y.dtype and x.tobytes()==y[:len(x)].tobytes())
    unchanged=all(oldmeta[k]==newmeta[k] for k in ('case','config','source_sha256','asset_hashes'))
    oldmetrics=json.loads((original/'metrics.json').read_text())
    outcome=(metrics['completion']=='COMPLETE' and abs(metrics['completed_s']-60)<1e-8 and tail['x_span_m']>.09 and tail['moves_positive_x'] and tail['moves_negative_x']) if oldmetrics['completion']=='COMPLETE' else (metrics['completion']=='STOP' and metrics['completed_s']==oldmetrics['completed_s'])
    return dict(status='PASS' if all(checks.values()) and unchanged and outcome else 'FAIL',prefix_checks=checks,
                unchanged_interaction_controller_model=unchanged,outcome_matches=outcome,
                completed_s=metrics['completed_s'],completion=metrics['completion'],post_18s_sliding=tail,
                source_sha256=newmeta['source_sha256'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original',type=Path,required=True);ap.add_argument('--extended',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--modes',default='K0,K1,K2,K3,K4');a=ap.parse_args()
    result={mode:validate(a.original/mode,a.extended/mode) for mode in a.modes.split(',')}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:dict(status=v['status'],completion=v['completion'],completed_s=v['completed_s']) for k,v in result.items()}))
    return 0 if all(v['status']=='PASS' for v in result.values()) else 2


if __name__=='__main__':raise SystemExit(main())
