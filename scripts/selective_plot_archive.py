"""Regenerate every published curve using only the tracked compact archive."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from scripts.selective_report import plot_case,plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);data={}
    for metric in json.loads((a.report/'metrics.json').read_text()):
        case,mode=metric['case'],metric['mode'];stem=f'{case}_{mode}'
        with np.load(a.report/'raw'/f'{stem}.npz') as z:
            p={k:z[k] for k in z.files if not k.startswith(('ctl_','baseline_'))}
            c={k[4:]:z[k] for k in z.files if k.startswith('ctl_')}
            b={k[9:]:z[k] for k in z.files if k.startswith('baseline_')}
        meta=json.loads((a.report/'raw'/f'{stem}_metadata.json').read_text())
        data[(case,mode)]=dict(p=p,c=c,b=b,m=metric,meta=meta)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    for f in json.loads((a.report/'figure_spec.json').read_text()):
        plot_case(data,f['case'],f['fields'],f['title'],a.out,f['window'],f['modes'],f['filename'])
    return 0


if __name__=='__main__':raise SystemExit(main())
