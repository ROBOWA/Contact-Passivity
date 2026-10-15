"""Read-only verification of the protected original and independently copied backup."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def verify(manifest):
    m=json.loads(Path(manifest).read_text());original=Path(m['original']['cwd']);failures=[]
    counts={}
    for group in ('tracked','untracked_ignored'):
        counts[group]=0
        for item in m[group]:
            p=original/item['path']
            if not p.is_file() or p.stat().st_size!=item['size'] or digest(p)!=item['sha256']:failures.append(str(p))
            if 'backup' in item:
                p=Path(item['backup'])
                if not p.is_file() or p.stat().st_size!=item['size'] or digest(p)!=item['sha256']:failures.append(str(p))
            counts[group]+=1
    def git(*args):return subprocess.run(['git','-C',str(original),*args],capture_output=True,text=True,check=True).stdout.strip()
    if git('rev-parse','HEAD')!=m['original']['head']:failures.append('original HEAD')
    if git('status','--porcelain')!=m['original']['status']:failures.append('original worktree status')
    for entry in m['original']['refs']:
        ref,sha=entry.split()
        if ref.startswith(('refs/heads/','refs/remotes/')) and git('rev-parse',ref)!=sha:failures.append(ref)
    return dict(status='PASS' if not failures else 'FAIL',counts=counts,backup_bytes=m['backup_bytes'],failures=failures,
                original_head=git('rev-parse','HEAD'),original_status=git('status','--porcelain'))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--manifest',required=True,type=Path);a.add_argument('--out',required=True,type=Path);args=a.parse_args()
    result=verify(args.manifest)
    with args.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result));raise SystemExit(0 if result['status']=='PASS' else 2)
