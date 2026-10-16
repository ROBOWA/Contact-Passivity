"""Inventory or verify protected worktrees without modifying their contents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def inventory(root):
    root = Path(root).resolve()
    tracked = set(git(root, 'ls-files').splitlines())
    records = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '.venv', '__pycache__', '.pytest_cache'})
        for name in sorted(files):
            p = Path(directory) / name
            rel = str(p.relative_to(root))
            if name in {'.git', '.DS_Store'} or p.is_symlink():
                continue
            records.append(dict(path=rel, size=p.stat().st_size, sha256=sha(p), tracked=rel in tracked))
    return dict(root=str(root), head=git(root, 'rev-parse', 'HEAD'),
                branch=git(root, 'branch', '--show-current'), status=git(root, 'status', '--porcelain'),
                refs=git(root, 'for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads', 'refs/remotes').splitlines(),
                files=records, excluded=['.git', '.venv', '__pycache__', '.pytest_cache', '.DS_Store', 'symlinks'])


def verify(manifest,mutable_refs=()):
    errors = []
    for tree in manifest['worktrees']:
        root = Path(tree['root'])
        for entry in tree['files']:
            p = root / entry['path']
            if not p.is_file() or p.stat().st_size != entry['size'] or sha(p) != entry['sha256']:
                errors.append(str(p))
        for key, args in [('head', ('rev-parse', 'HEAD')), ('branch', ('branch', '--show-current')),
                          ('status', ('status', '--porcelain'))]:
            if git(root, *args) != tree[key]:
                errors.append(str(root) + ':' + key)
        for ref in tree['refs']:
            name, old = ref.split()
            if name in mutable_refs:continue
            if git(root, 'rev-parse', name) != old:
                errors.append(name)
    return dict(status='PASS' if not errors else 'FAIL', errors=errors,
                worktrees=len(manifest['worktrees']),mutable_refs=list(mutable_refs),
                files=sum(len(x['files']) for x in manifest['worktrees']),
                bytes=sum(f['size'] for x in manifest['worktrees'] for f in x['files']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roots', nargs='+', type=Path)
    ap.add_argument('--verify', type=Path)
    ap.add_argument('--mutable-ref',action='append',default=[],help='Only the newly created, authorized output branch may change')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    if a.verify:
        result = verify(json.loads(a.verify.read_text()),a.mutable_ref)
    else:
        result = dict(worktrees=[inventory(p) for p in a.roots])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result if a.verify else {x['root']: len(x['files']) for x in result['worktrees']}))
    raise SystemExit(0 if not a.verify or result['status'] == 'PASS' else 2)
