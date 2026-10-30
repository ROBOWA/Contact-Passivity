"""Verify preserved research files and historical Git blobs after consolidation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'docs/consolidation/20261027/source_manifest.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def verify(manifest):
    errors = []
    objects = {}
    files = {}
    records = 0
    for source in manifest['sources']:
        tree = {}
        if 'head' in source:
            for entry in git('ls-tree', '-rz', '--full-tree', source['head']).split(b'\0'):
                if entry:
                    metadata, name = entry.split(b'\t', 1)
                    tree[name.decode()] = metadata.split()[2].decode()
        for item in source['files']:
            records += 1
            expected = (item['size'], item['sha256'])
            if item['stored_as'] == 'git':
                obj = item['git_blob']
                if tree.get(item['path']) != obj:
                    errors.append(f"Historical tree mismatch: {source['name']}/{item['path']}")
                mapping, key = objects, obj
            elif item['stored_as'] == 'file':
                path = (ROOT / item['destination']).resolve()
                if not path.is_relative_to(ROOT):
                    errors.append(f'File outside repository: {path}')
                    continue
                mapping, key = files, path
            else:
                errors.append(f"Unknown storage kind: {item['stored_as']}")
                continue
            if key in mapping and mapping[key] != expected:
                errors.append(f'Conflicting expected hashes: {key}')
            mapping[key] = expected

    for obj, expected in objects.items():
        try:
            content = git('cat-file', 'blob', obj)
            if (len(content), hashlib.sha256(content).hexdigest()) != expected:
                errors.append('Historical blob mismatch: ' + obj)
        except subprocess.CalledProcessError:
            errors.append('Missing historical blob: ' + obj)

    def check_file(entry):
        path, expected = entry
        try:
            if (path.stat().st_size, sha256(path)) != expected:
                return f'File mismatch: {path.relative_to(ROOT)}'
        except OSError as exc:
            return f'File unavailable: {path.relative_to(ROOT)}: {exc}'
        return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        errors.extend(error for error in pool.map(check_file, files.items()) if error)
    return dict(status='FAIL' if errors else 'PASS', source_records=records,
                unique_git_blobs=len(objects), unique_additional_files=len(files),
                source_bytes=sum(x['size'] for s in manifest['sources'] for x in s['files']),
                errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out', type=Path, help='Write a NEW verification result file')
    args = parser.parse_args()
    result = verify(json.loads(args.manifest.read_text()))
    output = json.dumps(result, indent=2) + '\n'
    if args.out:
        with args.out.open('x') as stream:
            stream.write(output)
    print(output, end='')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
