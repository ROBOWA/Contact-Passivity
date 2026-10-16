#!/usr/bin/env python3
"""Independently compare reused and fresh PoC raw runs, excluding timing only.

python scripts/poc_figures_reproduction.py --old-root OLD_STAGE2_RESULTS \
  --new-root NEW_POC_RESULTS --config FROZEN_JSON --out NEW_SUPPORT_JSON
No plotting helpers are imported. Numeric arrays, snapshots and configurations
are checked directly. A stopped stress case is expected to remain exit 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import osqp
import scipy
from kuka_table_sim.oracle_experiments import source_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files(path):
    return {p.name: dict(sha256=digest(p), bytes=p.stat().st_size)
            for p in sorted(Path(path).iterdir()) if p.is_file()}


def compare(left, right):
    result = {}
    names = ['guard.npz', 'baseline_log.npz', 'final_state.npz']
    if (Path(left) / 'initial_state.npz').exists() and (Path(right) / 'initial_state.npz').exists():
        names.append('initial_state.npz')
    for name in names:
        with np.load(Path(left) / name, allow_pickle=False) as a, np.load(Path(right) / name, allow_pickle=False) as b:
            excluded = sorted(k for k in set(a.files) | set(b.files)
                              if k == 'meta_json' or k == 'tick_seconds' or k.startswith('ctl__seconds_'))
            keys = sorted(set(a.files) - set(excluded))
            assert keys == sorted(set(b.files) - set(excluded)), (name, 'different fields')
            bad = []
            for key in keys:
                av, bv = a[key], b[key]
                equal = np.array_equal(av, bv, equal_nan=True) if av.dtype.kind in 'fc' else np.array_equal(av, bv)
                if not equal:
                    bad.append(key)
            result[name] = dict(compared_fields=keys, compared_count=len(keys),
                                excluded_fields=excluded, nonidentical_fields=bad,
                                exact_equal=not bad)
            assert not bad, (name, bad)
    return result


def check_metadata(directory, frozen, variants=True):
    metadata = json.loads((Path(directory) / 'metadata.json').read_text())
    assert metadata['source_sha256'] == frozen['numerical_source_sha256']
    assert metadata['source_end_sha256'] == frozen['numerical_source_sha256']
    assert metadata['versions'] == frozen['versions']
    assert metadata['config'] == frozen['guard']
    if variants:
        assert metadata['variant'] == 'D'
    return dict(numerical_source_equal=True, environment_versions_equal=True,
                guard_config_equal=True, metadata_sha256=digest(Path(directory) / 'metadata.json'))


def matching_inputs(left, right):
    a, b = [json.loads((Path(p) / 'metadata.json').read_text()) for p in [left, right]]
    fields = ['case', 'variant', 'config', 'duration', 'versions', 'asset_hashes',
              'source_sha256', 'source_end_sha256']
    assert all(a[k] == b[k] for k in fields), 'different simulation inputs'
    return dict(exactly_equal_fields=fields, assets_equal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', required=True, type=Path)
    parser.add_argument('--new-root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.config.read_text())
    assert source_hash() == frozen['numerical_source_sha256']
    current_versions = dict(python=platform.python_version(), mujoco=mujoco.__version__,
                            numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__)
    assert current_versions == frozen['versions']
    evidence = dict(schema_version=1, status='PASS', command=sys.argv,
                    numerical_source_sha256=source_hash(), config_sha256=digest(args.config),
                    environment_versions=current_versions,
                    method='direct np.array_equal of all non-timing arrays and complete final-state fields; no plotting helper',
                    tolerance='exact equality; equal-position NaNs allowed only for recorded optional diagnostics',
                    reused={}, fresh={})
    for case, name in [('main', 'repeat_main_D_01'), ('stress', 'repeat_stress_D_01')]:
        archived = args.old_root / 'final_matrix_01' / case / case / 'D'
        repeated = args.old_root / name
        m1, m2 = (json.loads((p / 'metrics.json').read_text()) for p in [archived, repeated])
        assert m1['failure'] is None if case == 'main' else m1['failure']['reason'] == 'no_verified_reserve_candidate'
        assert (m1['failure'] or {}).get('reason') == (m2['failure'] or {}).get('reason')
        evidence['reused'][case] = dict(archived=str(archived.resolve()), repeat=str(repeated.resolve()),
              archived_metadata=check_metadata(archived, frozen), repeat_metadata=check_metadata(repeated, frozen),
              input_equality=matching_inputs(archived, repeated),
              comparison=compare(archived, repeated), repeat_files=files(repeated),
              archived_files=files(archived),
              completion=m2['completion'], completed_s=m2['completed_s'], original_wall_s=m2['wall_seconds'],
              explanation='Previously checked fresh exact-source/config/environment repeat reused as allowed; not a new simulation in this build.')
    for case, name, expected_exit in [('nohuman-ample', 'fresh_nohuman_D_01', 0), ('stress', 'fresh_stress_D_01', 2)]:
        archived = args.old_root / 'final_matrix_01' / case / case / 'D'
        fresh = args.new_root / name
        process = json.loads((args.new_root / (name + '.process.json')).read_text())
        assert process['returncode'] == expected_exit
        assert process['actual_child_exit']
        result = json.loads((fresh / 'metrics.json').read_text())
        if expected_exit == 2:
            assert result['completion'] == 'FAIL'
            assert result['failure']['reason'] == 'no_verified_reserve_candidate'
            assert abs(result['completed_s'] - 9.533) < 1e-8
        else:
            assert result['completion'] == 'PASS'
            assert abs(result['completed_s'] - 18.) < 1e-8
        evidence['fresh'][case] = dict(directory=str(fresh.resolve()), metadata=check_metadata(fresh, frozen),
              input_equality=matching_inputs(archived, fresh),
              comparison=compare(archived, fresh), raw_files=files(fresh), process=process,
              completion=result['completion'], completed_s=result['completed_s'])
    first, second = (args.new_root / ('help_then_block_%02d' % n) for n in [1, 2])
    for p in [first, second]:
        check_metadata(p, frozen)
        m = json.loads((p / 'metadata.json').read_text())
        assert m['continuous_state'] and m['state_resets_after_initialization'] == 0
        assert m['profile'] == frozen['profile']
        assert m['frozen_config_sha256'] == digest(args.config)
        assert m['adapter_sha256'] == frozen['adapter_sha256']
    process = [json.loads((args.new_root / (p.name + '.process.json')).read_text()) for p in [first, second]]
    assert process[0]['returncode'] == process[1]['returncode']
    for d in process:
        assert d['actual_child_exit'] and d['returncode'] in (0, 2)
    m1, m2 = [json.loads((p / 'metrics.json').read_text()) for p in [first, second]]
    assert m1['phase_work'] == m2['phase_work']
    # Positive work under the first pulse and negative work under the second
    # assess the actual applied-port power, rather than merely force direction.
    assert m1['phase_work']['helping']['incoming_signed_J'] > 0
    assert m1['phase_work']['blocking']['incoming_signed_J'] < 0
    with np.load(first / 'guard.npz', allow_pickle=False) as z:
        t, w, energy = z['phys__t'], z['phys__work'], z['phys__energy']
        assert np.max(np.abs(energy[:, 0] - (frozen['budget_J'] + np.cumsum(w)))) < 1e-12
        assert np.array_equal(energy[:, 0], energy[:, 1])
        assert np.max(np.abs(np.diff(t) - .00025)) < 1e-10
        assert np.all(np.diff(t) > 0)
        complete = bool(len(t) == 72000 and abs(t[-1] + .00025 - 18.) < 1e-8)
        if process[0]['returncode'] == 0:
            assert complete
        field_summary = dict(physics_samples=len(t), monotonic=True, physics_dt_s=.00025,
                             full_requested_coverage=complete, ledger_identity_tolerance_J=1e-12,
                             equal_H_R=True, max_energy_J=float(np.max(energy[:, 0])))
    evidence['fresh']['help-then-block'] = dict(first=str(first.resolve()), second=str(second.resolve()),
          input_equality=matching_inputs(first, second),
          comparison=compare(first, second), raw_files_first=files(first), raw_files_second=files(second),
          process=process, completion=m1['completion'], phases=m1['phase_work'], sampling=field_summary,
          no_cap_implemented=True, no_reset=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(evidence, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(status=evidence['status'], output=str(args.out), sha256=digest(args.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
