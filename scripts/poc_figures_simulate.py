#!/usr/bin/env python3
"""Generate opt-in PoC evidence without changing frozen controller modules.

Freeze first, then run each named run in a distinct output directory:
  python scripts/poc_figures_simulate.py freeze --support SUPPORT
  python scripts/poc_figures_simulate.py run --config SUPPORT/new_runs_config.json --out NEW --name help_then_block_01
The run command launches a real child process and records its actual exit code.
Exit 2 denotes the existing explicit controller stop, never a passing completion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import osqp
import scipy

from kuka_table_sim.energy_experiments import metrics, development_cases, run_case
from kuka_table_sim.energy_guard import EnergyGuard
from kuka_table_sim.energy_prediction import GuardConfig
from kuka_table_sim.energy_state import HybridSnapshot, controller_bytes
from kuka_table_sim.joint_qp import FilterFailure
from kuka_table_sim.oracle_experiments import source_hash
from kuka_table_sim.oracle_physics import HumanPulse
from kuka_table_sim.recorder import git_info, _json_default
from kuka_table_sim.scenarios import make_force_sim


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as handle:
        json.dump(obj, handle, indent=2, default=_json_default, allow_nan=False)
        handle.write('\n')


def versions():
    return dict(python=platform.python_version(), mujoco=mujoco.__version__,
                numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__)


def freeze(support):
    """No evaluation occurs before this exclusive, immutable config write."""
    config = dict(schema_version=1, frozen_before_evaluation=True,
                  source=git_info(), numerical_source_sha256=source_hash(),
                  numerical_source_commit='81dffa64109752a9a027975e7b831daa4ff91e86',
                  adapter_sha256=sha(__file__), versions=versions(), seed=0,
                  seed_semantics='deterministic fixtures; no stochastic inputs',
                  duration_s=18., variant='D', budget_J=.12, body='link5',
                  local_m=[0., .04, .12], guard=asdict(GuardConfig()),
                  forecast='only current measured WORLD force passed to predictor; held through forecasts',
                  ledger='uncapped E_H=E_R=.12+sum(p_H*h); no reset across pulses',
                  profile=[
                      dict(name='helping', start=9.5, ramp=.2, hold=.2, magnitude=5., direction=[1., 0., 0.]),
                      dict(name='blocking', start=10.3, ramp=.2, hold=1., magnitude=5., direction=[-1., 0., 0.])],
                  analysis_windows_s=dict(task=[8.71, 18.], helping=[9.5, 10.1],
                                          between=[10.1, 10.3], blocking=[10.3, 11.7],
                                          post_release=[11.7, 18.]),
                  allowed_runs=['help_then_block_01', 'help_then_block_02',
                                'fresh_nohuman_D_01', 'fresh_stress_D_01', 'fresh_main_D_01'])
    path = Path(support) / 'new_runs_config.json'
    save(path, config)
    print(json.dumps(dict(config=str(path), sha256=sha(path))))


class FrozenPulseSequence:
    """Scenario input only: sum existing HumanPulse.force; no control changes."""
    def __init__(self, definitions):
        self.pulses = tuple(HumanPulse(**{k: v for k, v in d.items() if k != 'name'})
                            for d in definitions)

    def force(self, t):
        return sum((p.force(t) for p in self.pulses), start=np.zeros(3))


def continuous(config_path, out):
    frozen = json.loads(Path(config_path).read_text())
    if source_hash() != frozen['numerical_source_sha256']:
        raise ValueError('numerical source changed after scenario freeze')
    if versions() != frozen['versions']:
        raise ValueError('environment changed after scenario freeze')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    before = source_hash()
    cfg = GuardConfig(**frozen['guard'])
    duration = frozen['duration_s']
    profile = FrozenPulseSequence(frozen['profile'])
    sim = make_force_sim(duration=duration)
    guard = EnergyGuard(sim, frozen['variant'], profile, frozen['body'], frozen['local_m'],
                        (frozen['budget_J'],) * 2, config=cfg)
    # Preserve the exact initial plant/controller/account state before any step.
    HybridSnapshot.capture(sim, guard.energy, guard.previous, guard.base).save(out / 'initial_state.npz')
    ticks, failed = [], None
    started = time.perf_counter()
    try:
        while sim.data.time < duration - .000125:
            tick_started = time.perf_counter()
            sim.advance(4)
            ticks.append(time.perf_counter() - tick_started)
    except (FilterFailure, ValueError) as exc:
        failed = dict(t=float(sim.data.time), reason=getattr(exc, 'reason', str(exc)),
                      diagnostics=getattr(exc, 'diagnostics', {}),
                      seconds_failed_tick=time.perf_counter() - tick_started)
        guard.save_failure(out, sim)
        save(out / 'failure.json', failed)
    wall = time.perf_counter() - started
    HybridSnapshot.capture(sim, guard.energy, guard.previous, guard.base).save(
        out / 'final_state.npz',
        counterfactual_blob=np.frombuffer(controller_bytes(guard.counterfactual), dtype=np.uint8))
    log = sim.recorder.finish()
    ctl, phys = guard.arrays()
    log.save(out / 'baseline_log.npz')
    np.savez_compressed(out / 'guard.npz', **{'ctl__' + k: v for k, v in ctl.items()},
                        **{'phys__' + k: v for k, v in phys.items()}, tick_seconds=np.asarray(ticks))
    with gzip.open(out / 'events.json.gz', 'wt') as handle:
        json.dump(guard.events, handle, default=_json_default, allow_nan=True)
    # Metadata's single-envelope fields only identify outer interaction bounds.
    # The separately frozen two-pulse profile is the sole executed force input.
    first, last = frozen['profile'][0], frozen['profile'][-1]
    release = last['start'] + 2 * last['ramp'] + last['hold']
    case = dict(name='help-then-block', magnitude=5., start=first['start'], ramp=first['ramp'],
                hold=release - first['start'] - 2 * first['ramp'], direction=[-1., 0., 0.],
                budget=frozen['budget_J'], rate_factor=1., body=frozen['body'], local=frozen['local_m'],
                profile=frozen['profile'], envelope_only=True)
    result = metrics(log, ctl, phys, case, frozen['variant'], wall, ticks, failed, cfg)
    phase_work = {}
    for name, (start, end) in frozen['analysis_windows_s'].items():
        selected = (phys['t'] >= start - 1e-10) & (phys['t'] < end - 1e-10)
        work = phys['work'][selected]
        phase_work[name] = dict(incoming_signed_J=float(np.sum(work)),
                                outgoing_net_J=float(-np.sum(work)),
                                outgoing_gross_J=float(np.maximum(-work, 0).sum()),
                                positive_power_samples=int(np.sum(phys['power'][selected] > 0)),
                                negative_power_samples=int(np.sum(phys['power'][selected] < 0)))
    result['phase_work'] = phase_work
    save(out / 'metrics.json', result)
    save(out / 'metadata.json', dict(**git_info(), source_sha256=before,
         source_end_sha256=source_hash(), case=case, variant=frozen['variant'],
         config=asdict(cfg), duration=duration, versions=versions(), platform=platform.platform(),
         asset_hashes=log.meta['model_file_sha256'], frozen_config_sha256=sha(config_path),
         adapter_sha256=sha(__file__), profile=frozen['profile'], forecast=frozen['forecast'],
         continuous_state=True, state_resets_after_initialization=0, seed=frozen['seed'],
         actual_initial_state_sha256=sha(out / 'initial_state.npz'),
         runtime_wall_s=wall, sampling='pre-integration stage k, post-step signed left-rectangle p_H*h account',
         ledger=frozen['ledger']))
    print(json.dumps(dict(completion=result['completion'], final_s=float(sim.data.time),
                         wall_s=wall, phase_work=phase_work, failure=failed), default=_json_default), flush=True)
    return 0 if result['completion'] == 'PASS' else 2


def run(config_path, out, name):
    frozen = json.loads(Path(config_path).read_text())
    if name not in frozen['allowed_runs']:
        raise ValueError('run name not frozen')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), '_case', '--config', str(Path(config_path).resolve()),
               '--out', str(out.resolve()), '--name', name]
    stdout_path = out.parent / (name + '.stdout.txt')
    started = time.perf_counter()
    with stdout_path.open('x') as handle:
        process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    status = dict(command=command, cwd=str(ROOT), returncode=process.returncode,
                  expected_stop_exit=2, wall_seconds=time.perf_counter() - started,
                  output_directory=str(out.resolve()), config_sha256=sha(config_path),
                  stdout=str(stdout_path.resolve()), stdout_sha256=sha(stdout_path),
                  actual_child_exit=True, versions=versions())
    save(out.parent / (name + '.process.json'), status)
    print(json.dumps(status), flush=True)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'run', '_case'])
    parser.add_argument('--support', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.support)
        return 0
    if args.command == 'run':
        return run(args.config, args.out, args.name)
    frozen = json.loads(args.config.read_text())
    if args.name.startswith('help_then_block_'):
        return continuous(args.config, args.out)
    name = {'fresh_nohuman_D_01': 'nohuman-ample', 'fresh_stress_D_01': 'stress',
            'fresh_main_D_01': 'main'}[args.name]
    if source_hash() != frozen['numerical_source_sha256']:
        raise ValueError('numerical source changed')
    result = run_case(development_cases()[name], 'D', args.out, GuardConfig(**frozen['guard']))
    return 0 if result['completion'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
