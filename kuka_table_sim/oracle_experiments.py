"""Reproducible oracle scenarios, buffered logs and quantitative comparisons.

python -m kuka_table_sim.oracle_experiments case --name resistance --mode off --budget 1
python -m kuka_table_sim.oracle_experiments matrix --out results/oracle/final
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import mujoco
import numpy as np
import osqp
import scipy

from . import metrics as mx
from .joint_qp import FilterFailure, QPConfig
from .oracle import OracleFilter
from .oracle_physics import HumanPulse, integration_state
from .recorder import RunLog, _json_default, git_info
from .scenarios import make_force_sim


# Tolerances declared before final evaluations. Numerical replay limits have
# physical-unit independent tests; task limits are baseline engineering gates.
TOLERANCES = dict(replay_work_J=1e-11, replay_q_rad=1e-11, replay_qvel_rad_s=1e-10,
                  power_identity_W=1e-12, account_equality_J=0.,
                  energy_J=1e-10, torque_Nm=1e-7,
                  mean_force_N=.5, max_force_N=3., along_path_mm=15., cross_path_mm=3.,
                  orientation_deg=2., release_force_N=.5, release_along_mm=5.)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=_json_default, allow_nan=False) + "\n")


def source_hash():
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for p in sorted((root / "kuka_table_sim").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def percentiles(a):
    a = np.asarray(a)
    return dict(zip(("median", "p95", "p99", "max"), map(float, np.percentile(a, [50,95,99,100])))) if len(a) else None


def fixture(name):
    pulse = HumanPulse(start=9.5, ramp=.2, hold=1., magnitude=5.)
    body, local = "link5", (0., .04, .12)
    if name == "no-human":
        pulse = replace(pulse, magnitude=0.)
    elif name == "assist":
        pulse = replace(pulse, magnitude=-5.)
    elif name == "elbow":
        body, local = "link3", (.02, .04, .16)
    elif name == "varying":
        pulse = replace(pulse, varying=True)
    elif name == "held-out":
        pulse = replace(pulse, start=10.1, ramp=.15, hold=.8, magnitude=4.5)
    elif name != "resistance":
        raise ValueError(name)
    return pulse, body, local


def run_case(name, mode, budget, out, *, duration=18., config=None, label=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pulse, body, local = fixture(name)
    sim = make_force_sim(duration=duration)
    f = OracleFilter(sim, mode, pulse, body, local, (budget, budget), config=config)
    sim.oracle = f
    start = time.perf_counter()
    step_times = []
    failed = None
    try:
        for k in range(int(round(duration / sim.timing.control_period))):
            tick_start = time.perf_counter()
            sim.advance(sim.timing.control_interval)
            step_times.append(time.perf_counter() - tick_start)
    except FilterFailure as exc:
        failed = dict(reason=exc.reason, diagnostics=exc.diagnostics, t=float(sim.data.time))
    wall = time.perf_counter() - start
    log = sim.recorder.finish()
    ctl, phys = f.arrays()
    meta = dict(**git_info(), source_sha256=source_hash(), scenario=name, label=label,
                oracle=f.metadata(), tolerances=TOLERANCES, seed=0,
                versions=dict(mujoco=mujoco.__version__, numpy=np.__version__, osqp=osqp.__version__,
                              scipy=scipy.__version__, python=platform.python_version()),
                hardware=dict(platform=platform.platform(), machine=platform.machine(),
                              cpu=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                 capture_output=True, text=True).stdout.strip()),
                requested_duration=duration, completed_time=float(sim.data.time))
    if failed:
        np.savez_compressed(out / "failure_snapshot.npz", state=integration_state(sim.model, sim.data),
                            initial_tick_state=f.snapshot, energy=f.energy, previous=f.previous,
                            force=f.force, base_qfrc=f.base_qfrc)
        save_json(out / "failure.json", failed)
    save_json(out / "metadata.json", meta)
    save_json(out / "events.json", f.events)
    np.savez_compressed(out / "oracle.npz", **{f"ctl__{k}":v for k,v in ctl.items()},
                        **{f"phys__{k}":v for k,v in phys.items()}, tick_seconds=np.array(step_times))
    log.save(out / "baseline_log.npz")
    summary = summarize(log, ctl, phys, f, wall, step_times, failed)
    save_json(out / "metrics.json", summary)
    print(json.dumps(dict(case=label or name, mode=mode, budget=budget, **summary), default=_json_default), flush=True)
    return summary


def summarize(log, c, p, f, wall, step_times, failed):
    n = len(c.get("t", []))
    window = (8.71, float(log.phys["t"][-1]))
    if window[1] < window[0] or not n:
        return dict(status="FAIL", failure=failed, completed_time=window[1], wall_seconds=wall)
    force = mx.force_stats(log, window)
    steady = (log.phys["t"] >= window[0]) & (log.phys["t"] <= window[1])
    # The recovery FSM can lower/ramp f_d. Also report the ORIGINAL 10 N task
    # target so recovery does not make apparent tracking quality look better.
    fixed_error = 10. - log.phys["F_task"][steady,2]
    force["original_10N_mean_abs"] = float(np.mean(np.abs(fixed_error)))
    force["original_10N_max_abs"] = float(np.max(np.abs(fixed_error)))
    tracking = mx.tracking_stats(log, window)
    release = (max(f.pulse.end + 1., window[1] - 1.), window[1])
    recovery = None
    if release[1] > release[0]:
        recovery = dict(force=mx.force_stats(log, release), tracking=mx.tracking_stats(log, release),
                        integrals_max=dict(f=float(np.max(np.abs(log.ctl["f_int"]))),
                                           tangent=float(np.max(np.linalg.norm(log.ctl["tan_int"], axis=1))),
                                           rotation=float(np.max(np.linalg.norm(log.ctl["rot_int"], axis=1)))))
    prediction_error = c["predicted"] - c["actual"]
    finite = np.isfinite(prediction_error)
    affine_error = c["affine"] - c["predicted"]
    margin = c["energy_after"] - f.minimum
    intervention = c["intervention"] > 1e-7
    activity = dict(qp_ticks=int(np.sum(c["attempts"] > 0)),
                    solves=int(np.sum(c["attempts"])), retries=int(np.sum(np.maximum(c["attempts"] - 1, 0))),
                    intervention_ticks=int(np.sum(intervention)),
                    max_delta_Nm=float(np.max(c["intervention"])),
                    torque_limit_ticks=int(np.sum(np.any(np.abs(c["tau_exec"]) >= f.predictor.arm.torque_limit - 1e-6, axis=1))),
                    rate_limit_ticks=int(np.sum(np.any(np.abs(c["tau_exec"] - c["tau_previous"]) >= np.asarray(f.cfg.rate) * .001 - 1e-6, axis=1))),
                    energy_row_active_ticks=int(np.sum(intervention & (np.min(margin,axis=1) < 1e-6))))
    active_t = c["t"][intervention]
    activity["first_intervention_t"] = float(active_t[0]) if len(active_t) else None
    activity["last_intervention_t"] = float(active_t[-1]) if len(active_t) else None
    activity["failed_candidate_ticks"] = 1 if failed else 0
    activity["failed_attempts"] = len(failed.get("diagnostics",{}).get("attempts",[])) if failed else 0
    activity["total_attempts_including_failure"] = activity["solves"] + activity["failed_attempts"]
    activity["solver_recoveries"] = sum(a.get("status", "").startswith("active_set_recovery")
                                          for e in f.events for a in e.get("attempts", []))
    return dict(status="FAIL" if failed else "PASS", failure=failed, completed_time=float(f.predictor.model.opt.timestep * len(p["t"])),
                force=force, tracking=tracking, contact=mx.contact_absence(log, window),
                work_task_J=float(np.sum(c["task_work"])), work_human_J=float(np.sum(c["actual"][:,0])),
                work_residual_J=float(np.sum(c["actual"][:,1])), net_outgoing_J=-float(np.sum(c["actual"][:,0])),
                gross_outgoing_J=float(np.sum(c["gross"])), account_final_J=f.energy.tolist(),
                account_min_margin_J=float(np.min(margin)), largest_boundary_violation_J=max(0.,-float(np.min(margin))),
                observed_substep_min_margin_J=float(np.min(c["min_prefix"] - f.minimum)),
                power_identity_max_W=float(np.max(np.abs(p["power_identity"]))),
                account_equality_max_J=float(np.max(np.abs(c["energy_after"][:,0] - c["energy_after"][:,1]))),
                prediction_max_abs_J=float(np.max(np.abs(prediction_error[finite]))) if np.any(finite) else None,
                prediction_max_over_J=float(max(0., np.max(prediction_error[finite]))) if np.any(finite) else None,
                affine_max_over_J=float(max(0., np.nanmax(affine_error))) if np.any(np.isfinite(affine_error)) else None,
                human_point_speed_max_m_s=float(np.max(np.linalg.norm(p["velocity"], axis=1))),
                human_power_range_W=[float(np.min(p["power"])), float(np.max(p["power"]))],
                human_force_max_N=float(np.max(np.linalg.norm(p["force"],axis=1))),
                activity=activity, recovery=recovery, wall_seconds=wall,
                realtime_factor=float(log.phys["t"][-1] / wall), tick_seconds=percentiles(step_times),
                timing_seconds={k:percentiles(c[k]) for k in c if k.startswith("seconds_")},
                active_gradient_seconds=percentiles(c["seconds_gradient"][c["seconds_gradient"] > 0]),
                active_qp_seconds=percentiles(c["seconds_qp"][c["seconds_qp"] > 0]))


MATRIX = [
    ("no-human-off", "no-human", "off", 1.),
    ("no-human-shadow", "no-human", "shadow", 1.),
    ("no-human-active", "no-human", "active", 1.),
    ("ample-off", "resistance", "off", 1.),
    ("ample-active", "resistance", "active", 1.),
    ("tight-off", "resistance", "off", .12),
    ("tight-shadow", "resistance", "shadow", .12),
    ("tight-active", "resistance", "active", .12),
    ("assist", "assist", "active", .12),
    ("stress", "resistance", "active", .001),
    ("elbow", "elbow", "active", 1.),
    ("elbow-tight", "elbow", "active", .12),
    ("varying", "varying", "shadow", .12),
    ("held-out-off", "held-out", "off", .12),
    ("held-out-active", "held-out", "active", .12),
    ("task-cost", "resistance", "active", .12),
    ("lower-budget", "resistance", "active", .10),
    ("torque-only", "resistance", "active", .12),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    cp = sub.add_parser("case")
    cp.add_argument("--name", choices=["no-human", "resistance", "assist", "elbow", "varying", "held-out"], default="resistance")
    cp.add_argument("--mode", choices=["off", "shadow", "active"], default="shadow")
    cp.add_argument("--budget", type=float, default=.12)
    cp.add_argument("--duration", type=float, default=18.)
    cp.add_argument("--task-weight", type=float, default=1.)
    cp.add_argument("--out", type=Path, default=Path("results/oracle/single"))
    mp = sub.add_parser("matrix")
    mp.add_argument("--out", type=Path, default=Path("results/oracle/final"))
    args = ap.parse_args()
    if args.command == "case":
        if args.duration < 9 or not np.isfinite(args.duration):
            ap.error("duration must be finite and at least 9 s for contact metrics")
        r = run_case(args.name, args.mode, args.budget, args.out, duration=args.duration,
                     config=QPConfig(task_weight=args.task_weight))
        return 0 if r["status"] == "PASS" else 2
    results = {}
    for label, name, mode, budget in MATRIX:
        weight = 100. if label in ("task-cost", "lower-budget", "elbow-tight") else 0. if label == "torque-only" else 1.
        results[label] = run_case(name, mode, budget, args.out / label, label=label,
                                  config=QPConfig(task_weight=weight))
    save_json(args.out / "summary.json", results)
    # Completion here means trials finished, not task quality. Return a failure
    # when a held-out/optional active trial fails; preserve every case regardless.
    expected_failures = {"stress", "torque-only"}
    return 0 if all(v["status"] == "PASS" for k,v in results.items() if k not in expected_failures) else 2


if __name__ == "__main__":
    raise SystemExit(main())
