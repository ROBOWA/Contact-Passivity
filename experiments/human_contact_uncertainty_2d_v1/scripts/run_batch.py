"""Parallel paired-seed experiment runner.

  python scripts/run_batch.py --tag smoke --scenarios block_release no_human --controllers C0 C1 C3 \
      --seeds 0 30 [--workers 9] [--trace-seeds 3] [--override '{"planner":{"beta":1.0}}']

Writes outputs/runs/<tag>/metrics.jsonl (one line per episode), traces/<scenario>_<ctrl>_<seed>.npz for the
first --trace-seeds seeds, and run_manifest.json (config, git, timing, hardware).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process
import argparse, json, os, platform, subprocess, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params, scenarios, episode, auditor, planner, metrics


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def job(a):
    scen, seed, ckey, override, trace_dir, keep_trace, h_phys = a
    try:
        cfg = params.merge(params.default(), override)
        sc = scenarios.make(scen, seed, cfg, overrides=override.get("scenario_override"))
        C = planner.make(ckey, cfg)
        t0 = time.perf_counter()
        out = episode.run(cfg, sc, C, h_phys=h_phys)
        wall = time.perf_counter() - t0
        aud, ser = auditor.audit(out["log"], cfg, sc["accounts"])
        m = metrics.episode_metrics(out, aud, ser, cfg, sc, ckey)
        m["wall_s"] = wall
        m["h_phys"] = h_phys
        m["truth"] = scenarios.describe(sc)
        m["b_f"], m["b_s"] = sc["noise"]["b_f"].tolist(), sc["noise"]["b_s"]
        if keep_trace:
            tr = metrics.traces(out, ser)
            np.savez_compressed(Path(trace_dir) / f"{scen}_{ckey}_{seed}.npz", **tr,
                                plan_info=np.array(json.dumps(_clean(out["plan_info"]))))
        return _clean(m)
    except Exception as e:
        return {"scenario": scen, "seed": seed, "controller": ckey, "error": repr(e), "tb": traceback.format_exc()}


def git_info():
    def g(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    return {"head": g("rev-parse", "HEAD"), "status_folder": g("status", "--porcelain", "--", ".")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--controllers", nargs="+", required=True)
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 30])
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--trace-seeds", type=int, default=2)
    ap.add_argument("--override", default="{}")
    ap.add_argument("--h-phys", type=float, default=1e-3)
    a = ap.parse_args()
    override = json.loads(a.override)
    out = ROOT / "outputs" / "runs" / a.tag
    (out / "traces").mkdir(parents=True, exist_ok=True)
    jobs = [(s, seed, c, override, str(out / "traces"), seed - a.seeds[0] < a.trace_seeds, a.h_phys)
            for s in a.scenarios for seed in range(*a.seeds) for c in a.controllers]
    # longest first for load balance
    order = {"C0": 0, "F": 1, "Finst": 1, "ORACLE": 1, "C1": 2, "C2": 3}
    jobs.sort(key=lambda j: -order.get(j[2], 5))
    t0 = time.time()
    n_err = 0
    with open(out / "metrics.jsonl", "a") as f, Pool(a.workers) as pool:
        for i, m in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            f.write(json.dumps(m) + "\n")
            f.flush()
            n_err += "error" in m
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"[{a.tag}] {i + 1}/{len(jobs)} episodes, {time.time() - t0:.0f}s, errors={n_err}", flush=True)
    man = dict(tag=a.tag, scenarios=a.scenarios, controllers=a.controllers, seeds=a.seeds, workers=a.workers,
               override=override, h_phys=a.h_phys, n_jobs=len(jobs), n_errors=n_err, wall_s=time.time() - t0,
               config=params.merge(params.default(), override), git=git_info(),
               hardware=dict(machine=platform.machine(), platform=platform.platform(), cpus=os.cpu_count(),
                             python=sys.version.split()[0]),
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)))
    (out / f"run_manifest_{int(t0)}.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
