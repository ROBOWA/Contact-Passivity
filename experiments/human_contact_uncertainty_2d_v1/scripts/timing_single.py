"""Single-process planner timing (one process, BLAS 1 thread) on block_release, fresh seeds 300-302.
Writes outputs/timing_single.json. Other system load at measurement time is recorded (loadavg)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, platform, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params, scenarios, episode, planner

cfg = params.default()
out = {"loadavg_start": os.getloadavg(), "machine": platform.platform(), "python": sys.version.split()[0]}
for ck in ["C1", "C2", "C3", "C4"]:
    allt, eng, wall = [], [], []
    for seed in (300, 301, 302):
        sc = scenarios.make("block_release", seed, cfg)
        t0 = time.perf_counter()
        o = episode.run(cfg, sc, planner.make(ck, cfg))
        wall.append(time.perf_counter() - t0)
        allt += list(o["plan_time"])
        eng += [p["t_mc"] + p["t_lin"] + p["t_model"] for p in o["plan_info"] if p["engaged"]]
    a, e = np.array(allt) * 1e3, np.array(eng) * 1e3
    out[ck] = dict(all_p50=float(np.median(a)), all_p95=float(np.quantile(a, .95)), all_p99=float(np.quantile(a, .99)),
                   all_max=float(a.max()), engaged_p50=float(np.median(e)), engaged_p95=float(np.quantile(e, .95)),
                   engaged_max=float(e.max()), episode_wall_s=float(np.mean(wall)), n_calls=len(a))
    print(ck, {k: round(v, 1) for k, v in out[ck].items()})
out["loadavg_end"] = os.getloadavg()
(ROOT / "outputs" / "timing_single.json").write_text(json.dumps(out, indent=1))
