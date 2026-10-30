"""Belief-calibration study with forked truth (formulation §9, tier B/C evidence).

At every k-th engaged planning call, the TRUE simulator is deep-copied and the selected candidate is
executed open-loop for the horizon (no filter, no re-planning); realized stage-end h_H, Ec, BH_true
and human work are compared with the planner's first-order and Monte Carlo predictions.

  python scripts/calibration_fork.py --tag calib --controllers C3 C2 --seeds 200 230
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params, scenarios, episode, planner

SCEN = ["block_release", "assist_reverse", "normal_oblique", "moment_arm", "bias_ood"]
LEV = [0.5, 0.8, 0.9, 0.95, 0.99]
QL = [0.005, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.995]


def job(a):
    scen, seed, ck, every = a
    cfg = params.default()
    sc = scenarios.make(scen, seed, cfg)
    out = episode.run(cfg, sc, planner.make(ck, cfg), fork_every=every)
    recs = []
    for p in out["plan_info"]:
        if "fork" not in p:
            continue
        f, st = p["fork"], p["stage"]
        recs.append(dict(scenario=scen, seed=seed, controller=ck, t=p["t"], cand=p["cand"], fork=f, stage=st))
    return recs


def pit_from_quantiles(qv, x):
    """Approximate PIT of x under a distribution given by quantiles QL (linear interpolation, clipped)."""
    return float(np.interp(x, qv, QL, left=0.0, right=1.0))


def analyse(recs):
    out = {}
    for ck in sorted({r["controller"] for r in recs}):
        R = [r for r in recs if r["controller"] == ck]
        res = {"n_forks": len(R)}
        for name, mk, sk, real in [("hH_lin", "hH_m", "hH_sd", "hH"), ("hH_mc", "hH_mc_m", "hH_mc_sd", "hH"),
                                   ("Ec_lin", "Ec_m", "Ec_sd", "Ec"), ("BH_lin", "BH_m", "BH_sd", "BH")]:
            z = []
            for r in R:
                m, s = np.array(r["stage"][mk]), r["stage"].get(sk)
                if s is None:
                    continue
                s = np.array(s)
                x = np.array(r["fork"][real])
                k = np.arange(1, len(x))
                z.append((x[k] - m[k]) / np.maximum(s[k], 1e-9))
            if not z:
                continue
            z = np.array(z)
            zN = z[:, -1]
            res[name] = dict(z_mean=float(zN.mean()), z_sd=float(zN.std()),
                             cover={str(l): float(np.mean(np.abs(zN) <= norm.ppf(0.5 + l / 2))) for l in LEV},
                             cover_all_stages={str(l): float(np.mean(np.abs(z) <= norm.ppf(0.5 + l / 2))) for l in LEV},
                             lower_tail_exceed_01=float(np.mean(zN < norm.ppf(0.01))))
        pit, exceed_hi, exceed_lo, lin_vs_mc = [], [], [], []
        for r in R:
            q = np.array(r["stage"]["WH_mc_q"])[:, -1]
            x = r["fork"]["WH"][-1]
            pit.append(pit_from_quantiles(q, x))
            exceed_hi.append(x > q[-1])
            exceed_lo.append(x < q[0])
            bhm = np.array(r["stage"]["BH_m"])
            lin_vs_mc.append((bhm[0] - bhm[-1]) - r["stage"]["WH_mc_m"][-1])
        pit = np.array(pit)
        res["WH_mc"] = dict(pit_hist=np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
                            exceed_q995=float(np.mean(exceed_hi)), exceed_q005=float(np.mean(exceed_lo)),
                            cover={str(l): float(np.mean(np.abs(pit - 0.5) <= l / 2)) for l in LEV})
        d = np.array(lin_vs_mc)
        res["mean_power_correction"] = dict(note="first-order mean W_H minus MC mean W_H over the horizon (J); MC includes −tr Cov(f,v)",
                                            mean=float(d.mean()), median=float(np.median(d)), q05=float(np.quantile(d, .05)),
                                            q95=float(np.quantile(d, .95)))
        out[ck] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="calib")
    ap.add_argument("--controllers", nargs="+", default=["C3", "C2"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[200, 230])
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    jobs = [(s, seed, c, a.every) for s in SCEN for seed in range(*a.seeds) for c in a.controllers]
    t0 = time.time()
    with Pool(a.workers) as pool:
        recs = [r for rs in pool.imap_unordered(job, jobs) for r in rs]
    out = ROOT / "outputs" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}_records.json").write_text(json.dumps(recs))
    summ = analyse(recs)
    summ["meta"] = dict(scenarios=SCEN, seeds=a.seeds, every=a.every, wall_s=time.time() - t0,
                        note="forks execute the selected candidate open-loop in the TRUE simulator; no filter, no re-planning")
    (out / f"{a.tag}_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
