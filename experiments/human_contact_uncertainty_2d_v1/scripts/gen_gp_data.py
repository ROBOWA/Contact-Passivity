"""Generate synthetic, clearly labelled GP training/validation/test data (splits by trajectory seed,
intent family, location and gains; never by neighbouring time samples). Unprotected explorer controller.
Usage: python scripts/gen_gp_data.py [--workers 8]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params, scenarios, episode, controllers
from hcu2d.scenarios import FAMILIES_TRAIN

SPLITS = {  # name: (seed0, n, families, residual, gain_mismatch)
    "train_g0": (1000, 30, FAMILIES_TRAIN, False, 1.0),
    "train_res": (1100, 50, FAMILIES_TRAIN, True, 1.0),
    "val": (2000, 25, FAMILIES_TRAIN, True, 1.0),
    "test_g0": (2900, 25, FAMILIES_TRAIN, False, 1.0),
    "test_id": (3000, 30, FAMILIES_TRAIN, True, 1.0),
    "test_heldout": (4000, 20, ("oscillate", "fast_assist"), True, 1.0),
    "test_mismatch": (5000, 20, FAMILIES_TRAIN, True, 2.0),
}

def one(args):
    split, seed, fam, residual, mis = args
    cfg = params.default(); cfg["task"]["T"] = 5.0
    sc = scenarios.make("gp_data", seed, cfg, overrides=dict(family=fam, residual=residual, gain_mismatch=mis))
    out = episode.run(cfg, sc, controllers.Explorer())
    L = out["log"]; ev = sc["events"][0]
    k = np.where(L["engaged"] > 0)[0]
    t = np.arange(len(L["q"])) * 1e-3
    a = np.array([ev.intention(tt)[:4] for tt in t[k]])
    loc = ev.s + sc["noise"]["b_s"] + sc["noise"]["eps_s"]
    return dict(split=split, seed=seed, family=fam, residual=residual, mismatch=mis, s=ev.s, K=ev.K, D=ev.D,
                t_on=ev.t_on, t_off=ev.t_off, cause=ev.release_cause, q_on=L["q"][k[0]], idx=k, t=t[k], q=L["q"][k],
                qd=L["qd"][k], y=L["y"][k], f=L["f_true"][k], a=a, b_s=sc["noise"]["b_s"], loc=loc[: (len(k) // 40) + 1])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    out = ROOT / "outputs" / "gp_data"; out.mkdir(parents=True, exist_ok=True)
    jobs = [(sp, s0 + i, fams[i % len(fams)], res, mis) for sp, (s0, n, fams, res, mis) in SPLITS.items() for i in range(n)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(one, jobs)
    for sp in SPLITS:
        rows = [r for r in res if r["split"] == sp]
        np.save(out / f"{sp}.npy", np.array(rows, dtype=object), allow_pickle=True)
    meta = {sp: dict(seed0=v[0], n=v[1], families=list(v[2]), residual=v[3], gain_mismatch=v[4]) for sp, v in SPLITS.items()}
    meta["controller"] = "X_explorer_data_collection (unprotected, offline data only)"
    meta["seconds"] = time.time() - t0
    (out / "splits.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))

if __name__ == "__main__":
    main()
