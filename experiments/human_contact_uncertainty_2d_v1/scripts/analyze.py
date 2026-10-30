"""Aggregate a run tag: per (scenario, controller) rates with Wilson 95% CI and one-sided exact
(Clopper–Pearson) 95% upper bounds; task/interaction/timing summaries; paired differences.

  python scripts/analyze.py <tag> [<tag> ...]   -> outputs/runs/<tag>/summary.json, summary.md
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import beta as Beta

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cp_upper(k, n, a=0.05):
    return 1.0 if k >= n else float(Beta.ppf(1 - a, k + 1, n - k))


def load(tag):
    rows = [json.loads(l) for l in (ROOT / "outputs" / "runs" / tag / "metrics.jsonl").read_text().splitlines() if l.strip()]
    err = [r for r in rows if "error" in r]
    return [r for r in rows if "error" not in r], err


def nanstat(v, f):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return float(f(v)) if len(v) else None


RATE_KEYS = ["completed", "violation_Ec", "violation_BH", "violation_hH", "violation_any", "infeasible_init",
             "domain_violation"]
MEAN_KEYS = ["coverage", "tip_x_final", "force_rmse", "theta_rmse", "lag_rms", "contact_loss_s", "ref_drift_max",
             "du_rms", "sat_frac", "filt_frac", "WH_net_final", "WH_pos_final", "pH_peak", "fH_peak", "fn_peak",
             "Ec_min", "BH_true_min", "hH_true_min", "BH_err_maxabs", "recovery_s", "surge_v", "n_fallback",
             "n_fallback_engaged", "plan_ms_p50", "plan_ms_p95", "plan_ms_p99", "plan_ms_max", "plan_ms_engaged_p95",
             "wall_s", "residual_storage_plant_max", "residual_storage_quad_max", "credit_excess_vs_plant",
             "sel_sig_v_mean", "sel_sig_K_mean", "sel_sig_F_mean", "sel_backup_frac", "sel_brake_frac", "obj_se_mean", "ess_min",
             "s_err_gp", "s_out_of_patch_max", "filt_infeasible_ticks", "corner_min_y", "t_done"]


def summarize(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["scenario"], r["controller"])].append(r)
    out = {}
    for (s, c), rs in sorted(g.items()):
        n = len(rs)
        d = {"n": n}
        for k in RATE_KEYS:
            kk = int(sum(bool(r.get(k)) for r in rs))
            d[k] = dict(k=kk, rate=kk / n, wilson95=wilson(kk, n), cp_upper95=cp_upper(kk, n))
        for k in MEAN_KEYS:
            vals = [r.get(k) for r in rs]
            d[k] = dict(mean=nanstat(vals, np.mean), median=nanstat(vals, np.median),
                        q05=nanstat(vals, lambda v: np.quantile(v, .05)), q95=nanstat(vals, lambda v: np.quantile(v, .95)),
                        min=nanstat(vals, np.min), max=nanstat(vals, np.max))
        out[f"{s}|{c}"] = d
    return out


def paired(rows, a, b, key):
    """Paired difference a−b over common (scenario, seed)."""
    A = {(r["scenario"], r["seed"]): r.get(key) for r in rows if r["controller"] == a}
    B = {(r["scenario"], r["seed"]): r.get(key) for r in rows if r["controller"] == b}
    res = defaultdict(list)
    for k in A:
        if k in B and A[k] is not None and B[k] is not None:
            res[k[0]].append(float(A[k]) - float(B[k]))
    return {s: dict(n=len(v), mean=float(np.mean(v)), se=float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else None)
            for s, v in res.items()}


def md_table(summ, keys):
    lines = ["| scenario | ctrl | n | completed | viol Ec | viol BH | viol h_H | W_H net [J] | W_H+ [J] | coverage | force RMSE [N] | recovery [s] | plan p95 [ms] |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in summ.items():
        s, c = key.split("|")
        f = lambda k: f"{d[k]['k']}/{d['n']}"
        m = lambda k, fmt="{:.3f}": (fmt.format(d[k]["mean"]) if d[k]["mean"] is not None else "–")
        lines.append(f"| {s} | {c} | {d['n']} | {f('completed')} | {f('violation_Ec')} | {f('violation_BH')} | {f('violation_hH')} | "
                     f"{m('WH_net_final')} | {m('WH_pos_final')} | {m('coverage', '{:.2f}')} | {m('force_rmse', '{:.2f}')} | "
                     f"{m('recovery_s', '{:.2f}')} | {m('plan_ms_p95', '{:.1f}')} |")
    return "\n".join(lines)


def main():
    for tag in sys.argv[1:]:
        rows, err = load(tag)
        summ = summarize(rows)
        ctrls = sorted({r["controller"] for r in rows})
        pairs = {}
        if "C3" in ctrls:
            for other in ctrls:
                if other != "C3":
                    pairs[f"{other}-C3"] = {k: paired(rows, other, "C3", k) for k in
                                           ["WH_pos_final", "WH_net_final", "coverage", "force_rmse", "tip_x_final", "BH_true_min", "hH_true_min"]}
        res = dict(tag=tag, n_rows=len(rows), n_errors=len(err), errors=err[:5], summary=summ, paired=pairs)
        out = ROOT / "outputs" / "runs" / tag
        (out / "summary.json").write_text(json.dumps(res, indent=1))
        (out / "summary.md").write_text(md_table(summ, None) + "\n")
        print(f"== {tag}: {len(rows)} episodes, {len(err)} errors")
        print(md_table(summ, None))
        for e in err[:3]:
            print("ERROR", e["scenario"], e["seed"], e["controller"], e["error"])


if __name__ == "__main__":
    main()
