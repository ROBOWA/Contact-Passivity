"""Aggregate offline predictor metrics into outputs/offline/offline_summary.json (and print a table)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "outputs" / "offline" / "offline_records.json").read_text())
LEV = ["0.5", "0.8", "0.9", "0.95", "0.99"]
summ = {}
for split, rec in R.items():
    S = {}
    for m in ["S", "S0", "N", "GM"]:
        rm = np.array([r[m]["rmse"] for r in rec]); nl = np.array([r[m]["nlpd"] for r in rec])
        cov = {l: float(np.mean([r[m]["cover"][l] for r in rec])) for l in LEV}
        Wz = np.array([r[m]["W"]["z"] for r in rec])
        Wc = {l: float(np.mean([r[m]["W"]["cover"][l] for r in rec])) for l in LEV}
        werr = np.array([r[m]["W"]["mean"] - r[m]["W"]["truth"] for r in rec])
        d = dict(rmse=float(np.sqrt(np.mean(rm ** 2))), rmse_h1=float(np.sqrt(np.mean([r[m]["rmse_h"][0] ** 2 for r in rec]))),
                 rmse_h15=float(np.sqrt(np.mean([r[m]["rmse_h"][-1] ** 2 for r in rec]))), nlpd=float(np.mean(nl)),
                 cover=cov, W_rmse=float(np.sqrt(np.mean(werr ** 2))), W_z_sd=float(np.std(Wz)), W_cover=Wc, n=len(rec))
        if m in ("S", "S0"):
            Wzi = np.array([r[m]["W_indep"]["z"] for r in rec])
            d["W_indep_z_sd"] = float(np.std(Wzi)); d["W_indep_cover"] = {l: float(np.mean([r[m]["W_indep"]["cover"][l] for r in rec])) for l in LEV}
            d["W_sd_ratio_full_over_indep"] = float(np.median([r[m]["W"]["sd"] / max(r[m]["W_indep"]["sd"], 1e-12) for r in rec]))
        # response to candidate motion: compare predicted vs true force change at horizon end
        for pn in ["trans_x", "rot"]:
            pr = np.array([r[m]["resp"][pn] for r in rec]); tr = np.array([r["truth_resp"][pn] for r in rec])
            d[f"resp_{pn}_rmse"] = float(np.sqrt(np.mean((pr - tr) ** 2)))
            d[f"resp_{pn}_true_rms"] = float(np.sqrt(np.mean(tr ** 2)))
            d[f"resp_{pn}_slope"] = float(np.sum(pr * tr) / max(np.sum(tr * tr), 1e-12))
        S[m] = d
    S["HOLD"] = dict(rmse=float(np.sqrt(np.mean([r["HOLD"]["rmse"] ** 2 for r in rec]))))
    it = [r["intent"] for r in rec if "intent" in r]
    e = np.array([x["err"] for x in it]); sd = np.array([x["sd"] for x in it])
    z = e / sd
    S["intent"] = dict(rmse=float(np.sqrt(np.mean(e ** 2))), mean_post_sd=float(np.mean(sd)), prior_sd=it[0]["prior_sd"],
                       z_sd=float(np.std(z)), cover95=float(np.mean(np.abs(z) <= 1.96)),
                       s_err_rmse=float(np.sqrt(np.mean([(x["s_post"] - r["s"]) ** 2 for x, r in zip(it, rec)]))),
                       s_post_sd_mean=float(np.mean([x["s_post_sd"] for x in it])),
                       s_sensor_err_rmse=float(np.sqrt(np.mean([(r["s_hat_sensor"] - r["s"]) ** 2 for r in rec]))),
                       max_abs_corr_w_a_median=float(np.median([x["corr_w_a"] for x in it])))
    summ[split] = S
(ROOT / "outputs" / "offline" / "offline_summary.json").write_text(json.dumps(summ, indent=1))
for split, S in summ.items():
    print(f"== {split}  n={S['S']['n']}  HOLD rmse={S['HOLD']['rmse']:.2f}")
    for m in ["S", "S0", "N", "GM"]:
        d = S[m]
        print(f"  {m:3s} rmse={d['rmse']:.2f} (h1 {d['rmse_h1']:.2f}, h15 {d['rmse_h15']:.2f}) nlpd={d['nlpd']:6.2f} cov90={d['cover']['0.9']:.2f} cov99={d['cover']['0.99']:.2f} "
              f"| W rmse={d['W_rmse']*1e3:.1f}mJ zsd={d['W_z_sd']:.2f} Wcov90={d['W_cover']['0.9']:.2f} "
              + (f"indep zsd={d['W_indep_z_sd']:.2f} Wcov90i={d['W_indep_cover']['0.9']:.2f} sd_ratio={d['W_sd_ratio_full_over_indep']:.2f} " if 'W_indep_z_sd' in d else "")
              + f"| resp trans rmse={d['resp_trans_x_rmse']:.2f}/{d['resp_trans_x_true_rms']:.2f} slope={d['resp_trans_x_slope']:.2f} rot rmse={d['resp_rot_rmse']:.2f}/{d['resp_rot_true_rms']:.2f} slope={d['resp_rot_slope']:.2f}")
    print("  intent:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in S["intent"].items()})
