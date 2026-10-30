"""Offline predictor figures: example inference/forecast, coverage, response to candidate motion,
learned residual. Inputs: outputs/gp_data, outputs/models/gp_frozen.json, outputs/offline/*."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params
from hcu2d.gp import StructuredGP, NaiveGP, GaussMarkov, edge_terms
from hcu2d.kernels import RBFFeatures
from hcu2d.viz import plt, style, save, PRED, PRED_LABEL, INK, INK2, MUTED

FIG = ROOT / "outputs" / "figures"
M = json.loads((ROOT / "outputs" / "models" / "gp_frozen.json").read_text())
HE = params.default()["body"]["h_e"]


def example_inference(split="test_id", pick=("reverse", 3006), td=1.3):
    data = list(np.load(ROOT / "outputs" / "gp_data" / f"{split}.npy", allow_pickle=True))
    ev = [e for e in data if e["family"] == pick[0]][0]
    t_all = ev["t"]
    t0 = t_all[0]
    kd = int(round(td / 1e-3))
    ko = np.arange(max(0, kd - 1600), kd + 1)
    ko = ko[ko % 40 == 0]
    kf = kd + np.arange(1, 16) * 20
    n_loc = kd // 40 + 1
    sbar = float(np.mean(ev["loc"][:n_loc]))
    S = StructuredGP(M["structured"])
    dense = t_all[np.arange(0, kf[-1] + 1, 20)]
    post = S.condition(t_all[ko], ev["q"][ko], ev["qd"][ko], ev["y"][ko], t0, ev["q_on"], sbar, n_loc, dense, ep_times=dense[:1])
    c_on, _, u_on, _ = edge_terms(ev["q_on"][None], np.zeros((1, 3)), HE)
    ix = post["idx"]
    a_m, a_sd = [], []
    for i, a_ in enumerate(("ax", "ay")):
        G = np.zeros((len(dense), len(post["mu"])))
        G[np.arange(len(dense)), ix[a_].start + np.arange(len(dense))] = 1.0
        G[:, ix["s"].start] = u_on[0, i]
        a_m.append(c_on[0, i] + G @ post["mu"])
        a_sd.append(np.sqrt(np.einsum("ij,jk,ik->i", G, post["Sig"], G)))
    kk = np.arange(0, kf[-1] + 1, 20)
    a_true = ev["a"][kk, :2]
    # force forecasts given the TRUE future robot trajectory
    tf = t_all[kf]
    postf = S.condition(t_all[ko], ev["q"][ko], ev["qd"][ko], ev["y"][ko], t0, ev["q_on"], sbar, n_loc, tf, ep_times=tf)
    fS, cS, _, _ = S.force_map(postf, tf, ev["q"][kf], ev["qd"][kf], t0, ev["q_on"])
    sdS = np.sqrt(np.diag(cS)).reshape(2, -1).T
    gm = GaussMarkov(M["gauss_markov"])
    fG, cG = gm.predict(t_all[ko], ev["y"][ko], tf)
    sdG = np.sqrt(np.diag(cG))[:, None].repeat(2, 1)
    s_hat = sbar
    ng = NaiveGP(M["naive"])
    Xo = NaiveGP.inputs(t_all[ko], ev["q"][ko], ev["qd"][ko], t0, ev["q_on"], s_hat, HE)
    Xf = NaiveGP.inputs(tf, ev["q"][kf], ev["qd"][kf], t0, ev["q_on"], s_hat, HE)
    o = ng.predict(Xo, ev["y"][ko], Xf)
    fN = np.stack([o[0][0], o[1][0]], -1)
    sdN = np.stack([np.sqrt(np.diag(o[0][1])), np.sqrt(np.diag(o[1][1]))], -1)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
    for i, nm in enumerate("xy"):
        a = ax[0, i]
        tt = dense - t0
        a.fill_between(tt, a_m[i] - 2 * a_sd[i], a_m[i] + 2 * a_sd[i], color=PRED["S"], alpha=0.18, lw=0)
        a.plot(tt, a_m[i], color=PRED["S"], label="posterior mean ±2σ (structured GP)")
        a.plot(tt, a_true[:, i], color=INK, lw=1.2, label="hidden true intention a_H")
        a.plot(t_all[kk] - t0, ev["q"][kk, i] * 0 + np.nan, color=MUTED)
        a.axvline(td, color=MUTED, lw=0.8)
        a.text(td, a.get_ylim()[1], " decision time", color=MUTED, fontsize=7, va="top")
        a.set_title(f"({'ab'[i]}) latent intention a_H,{nm} [m] — posterior vs truth")
        a.set_xlabel("time since grasp [s]")
        if i == 0:
            a.legend(loc="best")
        b = ax[1, i]
        th = t_all[np.arange(max(0, kd - 600), kf[-1] + 1)] - t0
        b.plot(th, ev["f"][np.arange(max(0, kd - 600), kf[-1] + 1), i], color=INK, lw=1.0, label="true f_H")
        b.plot(t_all[ko] - t0, ev["y"][ko, i], "o", ms=3, color=MUTED, label="measured (25 Hz)")
        for f_, sd_, key in [(fS, sdS, "S"), (fG, sdG, "GM"), (fN, sdN, "N")]:
            b.fill_between(tf - t0, f_[:, i] - 2 * sd_[:, i], f_[:, i] + 2 * sd_[:, i], color=PRED[key], alpha=0.14, lw=0)
            b.plot(tf - t0, f_[:, i], color=PRED[key], label=PRED_LABEL[key] + " ±2σ")
        b.set_xlim(td - 0.6, tf[-1] - t0 + 0.02)
        b.axvline(td, color=MUTED, lw=0.8)
        b.set_title(f"({'cd'[i]}) 0.3 s force forecast f_H,{nm} [N] given the robot trajectory")
        b.set_xlabel("time since grasp [s]")
        if i == 0:
            b.legend(loc="best", fontsize=7)
    fig.suptitle(f"Held-out test episode (family '{ev['family']}', seed {ev['seed']}, true s = {ev['s']:+.3f} m)", fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "gp_inference_example.png")


def coverage_and_response():
    R = json.loads((ROOT / "outputs" / "offline" / "offline_records.json").read_text())
    Sm = json.loads((ROOT / "outputs" / "offline" / "offline_summary.json").read_text())
    lev = [0.5, 0.8, 0.9, 0.95, 0.99]
    splits = [("test_id", "in-distribution test"), ("test_heldout", "held-out intent families"), ("test_mismatch", "gain mismatch ×2")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    for j, (sp, nm) in enumerate(splits):
        a = ax[j]
        a.plot([0.45, 1], [0.45, 1], color=MUTED, lw=0.8)
        for m in ["S", "S0", "N", "GM"]:
            cv = [Sm[sp][m]["cover"][str(l)] for l in lev]
            a.plot(lev, cv, "o-", ms=4, color=PRED[m], label=PRED_LABEL[m])
        a.set_title(f"force-interval coverage: {nm}")
        a.set_xlabel("nominal level")
        a.set_ylabel("empirical coverage")
        a.set_ylim(0.3, 1.01)
        if j == 0:
            a.legend(loc="lower right", fontsize=7)
    a = ax[3]
    xs = np.arange(len(splits))
    for k, (key, lab, col, off) in enumerate([("W_z_sd", "full temporal covariance", PRED["S"], -0.18),
                                              ("W_indep_z_sd", "independent-step approx.", "#e34948", 0.18)]):
        v = [Sm[sp]["S"][key] for sp, _ in splits]
        a.bar(xs + off, v, width=0.34, color=col, label=lab)
        for x, y in zip(xs + off, v):
            a.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
    a.axhline(1.0, color=INK, lw=0.8)
    a.text(2.45, 1.0, "calibrated", fontsize=7, color=INK2, va="bottom", ha="right")
    a.set_xticks(xs, [n for _, n in splits], rotation=12, fontsize=7)
    a.set_title("work z-score s.d. (structured GP)")
    a.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, FIG / "offline_calibration.png")
    # response to candidate motion
    rec = R["test_id"] + R["val"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for j, (pn, nm) in enumerate([("trans_x", "+2 cm tangential ramp over 0.3 s"), ("rot", "+0.05 rad rotation ramp over 0.3 s")]):
        a = ax[j]
        tr = np.array([r["truth_resp"][pn] for r in rec])
        lim = np.abs(tr).max() * 1.1
        a.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8)
        for m in ["S", "N", "GM"]:
            pr = np.array([r[m]["resp"][pn] for r in rec])
            sl = Sm["test_id"][m][f"resp_{pn}_slope"]
            a.scatter(tr.ravel(), pr.ravel(), s=9, color=PRED[m], alpha=0.55, lw=0,
                      label=f"{PRED_LABEL[m]} (slope {sl:.2f})")
        a.set_xlabel("true counterfactual Δf_H at horizon end [N]")
        a.set_ylabel("predicted Δf_H [N]")
        a.set_title(f"response to candidate motion: {nm}")
        a.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, FIG / "offline_response_to_motion.png")


def residual():
    h = M["structured"]
    f = RBFFeatures(h["feat_centers"], h["feat_width"])
    chi = np.linspace(-0.3, 0.3, 200)
    P = f(chi)
    c = params.default()["human"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for i, nm in enumerate(["x (tangential)", "y (normal)"]):
        mw, Sw = np.array(h["mu_w"][i]), np.array(h["Sig_w"][i])
        m = P @ mw
        sd = np.sqrt(np.einsum("ij,jk,ik->i", P, Sw, P))
        a = ax[i]
        a.fill_between(chi, m - 2 * sd, m + 2 * sd, color=PRED["S"], alpha=0.18, lw=0)
        a.plot(chi, m, color=PRED["S"], label="learned residual ±2σ (train split)")
        a.plot(chi, -c["c_g"] * np.tanh(chi / c["v_g"]), color=INK, lw=1.2, label="hidden truth g(χ)")
        a.set_xlabel(f"patch velocity χ_{nm[0]} [m/s]")
        a.set_title(f"residual, axis {nm}")
        if i == 0:
            a.set_ylabel("force [N]")
            a.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, FIG / "offline_residual.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    example_inference()
    coverage_and_response()
    residual()
    print("ok")
