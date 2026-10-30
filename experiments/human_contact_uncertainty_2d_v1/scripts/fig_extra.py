"""Figures for the moment-arm x location-uncertainty sweep, forked-truth calibration and refinement."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params, model as M
from hcu2d.viz import plt, style, save, CTRL, LABEL, INK, INK2, MUTED

FIG = ROOT / "outputs" / "figures"
RUNS = ROOT / "outputs" / "runs"
SV, UV = ["-0.15", "0.0", "0.15"], ["0.01", "0.03", "0.06"]
SWC = {"C1": CTRL["C1"], "C3": CTRL["C3"], "C3_forceonly": "#6250d6"}
SWL = {"C1": "C1 (no margins)", "C3": "C3 (force + location uncertainty)", "C3_forceonly": "C3 force-only uncertainty"}


def rows(tag):
    p = RUNS / tag / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def fig_sweep():
    data = {(s, u): rows(f"sweep_s{s}_u{u}") for s in SV for u in UV}
    if not any(data.values()):
        return None
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))
    # (a) analytic: identical nominal force, generalised torque vs s
    P = M.Known(params.default())
    s_grid = np.linspace(-0.18, 0.18, 50)
    for f, lab, col in [((0.0, -6.0), "normal push 6 N", INK), ((-4.2, -4.2), "oblique 6 N", MUTED), ((6.0, 0.0), "tangential 6 N (along edge)", "#e34948")]:
        tau = M.gen_force(np.zeros((50, 3)) + [0, 0.1, 0], M.r_human(s_grid, P.he), np.tile(f, (50, 1)))
        ax[0].plot(s_grid, tau[:, 2], color=col, label=lab)
    ax[0].set_xlabel("human contact coordinate s [m]")
    ax[0].set_ylabel("human moment τ_H,θ [N m]")
    ax[0].set_title("(a) same force, different moment arm")
    ax[0].legend(fontsize=7)
    # (b) physical effect in closed loop: rotation caused by the same push at different s (u = 0.03)
    a = ax[1]
    for ck in ["C1", "C3"]:
        m = [np.mean([r["theta_maxabs"] for r in data[(s, "0.03")] if r["controller"] == ck]) for s in SV]
        a.plot([float(s) for s in SV], m, "o-", color=SWC[ck], label=SWL[ck])
    a.set_xlabel("true s [m] (location noise 0.03 m)")
    a.set_ylabel("max |θ| during episode [rad]")
    a.set_title("(b) closed-loop rotation caused by the push")
    a.legend(fontsize=7)
    # (c) BH ledger error vs location uncertainty (pooled over s)
    a = ax[2]
    for ck in ["C1", "C3", "C3_forceonly"]:
        m = [np.mean([abs(r["BH_err_maxabs"]) for s in SV for r in data[(s, u)] if r["controller"] == ck]) * 1e3 for u in UV]
        sd = [np.mean([r.get("sd_e_max") or 0 for s in SV for r in data[(s, u)] if r["controller"] == ck]) * 1e3 for u in UV]
        a.plot([float(u) for u in UV], m, "o-", color=SWC[ck], label=SWL[ck] + " |error|")
        if ck != "C1":
            a.plot([float(u) for u in UV], sd, "s--", color=SWC[ck], ms=4, lw=1, label=SWL[ck] + " belief σ_e (max)")
    a.set_xlabel("location sensor noise = bias s.d. [m]")
    a.set_ylabel("true − booked BH, max |·| [mJ]")
    a.set_title("(c) past-work error vs location uncertainty")
    a.legend(fontsize=6)
    # (d) violations of h_H (true) vs location uncertainty
    a = ax[3]
    for ck in ["C1", "C3", "C3_forceonly"]:
        k = [sum(r["violation_hH"] or r["violation_BH"] for s in SV for r in data[(s, u)] if r["controller"] == ck) for u in UV]
        n = [sum(1 for s in SV for r in data[(s, u)] if r["controller"] == ck) for u in UV]
        a.plot([float(u) for u in UV], [x / max(y, 1) for x, y in zip(k, n)], "o-", color=SWC[ck], label=SWL[ck])
        for x, y, kk, nn in zip(UV, k, k, n):
            a.text(float(x), kk / max(nn, 1), f" {kk}/{nn}", fontsize=6, color=INK2)
    a.set_xlabel("location sensor noise = bias s.d. [m]")
    a.set_ylabel("episodes with true BH or h_H violation")
    a.set_title("(d) violations vs location uncertainty")
    a.legend(fontsize=6)
    fig.tight_layout()
    save(fig, FIG / "moment_arm_sweep.png")
    out = {}
    for (s, u), rs in data.items():
        for ck in ["C1", "C3", "C3_forceonly"]:
            R = [r for r in rs if r["controller"] == ck]
            if R:
                out[f"s={s},u={u},{ck}"] = dict(n=len(R), viol_hH=sum(r["violation_hH"] for r in R), viol_BH=sum(r["violation_BH"] for r in R),
                                                 completed=sum(r["completed"] for r in R),
                                                 BH_err_maxabs_mJ=float(np.mean([r["BH_err_maxabs"] for r in R]) * 1e3),
                                                 sd_e_max_mJ=float(np.mean([r.get("sd_e_max") or 0 for r in R]) * 1e3),
                                                 theta_max=float(np.mean([r["theta_maxabs"] for r in R])),
                                                 WH_pos=float(np.mean([r["WH_pos_final"] for r in R])))
    (ROOT / "outputs" / "runs" / "sweep_summary.json").write_text(json.dumps(out, indent=1))
    return out


def fig_calibration():
    p = ROOT / "outputs" / "calibration" / "calib_by_scenario.json"
    if not p.exists():
        return None
    S = json.loads(p.read_text())
    lev = [0.5, 0.8, 0.9, 0.95, 0.99]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    for j, (grp, title) in enumerate([("in_distribution_pooled", "(a) in-distribution scenarios (4)"),
                                      ("bias_ood", "(b) misspecified: OOD intent + sensor bias")]):
        a = ax[j]
        for ck in ["C3", "C2"]:
            d = S[grp][ck]
            for key, ls, lab in [("hH_lin", "-", "h_H first-order"), ("hH_mc", "--", "h_H Monte Carlo")]:
                a.plot(lev, [d[key]["cover"][str(l)] for l in lev], "o" + ls, ms=4, color=CTRL[ck], label=f"{ck}: {lab}")
            a.plot(lev, [d["WH_mc"]["cover"][str(l)] for l in lev], "s:", ms=4, color=CTRL[ck], lw=1, label=f"{ck}: work W_H (MC quantiles)")
        a.plot([0.45, 1], [0.45, 1], color=MUTED, lw=0.8)
        a.set_ylim(0, 1.02)
        a.set_xlabel("nominal central interval")
        a.set_ylabel("empirical coverage at horizon end (0.3 s)")
        a.set_title(title + f"; n = {S[grp]['C3']['n_forks']} forks / predictor")
        a.legend(fontsize=6, loc="lower right" if j == 0 else "upper left")
    a = ax[2]
    for k, ck in enumerate(["C3", "C2"]):
        h = np.array(S["in_distribution_pooled"][ck]["WH_mc"]["pit_hist"], float)
        a.bar(np.arange(10) / 10 + 0.025 + 0.04 * k, h / h.sum(), width=0.04, color=CTRL[ck], label=ck)
    a.axhline(0.1, color=MUTED, lw=0.8)
    a.text(0.98, 0.1, "uniform", fontsize=7, color=INK2, ha="right", va="bottom")
    a.set_xlabel("PIT of realised human work over 0.3 s")
    a.set_ylabel("fraction of forks")
    a.set_title("(c) work PIT, in-distribution (U-shape = over-confident)")
    a.legend(fontsize=7)
    fig.tight_layout()
    save(fig, FIG / "calibration_forks.png")
    return S


def fig_refinement():
    hs = ["0.001", "0.0005", "0.00025"]
    R = {h: rows(f"refine_h{h}") for h in hs}
    if not all(R.values()):
        return None
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    out = {}
    for ck in ["C0", "C1", "C3"]:
        res = [np.max([r["residual_storage_plant_max"] for r in R[h] if r["controller"] == ck]) for h in hs]
        resq = [np.max([r["residual_storage_quad_max"] for r in R[h] if r["controller"] == ck]) for h in hs]
        ax[0].loglog([float(h) for h in hs], res, "o-", color=CTRL[ck], label=f"{ck}: RK4 channels")
        ax[0].loglog([float(h) for h in hs], resq, "s--", ms=4, color=CTRL[ck], lw=1, label=f"{ck}: auditor trapezoid")
        # paired differences of account minima relative to the finest step
        fine = {(r["scenario"], r["seed"]): r for r in R["0.00025"] if r["controller"] == ck}
        for j, key in enumerate(["hH_true_min", "Ec_min"]):
            d = [np.max([abs(r[key] - fine[(r["scenario"], r["seed"])][key]) for r in R[h] if r["controller"] == ck]) for h in hs[:2]]
            ax[1 + j].semilogy([float(h) for h in hs[:2]], d, "o-", color=CTRL[ck], label=ck)
        out[ck] = dict(residual_plant=res, residual_quad=resq)
    ax[0].set_xlabel("physics step h [s] (servo 1 kHz)")
    ax[0].set_ylabel("max |ΔS − W_a + W_H + ∫D_R + ∫D_T| [J]")
    ax[0].set_title("(a) energy-identity residual vs step")
    ax[0].legend(fontsize=6)
    for j, key in enumerate(["min h_H (true)", "min Ec"]):
        ax[1 + j].set_xlabel("physics step h [s]")
        ax[1 + j].set_ylabel(f"max |Δ {key}| vs h = 0.25 ms [J]")
        ax[1 + j].set_title(f"({'bc'[j]}) closed-loop {key} convergence")
        ax[1 + j].legend(fontsize=7)
    fig.tight_layout()
    save(fig, FIG / "refinement.png")
    (ROOT / "outputs" / "runs" / "refinement_summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    print("sweep", fig_sweep() is not None, "calib", fig_calibration() is not None, "refine", fig_refinement() is not None)
