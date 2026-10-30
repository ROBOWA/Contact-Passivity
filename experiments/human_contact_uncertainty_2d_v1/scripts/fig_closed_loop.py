"""Closed-loop figures from a run tag (default: main).

  python scripts/fig_closed_loop.py --tag main --example-scenario bias_ood --example-seed 100
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params
from hcu2d.viz import plt, style, save, CTRL, LABEL, INK, INK2, MUTED, GRID, AXIS

FIG = ROOT / "outputs" / "figures"
CFG = params.default()
ORDER = ["C0", "C1", "C2", "C3", "C4"]


def load_trace(tag, scen, ck, seed):
    p = ROOT / "outputs" / "runs" / tag / "traces" / f"{scen}_{ck}_{seed}.npz"
    if not p.exists():
        return None
    d = dict(np.load(p, allow_pickle=True))
    d["plan_info"] = json.loads(str(d["plan_info"]))
    return d


def body_poly(q):
    b = CFG["body"]
    L, H = b["L"], b["H"]
    c, s = np.cos(q[2]), np.sin(q[2])
    R = np.array([[c, -s], [s, c]])
    rect = np.array([[-L / 2, -H / 2], [L / 2, -H / 2], [L / 2, H / 2], [-L / 2, H / 2]])
    stem = np.array([[b["r_T"][0] - 0.012, -H / 2], [b["r_T"][0] + 0.012, -H / 2], [b["r_T"][0] + 0.004, b["r_T"][1]],
                     [b["r_T"][0] - 0.004, b["r_T"][1]]])
    return q[:2] + rect @ R.T, q[:2] + stem @ R.T


def draw_scene(ax, tr, k, title=None, show_belief=True, fscale=0.008, follow=False):
    """Planar scene at trace index k: wall, tool, task contact force, human contact (true + belief)."""
    b = CFG["body"]
    q = tr["q"][k]
    ax.add_patch(Rectangle((-0.5, -0.06), 1.1, 0.06, color="#e9e7e1", lw=0))
    ax.plot([-0.5, 0.6], [0, 0], color=AXIS, lw=1.2)
    ax.text(0.99, 0.03, "compliant wall", ha="right", va="bottom", fontsize=7, color=MUTED, transform=ax.transAxes)
    rect, stem = body_poly(q)
    ax.add_patch(Polygon(stem, closed=True, color="#b9b7ae", lw=0))
    ax.add_patch(Polygon(rect, closed=True, facecolor="#dcdad3", edgecolor=INK2, lw=0.8))
    c, s = np.cos(q[2]), np.sin(q[2])
    R = np.array([[c, -s], [s, c]])
    tip = q[:2] + R @ np.array(b["r_T"])
    ax.plot(*q[:2], "+", color=INK, ms=7)
    fn = tr["fn"][k]
    ax.add_patch(FancyArrowPatch(tip, tip + np.array([0, fn * fscale]), arrowstyle="-|>", mutation_scale=8, color="#6250d6", lw=1.4))
    ax.text(*(tip + np.array([0.012, fn * fscale * 0.6])), f"task f_n {fn:.1f} N", fontsize=7, color="#6250d6")
    # patch (top edge) and human contact
    e0 = q[:2] + R @ np.array([b["s_min"], b["h_e"]])
    e1 = q[:2] + R @ np.array([b["s_max"], b["h_e"]])
    ax.plot([e0[0], e1[0]], [e0[1], e1[1]], color=INK2, lw=2.2, solid_capstyle="butt")
    if tr["engaged"][k] > 0:
        s_true = tr["s_true"][k]
        pH = q[:2] + R @ np.array([s_true, b["h_e"]])
        if show_belief:
            sm, sd = tr["s_hat"][k], np.sqrt(tr["s_var"][k])
            for w_, al in [(2, 0.18), (1, 0.32)]:
                a0 = q[:2] + R @ np.array([sm - w_ * sd, b["h_e"]])
                a1 = q[:2] + R @ np.array([sm + w_ * sd, b["h_e"]])
                ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color="#2a78d6", lw=7, alpha=al, solid_capstyle="butt")
            ax.plot(*(q[:2] + R @ np.array([sm, b["h_e"]])), "|", color="#2a78d6", ms=12, mew=2)
        ax.plot(*pH, "o", color="#e34948", ms=6, mec="white", mew=1)
        f = tr["f_true"][k]
        ax.add_patch(FancyArrowPatch(pH, pH + f * fscale, arrowstyle="-|>", mutation_scale=8, color="#e34948", lw=1.4))
        ax.text(*(pH + f * fscale + np.array([0.005, 0.008])), f"human f_H |{np.hypot(*f):.1f}| N", fontsize=7, color="#e34948")
    ax.set_aspect("equal")
    if follow:
        xc = float(q[0])
        ax.set_xlim(xc - 0.30, xc + 0.30)
        ax.set_ylim(-0.035, 0.235)
    else:
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.05, 0.27)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def fig_geometry(tag, scen, seed, ck="C3"):
    tr = load_trace(tag, scen, ck, seed)
    if tr is None:
        return
    ks = np.where(tr["engaged"] > 0)[0]
    k = ks[len(ks) // 2]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    draw_scene(ax, tr, k, title=f"Planar tool, wall contact and uncertain human contact point ({scen}, {LABEL[ck]}, t = {tr['t'][k]:.2f} s)",
               follow=True, fscale=0.010)
    q = tr["q"][k]
    c, s = np.cos(q[2]), np.sin(q[2])
    R = np.array([[c, -s], [s, c]])
    b = CFG["body"]
    ax.annotate("COM (x, y), θ", q[:2], xytext=(q[0] - 0.22, q[1] - 0.005), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    tip = q[:2] + R @ np.array(b["r_T"])
    ax.annotate("task contact: off-centre tip r_T", tip, xytext=(tip[0] - 0.26, tip[1] + 0.012), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    e0 = q[:2] + R @ np.array([b["s_min"], b["h_e"]])
    ax.annotate("human patch (top edge), s ∈ [−0.18, 0.18] m", e0, xytext=(e0[0] - 0.02, e0[1] + 0.05), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.text(0.01, 0.98, "red dot = TRUE human contact point s (hidden from the controller); blue band = location belief ŝ ±1σ/±2σ\n"
            "(25 Hz sensor with persistent bias floor); q = (x, y, θ), M = diag(m, m, I)", transform=ax.transAxes,
            fontsize=7, color=INK2, va="top")
    save(fig, FIG / "geometry.png")


def fig_accounts(tag, scen, seed, cks=("C1", "C3")):
    trs = {ck: load_trace(tag, scen, ck, seed) for ck in cks}
    trs = {k: v for k, v in trs.items() if v is not None}
    if not trs:
        return
    kap = 2.326
    fig, ax = plt.subplots(len(trs), 2, figsize=(11, 3.1 * len(trs)), squeeze=False)
    for i, (ck, tr) in enumerate(trs.items()):
        t = tr["t"]
        a = ax[i, 0]
        mean = tr["BH_book"] + tr["m_e"]
        a.fill_between(t, mean - kap * tr["sd_e"], mean + kap * tr["sd_e"], color=CTRL[ck], alpha=0.2, lw=0,
                       label="belief: BH_book + E[e_BH] ± κσ_e")
        a.plot(t, tr["BH_book"], color=CTRL[ck], lw=1.2, label="booked BH (software ledger)")
        a.plot(t, tr["true_BH_true"], color=INK, lw=1.2, label="TRUE BH (auditor)")
        a.plot(t, tr["true_S"], color=MUTED, lw=1.0, label="storage S (backup reserve need)")
        a.axhline(0, color=AXIS, lw=0.8)
        eng = tr["engaged"] > 0
        if eng.any():
            a.axvspan(t[eng][0], t[eng][-1], color=GRID, alpha=0.5, lw=0)
        a.set_title(f"{LABEL[ck]}: human-output account [J] ({scen}, seed {seed})")
        a.set_xlabel("time [s]")
        a.legend(loc="upper right", fontsize=7)
        b = ax[i, 1]
        b.plot(t, tr["true_hH_true"], color=INK, lw=1.2, label="TRUE backup margin h_H = BH − S")
        b.plot(t, tr["Ec"], color=CTRL[ck], lw=1.2, label="Ec ledger (exact under ZOH + encoders)")
        b.axhline(0, color="#d03b3b", lw=0.8)
        b.text(t[-1], 0, " floor", color="#d03b3b", fontsize=7, va="bottom", ha="right")
        hmin = np.min(tr["true_hH_true"])
        b.set_title(f"backup margin and Ec [J]; min h_H = {hmin:+.4f} J")
        b.set_xlabel("time [s]")
        b.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, FIG / f"accounts_{scen}_{seed}.png")


def fig_rates(summ, scen_list, fname, keys=("violation_hH", "violation_BH"), title="violation rate"):
    fig, ax = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 0.55 * len(scen_list) + 1.6), squeeze=False)
    for j, key in enumerate(keys):
        a = ax[0, j]
        for i, sc in enumerate(scen_list):
            for m, ck in enumerate(ORDER):
                d = summ.get(f"{sc}|{ck}")
                if d is None:
                    continue
                y = i + (m - 2) * 0.14
                r, (lo, hi) = d[key]["rate"], d[key]["wilson95"]
                a.plot([lo, hi], [y, y], color=CTRL[ck], lw=1.6)
                a.plot(r, y, "o", color=CTRL[ck], ms=4.5, mec="white", mew=0.6,
                       label=LABEL[ck] if i == 0 else None)
                if r > 0:
                    a.text(hi + 0.01, y, f"{d[key]['k']}/{d['n']}", fontsize=6, va="center", color=INK2)
        a.set_yticks(range(len(scen_list)), scen_list)
        a.invert_yaxis()
        a.set_xlim(-0.02, 1.08)
        a.set_xlabel(f"{key.replace('violation_', 'true ')} violation rate (Wilson 95% CI)")
        if j == 0:
            a.legend(loc="lower right", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save(fig, FIG / fname)


def fig_tradeoff(summ, scen_list, fname="tradeoff.png"):
    n = len(scen_list)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), squeeze=False)
    for i, sc in enumerate(scen_list):
        a = ax[i // cols, i % cols]
        for ck in ORDER:
            d = summ.get(f"{sc}|{ck}")
            if d is None:
                continue
            x, y = d["WH_pos_final"]["mean"], d["coverage"]["mean"]
            a.plot(x, y, "o", color=CTRL[ck], ms=7, mec="white", mew=0.8)
            a.annotate(ck, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
        a.set_xscale("symlog", linthresh=0.05)
        a.set_title(sc, fontsize=9)
        a.set_xlabel("mean positive robot→human work [J]")
        a.set_ylabel("mean wiping coverage")
    for j in range(n, rows * cols):
        ax[j // cols, j % cols].axis("off")
    fig.suptitle("Task vs. interaction trade-off (C0 is UNPROTECTED)", fontsize=10)
    fig.tight_layout()
    save(fig, FIG / fname)


def fig_runtime(summ, scen_list, fname="runtime.png"):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for m, ck in enumerate(["C1", "C2", "C3", "C4"]):
        p50 = [summ[f"{s}|{ck}"]["plan_ms_p50"]["mean"] for s in scen_list if f"{s}|{ck}" in summ]
        p95e = [summ[f"{s}|{ck}"]["plan_ms_engaged_p95"]["mean"] for s in scen_list if f"{s}|{ck}" in summ]
        mx = [summ[f"{s}|{ck}"]["plan_ms_max"]["max"] for s in scen_list if f"{s}|{ck}" in summ]
        for j, (v, lab) in enumerate([(np.mean(p50), "p50 all calls"), (np.mean(p95e), "p95 engaged"), (np.max(mx), "max")]):
            x = j + (m - 1.5) * 0.2
            a = ax.bar(x, v, width=0.18, color=CTRL[ck], label=LABEL[ck] if j == 0 else None)
            ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.axhline(20, color="#d03b3b", lw=0.8)
    ax.text(2.45, 20, "20 ms control period", fontsize=7, color="#d03b3b", ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2], ["p50 (all planning calls)", "p95 (during human contact)", "max"])
    ax.set_ylabel("planner wall time per call [ms]")
    ax.set_title("Planner timing (Python/numpy, Apple M4; not a real-time claim)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, FIG / fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--example-scenario", default="bias_ood")
    ap.add_argument("--example-seed", type=int, default=100)
    a = ap.parse_args()
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.loads((ROOT / "outputs" / "runs" / a.tag / "summary.json").read_text())["summary"]
    scen = [s for s in ["no_human", "block_release", "assist_reverse", "normal_oblique", "moment_arm", "ood_intent", "sensor_bias",
                        "low_BH", "infeasible_init"] if any(k.startswith(s + "|") for k in S)]
    fig_geometry(a.tag, "moment_arm", a.example_seed)
    fig_accounts(a.tag, a.example_scenario, a.example_seed)
    fig_accounts(a.tag, "block_release", a.example_seed, cks=("C0", "C3"))
    fig_rates(S, scen, f"violations_{a.tag}.png", title=f"True-account violations per episode ({a.tag}); C0 is UNPROTECTED")
    fig_rates(S, scen, f"completion_{a.tag}.png", keys=("completed",), title=f"Task completion rate ({a.tag})")
    fig_tradeoff(S, scen, f"tradeoff_{a.tag}.png")
    fig_runtime(S, [s for s in scen], f"runtime_{a.tag}.png")
    print("ok")


if __name__ == "__main__":
    main()
