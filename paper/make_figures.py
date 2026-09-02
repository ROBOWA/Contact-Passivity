"""Generate the publication figures (Figs 1-9) and Table I for the PoC paper.

Data sources
------------
* Figs 2-7, 9 and Table I: the iteration-1 result logs in results_passivation/
  (these are the runs whose numbers are quoted in the paper text:
  RMSE(F_n) = 0.0348 N, first activations 6.21 s vs 14.73 s, peak powers
  0.1499 / 0.0997 W, E-window RMSE 0.859 vs 2.522 N, ...).
* Fig 8 (reference-governor ablation): results_passivation_iter2/governor_compare/
  (the governor exists only in the iteration-2 code path; it is a task-layer
  policy independent of the ledger rows, so the ablation is self-contained).
* Fig 1 is a drawn schematic (no simulation data).

All figures are vector PDFs sized for a two-column paper (single column
3.4 in, double column 7.0 in). Run from the repository root:

    python paper/make_figures.py
"""

from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R1 = os.path.join(ROOT, "results_passivation")          # iteration-1 logs
GOV = os.path.join(ROOT, "results_passivation_iter2", "governor_compare")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.1,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

SINGLE = 3.4          # single-column width [in]
DOUBLE = 7.0          # double-column width [in]

# Fixed controller identities (never re-ranked).
C_C0 = "#52514e"
C_C1 = "#eda100"
C_C2 = "#1baf7a"
C_C3 = "#2a78d6"
C_C4 = "#e87ba4"
C_REF = "#eb6834"     # desired / reference (dashed)
C_LIM = "#7a7975"     # limit lines
C_AUX = "#4a3aa7"     # secondary ledger / channel (violet)
GRID = "#e4e3df"
SHADE_HUMAN = "#f6e8ef"
SHADE_HELP = "#e8f3ee"


def style(ax, ylabel="", xlabel=None):
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#c9c8c4")
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def letter(ax, s):
    ax.text(0.012, 0.97, s, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top")


def load(scen_dir, mode):
    return dict(np.load(os.path.join(R1, scen_dir, mode, "log.npz")))


def shade_human(ax, t, f_h, split_sign=False):
    """Shade the human-force windows; optionally color by tangential sign."""
    on = np.linalg.norm(f_h, axis=1) > 1e-12
    if not on.any():
        return
    idx = np.flatnonzero(on)
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = [idx[0]] + list(idx[splits + 1])
    ends = list(idx[splits]) + [idx[-1]]
    for a, b in zip(starts, ends):
        color = SHADE_HUMAN
        if split_sign and f_h[(a + b) // 2, 0] > 0:
            color = SHADE_HELP
        ax.axvspan(t[a], t[b], color=color, zorder=0)


def wh_out(lg):
    dt = float(lg["meta_timestep"])
    return -np.cumsum(lg["sp_p_h_act"]) * dt


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Fig. 1 — system schematic + pipeline block diagram
# ---------------------------------------------------------------------------

def fig1():
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(DOUBLE, 2.5), width_ratios=[1.0, 1.55])

    # ---- (a) planar robot at a sliding configuration ----
    q1, q2 = 0.09, 1.55
    base = np.array([0.0, 0.7])
    elbow = base + 0.6 * np.array([np.cos(q1), -np.sin(q1)])
    ee = elbow + 0.6 * np.array([np.cos(q1 + q2), -np.sin(q1 + q2)])

    a.axhspan(-0.12, 0.0, color="#dddcd7", zorder=0)
    a.axhline(0.0, color="#8a8985", lw=0.8)
    for x in np.arange(-0.15, 1.05, 0.08):
        a.plot([x, x - 0.05], [0.0, -0.05], color="#b5b4b0", lw=0.5)

    a.plot([base[0], elbow[0], ee[0]], [base[1], elbow[1], ee[1]],
           "-", color="#3a3a38", lw=3, solid_capstyle="round", zorder=4)
    for p in (base, elbow):
        a.add_patch(Circle(p, 0.022, fc="white", ec="#3a3a38", lw=1.0,
                           zorder=5))
    a.add_patch(Circle(ee, 0.05, fc="#1baf7a", ec="none", zorder=5))
    a.add_patch(Rectangle((base[0] - 0.07, base[1]), 0.14, 0.045,
                          fc="#b5b4b0", ec="none", zorder=3))

    # direction triad at the contact
    cx = ee[0]
    a.annotate("", xy=(cx + 0.17, 0.015), xytext=(cx + 0.02, 0.015),
               arrowprops=dict(arrowstyle="-|>", color="#3a3a38", lw=1.0))
    a.text(cx + 0.185, 0.015, r"$\mathbf{t}$", va="center")
    a.annotate("", xy=(cx - 0.10, 0.19), xytext=(cx - 0.10, 0.03),
               arrowprops=dict(arrowstyle="-|>", color="#3a3a38", lw=1.0))
    a.text(cx - 0.10, 0.215, r"$\mathbf{n}$", ha="center")

    # wrenches
    a.annotate("", xy=(ee[0] - 0.09, ee[1] + 0.22),
               xytext=(ee[0] - 0.01, ee[1] + 0.02),
               arrowprops=dict(arrowstyle="-|>", color=C_REF, lw=1.4))
    a.text(ee[0] - 0.11, ee[1] + 0.25, r"$\mathbf{w}_T$", color=C_REF)
    a.annotate("", xy=(ee[0] - 0.22, ee[1] + 0.09),
               xytext=(ee[0] + 0.0, ee[1] + 0.09),
               arrowprops=dict(arrowstyle="-|>", color=C_C4, lw=1.4))
    a.text(ee[0] - 0.34, ee[1] + 0.10, r"$\mathbf{w}_H$", color=C_C4)
    a.annotate("", xy=(ee[0] + 0.24, ee[1] - 0.028),
               xytext=(ee[0] + 0.07, ee[1] - 0.028),
               arrowprops=dict(arrowstyle="-|>", color=C_C3, lw=1.2))
    a.text(ee[0] + 0.115, ee[1] - 0.085, r"$v_t^d$", color=C_C3)
    a.text(0.72, 0.42, r"$F_n^d = 5\,$N", fontsize=7.5)
    a.text(0.72, 0.30, r"$v_t^d = 0.05\,$m/s", fontsize=7.5)
    a.text(base[0] + 0.13, base[1] + 0.05, r"$q_1$", fontsize=7.5)
    a.text(elbow[0] + 0.05, elbow[1] - 0.11, r"$q_2$", fontsize=7.5)

    a.set_xlim(-0.25, 1.15)
    a.set_ylim(-0.13, 0.95)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    letter(a, "(a)")

    # ---- (b) pipeline block diagram ----
    b.set_xlim(0, 10)
    b.set_ylim(0, 6.2)
    b.axis("off")
    letter(b, "(b)")

    def box(x, y, w, h, text, fc="#f4f3f0", ec="#8a8985"):
        b.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8, zorder=3))
        b.text(x + w / 2, y + h / 2, text, ha="center", va="center",
               fontsize=6.6, zorder=4)

    def arrow(x0, y0, x1, y1, text="", tx=0.0, ty=0.12, color="#3a3a38"):
        b.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                    arrowstyle="-|>", mutation_scale=7,
                                    color=color, lw=0.9, zorder=2))
        if text:
            b.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text,
                   fontsize=6.4, ha="center", color=color)

    box(0.1, 4.6, 2.3, 1.2, "nominal controller\n(PI force + PD track)")
    box(3.6, 4.6, 2.4, 1.2, "task-weighted QP\n(7 hard rows, OSQP)")
    box(6.9, 4.6, 2.3, 1.2,
        r"$\tau=\tau_0+J_P^{\top}B_{tn}u$" + "\nMuJoCo plant")
    box(6.9, 2.2, 2.3, 1.1, "task-contact predictor\n" + r"$\widehat{w}_T$")
    box(3.6, 2.2, 2.4, 1.1,
        "residual\n" + r"$w_R=w_{\mathrm{meas}}-\widehat{w}_T$")
    box(0.1, 2.2, 2.3, 1.1, "energy ledgers\n" + r"$E_H,\;E_R$")
    box(0.1, 0.2, 4.6, 1.1,
        "reference governor (optional)\nfreeze / rebase / resume "
        + r"$x_g,v_g$")

    arrow(2.4, 5.2, 3.6, 5.2, r"$u_{\mathrm{nom}}$")
    arrow(6.0, 5.2, 6.9, 5.2, r"$u$")
    arrow(8.05, 4.6, 8.05, 3.3, r"$w_T,\,v_P$", tx=0.82, ty=0.0)
    arrow(6.9, 2.75, 6.0, 2.75, "")
    b.text(6.45, 3.02, r"$+\,w_H$", fontsize=6.4, color=C_C4, ha="center")
    arrow(3.6, 2.75, 2.4, 2.75, r"$p_H,\,p_R$", ty=0.18)
    arrow(1.25, 3.3, 1.25, 4.6, "constraints\n" + r"$E_i\geq E_{i,\min}$",
          tx=1.08, ty=0.0)
    # governor -> nominal controller (reference shaping, task layer)
    b.add_patch(FancyArrowPatch((3.0, 1.3), (3.0, 4.85),
                                arrowstyle="-", color="#3a3a38", lw=0.9))
    arrow(3.0, 4.85, 2.4, 4.85, "")
    b.text(3.12, 3.9, r"$x_g,\,v_g$", fontsize=6.4)
    # plant -> governor human-detection feedback (gray, task layer only)
    b.add_patch(FancyArrowPatch((9.6, 4.55), (9.6, 0.75),
                                arrowstyle="-", color="#b5b4b0", lw=0.9))
    arrow(9.6, 0.75, 4.7, 0.75, "", color="#b5b4b0")
    b.text(7.15, 0.9, r"detected $\|w_H\|$ (task layer only)",
           fontsize=6.2, color="#8a8985", ha="center")

    save(fig, "fig1_overview.pdf")


# ---------------------------------------------------------------------------
# Fig. 2 — Test A: false depletion under nominal sliding (C1 vs C3)
# ---------------------------------------------------------------------------

def fig2():
    c0 = load("A_exact_no_human", "C0_nominal")
    c1 = load("A_exact_no_human", "C1_whole_port_scalar")
    c3 = load("A_exact_no_human", "C3_dual_ledger_qp")
    t = c3["time"]

    # first ledger-driven intervention of C1
    act = c1["sp_active"][:, :5].any(axis=1) & c1["sp_active_any"].astype(bool)
    t_act = t[np.flatnonzero(act)[0]]

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 4.4), sharex=True)

    ax = axes[0]
    ax.plot(t, c1["sp_e_w"], color=C_C1, label=r"$E_W$ (C1)")
    ax.plot(t, c3["sp_e_r"], color=C_C3, label=r"$E_R$ (C3)")
    ax.plot(t, c3["sp_e_h"], color=C_AUX, label=r"$E_H$ (C3)")
    ax.axhline(0.02, color=C_LIM, lw=0.7, ls="--")
    ax.axhline(0.005, color=C_LIM, lw=0.7, ls=":")
    ax.text(11.8, 0.033, r"$E_{R,\min}$", fontsize=6.2, color=C_LIM,
            ha="right")
    ax.set_ylim(-0.34, 0.36)
    style(ax, "ledger energy [J]")
    ax.legend(frameon=False, loc="lower left", ncol=1)
    letter(ax, "(a)")

    ax = axes[1]
    ax.plot(t, c0["f_n"], color=C_C0, lw=0.9, label="C0")
    ax.plot(t, c1["f_n"], color=C_C1, lw=0.9, label="C1")
    ax.plot(t, c3["f_n"], color=C_C3, lw=0.9, label="C3")
    ax.plot(t, c3["f_n_desired"], color=C_REF, lw=0.8, ls="--",
            label=r"$F_n^d$")
    style(ax, r"$F_n$ [N]")
    ax.set_ylim(-0.5, 16)
    ax.legend(frameon=False, loc="upper left", ncol=4, columnspacing=1.0,
              bbox_to_anchor=(0.03, 1.0))
    letter(ax, "(b)")

    ax = axes[2]
    ax.plot(t, c0["ee_vel"][:, 0], color=C_C0, lw=0.9)
    ax.plot(t, c1["ee_vel"][:, 0], color=C_C1, lw=0.9)
    ax.plot(t, c3["ee_vel"][:, 0], color=C_C3, lw=0.9)
    ax.plot(t, c3["vx_desired"], color=C_REF, lw=0.8, ls="--")
    style(ax, r"$v_t$ [m/s]", "time [s]")
    ax.set_ylim(-0.06, 0.12)
    letter(ax, "(c)")

    for ax in axes:
        ax.axvline(t_act, color=C_C1, lw=0.7, ls=":")
    axes[0].annotate(f"C1 intervenes\n({t_act:.1f} s)",
                     xy=(t_act, 0.1), xytext=(t_act - 4.6, 0.16),
                     fontsize=6.4, color=C_C1,
                     arrowprops=dict(arrowstyle="->", color=C_C1, lw=0.7))
    fig.align_ylabels(axes)
    save(fig, "fig2_false_depletion.pdf")
    print(f"  fig2: C1 first ledger intervention at t = {t_act:.3f} s")


# ---------------------------------------------------------------------------
# Fig. 3 — Test B: mismatch stays in the residual port
# ---------------------------------------------------------------------------

def fig3():
    c3 = load("B_mismatch_no_human", "C3_dual_ledger_qp")
    c1 = load("B_mismatch_no_human", "C1_whole_port_scalar")
    t = c3["time"]

    def t_first(lg):
        act = (lg["sp_active"][:, :5].any(axis=1)
               & lg["sp_active_any"].astype(bool))
        idx = np.flatnonzero(act)
        return t[idx[0]] if idx.size else None

    ta_c1, ta_c3 = t_first(c1), t_first(c3)

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.2), sharex=True)

    ax = axes[0]
    ax.plot(t, c3["sp_e_h"], color=C_AUX, label=r"$E_H$")
    ax.plot(t, c3["sp_e_r"], color=C_C3, label=r"$E_R$")
    ax.axhline(0.02, color=C_LIM, lw=0.7, ls="--")
    ax.text(0.4, 0.035, r"$E_{R,\min}$", fontsize=6.2, color=C_LIM)
    ax.axvline(ta_c1, color=C_C1, lw=0.7, ls=":")
    ax.axvline(ta_c3, color=C_C3, lw=0.7, ls=":")
    ax.annotate(f"C1 whole-port\nactivates ({ta_c1:.2f} s)",
                xy=(ta_c1, 0.24), xytext=(ta_c1 + 1.2, 0.33), fontsize=6.4,
                color=C_C1,
                arrowprops=dict(arrowstyle="->", color=C_C1, lw=0.7))
    ax.annotate(f"C3 residual\nactivates ({ta_c3:.2f} s)",
                xy=(ta_c3, 0.05), xytext=(ta_c3 - 6.6, 0.12), fontsize=6.4,
                color=C_C3,
                arrowprops=dict(arrowstyle="->", color=C_C3, lw=0.7))
    ax.set_ylim(0, 0.4)
    style(ax, "ledger energy [J]")
    ax.legend(frameon=False, loc="upper right")
    letter(ax, "(a)")

    ax = axes[1]
    dt = float(c3["meta_timestep"])
    k = max(1, int(0.05 / dt))
    ker = np.ones(k) / k
    p_d = np.convolve(c3["sp_p_delta_act"], ker, mode="same")
    ax.plot(t, c3["sp_p_delta_act"], color=C_C2, lw=0.4, alpha=0.35)
    ax.plot(t, p_d, color=C_C2, label=r"$p_\Delta$ (50 ms mean)")
    ax.plot(t, c3["sp_p_h_act"], color=C_AUX, lw=0.9, label=r"$p_H$")
    ax.axhline(0.0, color=C_LIM, lw=0.5)
    style(ax, "port power [W]", "time [s]")
    ax.set_ylim(-0.09, 0.05)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    letter(ax, "(b)")

    fig.align_ylabels(axes)
    save(fig, "fig3_mismatch_residual.pdf")
    print(f"  fig3: first activation C1 = {ta_c1:.3f} s, C3 = {ta_c3:.3f} s")


# ---------------------------------------------------------------------------
# Fig. 4 — Test C: human power / energy limiting (C0 vs C2 vs C3)
# ---------------------------------------------------------------------------

def fig4():
    runs = {"C0": ("C0_nominal", C_C0), "C2": ("C2_residual_qp", C_C2),
            "C3": ("C3_dual_ledger_qp", C_C3)}
    logs = {k: load("C_oracle_blocking", m) for k, (m, _) in runs.items()}
    t = logs["C3"]["time"]

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 4.4), sharex=True)

    ax = axes[0]
    for k, (_, c) in runs.items():
        ax.plot(t, -logs[k]["sp_p_h_act"], color=c, lw=0.9, label=k)
    ax.axhline(0.10, color=C_LIM, lw=0.8, ls="--")
    ax.text(0.4, 0.107, r"$P_H^{\max}=0.10$ W", fontsize=6.2, color=C_LIM)
    style(ax, r"$-p_H$ [W]")
    ax.set_ylim(-0.12, 0.19)
    ax.legend(frameon=False, loc="upper right", ncol=3)
    letter(ax, "(a)")

    ax = axes[1]
    for k, (_, c) in runs.items():
        w = wh_out(logs[k])
        ax.plot(t, w, color=c, lw=0.9)
        ax.annotate(f"{w[-1]:.3f} J", xy=(t[-1], w[-1]),
                    xytext=(-2, 4), textcoords="offset points",
                    fontsize=6.2, color=c, ha="right")
    ax.axhline(0.045, color=C_LIM, lw=0.8, ls="--")
    ax.text(0.4, 0.06, r"$E_H(0)-E_{H,\min}=0.045$ J", fontsize=6.2,
            color=C_LIM)
    style(ax, r"$W_H^{\mathrm{out}}$ [J]")
    letter(ax, "(b)")

    ax = axes[2]
    for k, (_, c) in runs.items():
        ax.plot(t, logs[k]["sp_e_h"], color=c, lw=0.9)
    ax.axhline(0.005, color=C_LIM, lw=0.8, ls="--")
    ax.text(0.4, 0.02, r"$E_{H,\min}$", fontsize=6.2, color=C_LIM)
    style(ax, r"$E_H$ [J]", "time [s]")
    letter(ax, "(c)")

    for ax in axes:
        shade_human(ax, t, logs["C3"]["f_h"])
    fig.align_ylabels(axes)
    save(fig, "fig4_human_limiting.pdf")


# ---------------------------------------------------------------------------
# Fig. 5 — Test D: combined mismatch + human decomposition (C3)
# ---------------------------------------------------------------------------

def fig5():
    lg = load("D_mismatch_blocking", "C3_dual_ledger_qp")
    t = lg["time"]
    dt = float(lg["meta_timestep"])
    k = max(1, int(0.05 / dt))
    ker = np.ones(k) / k
    mh = np.convolve(-lg["sp_p_h_act"], ker, mode="same")
    md = np.convolve(-lg["sp_p_delta_act"], ker, mode="same")
    mr = np.convolve(-lg["sp_p_r_act"], ker, mode="same")

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.2), sharex=True)

    ax = axes[0]
    ax.plot(t, lg["sp_e_r"], color=C_C3, label=r"$E_R$")
    ax.plot(t, lg["sp_e_h"], color=C_AUX, label=r"$E_H$")
    ax.axhline(0.02, color=C_LIM, lw=0.7, ls="--")
    ax.axhline(0.005, color=C_LIM, lw=0.7, ls=":")
    ax.text(0.4, 0.035, r"$E_{R,\min}$", fontsize=6.2, color=C_LIM)
    style(ax, "ledger energy [J]")
    ax.legend(frameon=False, loc="upper right")
    letter(ax, "(a)")

    ax = axes[1]
    ax.fill_between(t, 0.0, mh, color=C_AUX, alpha=0.55, lw=0,
                    label=r"$-p_H$")
    ax.fill_between(t, mh, mh + md, color=C_C2, alpha=0.55, lw=0,
                    label=r"$-p_\Delta$")
    ax.plot(t, mr, color="#3a3a38", lw=0.8, label=r"$-p_R$")
    ax.axhline(0.0, color=C_LIM, lw=0.5)
    style(ax, "extracted power [W]", "time [s]")
    ax.set_ylim(-0.05, 0.14)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    letter(ax, "(b)")

    for ax in axes:
        shade_human(ax, t, lg["f_h"])
    fig.align_ylabels(axes)
    save(fig, "fig5_combined_decomposition.pdf")


# ---------------------------------------------------------------------------
# Fig. 6 — Test E: QP authority vs scalar attenuation (C3 vs C4)
# ---------------------------------------------------------------------------

def fig6():
    c3 = load("E_oracle_oblique", "C3_dual_ledger_qp")
    c4 = load("E_oracle_oblique", "C4_dual_ledger_scalar")
    t = c3["time"]

    # human-window F_n RMSE (matches the summary metric); phase 2 = SLIDE
    slide3 = c3["phase"] == 2
    hum = np.linalg.norm(c3["f_h"], axis=1) > 1e-12
    def rmse(lg):
        m = slide3 & hum
        return float(np.sqrt(np.mean((lg["f_n"][m] - lg["f_n_desired"][m]) ** 2)))
    r3, r4 = rmse(c3), rmse(c4)

    fig = plt.figure(figsize=(DOUBLE, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    ax.plot(t, c3["f_n"], color=C_C3, lw=0.9, label="C3")
    ax.plot(t, c4["f_n"], color=C_C4, lw=0.9, label="C4")
    ax.plot(t, c3["f_n_desired"], color=C_REF, lw=0.8, ls="--",
            label=r"$F_n^d$")
    shade_human(ax, t, c3["f_h"])
    ax.set_xlim(5.0, 10.5)
    ax.set_ylim(-0.3, 8.2)
    style(ax, r"$F_n$ [N]", "time [s]")
    ax.legend(frameon=False, loc="lower left", ncol=3, columnspacing=0.9)
    ax.text(0.98, 0.96,
            f"window RMSE:\nC3 {r3:.3f} N\nC4 {r4:.3f} N",
            transform=ax.transAxes, fontsize=6.4, va="top", ha="right")
    letter(ax, "(a)")

    for j, (lg, name, cc) in enumerate(
            [(c3, "C3", C_C3), (c4, "C4", C_C4)]):
        ax = fig.add_subplot(gs[j + 1])
        ax.plot(t, lg["sp_u_nom"][:, 0], color=C_C3, lw=0.7, ls="--",
                label=r"$u_{t,\mathrm{nom}}$")
        ax.plot(t, lg["sp_u"][:, 0], color=C_C3, lw=1.1, label=r"$u_t$")
        ax.plot(t, lg["sp_u_nom"][:, 1], color=C_C2, lw=0.7, ls="--",
                label=r"$u_{n,\mathrm{nom}}$")
        ax.plot(t, lg["sp_u"][:, 1], color=C_C2, lw=1.1, label=r"$u_n$")
        shade_human(ax, t, lg["f_h"])
        ax.set_xlim(5.0, 10.5)
        ax.set_ylim(-42, 62)
        style(ax, "command [N]" if j == 0 else "", "time [s]")
        ax.set_title(f"{name} commands", fontsize=7.5, color=cc, pad=2)
        if j == 0:
            ax.legend(frameon=False, loc="upper left", ncol=2,
                      columnspacing=0.8, handlelength=1.6)
        letter(ax, "(b)" if j == 0 else "(c)")

    save(fig, "fig6_qp_vs_scalar.pdf")
    print(f"  fig6: window RMSE C3 = {r3:.3f} N, C4 = {r4:.3f} N")


# ---------------------------------------------------------------------------
# Fig. 7 — Test F: ledger charging, capping, discharge
# ---------------------------------------------------------------------------

def fig7():
    lg = load("F_helping_blocking", "C3_dual_ledger_qp")
    t = lg["time"]

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.2), sharex=True)

    ax = axes[0]
    ax.plot(t, lg["sp_e_h"], color=C_AUX, label=r"$E_H$")
    ax.axhline(0.08, color=C_LIM, lw=0.8, ls="--")
    ax.axhline(0.005, color=C_LIM, lw=0.8, ls="--")
    ax.text(0.3, 0.0855, r"$E_{H,\max}=0.08$ J", fontsize=6.2, color=C_LIM)
    ax.text(0.3, 0.010, r"$E_{H,\min}=0.005$ J", fontsize=6.2, color=C_LIM)
    ax.text(6.15, 0.062, "helping", fontsize=6.4, ha="center",
            color="#3f7a63")
    ax.text(10.9, 0.062, "blocking", fontsize=6.4, ha="center",
            color="#a0577e")
    style(ax, r"$E_H$ [J]")
    ax.set_ylim(-0.002, 0.097)
    letter(ax, "(a)")

    ax = axes[1]
    ax.plot(t, lg["sp_p_h_act"], color=C_AUX)
    ax.axhline(0.0, color=C_LIM, lw=0.5)
    ax.axhline(-0.10, color=C_LIM, lw=0.7, ls="--")
    ax.text(0.3, -0.093, r"$-P_H^{\max}$", fontsize=6.2, color=C_LIM,
            va="top")
    style(ax, r"$p_H$ [W]", "time [s]")
    letter(ax, "(b)")

    for ax in axes:
        shade_human(ax, t, lg["f_h"], split_sign=True)
    fig.align_ylabels(axes)
    save(fig, "fig7_charge_discharge.pdf")


# ---------------------------------------------------------------------------
# Fig. 8 — reference-governor ablation (iteration-2 comparison runs)
# ---------------------------------------------------------------------------

def fig8():
    off = dict(np.load(os.path.join(GOV, "governor_off", "log.npz")))
    on = dict(np.load(os.path.join(GOV, "governor_on", "log.npz")))
    t = off["time"]
    t_rel = 9.0

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 4.4), sharex=True)

    ax = axes[0]
    ax.plot(t, off["ee_vel"][:, 0], color=C_C0, lw=0.9, label="governor off")
    ax.plot(t, on["ee_vel"][:, 0], color=C_C3, lw=0.9, label="governor on")
    ax.axhline(0.05, color=C_REF, lw=0.7, ls="--")
    style(ax, r"$v_t$ [m/s]")
    ax.set_ylim(-0.06, 0.75)
    ax.legend(frameon=False, loc="upper left")
    letter(ax, "(a)")

    ax = axes[1]
    ax.plot(t, off["f_n"], color=C_C0, lw=0.9)
    ax.plot(t, on["f_n"], color=C_C3, lw=0.9)
    ax.axhline(5.0, color=C_REF, lw=0.7, ls="--")
    style(ax, r"$F_n$ [N]")
    ax.set_ylim(-0.5, 26)
    letter(ax, "(b)")

    ax = axes[2]
    ax.plot(t, off["ee_pos"][:, 0], color=C_C0, lw=0.9, label=r"$x$ (off)")
    ax.plot(t, off["sp_x_ref_original"], color=C_C0, lw=0.7, ls=":",
            label=r"$x_d$ (time-indexed)")
    ax.plot(t, on["ee_pos"][:, 0], color=C_C3, lw=0.9, label=r"$x$ (on)")
    ax.plot(t, on["x_desired"], color=C_C3, lw=0.7, ls="--",
            label=r"$x_g$ (governed)")
    style(ax, r"$x$ [m]", "time [s]")
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.9)
    letter(ax, "(c)")

    for ax in axes:
        shade_human(ax, t, off["f_h"])
        ax.axvline(t_rel, color=C_LIM, lw=0.6, ls=":")
        ax.set_xlim(4.5, 12.0)
    axes[0].text(t_rel + 0.06, 0.66, "release", fontsize=6.2, color=C_LIM)
    fig.align_ylabels(axes)
    save(fig, "fig8_governor.pdf")
    r = (t >= t_rel)
    print(f"  fig8: peak v_t post-release off = {off['ee_vel'][r,0].max():.3f}"
          f" on = {on['ee_vel'][r,0].max():.3f}; "
          f"peak F_n off = {off['f_n'][r].max():.2f} on = {on['f_n'][r].max():.2f}")


# ---------------------------------------------------------------------------
# Fig. 9 — friction-model sweep
# ---------------------------------------------------------------------------

def fig9():
    rows = list(csv.DictReader(open(os.path.join(R1, "sweep", "sweep.csv"))))
    def get(with_h):
        sel = [r for r in rows if r["with_human"] == str(with_h)]
        sel.sort(key=lambda r: float(r["mu_hat"]))
        mu = [float(r["mu_hat"]) for r in sel]
        emin = [float(r["e_r_min_raw"]) for r in sel]
        tact = [float(r["t_first_ledger_active"])
                if r["t_first_ledger_active"] else np.nan for r in sel]
        rfn = [float(r["rmse_fn"]) for r in sel]
        return mu, emin, tact, rfn

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 1.9))
    fams = [(False, "no human", C_C3, "o"), (True, "blocking human",
                                             C_C4, "s")]
    for wh, lab, c, m in fams:
        mu, emin, tact, rfn = get(wh)
        axes[0].plot(mu, emin, marker=m, ms=3.5, color=c, label=lab)
        axes[1].plot(mu, tact, marker=m, ms=3.5, color=c, label=lab)
        axes[2].plot(mu, rfn, marker=m, ms=3.5, color=c, label=lab)
    axes[0].axhline(0.02, color=C_LIM, lw=0.7, ls="--")
    axes[0].text(0.302, 0.045, r"$E_{R,\min}$", fontsize=6.2, color=C_LIM)
    style(axes[0], r"$\min E_R$ [J]", r"$\widehat{\mu}$")
    style(axes[1], "first ledger activation [s]", r"$\widehat{\mu}$")
    style(axes[2], r"RMSE $F_n$ [N]", r"$\widehat{\mu}$")
    axes[1].annotate("no activation for\n" + r"$\widehat{\mu}\geq0.25$",
                     xy=(0.25, 14.0), fontsize=6.2, color=C_C3)
    axes[1].set_ylim(0, 17)
    axes[0].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.set_xticks([0.20, 0.25, 0.30, 0.35])
        letter(ax, "(%s)" % "abc"[list(axes).index(ax)])
    save(fig, "fig9_sweep.pdf")


# ---------------------------------------------------------------------------
# Table I — summary metrics -> LaTeX
# ---------------------------------------------------------------------------

def table1():
    rows = json.load(open(os.path.join(R1, "summary.json")))
    mode_short = {
        "C0_nominal": "C0", "C1_whole_port_scalar": "C1",
        "C2_residual_qp": "C2", "C3_dual_ledger_qp": "C3",
        "C4_dual_ledger_scalar": "C4",
    }

    def f(x, nd=3, none="--"):
        if x is None:
            return none
        return f"{x:.{nd}f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Summary metrics for all controller modes and tests. "
        r"RMSE values are over the sliding phase; $-p_H^{\max}$ is the peak "
        r"instantaneous power delivered to the human port; "
        r"$W_H^{\mathrm{out}}$ is the maximum cumulative human-directed "
        r"output energy; $\min E_H$, $\min E_R$ are raw ledger minima "
        r"(discharge is never clamped); ``act.'' is the fraction of "
        r"in-contact steps on which the filter modified the command; "
        r"$t_{\mathrm{act}}$ is the first ledger-driven intervention; "
        r"``fail'' counts QP solve failures handled by the bounded "
        r"emergency fallback.}",
        r"\label{tab:summary}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Test & Mode & RMSE $F_n$ & RMSE $v_t$ & $-p_H^{\max}$ & "
        r"$W_H^{\mathrm{out}}$ & $\min E_H$ & $\min E_R$ & act. & "
        r"$t_{\mathrm{act}}$ & fail \\",
        r" & & [N] & [m/s] & [W] & [J] & [J] & [J] & [\%] & [s] & \\",
        r"\midrule",
    ]
    last_scen = None
    for r in rows:
        scen = r["scenario"]
        if last_scen is not None and scen != last_scen:
            lines.append(r"\addlinespace[2pt]")
        last_scen = scen
        lines.append(" & ".join([
            scen,
            mode_short.get(r["mode"], r["mode"]),
            f(r["rmse_fn"], 3),
            f(r["rmse_vt"], 4),
            f(max(0.0, r["p_h_peak"]), 3),
            f(max(0.0, r["w_h_max"]), 3),
            f(r["e_h_min_raw"], 3),
            f(r["e_r_min_raw"], 3),
            f(100.0 * r["active_fraction"], 2),
            f(r.get("t_first_ledger_active"), 2),
            str(r["n_infeasible"]),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path = os.path.join(os.path.dirname(OUT), "table_summary.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", path)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    table1()
