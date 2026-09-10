"""Generate the publication figures (Figs. 1--8) and summary table.

Data sources
------------
* ALL result figures and Table I use the current robust (iteration-2)
  result logs in results_passivation_iter2/ — the dataset produced by
  `python -m mujoco_sliding.experiments --all --sweep --governor-compare`
  with the robust prediction-error-bounded rows, the whole-port QP baseline
  C1_whole_port_qp, and the safe-anchor scalar C4_dual_ledger_safe_scalar.
  No iteration-1 data is mixed in.
* In the report, C3 denotes the dual-ledger QP without the governor and C4
  denotes the same QP with the governor. The implementation's historical
  C4_dual_ledger_safe_scalar mode is a separate diagnostic and does not
  appear in the report.
* Fig. 1 pairs the planar schematic with a MuJoCo rendering of an existing
  logged state. Fig. 2 shows the controller architecture at full width.

All figures are vector PDFs sized for a two-column paper (single column
3.4 in, double column 7.0 in). Run from the repository root:

    python paper/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                # for mujoco_sliding
R1 = os.path.join(ROOT, "results_passivation_iter2")    # current robust logs
GOV = os.path.join(R1, "governor_compare")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

MODE_C1 = "C1_whole_port_qp"
MODE_C4 = "C4_dual_ledger_safe_scalar"


def human_bounds(key: str):
    """(rise start, full-force start, fall onset) of a scenario's human
    pulse, derived from its HumanForceConfig — never hard-coded."""
    from mujoco_sliding.experiments import SCENARIOS

    hp = SCENARIOS[key].human
    return (hp.t_start, hp.t_start + hp.t_rise,
            hp.t_start + hp.t_rise + hp.t_hold)


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


def label_blocking_phase(ax, t, f_h):
    """Label the shaded interval for a tangential blocking interaction."""
    blocking = f_h[:, 0] < -1e-12
    if not blocking.any():
        return
    idx = np.flatnonzero(blocking)
    t_mid = 0.5 * (t[idx[0]] + t[idx[-1]])
    ax.text(t_mid, 1.035, "human blocking phase", fontsize=6.2,
            color="#a0577e", ha="center", va="bottom",
            transform=ax.get_xaxis_transform())


def wh_out(lg):
    dt = float(lg["meta_timestep"])
    return -np.cumsum(lg["sp_p_h_act"]) * dt


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Fig. 1 — planar task schematic + MuJoCo snapshot
# ---------------------------------------------------------------------------

def render_simulation_snapshot():
    """Render an existing Test-C state; no simulation is rerun."""
    import mujoco

    from mujoco_sliding.config import MODEL_XML_PATH

    lg = load("C_oracle_blocking", "C3_dual_ledger_qp")
    idx = int(np.argmin(np.abs(lg["time"] - 7.0)))
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    data.qpos[:] = lg["qpos"][idx]
    data.qvel[:] = lg["qvel"][idx]
    data.ctrl[:] = lg["ctrl"][idx]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.55, 0.0, 0.36]
    camera.distance = 1.65
    camera.elevation = -8
    camera.azimuth = 90
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    renderer.update_scene(data, camera=camera, scene_option=options)
    frame = renderer.render().copy()
    renderer.close()

    snapshot_path = os.path.join(OUT, "fig1_simulation_snapshot.png")
    plt.imsave(snapshot_path, frame)
    print("wrote", snapshot_path)
    return frame, float(lg["time"][idx])


def fig1():
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(DOUBLE, 2.35), width_ratios=[1.0, 1.45])

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

    # ---- (b) actual MuJoCo visualization at a logged blocking state ----
    frame, t_snapshot = render_simulation_snapshot()
    b.imshow(frame)
    b.axis("off")
    b.text(0.018, 0.96, "(b)", transform=b.transAxes, color="white",
           fontsize=8, fontweight="bold", va="top",
           bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none",
                     alpha=0.55))
    b.text(0.018, 0.06, f"MuJoCo snapshot, $t={t_snapshot:.1f}$ s",
           transform=b.transAxes, color="white", fontsize=6.8,
           bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none",
                     alpha=0.55))
    # The scripted human force is not a rendered geom; show its direction as
    # a publication overlay at the contact pad.
    b.annotate("", xy=(0.37, 0.29), xytext=(0.50, 0.29),
               xycoords="axes fraction", textcoords="axes fraction",
               arrowprops=dict(arrowstyle="-|>", color=C_C4, lw=1.5))
    b.text(0.405, 0.33, r"$\mathbf{w}_H$", transform=b.transAxes,
           color=C_C4, fontsize=7.2, ha="center")
    fig.subplots_adjust(wspace=0.06)

    save(fig, "fig1_setup.pdf")


# ---------------------------------------------------------------------------
# Fig. 2 — controller architecture (full-width block diagram)
# ---------------------------------------------------------------------------

def fig2():
    fig, b = plt.subplots(figsize=(DOUBLE, 3.25))
    b.set_xlim(0, 10)
    b.set_ylim(0, 6.4)
    b.axis("off")

    def box(x, y, w, h, text, fc="#f4f3f0", ec="#8a8985", fs=8.0):
        b.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8, zorder=3))
        b.text(x + w / 2, y + h / 2, text, ha="center", va="center",
               fontsize=fs, zorder=4)

    def arrow(x0, y0, x1, y1, text="", tx=0.0, ty=0.12, color="#3a3a38"):
        b.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                    arrowstyle="-|>", mutation_scale=9,
                                    color=color, lw=1.0, zorder=2))
        if text:
            b.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text,
                   fontsize=7.6, ha="center", color=color)

    box(0.1, 4.6, 2.7, 1.2, "nominal controller\nPI force + PD track")
    box(3.8, 4.6, 2.4, 1.2, "task-weighted QP\nOSQP, 7 rows")
    box(7.0, 4.6, 2.6, 1.2,
        r"$\tau=\tau_0+J_P^{\top}B_{tn}u$" + "\nMuJoCo plant")
    box(7.0, 2.2, 2.6, 1.1, "task-contact\npredictor " + r"$\widehat{w}_T$")
    box(3.8, 2.2, 2.4, 1.1,
        "residual\n" + r"$w_R=w_{\mathrm{meas}}-\widehat{w}_T$")
    box(0.1, 2.2, 2.7, 1.1, "energy ledgers\n" + r"$E_H,\;E_R$")
    box(0.1, 0.2, 4.6, 1.1,
        "reference governor (optional)\nfreeze / rebase / resume "
        + r"$x_g,v_g$")

    arrow(2.8, 5.2, 3.8, 5.2, r"$u_{\mathrm{nom}}$")
    arrow(6.2, 5.2, 7.0, 5.2, r"$u$")
    arrow(8.3, 4.6, 8.3, 3.3, r"$w_T,\,v_P$", tx=0.78, ty=0.0)
    arrow(7.0, 2.75, 6.2, 2.75, "")
    b.text(6.6, 3.05, r"$+\,w_H$", fontsize=7.6, color=C_C4, ha="center")
    arrow(3.8, 2.75, 2.8, 2.75, r"$p_H,\,p_R$", ty=0.18)
    arrow(1.25, 3.3, 1.25, 4.6, "constraints\n" + r"$E_i\geq E_{i,\min}$",
          tx=0.92, ty=0.0)
    # governor -> nominal controller (reference shaping, task layer)
    b.add_patch(FancyArrowPatch((3.3, 1.3), (3.3, 4.85),
                                arrowstyle="-", color="#3a3a38", lw=0.9))
    arrow(3.3, 4.85, 2.8, 4.85, "")
    b.text(3.42, 3.9, r"$x_g,\,v_g$", fontsize=7.6)
    # plant -> governor human-detection feedback (gray, task layer only)
    b.add_patch(FancyArrowPatch((9.8, 4.55), (9.8, 0.75),
                                arrowstyle="-", color="#b5b4b0", lw=0.9))
    arrow(9.8, 0.75, 4.7, 0.75, "", color="#b5b4b0")
    b.text(7.25, 0.92, r"detected $\|w_H\|$ (task layer only)",
           fontsize=7.2, color="#8a8985", ha="center")

    save(fig, "fig2_architecture.pdf")


# ---------------------------------------------------------------------------
# Fig. 3 — Test A: false depletion under nominal sliding (C1 vs C3)
# ---------------------------------------------------------------------------

def fig3():
    c0 = load("A_exact_no_human", "C0_nominal")
    c1 = load("A_exact_no_human", MODE_C1)
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
    ax.text(0.35, 0.0265, r"$E_{R,\min}$", fontsize=6.2, color=C_LIM,
            va="bottom")
    ax.set_ylim(-0.02, 0.36)
    style(ax, "ledger energy [J]")
    ax.legend(frameon=False, loc="center left", ncol=1)
    letter(ax, "(a)")

    ax = axes[1]
    ax.plot(t, c0["f_n"], color=C_C0, lw=0.9, label="C0")
    ax.plot(t, c1["f_n"], color=C_C1, lw=0.9, label="C1")
    ax.plot(t, c3["f_n"], color=C_C3, lw=0.9, label="C3")
    ax.plot(t, c3["f_n_desired"], color=C_REF, lw=0.8, ls="--",
            label=r"$F_n^d$")
    style(ax, r"$F_n$ [N]")
    f_hi = max(c0["f_n"].max(), c1["f_n"].max(), c3["f_n"].max())
    ax.set_ylim(-0.4, f_hi * 1.35)
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
                     xy=(t_act + 0.05, 0.07), xytext=(t_act + 2.4, 0.15),
                     fontsize=6.4, color=C_C1,
                     arrowprops=dict(arrowstyle="->", color=C_C1, lw=0.7))
    fig.align_ylabels(axes)
    save(fig, "fig3_false_depletion.pdf")
    print(f"  fig3: C1 first ledger intervention at t = {t_act:.3f} s")


# ---------------------------------------------------------------------------
# Fig. 4 — Test B: mismatch stays in the residual port
# ---------------------------------------------------------------------------

def fig4():
    c3 = load("B_mismatch_no_human", "C3_dual_ledger_qp")
    c1 = load("B_mismatch_no_human", MODE_C1)
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
    ax.text(0.35, 0.0265, r"$E_{R,\min}$", fontsize=6.2, color=C_LIM,
            va="bottom")
    ax.axvline(ta_c1, color=C_C1, lw=0.7, ls=":")
    ax.axvline(ta_c3, color=C_C3, lw=0.7, ls=":")
    ax.annotate(f"C1 whole-port\nactivates ({ta_c1:.2f} s)",
                xy=(ta_c1, 0.24), xytext=(ta_c1 + 1.2, 0.33), fontsize=6.4,
                color=C_C1,
                arrowprops=dict(arrowstyle="->", color=C_C1, lw=0.7))
    ax.annotate(f"C3 residual\nactivates ({ta_c3:.2f} s)",
                xy=(ta_c3, 0.05), xytext=(ta_c3 - 7.4, 0.068), fontsize=6.4,
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
    ax.margins(y=0.10)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    letter(ax, "(b)")

    fig.align_ylabels(axes)
    save(fig, "fig4_mismatch_residual.pdf")
    print(f"  fig4: first activation C1 = {ta_c1:.3f} s, C3 = {ta_c3:.3f} s")


# ---------------------------------------------------------------------------
# Fig. 5 — Test C: human power / energy limiting (C0 vs C2 vs C3)
# ---------------------------------------------------------------------------

def fig5():
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
    ax.legend(frameon=False, loc="lower right", ncol=3)
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
    ax.text(11.8, 0.03, r"$E_{H,\min}$", fontsize=6.2, color=C_LIM, ha="right")
    style(ax, r"$E_H$ [J]", "time [s]")
    letter(ax, "(c)")

    for ax in axes:
        shade_human(ax, t, logs["C3"]["f_h"])
    label_blocking_phase(axes[0], t, logs["C3"]["f_h"])
    fig.align_ylabels(axes)
    save(fig, "fig5_human_limiting.pdf")


# ---------------------------------------------------------------------------
# Fig. 6 — Test D: combined mismatch + human decomposition (C3)
# ---------------------------------------------------------------------------

def fig6():
    """Test D dual-ledger decomposition, clipped to the interaction region.

    The view ends at the fall onset t_start + t_rise + t_hold (the instant
    the human force begins to decrease), derived from the scenario config.
    This is a PRESENTATION clip only: the simulation, logs and all
    certificate quantities cover the complete run; removing the release
    transient simply keeps the port decomposition on a readable scale.
    """
    lg = load("D_mismatch_blocking", "C3_dual_ledger_qp")
    t = lg["time"]
    dt = float(lg["meta_timestep"])
    t_rise0, t_full, t_fall = human_bounds("D")
    view = t <= t_fall
    tv = t[view]
    k = max(1, int(0.05 / dt))
    ker = np.ones(k) / k
    mh = np.convolve(-lg["sp_p_h_act"], ker, mode="same")[view]
    md = np.convolve(-lg["sp_p_delta_act"], ker, mode="same")[view]
    mr = np.convolve(-lg["sp_p_r_act"], ker, mode="same")[view]

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.2), sharex=True)

    ax = axes[0]
    ax.plot(tv, lg["sp_e_r"][view], color=C_C3, label=r"$E_R$")
    ax.plot(tv, lg["sp_e_h"][view], color=C_AUX, label=r"$E_H$")
    ax.axhline(0.02, color=C_LIM, lw=0.7, ls="--")
    ax.axhline(0.005, color=C_LIM, lw=0.7, ls=":")
    ax.text(0.35, 0.0265, r"$E_{R,\min}$", fontsize=6.2, color=C_LIM,
            va="bottom")
    lo = min(0.0, lg["sp_e_h"][view].min())
    hi = lg["sp_e_r"][view].max()
    ax.set_ylim(lo - 0.02, hi + 0.05)
    style(ax, "ledger energy [J]")
    ax.legend(frameon=False, loc="upper right")
    letter(ax, "(a)")

    ax = axes[1]
    ax.fill_between(tv, 0.0, mh, color=C_AUX, alpha=0.55, lw=0,
                    label=r"$-p_H$")
    ax.fill_between(tv, mh, mh + md, color=C_C2, alpha=0.55, lw=0,
                    label=r"$-p_\Delta$")
    ax.plot(tv, mr, color="#3a3a38", lw=0.8, label=r"$-p_R$")
    ax.axhline(0.0, color=C_LIM, lw=0.5)
    # Axis limits from the CLIPPED data, so the (excluded) release
    # transient cannot dominate the scale.
    span = max(mr.max(), (mh + md).max())
    ax.set_ylim(min(0.0, mh.min()) - 0.012, span * 1.55)
    style(ax, "extracted power [W]", "time [s]")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    letter(ax, "(b)")

    for ax in axes:
        shade_human(ax, tv, lg["f_h"][view])
        ax.set_xlim(t[0], t_fall)
        ax.axvline(t_full, color=C_LIM, lw=0.5, ls=":")
    label_blocking_phase(axes[0], tv, lg["f_h"][view])
    fig.align_ylabels(axes)
    save(fig, "fig6_combined_decomposition.pdf")
    print(f"  fig6: clipped to [{t[0]:.1f}, {t_fall:.1f}] s "
          f"(fall onset); E_R {lg['sp_e_r'][0]:.3f} -> "
          f"{lg['sp_e_r'][view][-1]:.3f} J, "
          f"E_H {lg['sp_e_h'][0]:.3f} -> {lg['sp_e_h'][view][-1]:.4f} J")


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
    ax.text(1.6, 0.0855, r"$E_{H,\max}=0.08$ J", fontsize=6.2, color=C_LIM)
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
    ax.text(0.3, -0.107, r"$-P_H^{\max}$", fontsize=6.2, color=C_LIM,
            va="top")
    style(ax, r"$p_H$ [W]", "time [s]")
    letter(ax, "(b)")

    for ax in axes:
        shade_human(ax, t, lg["f_h"], split_sign=True)
    fig.align_ylabels(axes)
    save(fig, "fig7_charge_discharge.pdf")


# ---------------------------------------------------------------------------
# Fig. 8 — C3/C4 reference-governor comparison
# ---------------------------------------------------------------------------

def fig8():
    """Comparison D: report C3 (governor off) vs C4 (governor on).

    Shows the complete interaction and release interval (never clipped) and
    the full mechanism: force trigger, position rebasing vs the
    comparison-only time-indexed reference, velocity/command traces, and
    the (inactive) tank energies. Release-jump metrics are computed over
    the fixed 1.0-s window starting when the force leaves its hold
    (t = 8.5 s), matching the experiment driver.
    """
    off = dict(np.load(os.path.join(
        GOV, "C3_dual_ledger_qp", "governor_off", "log.npz")))
    on = dict(np.load(os.path.join(
        GOV, "C3_dual_ledger_qp", "governor_on", "log.npz")))
    t = off["time"]
    t_rel = 8.5      # force leaves its full-force hold (ramp-out begins)
    # CBF-triggered governor: engagement (DECEL/HOLD/dwell or legacy
    # INTERACT) and, for the stop-time-anchor variant, the anchor capture.
    engaged = np.isin(on["sp_gov_mode"], [1, 3, 4, 5])
    idx = np.flatnonzero(engaged)
    t_trig = float(t[idx[0]]) if idx.size else None
    anchor = on.get("sp_gov_anchor")
    t_anchor = None
    if anchor is not None and np.isfinite(anchor).any():
        t_anchor = float(t[int(np.flatnonzero(np.isfinite(anchor))[0])])

    def mark_periods(ax):
        # Period A: QP-only (human present, before the CBF trigger);
        # Period B: governed interaction (DECEL shaded, then anchored HOLD);
        # Period C: post-release resume.
        if t_trig is not None:
            ax.axvspan(6.0, t_trig, color="#eef3f8", zorder=0)   # A
            ax.axvline(t_trig, color=C_AUX, lw=0.9, ls="-.")
        if t_anchor is not None:
            ax.axvspan(t_trig, t_anchor, color="#f2eef8", zorder=0)  # DECEL
            ax.axvline(t_anchor, color=C_AUX, lw=0.8, ls=":")
        ax.axvline(t_rel, color=C_LIM, lw=0.6, ls=":")

    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE, 5.4), sharex=True)

    ax = axes[0, 0]
    ax.plot(t, np.linalg.norm(off["f_h"], axis=1), color=C_AUX, lw=1.0)
    ax.axhline(0.1, color=C_LIM, lw=0.6, ls=":")
    ax.text(4.6, 2.55, r"$F_{\mathrm{detect}}=0.1$ N", fontsize=6.0,
            color=C_LIM)
    if t_trig is not None:
        lbl = f"CBF trigger {t_trig:.2f} s"
        if t_anchor is not None:
            lbl += f"\nanchor {t_anchor:.2f} s"
        ax.text(t_trig + 0.06, 1.35, lbl, fontsize=6.0, color=C_AUX)
    style(ax, r"$\|F_H\|$ [N]")
    letter(ax, "(a)")

    ax = axes[0, 1]
    ax.plot(t, off["sp_x_ref_original"], color=C_REF, lw=0.8, ls=":",
            label=r"$x_{\mathrm{ref,orig}}$ (comparison only)")
    ax.plot(t, off["ee_pos"][:, 0], color=C_C3, lw=0.9, ls="--",
            label=r"$x$ (C3)")
    ax.plot(t, on["ee_pos"][:, 0], color=C_C4, lw=1.0,
            label=r"$x$ (C4)")
    ax.plot(t, on["x_desired"], color=C_C4, lw=0.7, ls=":",
            label=r"$x_g$ (C4)")
    if t_anchor is not None:
        x_a = float(anchor[int(np.flatnonzero(np.isfinite(anchor))[0])])
        ax.axhline(x_a, color=C_AUX, lw=0.6, ls="--", alpha=0.7)
        ax.text(11.9, x_a - 0.018, r"$x_a$", fontsize=6.0, color=C_AUX,
                ha="right")
    style(ax, "position [m]")
    ax.legend(frameon=False, ncol=1, loc="upper left")
    letter(ax, "(b)")

    ax = axes[1, 0]
    ax.plot(t, off["ee_vel"][:, 0], color=C_C3, lw=0.9, ls="--",
            label=r"$v_t$ (C3)")
    ax.plot(t, on["ee_vel"][:, 0], color=C_C4, lw=1.0,
            label=r"$v_t$ (C4)")
    ax.plot(t, on["vx_desired"], color=C_C4, lw=0.7, ls=":",
            label=r"$v_g$ (C4)")
    ax.axhline(0.05, color=C_REF, lw=0.6, ls=":")
    style(ax, "velocity [m/s]")
    ax.legend(frameon=False, loc="upper left")
    letter(ax, "(c)")

    ax = axes[1, 1]
    ax.plot(t, off["sp_u"][:, 0], color=C_C3, lw=0.9, ls="--", label="C3")
    ax.plot(t, on["sp_u"][:, 0], color=C_C4, lw=1.0, label="C4")
    style(ax, r"$u_t$ [N]")
    ax.legend(frameon=False, loc="upper left")
    letter(ax, "(d)")

    ax = axes[2, 0]
    ax.plot(t, off["f_n"] - off["f_n_desired"], color=C_C3, lw=0.9,
            ls="--", label="C3")
    ax.plot(t, on["f_n"] - on["f_n_desired"], color=C_C4, lw=1.0,
            label="C4")
    ax.axhline(0.0, color=C_LIM, lw=0.5)
    style(ax, r"$F_n-F_n^d$ [N]", "time [s]")
    ax.legend(frameon=False, loc="upper left")
    letter(ax, "(e)")

    ax = axes[2, 1]
    ax.plot(t, off["sp_e_h"], color=C_C3, lw=0.9, ls="--",
            label=r"$E_H$ (C3)")
    ax.plot(t, on["sp_e_h"], color=C_C4, lw=1.0,
            label=r"$E_H$ (C4)")
    ax.plot(t, off["sp_e_r"], color=C_C3, lw=0.65, ls="--", alpha=0.55,
            label=r"$E_R$ (C3)")
    ax.plot(t, on["sp_e_r"], color=C_C4, lw=0.65, alpha=0.7,
            label=r"$E_R$ (C4)")
    ax.axhline(0.005, color=C_LIM, lw=0.6, ls=":")
    ax.axhline(0.02, color=C_LIM, lw=0.6, ls=":")
    style(ax, "tank energy [J]", "time [s]")
    ax.legend(frameon=False, ncol=2, loc="center left", fontsize=6.0)
    letter(ax, "(f)")

    for ax in axes.flat:
        shade_human(ax, t, off["f_h"])
        mark_periods(ax)
        ax.set_xlim(4.5, 12.0)
    axes[0, 0].text(t_rel + 0.06, 2.55, "release", fontsize=6.0,
                    color=C_LIM)
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    save(fig, "fig8_governor.pdf")

    # Release-jump metrics over [t_rel, t_rel + 1.0] (matches experiments).
    w = (t >= t_rel) & (t <= t_rel + 1.0)
    i_rel = int(np.argmin(np.abs(t - t_rel)))
    for name, lg in (("C3 (governor off)", off),
                     ("C4 (governor on)", on)):
        du = np.abs(np.diff(lg["sp_u"][w, 0])).max()
        print(f"  fig8 {name}: max|du_t|={du:.4f} N "
              f"peak|u_t|={np.abs(lg['sp_u'][w, 0]).max():.3f} N "
              f"peak|v_t|={np.abs(lg['ee_vel'][w, 0]).max():.4f} m/s "
              f"peak|Fn err|={np.abs(lg['f_n'][w] - lg['f_n_desired'][w]).max():.3f} N "
              f"debt={abs(lg['x_desired'][i_rel] - lg['ee_pos'][i_rel, 0]):.4f} m")


# ---------------------------------------------------------------------------
# Table I — summary metrics -> LaTeX
# ---------------------------------------------------------------------------

def table1():
    rows = json.load(open(os.path.join(R1, "summary.json")))
    mode_short = {
        "C0_nominal": "C0", MODE_C1: "C1",
        "C2_residual_qp": "C2", "C3_dual_ledger_qp": "C3",
    }

    scen_name = {"A": "exact", "B": "mismatch", "C": "block",
                 "D": "mis+block", "F": "help+block", "S": "stress"}

    def f(x, nd=3, none="--"):
        if x is None:
            return none
        return f"{x:.{nd}f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete-run summary. RMSE is evaluated during sliding; "
        r"``act.'' is the fraction of contact steps modified by the filter, "
        r"and ``fail'' counts solver failures requiring the bounded "
        r"fallback.}",
        r"\label{tab:summary}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Scenario & Mode & RMSE $F_n$ & RMSE $v_t$ & $-p_H^{\max}$ & "
        r"$W_H^{\mathrm{out}}$ & $\min E_H$ & $\min E_R$ & act. & "
        r"$t_{\mathrm{act}}$ & fail \\",
        r" & & [N] & [m/s] & [W] & [J] & [J] & [J] & [\%] & [s] & \\",
        r"\midrule",
    ]
    # This complete-run table covers the ungoverned C0--C3 configurations.
    # Paper C4 is the governed C3 run summarized in the text. The unrelated
    # implementation mode C4_dual_ledger_safe_scalar and scenario E remain
    # in the repository logs but are excluded here.
    rows = [r for r in rows
            if r["mode"] in mode_short and r["mode"] != MODE_C4
            and r["scenario"] != "E"]
    last_scen = None
    for r in rows:
        scen = r["scenario"]
        if last_scen is not None and scen != last_scen:
            lines.append(r"\addlinespace[2pt]")
        last_scen = scen
        rmse_fn = r["rmse_fn"]
        rmse_vt = r["rmse_vt"]
        lines.append(" & ".join([
            scen_name.get(scen, scen),
            mode_short.get(r["mode"], r["mode"]),
            f(rmse_fn, 3),
            f(rmse_vt, 4),
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
    table1()
