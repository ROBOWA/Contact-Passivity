"""Supplementary figures S1/S2: Tests D and F re-run as report configuration
C4, the C3 dual-ledger QP with its reference governor enabled. They are
counterparts of the paper's Fig. 6 (Test D) and Fig. 7 (Test F), which use
the governor-off C3 configuration.

These are supplementary/diagnostic: enabling the governor changes what the
two figures can show. Holding the fixed stop-time anchor brings the robot
to rest, and because the mismatch power is velocity-proportional
(p_Delta = F_Delta . v), the residual ledger also stops draining while the
robot is held — so the Test-D port attribution is less legible than in the
governor-off ablation (E_R ends at 0.145 J instead of 0.041 J). Unlike the
earlier continuous-rebase variant, the anchor prevents sustained retreat,
so E_H does NOT recharge to its cap: it dips to 0.013 J and stays there.
In Test F the helping phase is unchanged (the CBF trigger correctly ignores
a helping human — exactly one engagement, during the blocking pulse), while
the blocking discharge is arrested at 0.013 J instead of riding down to
0.006 J.

Styling is imported from make_figures so the supplementary figures match
the main ones exactly. Run from the repository root:

    python paper/make_supplementary_figures.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # make_figures (styling)
sys.path.insert(0, os.path.dirname(HERE))      # mujoco_sliding

import numpy as np
import matplotlib.pyplot as plt

import make_figures as mf
from mujoco_sliding.experiments import SCENARIOS, build_config
from mujoco_sliding.simulation import run_simulation


def governed_log(key: str) -> dict:
    """Run report C4: the C3 dual-ledger QP with its governor enabled."""
    cfg = build_config(SCENARIOS[key], "C3_dual_ledger_qp", governor=True)
    return run_simulation(cfg)


def trigger_time(log: dict):
    """Time the governor engages (CBF trigger), or None.

    Covers both variants: legacy INTERACT and the stop-time-anchor
    DECEL/HOLD/RELEASE_DWELL modes.
    """
    from mujoco_sliding.controller import GOV_ENGAGED_MODES

    idx = np.flatnonzero(np.isin(log["sp_gov_mode"],
                                 [int(m) for m in GOV_ENGAGED_MODES]))
    return float(log["time"][idx[0]]) if idx.size else None


def anchor_time(log: dict):
    """Time the stop-time anchor is captured (end of DECEL), or None."""
    a = log.get("sp_gov_anchor")
    if a is None or not np.isfinite(a).any():
        return None
    return float(log["time"][int(np.flatnonzero(np.isfinite(a))[0])])


def mark_events(ax, t_trig, t_anchor):
    """CBF trigger (dash-dot) and stop-time-anchor capture (dotted); the
    span between them is the DECEL phase."""
    if t_trig is not None:
        ax.axvline(t_trig, color=mf.C_AUX, lw=0.9, ls="-.")
    if t_anchor is not None:
        ax.axvspan(t_trig, t_anchor, color="#f2eef8", zorder=0)
        ax.axvline(t_anchor, color=mf.C_AUX, lw=0.8, ls=":")


def smooth(x, dt, win=0.05):
    k = max(1, int(win / dt))
    return np.convolve(x, np.ones(k) / k, mode="same")


# ---------------------------------------------------------------------------
# Supplementary Fig. S1 — Test D with the governor enabled
# (C4 counterpart of the paper's C3 Fig. 6)
# ---------------------------------------------------------------------------

def figS1(lg):
    t = lg["time"]
    dt = float(lg["meta_timestep"])
    t_trig = trigger_time(lg)
    t_anchor = anchor_time(lg)

    fig, axes = plt.subplots(2, 1, figsize=(mf.SINGLE, 3.4), sharex=True)

    ax = axes[0]
    ax.plot(t, lg["sp_e_r"], color=mf.C_C3, label=r"$E_R$")
    ax.plot(t, lg["sp_e_h"], color=mf.C_AUX, label=r"$E_H$")
    ax.axhline(0.02, color=mf.C_LIM, lw=0.7, ls="--")
    ax.axhline(0.005, color=mf.C_LIM, lw=0.7, ls=":")
    ax.text(0.35, 0.0225, r"$E_{R,\min}$", fontsize=6.2, color=mf.C_LIM,
            va="bottom")
    ax.set_ylim(-0.012, 0.335)
    mf.style(ax, "ledger energy [J]")
    ax.legend(frameon=False, loc="center left", ncol=1)
    mf.letter(ax, "(a)")
    # Event labels above the axes (blended transform: data-x, axes-y).
    if t_trig is not None:
        ax.text(t_trig, 1.04, f"CBF trigger {t_trig:.2f} s", fontsize=6.0,
                color=mf.C_AUX, ha="right",
                transform=ax.get_xaxis_transform())
    if t_anchor is not None:
        ax.text(t_anchor + 0.15, 1.04, f"anchor {t_anchor:.2f} s",
                fontsize=6.0, color=mf.C_AUX, ha="left",
                transform=ax.get_xaxis_transform())

    ax = axes[1]
    mh = smooth(-lg["sp_p_h_act"], dt)
    md = smooth(-lg["sp_p_delta_act"], dt)
    mr = smooth(-lg["sp_p_r_act"], dt)
    ax.fill_between(t, 0.0, mh, color=mf.C_AUX, alpha=0.55, lw=0,
                    label=r"$-p_H$")
    ax.fill_between(t, mh, mh + md, color=mf.C_C2, alpha=0.55, lw=0,
                    label=r"$-p_\Delta$")
    ax.plot(t, mr, color="#3a3a38", lw=0.8, label=r"$-p_R$")
    ax.axhline(0.0, color=mf.C_LIM, lw=0.5)
    ax.set_ylim(-0.062, 0.135)
    mf.style(ax, "extracted power [W]", "time [s]")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    mf.letter(ax, "(b)")

    for ax in axes:
        mf.shade_human(ax, t, lg["f_h"])
        mark_events(ax, t_trig, t_anchor)
    fig.align_ylabels(axes)
    mf.save(fig, "figS1_combined_governed.pdf")
    return t_trig, t_anchor


# ---------------------------------------------------------------------------
# Supplementary Fig. S2 — Test F with the governor enabled
# (C4 counterpart of the paper's C3 Fig. 7)
# ---------------------------------------------------------------------------

def figS2(lg):
    t = lg["time"]
    t_trig = trigger_time(lg)
    t_anchor = anchor_time(lg)

    fig, axes = plt.subplots(2, 1, figsize=(mf.SINGLE, 3.4), sharex=True)

    ax = axes[0]
    ax.plot(t, lg["sp_e_h"], color=mf.C_AUX, label=r"$E_H$")
    ax.axhline(0.08, color=mf.C_LIM, lw=0.8, ls="--")
    ax.axhline(0.005, color=mf.C_LIM, lw=0.8, ls="--")
    ax.text(1.6, 0.0855, r"$E_{H,\max}=0.08$ J", fontsize=6.2,
            color=mf.C_LIM)
    ax.text(0.3, 0.010, r"$E_{H,\min}=0.005$ J", fontsize=6.2,
            color=mf.C_LIM)
    ax.set_ylim(-0.002, 0.097)
    mf.style(ax, r"$E_H$ [J]")
    mf.letter(ax, "(a)")
    if t_trig is not None:
        lbl = f"CBF trigger {t_trig:.2f} s"
        if t_anchor is not None:
            lbl += f"\nanchor {t_anchor:.2f} s"
        ax.text(t_trig, 1.04, lbl, fontsize=6.0, color=mf.C_AUX,
                ha="right", va="bottom",
                transform=ax.get_xaxis_transform())
    ax.text(6.5, 0.062, "helping:\nno trigger", fontsize=6.0,
            color="#3f7a63", ha="center")

    ax = axes[1]
    ax.plot(t, lg["sp_p_h_act"], color=mf.C_AUX)
    ax.axhline(0.0, color=mf.C_LIM, lw=0.5)
    ax.axhline(-0.10, color=mf.C_LIM, lw=0.7, ls="--")
    ax.text(0.3, -0.107, r"$-P_H^{\max}$", fontsize=6.2, color=mf.C_LIM,
            va="top")
    mf.style(ax, r"$p_H$ [W]", "time [s]")
    mf.letter(ax, "(b)")

    for ax in axes:
        mf.shade_human(ax, t, lg["f_h"], split_sign=True)
        mark_events(ax, t_trig, t_anchor)
    fig.align_ylabels(axes)
    mf.save(fig, "figS2_charge_governed.pdf")
    return t_trig, t_anchor


if __name__ == "__main__":
    d = governed_log("D")
    t_d, a_d = figS1(d)
    print(f"  S1 (Test D, C4 governor on): trigger={t_d} anchor={a_d} "
          f"E_R end={d['sp_e_r'][-1]:.4f} J (min raw {d['sp_e_r_raw'].min():.4f}) "
          f"E_H min raw={d['sp_e_h_raw'].min():.4f} J "
          f"peak(-p_R)={np.max(-d['sp_p_r_act']):.4f} W "
          f"infeas={int(d['meta_n_infeasible'])}")

    f = governed_log("F")
    t_f, a_f = figS2(f)
    from mujoco_sliding.controller import GOV_ENGAGED_MODES
    eng = np.isin(f["sp_gov_mode"], [int(m) for m in GOV_ENGAGED_MODES])
    n_trig = int(eng[0] + np.sum(eng[1:] & ~eng[:-1]))
    print(f"  S2 (Test F, C4 governor on): trigger={t_f} anchor={a_f} "
          f"(events={n_trig}; helping phase never triggers) "
          f"E_H max={f['sp_e_h'].max():.4f} J "
          f"E_H min raw={f['sp_e_h_raw'].min():.4f} J "
          f"peak(-p_H)={np.max(-f['sp_p_h_act']):.4f} W "
          f"infeas={int(f['meta_n_infeasible'])}")
