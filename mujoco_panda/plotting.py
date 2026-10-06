"""Review plots for Panda mode comparisons."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "C0_nominal": "#777777", "C1_whole_port_qp": "#8e44ad",
    "C2_residual_qp": "#1677b8", "C3_dual_ledger_qp": "#d62728",
}
LABELS = {
    "C0_nominal": "C0 nominal", "C1_whole_port_qp": "C1 whole port",
    "C2_residual_qp": "C2 residual", "C3_dual_ledger_qp": "C3 residual + human",
}


def comparison(results: dict, path: str, title: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for mode, result in results.items():
        d, t = result.log, result.log["time"]
        color = COLORS.get(mode, "#d62728" if "on" in mode else "#777777")
        label = LABELS.get(mode, mode.replace("_", " "))
        axes[0, 0].plot(t, d["normal_force_filtered"], color=color, label=label, lw=1.3)
        axes[0, 1].plot(t, 1e3 * d["cart_error"], color=color, lw=1.3)
        axes[1, 0].plot(t, np.maximum(-d["p_human"], 0), color=color, lw=1.3)
        primary = d["e_whole"] if mode == "C1_whole_port_qp" else d["e_residual"]
        axes[1, 1].plot(t, primary, color=color, lw=1.3)
        if mode in ("C0_nominal", "C3_dual_ledger_qp") or "governor" in mode:
            axes[1, 1].plot(t, d["e_human"], color=color, lw=1.0, ls="--")
        axes[2, 0].plot(t, d["governed_progress"], color=color, lw=1.3)
        axes[2, 1].plot(t, d["qvel_prediction_error"], color=color, lw=1.0)
        human = np.linalg.norm(np.c_[d["human_fx"], d["human_fy"], d["human_fz"]], axis=1) > .05
        if np.any(human):
            lo, hi = t[human][0], t[human][-1]
            for ax in axes.flat:
                ax.axvspan(lo, hi, color="#ff6f91", alpha=.10, lw=0)
    axes[0, 0].axhline(5, color="black", lw=.8, ls=":")
    axes[0, 0].set_ylabel("filtered normal force [N]")
    axes[0, 1].set_ylabel("planar tracking error [mm]")
    axes[1, 0].set_ylabel("human output power [W]")
    axes[1, 1].set_ylabel("ledger energy [J]\n(C1 solid=whole; otherwise residual; dashed=human)")
    axes[2, 0].set_ylabel("task progress [m]")
    axes[2, 1].set_ylabel("qvel prediction error [rad/s]")
    for ax in axes[2]: ax.set_xlabel("time [s]")
    axes[0, 0].legend(ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sweep_plot(rows: list[dict], path: str) -> None:
    budget = np.array([r["whole_budget_J"] for r in rows])
    order = np.argsort(budget); budget = budget[order]
    output = np.array([r["human_output_J"] for r in rows])[order]
    tracking = 1e3 * np.array([r["cartesian_rmse_m"] for r in rows])[order]
    fallback = np.array([r["qp_fallback_steps"] for r in rows])[order]
    fig, ax = plt.subplots(figsize=(7, 4.2)); ax2 = ax.twinx()
    ax.plot(budget, output, "o-", label="human output energy", color="#d62728")
    ax2.plot(budget, tracking, "s--", label="tracking RMSE", color="#1677b8")
    for x, y, n in zip(budget, output, fallback):
        if n: ax.annotate(f"{n} fallback", (x, y), fontsize=7)
    ax.set(xlabel="whole-port usable budget [J]", ylabel="human output energy [J]")
    ax2.set_ylabel("planar tracking RMSE [mm]")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)
