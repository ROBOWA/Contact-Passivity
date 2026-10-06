"""Generate the figures for the Panda selective-passivation report."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import mujoco
import numpy as np

from mujoco_panda.simulation import load_model
from mujoco_panda.visualization import add_force_arrow, add_reference_marker


DATA = ROOT / "results_panda"
OUT = Path(__file__).resolve().parent / "figs"

COLORS = {
    "C0": "#666666",
    "C1": "#8e44ad",
    "C2": "#1677b8",
    "C3": "#d62728",
    "C4": "#e68613",
}


plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
})


def load(relative: str) -> dict[str, np.ndarray]:
    return dict(np.load(DATA / relative / "log.npz"))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)


def human_interval(log: dict[str, np.ndarray]) -> tuple[float, float] | None:
    force = np.c_[log["human_fx"], log["human_fy"], log["human_fz"]]
    mask = np.linalg.norm(force, axis=1) > 0.05
    if not np.any(mask):
        return None
    return float(log["time"][mask][0]), float(log["time"][mask][-1])


def shade_human(ax: plt.Axes, log: dict[str, np.ndarray]) -> None:
    interval = human_interval(log)
    if interval is not None:
        ax.axvspan(*interval, color="#ff6f91", alpha=0.15, lw=0,
                   label="human interaction")


def finish_axes(axes) -> None:
    for ax in np.asarray(axes).flat:
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)


def render_frame(relative: str, when: float, location: str) -> np.ndarray:
    log = load(relative)
    k = int(np.argmin(np.abs(log["time"] - when)))
    model, data, handles = load_model()
    data.qpos[handles.qpos_ids] = [log[f"q{i}"][k] for i in range(1, 8)]
    data.qvel[handles.dof_ids] = [log[f"qd{i}"][k] for i in range(1, 8)]
    data.ctrl[handles.actuator_ids] = [log[f"tau{i}"][k] for i in range(1, 8)]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=540, width=720)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    renderer.update_scene(data, camera="overview", scene_option=option)
    site = handles.tool_site if location == "tool" else handles.forearm_site
    force = np.array([log["human_fx"][k], log["human_fy"][k], log["human_fz"][k]])
    add_force_arrow(renderer.scene, data.site_xpos[site], force,
                    nominal=max(7.0, np.linalg.norm(force)))
    add_reference_marker(renderer.scene,
                         [log["x_des"][k], log["y_des"][k], log["z"][k]])
    image = renderer.render().copy()
    renderer.close()
    return image


def setup_figure() -> None:
    ee = render_frame("C_ee_oblique_blocking/C3_dual_ledger_qp", 4.70, "tool")
    forearm = render_frame("D_forearm_wiping/C3_dual_ledger_qp", 4.70, "forearm")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.15))
    for ax, image, title in zip(
            axes, (ee, forearm),
            ("(a) end-effector blocking", "(b) forearm interaction during wiping")):
        ax.imshow(image)
        ax.set_title(title, pad=4)
        ax.axis("off")
    fig.text(.5, .01, "Magenta: scripted human force; green: governed task setpoint.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.07, wspace=.03)
    save(fig, "fig1_setup.png")


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.25))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, color="#eef3f7", edge="#35566f"):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.04,rounding_size=0.08",
                               facecolor=color, edgecolor=edge, linewidth=1.1)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")

    def arrow(x1, y1, x2, y2, color="#36454f", style="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=1.1,
                                     linestyle=style, color=color))

    box(.2, 3.5, 1.35, .75, "task\nreference")
    box(2.0, 3.5, 1.45, .75, "C4 reference\ngovernor", "#fff2da", "#c27a00")
    box(3.95, 3.5, 1.65, .75, "force--motion\ncontroller")
    box(6.25, 3.5, 1.65, .75, "7-D torque QP", "#f9e8e8", "#a52a2a")
    box(8.55, 3.5, 1.35, .75, "Panda")
    box(10.5, 3.5, 1.25, .75, "table +\nhuman")
    for x1, x2 in ((1.55, 2.0), (3.45, 3.95), (5.6, 6.25),
                   (7.9, 8.55), (9.9, 10.5)):
        arrow(x1, 3.875, x2, 3.875)

    box(.55, .55, 1.55, .78, "measured task\ncontact $\\tau_T$")
    box(2.65, .55, 1.55, .78, "task model\n$\\widehat{\\tau}_T$")
    box(4.65, .55, 1.55, .78, "residual port\n$\\tau_R$")
    box(6.75, .55, 1.8, .78, "residual + human\nenergy ledgers", "#f9e8e8", "#a52a2a")
    box(9.35, .55, 2.0, .78, "known human\nport $\\tau_H$")
    arrow(2.1, .94, 2.65, .94)
    arrow(4.2, .94, 4.65, .94)
    arrow(6.2, .94, 6.75, .94)
    arrow(9.35, .94, 8.55, .94)
    arrow(7.65, 1.33, 7.1, 3.5, color="#a52a2a")
    arrow(9.35, .94, 7.9, 3.5, color="#a52a2a", style="--")
    arrow(9.18, 3.5, 1.32, 1.34, color="#607d8b", style="--")
    ax.text(6.0, 2.35, "prediction-error-robust power constraints",
            color="#8b1a1a", ha="center", fontsize=8)
    ax.text(2.72, 4.55, "C3 bypasses this box", ha="center", fontsize=8,
            color="#8a5a00")
    save(fig, "fig2_architecture.pdf")


def accounting_figure() -> None:
    exact_c1 = load("A_accurate_no_human/C1_whole_port_qp")
    exact_c3 = load("A_accurate_no_human/C3_dual_ledger_qp")
    mismatch = load("B_mismatch_no_human/C3_dual_ledger_qp")
    sweep = __import__("json").loads((DATA / "whole_port_sweep.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.65))
    axes[0].plot(exact_c1["time"], exact_c1["e_whole"], color=COLORS["C1"],
                 label="C1 whole ledger")
    axes[0].plot(exact_c3["time"], exact_c3["e_residual"], color=COLORS["C3"],
                 label="C3 residual ledger")
    axes[0].set(title="(a) accurate task model", xlabel="time [s]",
                ylabel="energy [J]", ylim=(0, .32))
    axes[0].legend(frameon=False, loc="lower left")

    axes[1].plot(mismatch["time"], mismatch["e_residual"], color=COLORS["C3"],
                 label="residual ledger")
    axes[1].plot(mismatch["time"], mismatch["e_human"], color=COLORS["C2"],
                 ls="--", label="human ledger")
    axes[1].set(title="(b) model mismatch, no human", xlabel="time [s]",
                ylabel="energy [J]", ylim=(0, .32))
    axes[1].legend(frameon=False, loc="center right")

    budget = np.array([row["whole_budget_J"] for row in sweep])
    fallback = np.array([row["qp_fallback_steps"] for row in sweep])
    output = np.array([row["human_output_J"] for row in sweep])
    order = np.argsort(budget); budget, fallback, output = (
        budget[order], fallback[order], output[order])
    axes[2].bar(budget, fallback, width=.045, color="#b99ac8", label="fallback steps")
    axes[2].set(title="(c) C1 budget sweep", xlabel="usable budget [J]",
                ylabel="fallback steps", ylim=(0, 2900))
    ax2 = axes[2].twinx()
    ax2.plot(budget, output, "o-", color=COLORS["C3"], label="human output")
    ax2.set_ylabel("human output [J]", color=COLORS["C3"])
    ax2.tick_params(axis="y", colors=COLORS["C3"])
    lines = [axes[2].patches[0], ax2.lines[0]]
    axes[2].legend(lines, ["fallback steps", "human output"], frameon=False,
                   loc="upper right")
    finish_axes(axes)
    ax2.grid(False); ax2.spines["top"].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig3_accounting.pdf")


def interaction_comparison(relative: str, modes: list[tuple[str, str]],
                           name: str, titles: tuple[str, str, str]) -> None:
    logs = [(label, load(f"{relative}/{mode}")) for label, mode in modes]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.55), sharex=True)
    for label, log in logs:
        color = COLORS[label]
        t = log["time"]
        axes[0].plot(t, log["human_output"], color=color, label=label, lw=1.35)
        axes[1].plot(t, np.maximum(-log["p_human"], 0), color=color, lw=1.25)
        axes[2].plot(t, 1e3 * log["cart_error"], color=color, lw=1.25)
    for ax in axes:
        shade_human(ax, logs[-1][1])
    axes[0].set_ylabel("cumulative human output [J]")
    axes[1].set_ylabel("human output power [W]")
    axes[2].set_ylabel("planar error [mm]")
    for ax, title in zip(axes, titles):
        ax.set_title(title); ax.set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:len(modes)], labels[:len(modes)], frameon=False)
    finish_axes(axes); fig.tight_layout(w_pad=1.0)
    save(fig, name)


def human_limiting_figure() -> None:
    interaction_comparison(
        "C_ee_oblique_blocking",
        [("C0", "C0_nominal"), ("C2", "C2_residual_qp"),
         ("C3", "C3_dual_ledger_qp")],
        "fig4_human_limiting.pdf",
        ("(a) transferred energy", "(b) instantaneous transfer",
         "(c) task tradeoff"),
    )


def forearm_figure() -> None:
    interaction_comparison(
        "D_forearm_wiping",
        [("C0", "C0_nominal"), ("C3", "C3_dual_ledger_qp")],
        "fig5_forearm.pdf",
        ("(a) transferred energy", "(b) forearm-port power",
         "(c) wiping error"),
    )


def governor_figure() -> None:
    off = load("governor_ablation/governor_off")
    on = load("governor_ablation/governor_on")
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.55), sharex=True)
    for label, log in (("C3", off), ("C4", on)):
        color = COLORS[label]; t = log["time"]
        axes[0].plot(t, log["human_output"], color=color, label=label, lw=1.35)
        axes[1].plot(t, 1e3 * log["cart_error"], color=color, lw=1.25)
        lag = 1e3 * (log["original_progress"] - log["governed_progress"])
        axes[2].plot(t, lag, color=color, lw=1.25)
    for ax in axes: shade_human(ax, on)
    axes[0].set(title="(a) same energy bound", ylabel="human output [J]")
    axes[1].set(title="(b) task error", ylabel="planar error [mm]")
    axes[2].set(title="(c) governed reference", ylabel="reference lag [mm]")
    for ax in axes: ax.set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:2], labels[:2], frameon=False)
    finish_axes(axes); fig.tight_layout(w_pad=1.0)
    save(fig, "fig6_governor.pdf")


def smoothing_figure() -> None:
    old = load("smoothing/original")
    revised = load("smoothing/revised")
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.55))
    for label, log, color in (("original", old, COLORS["C2"]),
                              ("revised", revised, COLORS["C4"])):
        t = log["time"]
        axes[0].plot(t, log["normal_force"], color=color, label=label,
                     lw=.65, alpha=.88)
        axes[2].plot(t, 1e3 * log["vz"], color=color, lw=.75)
    t = revised["time"]; zoom = (t >= 4.0) & (t <= 4.25)
    axes[1].plot(t[zoom], revised["normal_force"][zoom], color=COLORS["C2"],
                 label="raw", lw=.9, drawstyle="steps-post")
    axes[1].plot(t[zoom], revised["normal_force_filtered"][zoom],
                 color=COLORS["C4"], label="70 ms filtered", lw=1.4)
    axes[0].axhline(5, color="black", ls=":", lw=.8)
    axes[0].set(title="(a) raw 1 kHz contact", ylabel="normal force [N]",
                xlabel="time [s]", xlim=(2.8, 5.0), ylim=(-.5, 14))
    axes[0].legend(frameon=False)
    axes[1].set(title="(b) revised, 250 ms close-up", ylabel="force [N]",
                xlabel="time [s]")
    axes[1].legend(frameon=False)
    axes[2].set(title="(c) physical normal motion", ylabel="normal velocity [mm/s]",
                xlabel="time [s]", xlim=(2.8, 5.0), ylim=(-5, 5))
    finish_axes(axes); fig.tight_layout(w_pad=1.0)
    save(fig, "fig7_smoothing.pdf")


def main() -> None:
    setup_figure()
    architecture_figure()
    accounting_figure()
    human_limiting_figure()
    forearm_figure()
    governor_figure()
    smoothing_figure()


if __name__ == "__main__":
    main()
