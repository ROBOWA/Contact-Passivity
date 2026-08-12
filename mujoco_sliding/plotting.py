"""Headless matplotlib results figure and x-z plane animation."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe; set before pyplot import

import matplotlib.pyplot as plt
import numpy as np

from .controller import Phase

# Validated categorical palette (light mode) + text/grid tokens.
BLUE = "#2a78d6"      # measured / actual
ORANGE = "#eb6834"    # desired / reference
AQUA = "#1baf7a"      # secondary measured channel
YELLOW = "#eda100"    # tertiary channel
MAGENTA = "#e87ba4"
VIOLET = "#4a3aa7"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

PHASE_NAMES = {0: "approach", 1: "ramp", 2: "slide"}
PHASE_SHADE = {0: "#f0f0ee", 1: "#faf3e6", 2: "#ecf3fb"}


def _style(ax, title, ylabel=""):
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(labelsize=8, colors=TEXT_2)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_facecolor(SURFACE)


def _shade_phases(ax, t, phase):
    """Recessive background bands marking the controller phase."""
    changes = np.flatnonzero(np.diff(phase)) + 1
    bounds = np.concatenate(([0], changes, [len(t) - 1]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        ax.axvspan(t[a], t[b], color=PHASE_SHADE[int(phase[a])], zorder=0)


def plot_results(log: dict, output_dir: str, stem: str = "results") -> str:
    """Save the 3x3 results figure and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    t = log["time"]
    phase = log["phase"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    # 1. EE position
    ax = axes[0, 0]
    ax.plot(t, log["ee_pos"][:, 0], color=BLUE, lw=1.4, label="x")
    ax.plot(t, log["ee_pos"][:, 2], color=AQUA, lw=1.4, label="z")
    ax.plot(t, log["x_desired"], color=ORANGE, lw=1.2, ls="--", label="x desired")
    _style(ax, "End-effector position", "m")
    ax.legend(fontsize=7, loc="center left")

    # 2. Tangential velocity
    ax = axes[0, 1]
    ax.plot(t, log["ee_vel"][:, 0], color=BLUE, lw=1.2, label="v_x")
    ax.plot(t, log["vx_desired"], color=ORANGE, lw=1.2, ls="--", label="v_x desired")
    _style(ax, "Tangential velocity", "m/s")
    ax.legend(fontsize=7, loc="lower right")

    # 3. Normal force
    ax = axes[0, 2]
    ax.plot(t, log["f_n"], color=BLUE, lw=1.2, label="F_n measured")
    ax.plot(t, log["f_n_desired"], color=ORANGE, lw=1.2, ls="--", label="F_d")
    _style(ax, "Normal force", "N")
    ax.legend(fontsize=7, loc="lower right")

    # 4. Normal + friction force
    ax = axes[1, 0]
    ax.plot(t, log["f_n"], color=BLUE, lw=1.2, label="F_n")
    ax.plot(t, log["f_t"], color=AQUA, lw=1.2, label="F_t (friction)")
    ax.plot(t, log["f_h"][:, 0], color=MAGENTA, lw=1.2, label="f_h,x")
    _style(ax, "Contact and human forces", "N")
    ax.legend(fontsize=7, loc="center right")

    # 5. Contact distance / penetration
    ax = axes[1, 1]
    dist = np.where(np.isfinite(log["contact_dist"]), log["contact_dist"], np.nan)
    ax.plot(t, 1e3 * dist, color=BLUE, lw=1.2, label="gap (dist)")
    ax.plot(t, -1e3 * log["penetration"], color=ORANGE, lw=1.2, label="-penetration")
    ax.set_ylim(-2, 20)
    _style(ax, "Contact distance / penetration", "mm")
    ax.legend(fontsize=7, loc="upper right")

    # 6. Joint torques
    ax = axes[1, 2]
    ax.plot(t, log["ctrl"][:, 0], color=BLUE, lw=1.2, label="shoulder")
    ax.plot(t, log["ctrl"][:, 1], color=AQUA, lw=1.2, label="elbow")
    _style(ax, "Commanded joint torques", "N·m")
    ax.legend(fontsize=7, loc="center right")

    # 7. Task-contact power
    ax = axes[2, 0]
    ax.plot(t, log["p_task"], color=BLUE, lw=1.2, label="P_task")
    _style(ax, "Task-contact power", "W")
    ax.legend(fontsize=7, loc="lower right")

    # 8. Human-interaction power
    ax = axes[2, 1]
    ax.plot(t, log["p_h_to_r"], color=MAGENTA, lw=1.2, label="P_h→r")
    _style(ax, "Human-interaction power", "W")
    ax.legend(fontsize=7, loc="upper right")

    # 9. Controller phase
    ax = axes[2, 2]
    ax.step(t, phase, color=VIOLET, lw=1.4, where="post")
    ax.set_yticks([0, 1, 2], [PHASE_NAMES[i] for i in range(3)])
    _style(ax, "Controller phase")

    for ax in axes.flat:
        _shade_phases(ax, t, phase)
        ax.set_xlabel("time [s]", fontsize=8, color=TEXT_2)

    path = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _forward_kinematics(log: dict, i: int):
    """Base, elbow, and EE points in the x-z plane at log index i."""
    base = log["meta_base_pos"][[0, 2]]
    l1, l2 = log["meta_link_lengths"]
    q1, q2 = log["qpos"][i]
    # Hinge about +y: direction(theta) = (cos theta, -sin theta) in x-z.
    e = base + l1 * np.array([np.cos(q1), -np.sin(q1)])
    ee = e + l2 * np.array([np.cos(q1 + q2), -np.sin(q1 + q2)])
    return base, e, ee


def animate(log: dict, output_dir: str, stem: str = "animation",
            fps: int = 25, stride: int = 40) -> str | None:
    """Save an x-z plane GIF built from the logged trajectory.

    Returns the path, or None if no GIF writer is available (the headless
    pipeline then simply skips animation export).
    """
    from matplotlib import animation as mpl_animation

    if not mpl_animation.writers.is_available("pillow"):
        return None

    os.makedirs(output_dir, exist_ok=True)
    idx = np.arange(0, len(log["time"]), stride)
    pad_r = float(log["meta_pad_radius"])
    f_scale = 0.02  # m per N for force arrows

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.35, 1.3)
    ax.set_ylim(-0.15, 0.95)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(labelsize=8, colors=TEXT_2)
    ax.set_xlabel("x [m]", fontsize=9, color=TEXT_2)
    ax.set_ylabel("z [m]", fontsize=9, color=TEXT_2)

    ax.axhspan(-0.15, 0.0, color="#d8d7d2", zorder=0)  # surface
    ax.axhline(0.0, color=TEXT_2, lw=1.0)

    links, = ax.plot([], [], "-o", color=TEXT, lw=3, ms=5, zorder=4)
    pad = plt.Circle((0, 0), pad_r, color=AQUA, zorder=5)
    ax.add_patch(pad)
    trail, = ax.plot([], [], color=BLUE, lw=1.0, alpha=0.7, zorder=3)
    f_task_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
    f_h_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=2))
    label = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10,
                    color=TEXT, va="top")
    ax.plot([], [], color=ORANGE, lw=2, label="f_task")
    ax.plot([], [], color=MAGENTA, lw=2, label="f_h")
    ax.legend(fontsize=8, loc="upper right")

    ee_hist_x, ee_hist_z = [], []

    def frame(k):
        i = idx[k]
        base, elbow, ee = _forward_kinematics(log, i)
        links.set_data([base[0], elbow[0], ee[0]], [base[1], elbow[1], ee[1]])
        pad.center = (ee[0], ee[1])
        ee_hist_x.append(ee[0])
        ee_hist_z.append(ee[1])
        trail.set_data(ee_hist_x, ee_hist_z)

        ft = log["f_task"][i]
        fh = log["f_h"][i]
        f_task_arrow.xy = (ee[0] + f_scale * ft[0], ee[1] + f_scale * ft[2])
        f_task_arrow.set_position((ee[0], ee[1]))
        f_h_arrow.xy = (ee[0] + f_scale * fh[0], ee[1] + f_scale * fh[2])
        f_h_arrow.set_position((ee[0], ee[1]))

        label.set_text(
            f"t = {log['time'][i]:5.2f} s   phase: {PHASE_NAMES[int(log['phase'][i])]}"
            + ("   human force ON" if np.any(fh) else "")
        )
        return links, pad, trail

    anim = mpl_animation.FuncAnimation(fig, frame, frames=len(idx), blit=False)
    path = os.path.join(output_dir, f"{stem}.gif")
    try:
        anim.save(path, writer=mpl_animation.PillowWriter(fps=fps))
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return path
