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


# Fixed controller-mode identities (never cycled or re-ranked).
MODE_COLORS = {
    "C0_nominal": TEXT_2,
    "C1_whole_port_scalar": YELLOW,
    "C2_residual_qp": AQUA,
    "C3_dual_ledger_qp": BLUE,
    "C4_dual_ledger_scalar": MAGENTA,
}
MODE_SHORT = {
    "C0_nominal": "C0 nominal",
    "C1_whole_port_scalar": "C1 whole-port",
    "C2_residual_qp": "C2 residual QP",
    "C3_dual_ledger_qp": "C3 dual-ledger QP",
    "C4_dual_ledger_scalar": "C4 dual-ledger scalar",
}
_QP_ROW_NAMES = ("E_H", "E_R", "CBF_H", "CBF_R", "P_H", "tau_0", "tau_1")

# Ledger parameters mirrored for plotting reference lines (kept in sync with
# config.PassivationConfig defaults through the tests).
_E_H_LINES = (0.005, 0.08)
_E_R_LINES = (0.02, 0.50)
_P_H_MAX = 0.10


def _shade_human(ax, t, f_h):
    """Recessive band marking where the scripted human force is nonzero."""
    on = np.linalg.norm(f_h, axis=1) > 1e-12
    if not on.any():
        return
    idx = np.flatnonzero(on)
    splits = np.flatnonzero(np.diff(idx) > 1)
    seg_starts = [idx[0]] + list(idx[splits + 1])
    seg_ends = list(idx[splits]) + [idx[-1]]
    for a, b in zip(seg_starts, seg_ends):
        ax.axvspan(t[a], t[b], color="#f6e8ef", zorder=0)


def plot_passivation_run(log: dict, output_dir: str, stem: str = "run",
                         title: str = "") -> str:
    """3x3 per-run figure for a passivation-enabled simulation log."""
    os.makedirs(output_dir, exist_ok=True)
    t = log["time"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    if title:
        fig.suptitle(title, fontsize=12, color=TEXT)

    # 1. Tangential position tracking
    ax = axes[0, 0]
    ax.plot(t, log["ee_pos"][:, 0], color=BLUE, lw=1.4, label="x")
    ax.plot(t, log["x_desired"], color=ORANGE, lw=1.2, ls="--", label="x_d")
    _style(ax, "Tangential position", "m")
    ax.legend(fontsize=7, loc="upper left")

    # 2. Tangential velocity tracking
    ax = axes[0, 1]
    ax.plot(t, log["ee_vel"][:, 0], color=BLUE, lw=1.2, label="v_t")
    ax.plot(t, log["vx_desired"], color=ORANGE, lw=1.2, ls="--", label="v_d")
    _style(ax, "Tangential velocity", "m/s")
    ax.legend(fontsize=7, loc="upper left")

    # 3. Normal force regulation
    ax = axes[0, 2]
    ax.plot(t, log["f_n"], color=BLUE, lw=1.2, label="F_n")
    ax.plot(t, log["f_n_desired"], color=ORANGE, lw=1.2, ls="--", label="F_d")
    _style(ax, "Normal force", "N")
    ax.legend(fontsize=7, loc="lower right")

    # 4. Force decomposition (tangential components)
    ax = axes[1, 0]
    ax.plot(t, log["f_task"][:, 0], color=BLUE, lw=1.2, label="F_T,t")
    ax.plot(t, log["sp_f_hat"][:, 0], color=ORANGE, lw=1.0, ls="--",
            label="F_T_hat,t")
    ax.plot(t, log["sp_f_r"][:, 0], color=VIOLET, lw=1.2, label="F_R,t")
    ax.plot(t, log["f_h"][:, 0], color=MAGENTA, lw=1.2, label="F_H,t")
    _style(ax, "Force decomposition (tangential)", "N")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    # 5. Physical port powers (actual, held-force x next velocity)
    ax = axes[1, 1]
    ax.plot(t, log["sp_p_h_act"], color=MAGENTA, lw=1.2, label="p_H")
    ax.plot(t, log["sp_p_r_act"], color=VIOLET, lw=1.2, label="p_R")
    ax.plot(t, log["sp_p_delta_act"], color=YELLOW, lw=1.0, label="p_Delta")
    ax.axhline(-_P_H_MAX, color=TEXT_2, lw=0.8, ls=":",
               label="-P_H_max")
    _style(ax, "Port powers", "W")
    ax.legend(fontsize=7, loc="lower left", ncol=2)

    # 6. Human ledger
    ax = axes[1, 2]
    ax.plot(t, log["sp_e_h"], color=MAGENTA, lw=1.4, label="E_H")
    ax.plot(t, log["sp_e_h_raw"], color=MAGENTA, lw=0.8, ls=":",
            label="E_H raw")
    for y in _E_H_LINES:
        ax.axhline(y, color=TEXT_2, lw=0.8, ls="--")
    _style(ax, "Human ledger (min/max dashed)", "J")
    ax.legend(fontsize=7, loc="center right")

    # 7. Residual + whole-port ledgers
    ax = axes[2, 0]
    ax.plot(t, log["sp_e_r"], color=VIOLET, lw=1.4, label="E_R")
    ax.plot(t, log["sp_e_w"], color=YELLOW, lw=1.2, label="E_W (whole)")
    for y in _E_R_LINES:
        ax.axhline(y, color=TEXT_2, lw=0.8, ls="--")
    _style(ax, "Residual / whole-port ledgers (min/max dashed)", "J")
    ax.legend(fontsize=7, loc="center right")

    # 8. Nominal vs filtered commands
    ax = axes[2, 1]
    ax.plot(t, log["sp_u_nom"][:, 0], color=BLUE, lw=0.9, ls="--",
            label="u_t nom")
    ax.plot(t, log["sp_u"][:, 0], color=BLUE, lw=1.3, label="u_t")
    ax.plot(t, log["sp_u_nom"][:, 1], color=AQUA, lw=0.9, ls="--",
            label="u_n nom")
    ax.plot(t, log["sp_u"][:, 1], color=AQUA, lw=1.3, label="u_n")
    _style(ax, "Nominal vs filtered command", "N")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    # 9. Active constraints + solver
    ax = axes[2, 2]
    act = log["sp_active"]
    for i in range(act.shape[1]):
        on = act[:, i] > 0.5
        if on.any():
            ax.plot(t[on], np.full(on.sum(), i), ".", color=VIOLET, ms=2)
    infeas = log["sp_infeasible"].astype(bool)
    if infeas.any():
        ax.plot(t[infeas], np.full(infeas.sum(), 7.0), ".", color=ORANGE,
                ms=2, label="infeasible")
        ax.legend(fontsize=7, loc="upper left")
    ax.set_yticks(range(8), [*_QP_ROW_NAMES, "infeas"])
    ax.set_ylim(-0.5, 7.5)
    st = log["sp_solve_time"]
    ax.text(0.98, 0.02,
            f"solve med {1e3 * np.median(st):.3f} ms | "
            f"p99 {1e3 * np.percentile(st, 99):.3f} ms",
            transform=ax.transAxes, fontsize=7, color=TEXT_2,
            ha="right", va="bottom")
    _style(ax, "Active constraints / solver")

    for ax in axes.flat:
        _shade_human(ax, t, log["f_h"])
        ax.set_xlabel("time [s]", fontsize=8, color=TEXT_2)

    path = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_passivation_comparison(logs: dict[str, dict], output_dir: str,
                                stem: str = "comparison",
                                title: str = "") -> str:
    """Cross-controller comparison figure (one scenario, several modes)."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    if title:
        fig.suptitle(title, fontsize=12, color=TEXT)

    any_log = next(iter(logs.values()))
    dt = float(any_log["meta_timestep"])

    panels = [
        ("Tangential velocity", "m/s",
         lambda lg: lg["ee_vel"][:, 0], axes[0, 0]),
        ("Normal force", "N", lambda lg: lg["f_n"], axes[0, 1]),
        ("Human ledger E_H", "J", lambda lg: lg["sp_e_h"], axes[0, 2]),
        ("Residual ledger E_R", "J", lambda lg: lg["sp_e_r"], axes[1, 0]),
        ("Cumulative human extraction W_H", "J",
         lambda lg: -np.cumsum(lg["sp_p_h_act"]) * dt, axes[1, 1]),
        ("Tangential tracking error", "m",
         lambda lg: lg["ee_pos"][:, 0] - lg["x_desired"], axes[1, 2]),
    ]
    for name, unit, getter, ax in panels:
        for mode, lg in logs.items():
            ax.plot(lg["time"], getter(lg), color=MODE_COLORS[mode], lw=1.2,
                    label=MODE_SHORT[mode])
        _style(ax, name, unit)
        _shade_human(ax, any_log["time"], any_log["f_h"])
        ax.set_xlabel("time [s]", fontsize=8, color=TEXT_2)

    # Reference lines where meaningful.
    axes[0, 0].axhline(float(any_log["meta_v_slide"]), color=ORANGE, lw=0.9,
                       ls="--")
    axes[0, 1].axhline(float(any_log["meta_f_desired"]), color=ORANGE, lw=0.9,
                       ls="--")
    axes[0, 2].axhline(_E_H_LINES[0], color=TEXT_2, lw=0.8, ls="--")
    axes[1, 0].axhline(_E_R_LINES[0], color=TEXT_2, lw=0.8, ls="--")
    axes[1, 1].axhline(0.05 - 0.005, color=TEXT_2, lw=0.8, ls="--")
    axes[0, 0].legend(fontsize=7, loc="lower left")

    path = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_sweep(rows: list[dict], output_dir: str, stem: str = "sweep") -> str:
    """mu_hat sweep summary: residual response and tracking vs mu_hat."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    groups = {False: ("no human", BLUE, "o"), True: ("blocking human",
                                                     MAGENTA, "s")}
    for with_human, (label, color, marker) in groups.items():
        sel = [r for r in rows if r["with_human"] == with_human]
        sel.sort(key=lambda r: r["mu_hat"])
        mu = [r["mu_hat"] for r in sel]
        axes[0].plot(mu, [r["e_r_min_raw"] for r in sel], marker=marker,
                     color=color, lw=1.2, label=label)
        axes[1].plot(mu, [r["t_first_ledger_active"]
                          if r["t_first_ledger_active"] is not None
                          else np.nan for r in sel],
                     marker=marker, color=color, lw=1.2, label=label)
        axes[2].plot(mu, [r["rmse_fn"] for r in sel], marker=marker,
                     color=color, lw=1.2, label=label)
    axes[0].axhline(_E_R_LINES[0], color=TEXT_2, lw=0.8, ls="--")
    _style(axes[0], "min E_R (raw)", "J")
    _style(axes[1], "first activation time", "s")
    _style(axes[2], "RMSE F_n (slide)", "N")
    for ax in axes:
        ax.set_xlabel("mu_hat (true mu = 0.30)", fontsize=8, color=TEXT_2)
        ax.legend(fontsize=7)

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
