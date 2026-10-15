"""Plots from a RunLog (recorder.py). matplotlib is an optional dependency: pip install matplotlib.

    python -m kuka_table_sim.plot run.npz --kind force --out run.png
    kinds: force (force / position / tracking), free-air (FT estimates), recovery (zoom on a loss event), work (power and work)

Every panel has ONE y-axis. Nothing is smoothed, resampled or filtered: control-rate traces are the logged
control updates and physics-rate traces are the logged physics substeps (every 0.25 ms by default).
Phases are shaded: approach, contact, slide, search, hold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .recorder import RunLog

# Categorical slots 1-4 of the validated reference palette (light surface). Text, axes and grid use neutral
# inks, never the series colours; every panel carries a legend (relief for the low-contrast aqua/yellow).
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PHASE_FILL = {0: "#f1f1ee", 1: "#e8eef7", 2: "#fbeee6", 3: "#e6f3ec", 4: "#f6ecd2"}
PHASE_NAME = {0: "approach", 1: "contact", 2: "slide", 3: "search", 4: "hold"}


def _mpl(show: bool = False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(fig, axes):
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=1.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)


def _phase_bands(axes, log: RunLog, window=None, label_top=True):
    c = log.ctl
    t, ph = c["t"], c["phase"].astype(int)
    edges = np.flatnonzero(np.diff(ph) != 0) + 1
    starts, ends = np.concatenate(([0], edges)), np.concatenate((edges, [len(t)]))
    for ax in np.atleast_1d(axes):
        for a, b in zip(starts, ends):
            ax.axvspan(t[a], t[b - 1], color=PHASE_FILL[ph[a]], lw=0, zorder=0)
    if label_top:
        ax = np.atleast_1d(axes)[0]
        lo, hi = window if window else (t[0], t[-1])
        for a, b in zip(starts, ends):
            x0, x1 = max(t[a], lo), min(t[b - 1], hi)
            if x1 - x0 > 0.06 * (hi - lo):
                ax.text(0.5 * (x0 + x1), 0.97, PHASE_NAME[ph[a]], transform=ax.get_xaxis_transform(), ha="center",
                        va="top", fontsize=9, color=INK_2)


def _sel(t, window):
    return np.ones(len(t), bool) if window is None else (t >= window[0]) & (t <= window[1])


def plot_force_position(log: RunLog, out: Path | None, title: str | None = None, window=None, show: bool = False,
                        slide_dir=(1.0, 0.0, 0.0), surface_z: float | None = None):
    plt = _mpl(show)
    c, p = log.ctl, log.phys
    cs, ps = _sel(c["t"], window), _sel(p["t"], window)
    d = np.asarray(slide_dir, float); d /= np.linalg.norm(d)
    x0 = c["x"][0, :2]
    s_meas = (c["x"][:, :2] - x0) @ d[:2]
    s_ref = (c["x_d"][:, :2] - x0) @ d[:2]
    e_al = s_ref - s_meas
    cross = np.linalg.norm((c["x_d"][:, :2] - c["x"][:, :2]) - np.outer(e_al, d[:2]), axis=1)
    fig, axes = plt.subplots(5, 1, figsize=(11, 12.5), sharex=True, constrained_layout=True)
    _style(fig, axes)
    _phase_bands(axes, log, window)
    ax = axes[0]
    ax.plot(p["t"][ps], p["F_task"][ps, 2], color=C3, lw=2.0, label="contact force, simulator truth (physics rate)")
    ax.plot(c["t"][cs], c["f_n"][cs], color=C1, lw=2.0, label="normal force from FT sensor, corrected (used by control)")
    ax.plot(c["t"][cs], c["f_d"][cs], color=C2, lw=2.0, label="desired force")
    ax.set_ylabel("normal force [N]", color=INK); ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[1]
    ax.plot(c["t"][cs], c["z_d"][cs] * 1e3, color=C2, lw=2.0, label="normal position reference (only while position-controlled)")
    ax.plot(p["t"][ps], p["tcp"][ps, 2] * 1e3, color=C1, lw=2.0, label="tip centre height")
    if surface_z is not None:
        ax.axhline(surface_z * 1e3, color=INK_2, lw=1.0, label="expected contact height")
    ax.set_ylabel("tip height [mm]", color=INK); ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[2]
    ax.plot(c["t"][cs], s_ref[cs] * 1e3, color=C2, lw=2.0, label="active in-plane reference along the slide direction")
    ax.plot(c["t"][cs], s_meas[cs] * 1e3, color=C1, lw=2.0, label="tip position along the slide direction")
    ax.set_ylabel("tangential position [mm]", color=INK); ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[3]
    ax.plot(c["t"][cs], e_al[cs] * 1e3, color=C1, lw=2.0, label="along-track error (reference - measured)")
    ax.plot(c["t"][cs], cross[cs] * 1e3, color=C3, lw=2.0, label="cross-track error")
    ax.axhline(0, color=GRID, lw=1.0)
    ax.set_ylabel("tracking error [mm]", color=INK); ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[4]
    lim = np.asarray(log.meta["torque_limit"])
    ax.plot(p["t"][ps], (np.abs(p["tau_applied"]) / lim).max(axis=1)[ps] * 100, color=C4, lw=2.0, label="largest applied torque / limit (physics rate)")
    ax.set_ylabel("torque utilisation [%]", color=INK); ax.set_xlabel("simulated time [s]", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(title or "force / position", color=INK, fontsize=12)
    if out is not None:
        fig.savefig(out, dpi=125)
    if show:
        plt.show()
    plt.close(fig)


def plot_free_air(log: RunLog, out: Path | None, title: str | None = None, window=None, show: bool = False):
    plt = _mpl(show)
    p = log.phys
    ps = _sel(p["t"], window)
    t = p["t"][ps]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
    _style(fig, axes)
    ax = axes[0]
    ax.plot(t, p["a_com"][ps, 2], color=C1, lw=2.0, label="tool COM acceleration, z (coordinate acceleration, ideal)")
    ax.plot(t, np.linalg.norm(p["a_com"][ps], axis=1), color=C2, lw=2.0, label="|tool COM acceleration|")
    ax.set_ylabel("acceleration [m/s^2]", color=INK); ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[1]
    ax.plot(t, p["ft_grav"][ps, 2], color=C2, lw=2.0, label="gravity-only estimate  -R S - m g")
    ax.plot(t, p["ft_corr"][ps, 2], color=C1, lw=2.0, label="corrected estimate  -R S - m g + m a_com")
    ax.plot(t, p["F_task"][ps, 2], color=C3, lw=2.0, label="ground truth (tip/table contact force)")
    ax.axhline(1.0, color=INK_2, lw=1.0, label="1 N contact-entry threshold")
    ax.set_ylabel("normal force estimate [N]", color=INK); ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[2]
    ax.plot(t, p["ft_raw"][ps, 2], color=C4, lw=2.0, label="raw FT reading, sensor-frame z (tool weight and inertia included)")
    ax.set_ylabel("raw FT force z [N]", color=INK); ax.set_xlabel("simulated time [s]", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(title or "free-air FT estimates", color=INK, fontsize=12)
    if out is not None:
        fig.savefig(out, dpi=125)
    if show:
        plt.show()
    plt.close(fig)


def plot_recovery(log: RunLog, out: Path | None, window: tuple[float, float], title: str | None = None, show: bool = False,
                  surface_z: float | None = None, floor_z: float | None = None):
    """Zoom on a loss event, physics-rate traces where available, phases shaded."""
    plt = _mpl(show)
    c, p = log.ctl, log.phys
    cs, ps = _sel(c["t"], window), _sel(p["t"], window)
    dtc = log.meta["control_period"]
    xd = c["x_d"]
    spd_t = np.linalg.norm(np.diff(xd[:, :2], axis=0), axis=1) / dtc
    spd_n = np.abs(np.diff(xd[:, 2])) / dtc
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
    _style(fig, axes)
    _phase_bands(axes, log, window)
    ax = axes[0]
    ax.plot(p["t"][ps], p["F_task"][ps, 2], color=C3, lw=2.0, label="contact force, simulator truth")
    ax.plot(c["t"][cs], c["f_n"][cs], color=C1, lw=2.0, label="normal force from FT sensor, corrected (used by control)")
    ax.plot(c["t"][cs], c["f_d"][cs], color=C2, lw=2.0, label="desired force")
    ax.set_ylabel("normal force [N]", color=INK); ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[1]
    ax.plot(c["t"][cs], c["z_d"][cs] * 1e3, color=C2, lw=2.0, label="normal reference (position-controlled phases)")
    ax.plot(p["t"][ps], p["tcp"][ps, 2] * 1e3, color=C1, lw=2.0, label="tip centre height")
    if surface_z is not None:
        ax.axhline(surface_z * 1e3, color=INK_2, lw=1.0, label="expected contact height")
    if floor_z is not None:
        ax.axhline(floor_z * 1e3, color=C4, lw=1.0, label="search floor")
    ax.set_ylabel("height [mm]", color=INK); ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[2]
    ax.plot(c["t"][cs], (c["x_d"][cs, 0] - c["x_d"][0, 0]) * 1e3, color=C2, lw=2.0, label="active reference x")
    ax.plot(c["t"][cs], (c["x"][cs, 0] - c["x_d"][0, 0]) * 1e3, color=C1, lw=2.0, label="tip x")
    ax.set_ylabel("tangential position x [mm]", color=INK); ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[3]
    ts = c["t"][1:]
    m = _sel(ts, window)
    ax.plot(ts[m], spd_t[m] * 1e3, color=C2, lw=2.0, label="tangential reference speed (finite difference of the active reference)")
    ax.plot(ts[m], spd_n[m] * 1e3, color=C1, lw=2.0, label="normal reference speed (finite difference)")
    ax.set_ylabel("reference speed [mm/s]", color=INK); ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[4]
    lim = np.asarray(log.meta["torque_limit"])
    ax.plot(p["t"][ps], (np.abs(p["tau_applied"]) / lim).max(axis=1)[ps] * 100, color=C4, lw=2.0, label="largest applied torque / limit")
    ax.set_ylabel("torque utilisation [%]", color=INK); ax.set_xlabel("simulated time [s]", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    axes[-1].set_xlim(window)
    fig.suptitle(title or "contact loss and recovery", color=INK, fontsize=12)
    if out is not None:
        fig.savefig(out, dpi=125)
    if show:
        plt.show()
    plt.close(fig)


def plot_work(log: RunLog, out: Path | None, title: str | None = None, window=None, show: bool = False):
    plt = _mpl(show)
    p, c = log.phys, log.ctl
    ps, cs = _sel(p["t"], window), _sel(c["t"], window)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    _style(fig, axes)
    _phase_bands(axes, log, window)
    ax = axes[0]
    ax.plot(p["t"][ps], p["P_task"][ps] * 1e3, color=C1, lw=1.6, label="task-contact power P (physics rate), environment on robot")
    ax.axhline(0, color=GRID, lw=1.0)
    ax.set_ylabel("power [mW]", color=INK); ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[1]
    ax.plot(c["t"][cs], c["W_interval"][cs] * 1e6, color=C2, lw=1.6, label="work per control interval (sum of 4 substeps)")
    ax.axhline(0, color=GRID, lw=1.0)
    ax.set_ylabel("work per interval [uJ]", color=INK); ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax = axes[2]
    ax.plot(p["t"][ps], p["W_cum"][ps], color=C3, lw=2.0, label="accumulated signed work (rectangle rule)")
    ax.set_ylabel("work [J]", color=INK); ax.set_xlabel("simulated time [s]", color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(title or "contact power and work", color=INK, fontsize=12)
    if out is not None:
        fig.savefig(out, dpi=125)
    if show:
        plt.show()
    plt.close(fig)


def plot_scenario(res, out: Path, show: bool = False) -> None:
    """Dispatch on the scenario name (scenarios.ScenarioResult)."""
    log = res.log
    ctl = res.sim.controller
    name = res.name
    if name.startswith("free-air"):
        plot_free_air(log, out, title=name)
    elif name.startswith("loss") or name.startswith("removal"):
        t_ev = res.info.get("pull", {}).get("t_start", res.info.get("t_remove", 12.0))
        t_end = float(log.phys["t"][-1])
        hi = res.info.get("t_resume")
        hi = (hi + 6.0) if hi else min(t_end, t_ev + 8.0)
        plot_recovery(log, out, (t_ev - 1.0, min(hi, t_end)), title=name, surface_z=getattr(ctl, "z_surface", None),
                      floor_z=getattr(ctl, "z_floor", None))
    else:
        cfg = getattr(ctl, "cfg", None)
        plot_force_position(log, out, title=name, slide_dir=cfg.slide_direction if cfg else (1, 0, 0),
                            surface_z=getattr(ctl, "z_surface", None))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="RunLog .npz")
    ap.add_argument("--kind", choices=("force", "free-air", "recovery", "work"), default="force")
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("T0", "T1"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    log = RunLog.load(args.log)
    out = args.out or args.log.with_suffix(".png")
    win = tuple(args.window) if args.window else None
    if args.kind == "force":
        plot_force_position(log, out, window=win, show=args.show)
    elif args.kind == "free-air":
        plot_free_air(log, out, window=win, show=args.show)
    elif args.kind == "recovery":
        plot_recovery(log, out, win or (float(log.ctl["t"][0]), float(log.ctl["t"][-1])), show=args.show)
    else:
        plot_work(log, out, window=win, show=args.show)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
