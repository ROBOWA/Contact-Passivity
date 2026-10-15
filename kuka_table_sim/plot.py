"""Plot a force-controller log (.npz from `run --controller force headless --log`).

    python -m kuka_table_sim.plot run.npz [--out run.png] [--show]

Four panels on a shared time axis (one y-axis each, never a dual axis):
normal force (sensor, ground truth, desired), tip height against its
set-point, tangential position along the slide direction against its
set-point, and the tangential tracking error. Phases are shaded.
matplotlib is an optional dependency: pip install matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Categorical slots 1-3 of the validated reference palette (light surface);
# text and grid use neutral inks, never the series colours.
C_MEASURED = "#2a78d6"   # sensor / actual
C_REFERENCE = "#eb6834"  # desired / set-point
C_TRUTH = "#1baf7a"      # simulator ground truth
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
PHASE_FILL = {0: "#f1f1ee", 1: "#e8eef7", 2: "#fbeee6"}
PHASE_NAME = {0: "approach", 1: "contact", 2: "slide"}


def plot_log(log_path: Path, out: Path | None = None, show: bool = False, title: str | None = None) -> Path | None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = np.load(log_path)
    t = L["t"]
    phase = L["phase"].astype(int)
    pos = L["tcp_pos"]
    slide_dir = L["slide_dir"] if "slide_dir" in L else np.array([1.0, 0.0, 0.0])
    x_ref = np.array([L["xd"][0], L["yd"][0]])
    s_actual = (pos[:, :2] - x_ref) @ slide_dir[:2]
    s_ref = L["s_d"]
    e_along = (np.stack([L["xd"], L["yd"]], 1) - pos[:, :2]) @ slide_dir[:2]
    e_cross = np.linalg.norm((np.stack([L["xd"], L["yd"]], 1) - pos[:, :2]) - np.outer(e_along, slide_dir[:2]), axis=1)
    f_d = float(L["force_desired"])

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, linewidth=1.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        # phase bands
        edges = np.flatnonzero(np.diff(phase) != 0) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [len(t)]))
        for a, b in zip(starts, ends):
            ax.axvspan(t[a], t[b - 1], color=PHASE_FILL[phase[a]], lw=0, zorder=0)
    # phase labels on the top panel only
    for a, b in zip(starts, ends):
        if t[b - 1] - t[a] > 0.8:
            axes[0].text(0.5 * (t[a] + t[b - 1]), 0.97, PHASE_NAME[phase[a]], transform=axes[0].get_xaxis_transform(),
                         ha="center", va="top", fontsize=9, color=INK_2)

    ax = axes[0]
    ax.plot(t, L["f_true_n"], color=C_TRUTH, lw=2.0, solid_joinstyle="round", label="contact force (simulator truth)")
    ax.plot(t, L["f_n"], color=C_MEASURED, lw=2.0, solid_joinstyle="round", label="normal force from FT sensor")
    ax.plot(t, L["f_d"], color=C_REFERENCE, lw=2.0, solid_joinstyle="round", label="desired force")
    ax.set_ylabel("normal force [N]", color=INK)
    ax.set_ylim(-0.5, max(f_d * 1.25, float(np.nanmax(L["f_true_n"])) * 1.05 + 0.5))
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)

    ax = axes[1]
    surface = float(np.min(L["z_d"])) if "z_d" in L else np.nan
    ax.plot(t, L["z_d"] * 1000, color=C_REFERENCE, lw=2.0, label="set-point z_d")
    ax.plot(t, pos[:, 2] * 1000, color=C_MEASURED, lw=2.0, label="tip centre z")
    ax.set_ylabel("tip height [mm]", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)

    ax = axes[2]
    ax.plot(t, s_ref * 1000, color=C_REFERENCE, lw=2.0, label="set-point s_d along slide direction")
    ax.plot(t, s_actual * 1000, color=C_MEASURED, lw=2.0, label="tip position along slide direction")
    ax.set_ylabel("tangential position [mm]", color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)

    ax = axes[3]
    ax.plot(t, e_along * 1000, color=C_MEASURED, lw=2.0, label="along-track error (s_d - s)")
    ax.plot(t, e_cross * 1000, color=C_TRUTH, lw=2.0, label="cross-track error")
    ax.axhline(0, color=GRID, lw=1.0)
    ax.set_ylabel("tracking error [mm]", color=INK)
    ax.set_xlabel("simulated time [s]", color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)

    fig.suptitle(title or f"{Path(log_path).name}: force / position", color=INK, fontsize=12)
    if out is not None:
        fig.savefig(out, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path, help=".npz written by `run --controller force headless --log`")
    p.add_argument("--out", type=Path, default=None, help="PNG path (default: next to the log)")
    p.add_argument("--show", action="store_true", help="open an interactive window")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)
    out = args.out if args.out is not None else args.log.with_suffix(".png")
    plot_log(args.log, out, args.show, args.title)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
