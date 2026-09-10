"""Create synchronized MuJoCo/plot videos for paper Figures 3, 4, and 8.

Each 16:9 MP4 contains two titled MuJoCo views across the top and a live,
left-to-right rendering of the corresponding paper plot below. The videos
replay the exact NPZ logs used by ``make_figures.py``; they do not rerun the
controller. A magenta 3-D arrow makes a scripted human wrench visible in the
MuJoCo scene, and a matching status card reports its magnitude.

From the repository root:

    python paper/make_videos.py
    python paper/make_videos.py --figures 8 --fps 30

Outputs are written to ``paper/videos`` by default. H.264 with a yuv420p
pixel format is used for reliable PowerPoint playback.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import make_figures as mf
from mujoco_sliding.config import MODEL_XML_PATH
from mujoco_sliding.contact_extraction import ModelHandles
from mujoco_sliding.controller import GovernorMode
from mujoco_sliding.visualization import (
    add_human_force_arrow,
    add_reference_setpoint_marker,
)


VIDEO_BG = "#f7f7f5"
CURSOR = "#b13b35"
HUMAN = "#e8338a"
REFERENCE = "#e5a400"


def load(path: Path) -> dict:
    return dict(np.load(path))


def result_log(scenario: str, mode: str) -> dict:
    return load(ROOT / "results_passivation_iter2" / scenario / mode / "log.npz")


def governor_log(state: str) -> dict:
    return load(
        ROOT
        / "results_passivation_iter2"
        / "governor_compare"
        / "C3_dual_ledger_qp"
        / state
        / "log.npz"
    )


def extended_governor_log(enabled: bool) -> dict:
    """Run the Figure 8 controller for two real seconds beyond the paper log."""
    from mujoco_sliding.experiments import SCENARIOS, build_config
    from mujoco_sliding.simulation import run_simulation

    extended_spec = replace(SCENARIOS["C"], duration=14.0)
    config = build_config(
        extended_spec,
        "C3_dual_ledger_qp",
        governor=enabled,
    )
    return run_simulation(config)


class LogRenderer:
    """Off-screen MuJoCo renderer driven by an existing simulation log."""

    def __init__(self, width: int, height: int):
        self.model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.handles = ModelHandles.from_model(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.57, 0.0, 0.36]
        self.camera.distance = 1.65
        self.camera.elevation = -8
        self.camera.azimuth = 90

        self.options = mujoco.MjvOption()
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    def frame(
        self,
        log: dict,
        index: int,
        *,
        reference_x: float | None = None,
    ) -> np.ndarray:
        self.data.qpos[:] = log["qpos"][index]
        self.data.qvel[:] = log["qvel"][index]
        self.data.ctrl[:] = log["ctrl"][index]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(
            self.data, camera=self.camera, scene_option=self.options
        )
        add_human_force_arrow(
            self.renderer.scene,
            self.data.site_xpos[self.handles.ee_site],
            log["f_h"][index],
        )
        if reference_x is not None:
            add_reference_setpoint_marker(self.renderer.scene, reference_x)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


@dataclass
class LiveLayout:
    figure: plt.Figure
    view_axes: tuple[plt.Axes, plt.Axes]
    view_images: tuple
    status_texts: tuple
    plot_axes: list[plt.Axes]
    x_min: float
    x_max: float
    bindings: list[tuple] = field(default_factory=list)
    cursors: list = field(default_factory=list)

    def bind(self, line, x: np.ndarray, y: np.ndarray) -> None:
        self.bindings.append((line, np.asarray(x), np.asarray(y)))

    def finish(self) -> "LiveLayout":
        for ax in self.plot_axes:
            self.cursors.append(
                ax.axvline(
                    self.x_min,
                    color=CURSOR,
                    lw=1.15,
                    alpha=0.9,
                    zorder=9,
                )
            )
        return self

    def update_plot(self, time_now: float) -> None:
        for line, x, y in self.bindings:
            lo = int(np.searchsorted(x, self.x_min, side="left"))
            hi = int(np.searchsorted(x, time_now, side="right"))
            line.set_data(x[lo:hi], y[lo:hi])
        for cursor in self.cursors:
            cursor.set_xdata([time_now, time_now])


def video_style(ax: plt.Axes, ylabel: str = "", xlabel: str | None = None) -> None:
    mf.style(ax, ylabel, xlabel)
    ax.tick_params(labelsize=8.5)
    ax.yaxis.label.set_size(9.5)
    if xlabel is not None:
        ax.xaxis.label.set_size(9.5)


def set_full_data_ylim(ax: plt.Axes, *arrays, include=()) -> None:
    """Match Matplotlib's default 5% y margin using complete plot data."""
    values = [np.ravel(np.asarray(array, dtype=float)) for array in arrays]
    if include:
        values.append(np.asarray(include, dtype=float))
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    if span <= 1e-12:
        span = max(abs(high), 1.0) * 0.1
    margin = 0.05 * span
    ax.set_ylim(low - margin, high + margin)


def panel_letter(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.008,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        zorder=10,
    )


def base_layout(
    *,
    title: str,
    left_title: str,
    right_title: str,
    rows: int,
    cols: int,
    x_min: float,
    x_max: float,
    width: int,
    height: int,
) -> LiveLayout:
    dpi = 120
    fig = plt.figure(
        figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=VIDEO_BG
    )
    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=0.992,
        color="#30302e",
    )
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.48, 0.52],
        left=0.045,
        right=0.985,
        bottom=0.055,
        top=0.905,
        hspace=0.17,
        wspace=0.035,
    )
    left_ax = fig.add_subplot(outer[0, 0])
    right_ax = fig.add_subplot(outer[0, 1])
    blank = np.zeros((450, 800, 3), dtype=np.uint8)
    images = (left_ax.imshow(blank), right_ax.imshow(blank))

    status = []
    for ax, pane_title, color in (
        (left_ax, left_title, mf.C_C1 if "C1" in left_title else mf.C_C3),
        (right_ax, right_title, mf.C_C4 if "C4" in right_title else mf.C_C3),
    ):
        ax.set_title(
            pane_title,
            fontsize=12.5,
            fontweight="bold",
            color=color,
            pad=5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#aaa9a5")
            spine.set_linewidth(1.2)
        status.append(
            ax.text(
                0.018,
                0.04,
                "",
                transform=ax.transAxes,
                color="white",
                fontsize=8.5,
                va="bottom",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.30",
                    fc="black",
                    ec="none",
                    alpha=0.67,
                ),
            )
        )

    sub = outer[1, :].subgridspec(rows, cols, hspace=0.13, wspace=0.16)
    axes: list[plt.Axes] = []
    shared = None
    for row in range(rows):
        for col in range(cols):
            ax = fig.add_subplot(sub[row, col], sharex=shared)
            if shared is None:
                shared = ax
            if row < rows - 1:
                ax.tick_params(labelbottom=False)
            axes.append(ax)
    for ax in axes:
        ax.set_xlim(x_min, x_max)

    return LiveLayout(
        figure=fig,
        view_axes=(left_ax, right_ax),
        view_images=images,
        status_texts=tuple(status),
        plot_axes=axes,
        x_min=x_min,
        x_max=x_max,
    )


def _first_active_time(log: dict) -> float | None:
    active = (
        log["sp_active"][:, :5].any(axis=1)
        & log["sp_active_any"].astype(bool)
    )
    idx = np.flatnonzero(active)
    return float(log["time"][idx[0]]) if idx.size else None


def build_figure3(width: int, height: int):
    c0 = result_log("A_exact_no_human", "C0_nominal")
    c1 = result_log("A_exact_no_human", "C1_whole_port_qp")
    c3 = result_log("A_exact_no_human", "C3_dual_ledger_qp")
    t = c3["time"]
    t_act = _first_active_time(c1)
    layout = base_layout(
        title="Figure 3 — Predictable task contact should not consume the passivity budget",
        left_title="C1 — whole-port passivity",
        right_title="C3 — model-aware selective passivation",
        rows=3,
        cols=1,
        x_min=float(t[0]),
        x_max=float(t[-1]),
        width=width,
        height=height,
    )
    a, b, c = layout.plot_axes

    for y, color, label in (
        (c1["sp_e_w"], mf.C_C1, r"$E_W$ (C1)"),
        (c3["sp_e_r"], mf.C_C3, r"$E_R$ (C3)"),
        (c3["sp_e_h"], mf.C_AUX, r"$E_H$ (C3)"),
    ):
        line, = a.plot([], [], color=color, label=label)
        layout.bind(line, t, y)
    a.axhline(0.02, color=mf.C_LIM, lw=0.8, ls="--")
    a.axhline(0.005, color=mf.C_LIM, lw=0.8, ls=":")
    a.text(
        0.35,
        0.0265,
        r"$E_{R,\min}$",
        fontsize=8,
        color=mf.C_LIM,
        va="bottom",
    )
    a.set_ylim(-0.02, 0.36)
    video_style(a, "ledger energy [J]")
    a.legend(frameon=False, loc="center left", fontsize=8, ncol=3)
    panel_letter(a, "(a)")

    for y, color, label in (
        (c0["f_n"], mf.C_C0, "C0"),
        (c1["f_n"], mf.C_C1, "C1"),
        (c3["f_n"], mf.C_C3, "C3"),
        (c3["f_n_desired"], mf.C_REF, r"$F_n^d$"),
    ):
        line, = b.plot(
            [],
            [],
            color=color,
            lw=1.0,
            ls="--" if label == r"$F_n^d$" else "-",
            label=label,
        )
        layout.bind(line, t, y)
    f_hi = max(c0["f_n"].max(), c1["f_n"].max(), c3["f_n"].max())
    b.set_ylim(-0.4, f_hi * 1.35)
    video_style(b, r"$F_n$ [N]")
    b.legend(frameon=False, loc="upper left", ncol=4, fontsize=8)
    panel_letter(b, "(b)")

    for y, color, linestyle in (
        (c0["ee_vel"][:, 0], mf.C_C0, "-"),
        (c1["ee_vel"][:, 0], mf.C_C1, "-"),
        (c3["ee_vel"][:, 0], mf.C_C3, "-"),
        (c3["vx_desired"], mf.C_REF, "--"),
    ):
        line, = c.plot([], [], color=color, lw=1.0, ls=linestyle)
        layout.bind(line, t, y)
    c.set_ylim(-0.06, 0.12)
    video_style(c, r"$v_t$ [m/s]", "time [s]")
    panel_letter(c, "(c)")

    for ax in (a, b, c):
        ax.axvline(t_act, color=mf.C_C1, lw=0.8, ls=":")
    a.annotate(
        f"C1 intervenes ({t_act:.1f} s)",
        xy=(t_act, 0.07),
        xytext=(t_act + 1.4, 0.15),
        fontsize=8,
        color=mf.C_C1,
        arrowprops=dict(arrowstyle="->", color=mf.C_C1, lw=0.8),
    )
    return layout.finish(), c1, c3, "c1", "c3"


def build_figure4(width: int, height: int):
    c1 = result_log("B_mismatch_no_human", "C1_whole_port_qp")
    c3 = result_log("B_mismatch_no_human", "C3_dual_ledger_qp")
    t = c3["time"]
    ta_c1, ta_c3 = _first_active_time(c1), _first_active_time(c3)
    layout = base_layout(
        title="Figure 4 — Contact-model error remains in the residual channel",
        left_title="C1 — whole-port accounting",
        right_title="C3 — residual accounting",
        rows=2,
        cols=1,
        x_min=float(t[0]),
        x_max=float(t[-1]),
        width=width,
        height=height,
    )
    a, b = layout.plot_axes

    for y, color, label in (
        (c3["sp_e_h"], mf.C_AUX, r"$E_H$"),
        (c3["sp_e_r"], mf.C_C3, r"$E_R$"),
    ):
        line, = a.plot([], [], color=color, label=label)
        layout.bind(line, t, y)
    a.axhline(0.02, color=mf.C_LIM, lw=0.8, ls="--")
    a.text(
        0.35,
        0.0265,
        r"$E_{R,\min}$",
        fontsize=8,
        color=mf.C_LIM,
        va="bottom",
    )
    a.axvline(ta_c1, color=mf.C_C1, lw=0.8, ls=":")
    a.axvline(ta_c3, color=mf.C_C3, lw=0.8, ls=":")
    a.annotate(
        f"C1 whole-port activates ({ta_c1:.2f} s)",
        xy=(ta_c1, 0.24),
        xytext=(ta_c1 + 0.9, 0.34),
        fontsize=8,
        color=mf.C_C1,
        arrowprops=dict(arrowstyle="->", color=mf.C_C1, lw=0.8),
    )
    a.annotate(
        f"C3 residual activates ({ta_c3:.2f} s)",
        xy=(ta_c3, 0.05),
        xytext=(ta_c3 - 6.5, 0.075),
        fontsize=8,
        color=mf.C_C3,
        arrowprops=dict(arrowstyle="->", color=mf.C_C3, lw=0.8),
    )
    a.set_ylim(0, 0.4)
    video_style(a, "ledger energy [J]")
    a.legend(frameon=False, loc="upper right", fontsize=8)
    panel_letter(a, "(a)")

    dt = float(c3["meta_timestep"])
    count = max(1, int(0.05 / dt))
    p_delta_mean = np.convolve(
        c3["sp_p_delta_act"], np.ones(count) / count, mode="same"
    )
    raw, = b.plot([], [], color=mf.C_C2, lw=0.5, alpha=0.35)
    mean, = b.plot([], [], color=mf.C_C2, label=r"$p_\Delta$ (50 ms mean)")
    human, = b.plot([], [], color=mf.C_AUX, lw=1.0, label=r"$p_H$")
    layout.bind(raw, t, c3["sp_p_delta_act"])
    layout.bind(mean, t, p_delta_mean)
    layout.bind(human, t, c3["sp_p_h_act"])
    b.axhline(0.0, color=mf.C_LIM, lw=0.6)
    ymin = min(c3["sp_p_delta_act"].min(), p_delta_mean.min())
    ymax = max(c3["sp_p_delta_act"].max(), p_delta_mean.max())
    pad = 0.10 * max(ymax - ymin, 1e-3)
    b.set_ylim(ymin - pad, ymax + pad)
    video_style(b, "port power [W]", "time [s]")
    b.legend(frameon=False, loc="lower left", ncol=2, fontsize=8)
    panel_letter(b, "(b)")
    return layout.finish(), c1, c3, "c1", "c3"


def _governor_events(log: dict):
    engaged = np.isin(log["sp_gov_mode"], [1, 3, 4, 5])
    idx = np.flatnonzero(engaged)
    trigger = float(log["time"][idx[0]]) if idx.size else None
    anchor = log.get("sp_gov_anchor")
    anchor_time = None
    if anchor is not None and np.isfinite(anchor).any():
        anchor_time = float(
            log["time"][np.flatnonzero(np.isfinite(anchor))[0]]
        )
    return trigger, anchor_time


def build_figure8(width: int, height: int):
    off = governor_log("governor_off")
    on = governor_log("governor_on")
    t = off["time"]
    t_release = 8.5
    trigger, anchor_time = _governor_events(on)
    layout = base_layout(
        title="Figure 8 — The reference governor removes the release transient",
        left_title="C3 — no reference governor",
        right_title="C4 — reference governor enabled",
        rows=3,
        cols=2,
        x_min=4.5,
        x_max=12.0,
        width=width,
        height=height,
    )
    a, b, c, d, e, f = layout.plot_axes

    def periods(ax):
        mf.shade_human(ax, t, off["f_h"])
        if trigger is not None:
            ax.axvspan(6.0, trigger, color="#eef3f8", zorder=0)
            ax.axvline(trigger, color=mf.C_AUX, lw=0.9, ls="-.")
        if anchor_time is not None:
            ax.axvspan(trigger, anchor_time, color="#f2eef8", zorder=0)
            ax.axvline(anchor_time, color=mf.C_AUX, lw=0.8, ls=":")
        ax.axvline(t_release, color=mf.C_LIM, lw=0.7, ls=":")

    line, = a.plot([], [], color=mf.C_AUX, lw=1.0)
    layout.bind(line, t, np.linalg.norm(off["f_h"], axis=1))
    a.axhline(0.1, color=mf.C_LIM, lw=0.7, ls=":")
    a.text(
        4.62,
        2.50,
        r"$F_{\mathrm{detect}}=0.1$ N",
        fontsize=7.5,
        color=mf.C_LIM,
    )
    if trigger is not None:
        a.text(
            trigger + 0.05,
            1.30,
            f"CBF trigger {trigger:.2f} s",
            fontsize=7.5,
            color=mf.C_AUX,
        )
    set_full_data_ylim(a, np.linalg.norm(off["f_h"], axis=1), include=(0.1,))
    video_style(a, r"$\|F_H\|$ [N]")
    panel_letter(a, "(a)")

    for y, color, linestyle, label, width_line in (
        (
            off["sp_x_ref_original"],
            mf.C_REF,
            ":",
            r"$x_{\mathrm{ref,orig}}$",
            0.9,
        ),
        (off["ee_pos"][:, 0], mf.C_C3, "--", r"$x$ (C3)", 1.0),
        (on["ee_pos"][:, 0], mf.C_C4, "-", r"$x$ (C4)", 1.1),
        (on["x_desired"], mf.C_C4, ":", r"$x_g$ (C4)", 0.9),
    ):
        line, = b.plot(
            [],
            [],
            color=color,
            ls=linestyle,
            lw=width_line,
            label=label,
        )
        layout.bind(line, t, y)
    if anchor_time is not None:
        anchor = on["sp_gov_anchor"]
        x_anchor = float(anchor[np.flatnonzero(np.isfinite(anchor))[0]])
        b.axhline(
            x_anchor,
            color=mf.C_AUX,
            lw=0.7,
            ls="--",
            alpha=0.7,
        )
        b.text(
            11.85,
            x_anchor - 0.018,
            r"$x_a$",
            fontsize=7.5,
            color=mf.C_AUX,
            ha="right",
        )
    set_full_data_ylim(
        b,
        off["sp_x_ref_original"],
        off["ee_pos"][:, 0],
        on["ee_pos"][:, 0],
        on["x_desired"],
        include=(x_anchor,) if anchor_time is not None else (),
    )
    video_style(b, "position [m]")
    b.legend(frameon=False, ncol=2, loc="upper left", fontsize=7)
    panel_letter(b, "(b)")

    for y, color, linestyle, label, width_line in (
        (off["ee_vel"][:, 0], mf.C_C3, "--", r"$v_t$ (C3)", 1.0),
        (on["ee_vel"][:, 0], mf.C_C4, "-", r"$v_t$ (C4)", 1.1),
        (on["vx_desired"], mf.C_C4, ":", r"$v_g$ (C4)", 0.9),
    ):
        line, = c.plot(
            [],
            [],
            color=color,
            ls=linestyle,
            lw=width_line,
            label=label,
        )
        layout.bind(line, t, y)
    c.axhline(0.05, color=mf.C_REF, lw=0.7, ls=":")
    set_full_data_ylim(
        c,
        off["ee_vel"][:, 0],
        on["ee_vel"][:, 0],
        on["vx_desired"],
        include=(0.05,),
    )
    video_style(c, "velocity [m/s]")
    c.legend(frameon=False, loc="upper left", fontsize=7)
    panel_letter(c, "(c)")

    for y, color, linestyle, label in (
        (off["sp_u"][:, 0], mf.C_C3, "--", "C3"),
        (on["sp_u"][:, 0], mf.C_C4, "-", "C4"),
    ):
        line, = d.plot(
            [], [], color=color, ls=linestyle, lw=1.0, label=label
        )
        layout.bind(line, t, y)
    set_full_data_ylim(d, off["sp_u"][:, 0], on["sp_u"][:, 0])
    video_style(d, r"$u_t$ [N]")
    d.legend(frameon=False, loc="upper left", fontsize=7)
    panel_letter(d, "(d)")

    for log, color, linestyle, label in (
        (off, mf.C_C3, "--", "C3"),
        (on, mf.C_C4, "-", "C4"),
    ):
        line, = e.plot(
            [], [], color=color, ls=linestyle, lw=1.0, label=label
        )
        layout.bind(line, t, log["f_n"] - log["f_n_desired"])
    e.axhline(0.0, color=mf.C_LIM, lw=0.6)
    set_full_data_ylim(
        e,
        off["f_n"] - off["f_n_desired"],
        on["f_n"] - on["f_n_desired"],
        include=(0.0,),
    )
    video_style(e, r"$F_n-F_n^d$ [N]", "time [s]")
    e.legend(frameon=False, loc="upper left", fontsize=7)
    panel_letter(e, "(e)")

    for y, color, linestyle, label, width_line, alpha in (
        (off["sp_e_h"], mf.C_C3, "--", r"$E_H$ (C3)", 1.0, 1.0),
        (on["sp_e_h"], mf.C_C4, "-", r"$E_H$ (C4)", 1.1, 1.0),
        (
            off["sp_e_r"],
            mf.C_C3,
            "--",
            r"$E_R$ (C3)",
            0.8,
            0.55,
        ),
        (on["sp_e_r"], mf.C_C4, "-", r"$E_R$ (C4)", 0.8, 0.7),
    ):
        line, = f.plot(
            [],
            [],
            color=color,
            ls=linestyle,
            lw=width_line,
            alpha=alpha,
            label=label,
        )
        layout.bind(line, t, y)
    f.axhline(0.005, color=mf.C_LIM, lw=0.7, ls=":")
    f.axhline(0.02, color=mf.C_LIM, lw=0.7, ls=":")
    set_full_data_ylim(
        f,
        off["sp_e_h"],
        on["sp_e_h"],
        off["sp_e_r"],
        on["sp_e_r"],
        include=(0.005, 0.02),
    )
    video_style(f, "tank energy [J]", "time [s]")
    f.legend(frameon=False, ncol=2, loc="center left", fontsize=6.5)
    panel_letter(f, "(f)")

    for ax in layout.plot_axes:
        periods(ax)
        ax.set_xlim(4.5, 12.0)
    a.text(
        t_release + 0.05,
        2.52,
        "release",
        fontsize=7.5,
        color=mf.C_LIM,
    )
    # The published plot ends at 12 s. Continue both actual simulations to
    # 14 s for the viewer panes so the additional video time contains normal
    # post-release robot motion rather than a duplicated final frame.
    off_view = extended_governor_log(False)
    on_view = extended_governor_log(True)
    return layout.finish(), off_view, on_view, "c3", "c4"


def _constraint_status(log: dict, index: int, role: str) -> str:
    if "sp_active" not in log:
        return "nominal controller"
    active = set(
        np.flatnonzero(log["sp_active"][index].astype(bool)).tolist()
    )
    if role == "c1":
        names = {0: "whole-port energy floor", 2: "whole-port CBF"}
    else:
        names = {
            0: "human energy floor",
            1: "residual energy floor",
            2: "human-energy CBF",
            3: "residual-energy CBF",
            4: "human power limit",
        }
    labels = [names[i] for i in sorted(active) if i in names]
    return "QP: " + (
        ", ".join(labels) if labels else "no safety limit active"
    )


def _governor_status(log: dict, index: int, role: str) -> str | None:
    if role != "c4" or "sp_gov_mode" not in log:
        return None
    mode = GovernorMode(int(log["sp_gov_mode"][index]))
    labels = {
        GovernorMode.TRACK: "Governor: normal tracking",
        GovernorMode.INTERACT: "Governor: interaction",
        GovernorMode.RESUME: "Governor: smooth resume",
        GovernorMode.DECEL: "Governor: decelerating",
        GovernorMode.HOLD: "Governor: anchored hold",
        GovernorMode.RELEASE_DWELL: "Governor: release dwell",
    }
    return labels[mode]


def _pane_status(
    log: dict,
    index: int,
    role: str,
    *,
    show_reference: bool = False,
) -> str:
    time_now = float(log["time"][index])
    force = log["f_h"][index]
    magnitude = float(np.linalg.norm(force))
    lines = [f"t = {time_now:5.2f} s", _constraint_status(log, index, role)]
    governor = _governor_status(log, index, role)
    if governor:
        lines.append(governor)
    if show_reference:
        symbol = "x_g" if role == "c4" else "x_d"
        lines.append(f"Reference {symbol}: {float(log['x_desired'][index]):.3f} m")
    if magnitude > 1e-3:
        interaction = "blocking" if force[0] < 0 else "helping"
        lines.append(f"Human {interaction}: {magnitude:.2f} N")
    else:
        lines.append("Human force: 0 N")
    return "\n".join(lines)


def _set_human_border(ax: plt.Axes, active: bool) -> None:
    for spine in ax.spines.values():
        spine.set_color(HUMAN if active else "#aaa9a5")
        spine.set_linewidth(3.0 if active else 1.2)


def _open_encoder(path: Path, width: int, height: int, fps: int):
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is required; install the repository requirements"
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def render_video(
    figure_number: int,
    output: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> None:
    builders = {3: build_figure3, 4: build_figure4, 8: build_figure8}
    layout, left_log, right_log, left_role, right_role = builders[
        figure_number
    ](width, height)
    render_width = max(640, int(width * 0.46))
    render_height = max(360, int(height * 0.38))
    renderers = (
        LogRenderer(render_width, render_height),
        LogRenderer(render_width, render_height),
    )
    logs = (left_log, right_log)
    roles = (left_role, right_role)
    show_reference = figure_number == 8
    if show_reference:
        for ax in layout.view_axes:
            ax.text(
                0.982,
                0.955,
                "\u2193 reference setpoint",
                transform=ax.transAxes,
                color=REFERENCE,
                fontsize=8.5,
                fontweight="bold",
                va="top",
                ha="right",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="black",
                    ec="none",
                    alpha=0.62,
                ),
            )

    start = layout.x_min
    video_end = (
        min(float(left_log["time"][-1]), float(right_log["time"][-1]))
        if figure_number == 8
        else layout.x_max
    )
    frame_times = np.arange(start, video_end + 0.5 / fps, 1.0 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = _open_encoder(output, width, height, fps)
    if encoder.stdin is None:
        raise RuntimeError("failed to open video encoder input")

    try:
        for frame_number, time_now in enumerate(frame_times):
            for pane in range(2):
                log = logs[pane]
                index = int(
                    np.clip(
                        np.searchsorted(log["time"], time_now),
                        0,
                        len(log["time"]) - 1,
                    )
                )
                reference_x = (
                    float(log["x_desired"][index]) if show_reference else None
                )
                layout.view_images[pane].set_data(
                    renderers[pane].frame(
                        log,
                        index,
                        reference_x=reference_x,
                    )
                )
                layout.status_texts[pane].set_text(
                    _pane_status(
                        log,
                        index,
                        roles[pane],
                        show_reference=show_reference,
                    )
                )
                _set_human_border(
                    layout.view_axes[pane],
                    float(np.linalg.norm(log["f_h"][index])) > 1e-3,
                )

            # Figure 8's published plot is complete at 12 s; during the two
            # added simulation seconds it remains complete while both robot
            # views and their reference markers continue moving.
            layout.update_plot(float(min(time_now, layout.x_max)))
            layout.figure.canvas.draw()
            rgba = np.asarray(layout.figure.canvas.buffer_rgba())
            rgb = np.ascontiguousarray(rgba[:, :, :3])
            encoder.stdin.write(rgb.tobytes())

            if frame_number % fps == 0 or frame_number == len(frame_times) - 1:
                elapsed = time_now - start
                duration = video_end - start
                print(
                    f"  Figure {figure_number}: {elapsed:5.1f}/{duration:.1f} s",
                    flush=True,
                )
    finally:
        for renderer in renderers:
            renderer.close()
        plt.close(layout.figure)
        if encoder.stdin is not None:
            encoder.stdin.close()

    error = (
        encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.stderr
        else ""
    )
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {output}:\n{error}")
    print(f"wrote {output}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=(3, 4, 8),
        default=(3, 4, 8),
        help="paper figure numbers to render",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "videos",
    )
    args = parser.parse_args(argv)

    if args.width % 2 or args.height % 2:
        parser.error("H.264 output width and height must both be even")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    names = {
        3: "figure3_false_depletion.mp4",
        4: "figure4_mismatch_residual.mp4",
        8: "figure8_governor_comparison.mp4",
    }
    for figure_number in args.figures:
        render_video(
            figure_number,
            args.output_dir / names[figure_number],
            fps=args.fps,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
