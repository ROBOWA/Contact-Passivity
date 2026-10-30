"""Side-by-side animation of the SAME scenario/seed across controllers (GIF + MP4).

  python scripts/make_animation.py --tag main --scenario block_release --seed 100 --controllers C0 C1 C3 C4
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from hcu2d.viz import plt, style, CTRL, LABEL, INK, INK2, MUTED
from fig_closed_loop import load_trace, draw_scene
from matplotlib.animation import FuncAnimation, PillowWriter

FFMPEG = "/opt/anaconda3/envs/517_drake/bin/ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--scenario", default="block_release")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--controllers", nargs="+", default=["C0", "C1", "C3", "C4"])
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    style()
    trs = [(ck, load_trace(a.tag, a.scenario, ck, a.seed)) for ck in a.controllers]
    trs = [(ck, tr) for ck, tr in trs if tr is not None]
    n = len(trs)
    fig = plt.figure(figsize=(13.5, 3.4 * ((n + 1) // 2) + 0.4))
    gs = fig.add_gridspec((n + 1) // 2, 4, width_ratios=[3.0, 1.15, 3.0, 1.15], wspace=0.32, hspace=0.32,
                          left=0.02, right=0.93, top=0.90, bottom=0.03)
    axes = [(fig.add_subplot(gs[i // 2, 2 * (i % 2)]), fig.add_subplot(gs[i // 2, 2 * (i % 2) + 1])) for i in range(n)]
    T = min(len(tr["t"]) for _, tr in trs)
    step = max(1, int(round(50 / a.fps)))
    frames = range(0, T, step)
    vmax = {ck: max(0.05, float(np.nanmax(tr["Ec"][:T])), float(np.nanmax(tr["true_BH_true"][:T]))) for ck, tr in trs}

    def draw(k):
        for (ck, tr), (ax, bx) in zip(trs, axes):
            ax.cla()
            bx.cla()
            short = {"C0": "C0 nominal — UNPROTECTED", "ORACLE": "ORACLE — truth access"}.get(ck, LABEL[ck])
            draw_scene(ax, tr, k, title=f"{short}   t = {tr['t'][k]:.2f} s", follow=True, fscale=0.012)
            # account bars (true BH, booked BH, backup margin, Ec)
            vals = [tr["Ec"][k], tr["true_BH_true"][k], tr["BH_book"][k], tr["true_hH_true"][k]]
            names = ["Ec", "BH true", "BH booked", "h_H=BH−S"]
            for i, (v, nm) in enumerate(zip(vals, names)):
                col = CTRL[ck] if v >= 0 else "#d03b3b"
                bx.barh(i, v, color=col, height=0.6)
                bx.text(1.03, i, f"{v:+.3f} J" + (" VIOL." if v < -1e-6 else ""), transform=bx.get_yaxis_transform(),
                        ha="left", va="center", fontsize=6.5, color="#d03b3b" if v < -1e-6 else INK2, clip_on=False)
            bx.axvline(0, color=INK, lw=0.8)
            lim = vmax[ck]
            bx.set_xlim(-0.25 * lim, 1.05 * lim)
            bx.set_yticks(range(4), names, fontsize=7)
            bx.invert_yaxis()
            bx.set_xticks([])
            bx.grid(False)
            c = tr["cand"][k]
            bx.set_title(("backup u=0" if -1.5 < c[0] < 0 else "brake u=-Dq̇" if c[0] <= -1.5 else f"σ_v={c[0]:.1f} σ_K={c[1]:.2f} σ_F={c[2]:.1f}")
                         + ("  FILTER" if tr["filt"][k] > 0 else ""), fontsize=7, color=INK2)
        fig.suptitle(f"Scenario '{a.scenario}', seed {a.seed} (same hidden human intention, gains, location and noise for all panels). "
                     "Red dot: true human contact; blue band: location belief ±1σ/±2σ; arrows: forces", fontsize=8.5)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / a.fps, blit=False)
    out = ROOT / "outputs" / "animation"
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{a.scenario}_seed{a.seed}_{'_'.join(c for c, _ in trs)}"
    anim.save(str(base) + ".gif", writer=PillowWriter(fps=a.fps), dpi=72)
    if Path(FFMPEG).exists():
        from matplotlib.animation import FFMpegWriter
        plt.rcParams["animation.ffmpeg_path"] = FFMPEG
        anim.save(str(base) + ".mp4", writer=FFMpegWriter(fps=a.fps, bitrate=2400), dpi=110)
    print(base)


if __name__ == "__main__":
    main()
