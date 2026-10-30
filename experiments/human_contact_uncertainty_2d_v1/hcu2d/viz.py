"""Shared figure style (validated categorical palette, entity-fixed controller colours)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Validated light-mode categorical slots (validate_palette.js: all hard gates pass; slots 3-5 < 3:1
# contrast -> direct labels / table views provided). Colour follows the entity, never its rank.
CTRL = {"C3": "#2a78d6", "C4": "#eb6834", "C2": "#1baf7a", "C1": "#eda100", "C0": "#e87ba4",
        "ORACLE": "#008300", "F": "#6250d6", "Finst": "#e34948"}
LABEL = {"C0": "C0 nominal (UNPROTECTED)", "C1": "C1 deterministic + filter", "C2": "C2 Gauss–Markov",
         "C3": "C3 structured GP, β=0", "C4": "C4 structured GP, β=2", "ORACLE": "ORACLE (truth access)",
         "F": "filter only", "Finst": "filter only (instantaneous)"}
PRED = {"S": "#2a78d6", "S0": "#6250d6", "N": "#eda100", "GM": "#1baf7a", "HOLD": "#e87ba4", "truth": INK}
PRED_LABEL = {"S": "structured GP (+residual)", "S0": "structured GP (no residual)", "N": "naive GP",
              "GM": "Gauss–Markov", "HOLD": "hold-last", "truth": "truth"}


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": SURF, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.6, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": ["Helvetica Neue", "Arial", "PingFang SC", "sans-serif"]})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
