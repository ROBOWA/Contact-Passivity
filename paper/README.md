# Paper bundle — results section + figures

Drop-in material for Overleaf, generated entirely from the implemented PoC
and one coherent fresh dataset.

## Contents

- `main.tex` — standalone single-column report (set it as Overleaf's Main
  document). It contains the PoC setup and the two setup figures before
  importing `results_section.tex`.
- `results_section.tex` — the formal Results section (`\section{Results}`),
  organized around the four core selective-passivation comparisons:
  C0 vs C3 (why a filter), C1 vs C3 (model-work exemption), C2 vs C3
  (dedicated human port), and the C3/C4 governor comparison. Includes a
  compact controller-roles table. Requires `graphicx`, `booktabs`,
  `amsmath`.
- `table_summary.tex` — supplementary complete-run metrics for C0--C3. The
  concise standalone report states the headline values in the text instead
  of including this dense table.
- `figs/fig1…fig8.pdf` — figures numbered to match the compiled paper.
  Figure 1 combines the planar drawing with a rendered MuJoCo snapshot;
  Figure 2 is the full-width vector controller architecture. `figS1` and
  `figS2` are the supplementary governor-on counterparts of Figs. 6 and 7
  (not referenced from the LaTeX).
- `make_figures.py` — regenerates every figure and the table from the
  experiment logs, including the Figure 1 snapshot from a recorded Test-C
  state (run from the repository root:
  `python paper/make_figures.py`).
- `make_videos.py` — creates presentation videos for Figures 3, 4, and 8.
  Each 16:9 video synchronizes two titled MuJoCo views with a live version
  of the corresponding paper plot. Scripted human force is shown as a
  magenta arrow attached to the end effector. Figure 8 also shows each
  controller's tangential reference setpoint as a gold downward arrow.

Generate all three PowerPoint-ready H.264 videos from the repository root:

```bash
python paper/make_videos.py
```

The outputs are written to `paper/videos/`. Use `--figures 8` to render only
one video, or change `--fps`, `--width`, and `--height` for a smaller preview.

## Data provenance (single coherent dataset)

All figures and the table are generated from `results_passivation_iter2/`,
produced by the implemented PoC via

```bash
python -m mujoco_sliding.experiments --all --sweep --governor-compare
```

The paper compares C0 (nominal), C1 (whole-port baseline), C2
(residual-only), C3 (dual-ledger QP without the governor), and C4 (the same
dual-ledger QP with the governor). In implementation terms, paper C3 and C4
are both `C3_dual_ledger_qp`, with the governor flag off and on respectively.
The historical `C4_dual_ledger_safe_scalar` mode remains implemented and
tested as a separate diagnostic, but does not appear in the report. No
iteration-1 data or mode names are used anywhere.

### Clipping convention

Some filter-behavior figures clip post-interaction intervals whose only
purpose is recovery (the governor figure always shows the complete
interaction and release). **Only evaluation and presentation are ever
clipped**: the simulations run the full duration, the control commands /
forces / ledger updates / saved logs are never clipped, and all
certificate quantities (ledger minima, violation counts) use the complete
runs.

## Overleaf import

Upload the folder contents to your project **root** (`main.tex`,
`results_section.tex`, `table_summary.tex`, `figs/`). For standalone
compilation set *Menu → Settings → Main document* to `main.tex`. For the
full paper, the setup figures can be taken from `main.tex` and the results
from `results_section.tex`; narrow figures use `\figwidth` (defaults to
`\columnwidth`; predefine it for single-column layouts as `main.tex` does).

## Known provenance limitation

The Results section describes the robust (iteration-2) constraint rows
(prediction-error-bounded lower power estimates, conditional certificate).
If your methods section still shows the older row formulas
(`min(0, ·) − δ_p`) or the old mode names, update it to match — see
`mujoco_sliding/passivity_qp.py` and the README section
"Selective-passivation layer" for the current formulation.
