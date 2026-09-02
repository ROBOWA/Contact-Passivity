# Paper bundle — results section + figures

Drop-in material for Overleaf, generated entirely from the CURRENT robust
(iteration-2) implementation and one coherent fresh dataset.

## Contents

- `main.tex` — standalone single-column wrapper: compiles the Results
  section on its own (set it as Overleaf's Main document). Contains a
  labeled stub setup section so all references resolve; for the full
  paper, skip it and `\input{results_section}` from your own main file.
- `results_section.tex` — the formal Results section (`\section{Results}`),
  one subsection per implementation claim, including the **Test E1
  filter-allocation ablation** (evaluated on the full-force plateau only).
  Requires `graphicx`, `booktabs`, `amsmath`.
- `table_summary.tex` — Table I (all modes × tests). The Test E1 rows
  report plateau-window task RMSE (marked †, explained in the caption);
  all certificate columns are full-run quantities.
- `figs/fig1_overview.pdf` … `figs/fig9_sweep.pdf` — vector figures sized
  for a two-column layout (single column 3.4 in, `figure*` 7.0 in).
- `make_figures.py` — regenerates every figure and the table from the
  experiment logs (run from the repository root:
  `python paper/make_figures.py`).

## Data provenance (single coherent dataset)

All figures and the table are generated from `results_passivation_iter2/`,
produced by the current robust implementation via

```bash
python -m mujoco_sliding.experiments --all --sweep --governor-compare
```

Controller modes are the current ones: `C1_whole_port_qp` (whole-port QP
baseline) and `C4_dual_ledger_safe_scalar` (safe-anchor scalar
interpolation). No iteration-1 data or mode names are used anywhere.

### Test E1 clipping convention

E1 metrics and Fig. 6 are clipped to the full-magnitude plateau
`[t_start + t_rise, t_start + t_rise + t_hold]` (= 6.5–8.5 s), derived
programmatically from the scenario's `HumanForceConfig` — the bounds are
not hard-coded. **Only evaluation and presentation are clipped**: the
simulations run the full duration, the control commands / forces / ledger
updates / saved logs are never clipped, and all certificate quantities
(ledger minima, violation counts) use the complete runs.

## Overleaf import

Upload the folder contents to your project **root** (`main.tex` optional,
`results_section.tex`, `table_summary.tex`, `figs/`). For standalone
compilation set *Menu → Settings → Main document* to `main.tex`. For the
full paper, `\input{results_section}` from your main file; the narrow
figures use `\figwidth` (defaults to `\columnwidth`; predefine it for
single-column layouts as `main.tex` does). One placeholder remains:
`\ref{sec:task}` should point at your task-setup section.

## Known provenance limitation

The Results section describes the robust (iteration-2) constraint rows
(prediction-error-bounded lower power estimates, conditional certificate).
If your methods section still shows the older row formulas
(`min(0, ·) − δ_p`) or the old mode names, update it to match — see
`mujoco_sliding/passivity_qp.py` and the README section
"Selective-passivation layer" for the current formulation.
