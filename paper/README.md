# Paper bundle — results section + figures

Drop-in material for Overleaf.

## Contents

- `main.tex` — standalone wrapper: compiles the Results section on its own
  (set it as Overleaf's Main document). Contains a labeled stub setup
  section so all references resolve; for the full paper, skip it and
  `\input{results_section}` from your own main file.
- `results_section.tex` — the formal Results section (`\section{Results}`),
  with one subsection per implementation claim and all figure/table
  references wired up. Requires `graphicx`, `booktabs`, `amsmath`.
- `table_summary.tex` — Table I (all modes × tests), `\input` at the end of
  the results section; move it wherever your layout prefers.
- `figs/fig1_overview.pdf` … `figs/fig9_sweep.pdf` — vector figures sized
  for a two-column layout (single column 3.4 in, `figure*` 7.0 in).
- `make_figures.py` — regenerates every figure and the table from the
  experiment logs (run from the repository root:
  `python paper/make_figures.py`).

## Overleaf import

Upload `results_section.tex`, `table_summary.tex`, and the `figs/` folder,
then `\input{results_section}` from your main file. One `\ref` placeholder
(`sec:task`, in the opening paragraph) points at your task-setup section —
adjust or delete it.

## Data provenance

- Figs 2–7, 9 and Table I are generated from `results_passivation/`
  (iteration-1 runs). These are the runs whose numbers the paper text
  quotes: RMSE(F_n) = 0.0348 N, first activations 6.21 s vs 14.73 s, peak
  powers 0.1499 / 0.0997 W, cumulative 0.361 / 0.280 / 0.045 J, Test-E
  window RMSE 0.859 vs 2.522 N.
- Fig 8 (governor ablation) is generated from
  `results_passivation_iter2/governor_compare/` — the governor exists only
  in the iteration-2 code path. It is a task-layer policy independent of
  the ledger-row formulation, so the ablation is self-contained; its
  quoted numbers (0.68 → 0.050 m/s, 24 → 5.1 N) come from that pair of
  runs. If you later migrate the whole paper to the iteration-2 robust
  formulation, rerun the experiments and regenerate — the script only
  needs the log folders.
