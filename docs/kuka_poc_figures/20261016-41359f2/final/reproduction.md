# Reproduce the results-only build

Run from the repository root with the compatible Python environment and tectonic installed.
Inputs are preserved full-resolution executed signals, config/metadata, events, and independently audited all-matrix evidence. No simulation is run by the build. Output must not exist.

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/carl/code/contactSimulation/.venv/bin/python scripts/kuka_poc_figures.py --inputs docs/kuka_poc_figures/20261016-41359f2/inputs --out NEW_OUTPUT --reviews docs/kuka_poc_figures/20261016-41359f2/reviews
```

New simulation commands and frozen profiles are in `../support/new_runs_config.json` and `scripts/poc_figures_simulate.py --help`.

All power curves use actual point velocity; work uses signed physics increments. No missing prediction is replaced by zero. Reserve is an unexecuted forecast.
