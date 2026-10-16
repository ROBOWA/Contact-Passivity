# Reproduce the selective comparisons

Worktree: `/Users/carl/code/contactSimulation-selective-20261016-263136e`.
Branch: `feature/kuka-selective-comparisons-20261016-263136e`.
Numerical freeze: `dfc09dfcb402cdc99dd8138d5d71f0c6e8ea205e`.
Reporting files may be committed later. All earlier Python/control/plant files
are unchanged; the implementation adds the five `selective_*.py` modules.

From this worktree, with the existing environment:

```sh
SELECTIVE_PYTHON=/Users/carl/code/contactSimulation/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$SELECTIVE_PYTHON" -m pytest -q --junitxml=tests_reproduction.xml
"$SELECTIVE_PYTHON" scripts/selective_matrix.py --out results/selective/reproduction --jobs 4
```

The final matrix contains 30 primary runs (six scenarios times five controllers)
and two small-budget challenges. The matrix exits **2** if any declared trial
stops, **1** for unexpected subprocess/missing-output errors, and **0** only if
all trials complete. Exit 2 is retained in the recorded final evaluation; it
must not be relabeled as every controller passing. Per-case CLI behavior is
the same: 0 complete, 2 explicit stop. Output directories must be new.

Single run:

```sh
"$SELECTIVE_PYTHON" -m kuka_table_sim.selective_experiments case \
  --case C_blocking --mode K4 --out results/selective/single_reproduction
```

The final reporting command was:

```sh
"$SELECTIVE_PYTHON" scripts/selective_report.py \
  --root results/selective/final \
  --out docs/selective_comparisons/20261016-263136e/final
```

For another report directory, place `tests_full.xml`, `tests_report_audit.xml`,
`nominal_regression.json` and `preservation_after.json` in its parent; the report
reads actual JUnit counts and independently recalculates physical/account
evidence. `--refresh` explicitly permits regeneration of reporting artifacts in
an existing directory; it does not modify the numerical runs. Tectonic is invoked
at `/opt/homebrew/bin/tectonic`. Change that reporting-only path on other
platforms. A new report output directory is required. The report command can
exit 0 with stopped trials when the evidence-consistency audit passes; trial
completion is always a separate column.

`final/raw/` contains full-rate (.25 ms physics, 1 ms control), float64
account/force/progress data for every run, metadata, final/failure integration
states and lossless gzip QP histories, including rejected/failed attempts.
Controller state blobs are trusted local pickle artifacts; reproduction uses
matching code. The complete original logs retain generalized forces, contact
moment/material-velocity samples, joint state, prediction errors and task
recorder data under ignored `results/selective/final/`. Those larger logs are
required to repeat the full physical-power audit; the compact tracked account
data suffices to regenerate the published curves and energy identities.

Regenerate all figures from the Git-tracked archive, without MuJoCo reruns or
the ignored full logs:

```sh
"$SELECTIVE_PYTHON" scripts/selective_plot_archive.py \
  --report docs/selective_comparisons/20261016-263136e/final \
  --out results/selective/figures_from_archive
```

The independent audit reconstructs cap discard, energy, human force/point
velocity and generalized-power identities, actual and nominal task power,
protected energy/CBF/peak constraints, actuator/rate/trust limits, accepted
affine/nonlinear residuals and all QP attempt counts. It also verifies exact
no-human transparency and bitwise K3/K4 agreement before the first governor
action. Report-audit corruption tests must reject incorrect cap/energy and
human-power signs and missing failed-QP counts.

Timings were collected during a four-process matrix, with no visualization or
per-step disk writes. They are reproducible telemetry, not a dedicated latency
benchmark or evidence of a real-time 1 kHz implementation.

`design.md` and `frozen_config.json` predate physical evaluation. Exploratory
`results/selective/dev_01` and `dev_02` remain intact. The only implementation
change after the exploratory cases added executed contact moment/material
velocity telemetry; no gains, constraints, budgets or governor parameters were
retuned. Final cases all use the committed numerical freeze. Preservation
before/after manifests cover 4,417 files (14,241,995,578 bytes) in the three
earlier worktrees. The sole allowed ref change is this new comparison branch.
