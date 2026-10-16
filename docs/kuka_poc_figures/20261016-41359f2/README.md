# KUKA adaptations of the original 2-DOF PoC Figures 3–8

The results-first, editable English report is [final/report.pdf](final/report.pdf)
([Markdown](final/report.md), [TeX](final/report.tex)). Vector PDFs and PNGs are
in [final/figures](final/figures). The figure crosswalk, named plot data, exact
metrics, curve provenance and four-round review summary accompany the report.

Starting synchronized Stage 2: `41359f20430c930eec5afaebffa2cfbc482f07da`, identical
to the task reference. Numerical package archive:
`81dffa64109752a9a027975e7b831daa4ff91e86`, package SHA256
`3662a23fbcf913f4047e66361072808c75f98465dacf763d3136b13700cec80b`.
New branch: `feature/kuka-poc-figures-20261016-41359f2`.

No controller, model, gains, budgets, energy rules or safety logic were changed.
The new help-then-block profile composes two existing `HumanPulse` instances,
through the existing `force(t)` interface, in one simulation. Only the current
measured world force reaches the predictor. No hardware was accessed.

## Rebuild everything from preserved inputs

Run from the new worktree, `/Users/carl/code/contactSimulation-poc-20261016-41359f2`:

```sh
bash docs/kuka_poc_figures/20261016-41359f2/reproduce.sh NEW_OUTPUT_DIRECTORY
```

The output must not exist. This single entry point recomputes metrics, all six
figure sets, the editable report, PDF, and every rendered PDF page. It does not
run a simulation or query a future force profile. Dependencies are NumPy,
Matplotlib, Tectonic and Poppler (`pdftoppm`); installed versions and commands
are recorded in each build manifest. The wrapper uses the existing compatible
Python environment without installing into it. Set `POC_PYTHON=/path/to/python`
to use another environment. Numerical-run versions are separately recorded in
each input's provenance.

The direct implementation is `scripts/kuka_poc_figures.py`. Inputs live in
`inputs/`: full-resolution named executed samples for 13 selected/repeated
traces, configurations, complete event histories, source metadata/hashes,
independent 65-case audit evidence, and actual continuation logs. Original
new-run files (including q/qd and full initial/final states) for both help/block
repeats, the fresh no-human run and the fresh explicit-stop run are preserved
in `inputs/new_run_raw/`. All original archived full logs remain unchanged at
their exact hash/path references in each `provenance.json`; compact historical
archives remain in the inherited Stage 2 report tree.

No plotted trajectory includes a rejected rollout or post-stop padding. Energy
is plotted at `t+h`, power at its stage-coherent evaluation time `t`; work uses
all signed 0.25 ms increments. Raw energy is never lower-clamped. Undefined
normal position/velocity references under force control remain NaN and are
explicitly excluded from finite along-path plots. The curve manifest separates
executed observations, online references, offline diagnostics, unexecuted
reserve predictions and measured wall-clock timing.

## Independent verification

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/carl/code/contactSimulation/.venv/bin/python \
  scripts/poc_figures_audit.py \
  --bundle docs/kuka_poc_figures/20261016-41359f2/inputs \
  --reported-metrics NEW_OUTPUT_DIRECTORY/metrics.json --out NEW_AUDIT.json
```

The audit imports no plotting/controller metric helpers. It recomputes signed
and gross work, raw prefixes, force/tracking/recovery, common windows, activity
and progress. Retained signals are compared bytewise with the full originals
when those paths are available. Independent actual-point Jacobian and complete
task force/moment power checks on full raw logs are in the support audits.
Tests use actual subprocess exit codes and structured JUnit counts.

## Re-simulate the new frozen profile separately

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/carl/code/contactSimulation/.venv/bin/python \
  scripts/poc_figures_simulate.py run \
  --config docs/kuka_poc_figures/20261016-41359f2/inputs/new_runs_config.json \
  --name help_then_block_01 --out NEW_RUN_ROOT/help_then_block_01
```

Repeat with name/output `help_then_block_02`. Use `fresh_nohuman_D_01` or
`fresh_stress_D_01` for the corresponding regression runs. The stress run
returns **2**, saves its failure state, and remains stopped; exit 2 is never a
successful completion. Actual commands, child exits, wall times, seeds and
source/config/version checks are preserved next to the four raw runs.

For a fresh portable input bundle, use `scripts/poc_figures_prepare.py --help`.
Supply the preserved Stage 2 run root, the new-run root, support folder,
`--matrix-audit support/matrix_audit_02.json`,
`--reproduction-proof support/new_runs_reproduction_02.json`, and a new `--out`.
The standalone `scripts/poc_figures_reproduction.py --help` compares the fresh
repeats and reused exact-source main D evidence independently. The first checker
attempt encountered NumPy-bool JSON serialization; its partial files remain
preserved locally and are explicitly excluded from the verified input bundle.
Only the complete `_02.json` proof is used.

## Scientific scope and preservation

F3/F5/F7/F8 are adapted comparisons. F4/F6's independent mismatch-accounting
questions are unsupported; one combined data diagnostic shows the implemented
H=R identity. There is no exact original-controller reproduction, no actual
KUKA whole-port baseline, no isolated governor toggle and no finite energy cap
or enforced human peak-power limit. New continuous charging is demonstrated;
the extra task-QP rows remain inactive in the recorded D scenarios. Recorded
solver counts do not exclude missing telemetry from discarded candidates.
The tiny-budget case stops, and 1 ms simulated control does not establish 1 kHz
wall-clock execution.

The original Stage 1 and Stage 2 worktrees, existing refs, tracked sources and
local-only evidence were inventoried before this work. Before/after SHA256
checks and exact counts are in `support/preservation_before.json` and the final
preservation records. Only this newly created output branch is allowed to
advance. Historical results/reports were not overwritten, reset or cleaned.
