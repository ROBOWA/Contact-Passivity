# Evidence-backed review log

## Round 1 — physics and implementation

- Evidence: original baseline `results/oracle/baseline.txt`: PASS; original full
  suite: 50 passed in 60.96 s. New physical tests plus recorder regressions:
  `.venv/bin/python -m pytest -q tests/test_oracle_physics.py tests/test_recorder_infra.py`
  → 10 passed in 4.09 s.
- Independently checked: J^T f vs `mj_applyFT`, equivalent force/moment at body COM,
  positive/negative human power, nonzero-moment condim=6 task contact vs generalized
  constraint power, replay through independent `mj_step`, full-state isolation,
  candidate-order invariance, exact no-human H=R=0, half/base/double gradients.
- Trace: step1(x_k) → nominal compute once → integration-state snapshot → current
  world force held → freshly recomputed point/J → applyFT after restoring base
  qfrc → step2 → pre-integration human power + same-evaluation task power →
  left rectangle → account update only after four substeps. Prediction has a
  separate MjData and no reference/integrator/recorder access. Callback/unsupported
  actuator paths fail closed.
- Finding (fixture, blocking before comparison): the initial default pulse at
  12 s spans a direction reversal; the nominal point x velocity changes sign
  near 12.3 s. Sampling at 10–11 s shows +0.038 to +0.043 m/s; use a declared
  −x resistance pulse starting at 9.5 s, ending at 10.9 s, for main cases.
  This is fixture selection before evaluation, not a plant/trajectory change.
- Decision: basic physical gate passes; active candidate checks can now be
  exercised. Extended scan calibration and final baseline bitwise reproduction
  remain required for final acceptance.

## Round 2 — pending

## Round 3 — pending
