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

## Round 2 — control, numerics and measured repairs (in progress)

- 1155 directional samples across 15 snapshots, including resistance onset/hold/
  release, reversal, held-out magnitude/timing and elbow. Eta .01/.02/.04 N m
  gradient differences <9e-18 J/(N m), but large trust-region commands change
  contact activity. Initial 2e-7 J affine margin falsified: 1.837e-5 J max
  calibration error, 1.951e-5 J held-out. Use 2.5e-5 J at 5 N, scaled by current
  force magnitude; normalized held-out maximum is 2.168e-5 J. This is an empirical
  envelope, not a uniform certificate. Exact no-human input still gives zero.
- Found a fast-path inconsistency: nominal acceptance initially omitted the
  affine reserve required by the declared QP. Fixed nominal feasibility to
  include the same force-scaled reserve; nonlinear check still uses prediction
  error only. This is a stricter criterion, not a weakened energy gate.
- Original .12 J / 2e-7 J run failed at 10.662 s. .20/.22/.235/.238 J budget
  probes also failed (latter two before the reserve repair); budget feasibility
  is nonmonotonic for a greedy one-tick controller. Retain these diagnostics.
- Explicit authority probe (240/160/100/40 N m trust and larger rates) still
  failed at 10.86 s and degraded the task; rejected it. Final bounds remain
  the ORIGINAL 80/60/40/20 N m trust and corresponding rates.
- With calibrated margin and repaired fast path, torque-only .12 J failed at
  10.677 s. Task cost 1 completed: output .1199999984 J vs .2416232661 J off,
  min endpoint margin 1.55e-9 J. However peak force error 53.1 N and contact
  loss are unacceptable as an unchanged-quality sliding claim.
- Task cost 100 completed with .1199998801 J output, 2.47e-8 J min endpoint
  margin, peak force error 9.31 N, cross error 1.57 mm, orientation .917 deg,
  but .861 s geometric contact absence. Continue a measured normal-priority
  objective evaluation before freezing the task tradeoff. All these failures
  and tradeoffs must appear in the final report; no plant parameters changed.

Round 2 continued:
- Normal-priority cost (.1 N normal-force scale, .001 m/s normal-velocity scale)
  did not improve loss duration; rejected the extra weighting. Bounds unchanged.
- Solver whitening and exact consolidation of the equal energy rows preserve
  the QP optimum (independent dense-H KKT test). A maximum-iteration elbow
  snapshot had an affine-feasible box; added a seven-variable bound-active KKT
  recovery, tested by forcing OSQP to one iteration. Recovered the elbow solve
  and progressed to a distinct true affine-box infeasibility at 10.884 s.
  Never accepted an unconverged OSQP iterate or changed bounds/slack.
- Whitening exposed trajectory sensitivity of task weight 100: the .12 J case
  that completed with the old numerical solve now becomes affine-infeasible.
  Keep that failed comparison; do not select numerical settings solely to pass.
  Weight 1 .12 J completed with either parameterization (last measured net
  output .1199989829 J, endpoint margin 1.862e-7 J), while force error 53.1 N
  and 1.150 s contact loss remain real task-quality defects.
- .10 J / task weight 100 is another feasible measured fixture; .18 J and the
  held-out .12 J timing failed. Feasibility is nonmonotonic; no broad budget
  threshold is claimed. Stronger task constraints or viability management are
  outside this run. Final delivery will be marked PARTIAL even if its main
  active energy gate passes. Final source/configuration frozen before Round 3.

## Round 3 — final audit and reproduction

- Reviewed final command/step/account flow and diff for changed plant settings,
  hidden clipping/recharge, solver-success-only acceptance, shifted objectives,
  discarded failures and stale evidence. Plant XML is unchanged; nominal
  controller change is only the previous-intervention conditional integration.
- Finding (solver robustness): a feasible energy halfspace can intersect the
  torque box at exactly one vertex. The independent bound-active solver initially
  skipped the all-bound/nonzero-energy-multiplier case. Added multiplier-interval
  checks for that degeneracy, without relaxing constraints. Exact rerun:
  `.venv/bin/python -m pytest -q tests/test_joint_qp.py` → 5 passed in .11 s,
  including forced one-iteration OSQP and the unique feasible vertex.
- Core source frozen first at `daec55f`; the degeneracy repair is a follow-up
  source commit before final matrix reproduction. Fresh full suite, validation,
  baseline and sequential matrix must all use that final source hash.
- Final numerical/control revision `43a6661`. Fresh full suite:
  `.venv/bin/python -m pytest -q` → **61 passed in 70.03 s** (`final_tests.txt`).
- Fresh baseline `.venv/bin/python -m kuka_table_sim.scenarios nominal --log
  results/oracle/baseline_final.npz` → PASS, every physics/control array bitwise
  equal to the original 25 s baseline. `git diff dbff1c1 -- assets` is empty.
- `.venv/bin/python -m kuka_table_sim.oracle_validation --out
  results/oracle/validation_final` → PASS: 1155 samples / 15 snapshots,
  independent replay work error 0 J; max eta difference 8.47e-18 J/(N m).
  Normalized held-out affine overprediction 2.16785e-5 J < 2.5e-5 J.
- Sequential fresh-process matrix `.venv/bin/python -m
  kuka_table_sim.oracle_experiments matrix --out results/oracle/final` ran all
  18 cases, deliberately exits **2**. Main .12 J / task weight 1 completes:
  net output .119998982855545 J vs .241623266070888 J off, endpoint margin
  +1.86166e-7 J. Held-force prediction error 0 J. Both accounts identical.
- Independently reassembled logged physical constraints: largest torque/rate/
  trust residual 1.11512e-9 N m (<1e-7); affine energy residual 2.14811e-16 J
  (<1e-10). Tests independently challenge shifted centers, previous EXECUTED
  rate bounds, zero/near-zero gradients, infeasible rows and nonlinear retry.
- Manual numeric interval trace (`interval_trace.json`): first intervention at
  10.285 s; four powers (-.2261883, -.1680011, -.1097810, -.0515230) W give
  W_H=-.000138873346699213 J. E_before=.000161905068564909 J →
  E_after=.0000230317218656961 J. Torque mapping error 0 N m, account arithmetic
  error 2.71e-20 J. Task nominal and actual work equal; R=H.
- Blocking task finding: main active force error 53.104 N, geometric contact
  absence 1.14975 s; fixed ORIGINAL 10 N errors are reported as well as the
  FSM's varying f_d so SEARCH cannot mask this failure. Last-second recovery
  passes (.04581 N max force error, 3.840 mm along error); integrators remain
  bounded and recover. A successful energy gate is not successful task preservation.
- The main substep minimum is -1.11334e-6 J despite valid endpoints. Explicitly
  prohibit a continuous-time claim. Held-out active stops at 10.753 s with an
  empty affine halfspace/box intersection; tight elbow at 10.884 s. Failure
  snapshots and candidate histories are retained. The final .001 J stress
  case succeeds (net .0009996804 J), with its own substep undershoot; feasibility
  is nonmonotonic and a larger budget cannot be claimed universally feasible.
- Postprocessing findings/fixes: (i) the report originally assumed stress must
  fail, but the contract permits success; now validates either explicit outcome;
  (ii) plotting every eighth physics sample skipped the short 63.104 N force
  peak, so figures now use full physics force/power and show substep minima;
  (iii) numerical source provenance is verified against its recorded Git tree,
  permitting reporting-only revisions; (iv) retain nine representative compact
  raw cases, all failed snapshots and relevant candidate histories, about 7 MB
  including plots/reports, instead of committing full logs.
  Reporting commits: `3dba2f1`, `ddc02b6`, `25a8bf6`. Re-ran the report generator
  after each affected fix; simulation sources and final physics logs unchanged.
- Visually inspected final `active_comparison.png`, `failures_and_recovery.png`,
  `gradient_response.png`, `forecast_and_runtime.png`; figures agree with raw
  metrics. Force-forecast mismatch is measured separately: max 7.11783e-7 J
  when the plant varies force within a tick and prediction holds it constant.
- Runtime on Apple M4: main tick median .640 ms, p95 .684 ms, p99 3.013 ms,
  max 9.497 ms; total 11.540 s for 18 s simulated. No 1 kHz wall-time claim.
- Independent final standalone process reproduces main physics/control/work/
  candidate/account arrays bitwise (`fresh_process_reproduction.json`).
- Checked every physics row in all 18 cases: no unexpected contacts
  (`unexpected_contacts.json`), so the oracle's two modeled ports are not
  omitting an observed third contact. All installed global callback slots are
  unset (`callback_inventory.json`), including contactfilter/time callbacks.
- Final manifest: 13 PASS, 2 FAIL (task quality and held-out active); overall
  **PARTIAL**, not complete. The remaining blockers need a new scope decision
  (task constraints / viability reserve), not hidden changes to this fixture.

| round | evidence / outcome | implementation commit |
|---|---|---|
| 1 physics | independent point/wrench/frame/work/replay checks pass | `a57d37f` |
| 2 control | empirical margin, reserve fast path, scaling and failure evidence | `daec55f` |
| 3 reproduction | vertex repair, 61 regressions, final matrix/figures; partial result explicit | `43a6661` + reporting commits above |
