# Stage 2 evidence-backed reviews

## Round 1 — mechanics, preservation and numerics

Frozen reproduction: 61 tests passed in 76.27 s. Main/historical/no-human/ample
new-namespace runs reproduce the published same-platform outcomes, including
53.104212 N maximum force error, 1.14975 s absence and historical stop at 10.753 s.

Eight focused tests pass (4.74 s). Checks include independent `mj_applyFT`
generalized-force power at all four prefixes; independently executed state and
observer replay; A-B-A candidate ordering; controller deep isolation and trusted
snapshot roundtrip; exactly-once accepted compute/observe; reset; clock integral
and chain rule; unequal account minima; coupled-active analytic QP optimum
(0.6,0.4); a contradictory multi-row problem; and contact backup convergence,
prefix reserve recomputation and exact cached held-force execution.

Concrete findings and repairs:

- The one-second backup horizon did not reach persistent human-point settling.
  At a representative main snapshot it settled at 1.55 s. Extended the declared
  maximum to 2 s; 20/50/100/250/500/1000 ms unresolved results remain logged.
- Pickle memo/array alias representation changed despite equal plant/controller
  values, disabling cache reuse. Replaced byte-representation hashing with exact
  recursive dynamic-value hashing; no rounding or force tolerance. The contact
  test now exercises three exact cache hits and independently measured work.
- Prefix account rows originally assumed zero minima internally. Parameterized
  both account minima and added unequal-margin testing.
- A reporting window passed `None` to the frozen metrics helper. Changed it to
  the actual final evaluation time. Failed development raw logs remain intact.

The new solver never imports or calls the old one-halfspace active-set recovery.
It checks every energy/task row and box in physical units after OSQP. QP failures
retain the full affine matrix and bounds, distinguishing solver nonconvergence
from affine infeasibility. Affine infeasibility is not global nonlinear proof.

The old numerical files, plant assets, force target and gains remain unmodified.
The original-namespace hash/ref verification is recorded separately. The backup
reserve is a finite-horizon policy estimate, not a certificate. The 20 ms allowed
reaction delay is included in reserve, prefix and task validation; all prediction
uses only the currently measured world-force vector.

## Round 2 — control behavior and causal evidence

Development comparisons are in `development_01`, `development_02`,
`development_03`, and `development_transparency_01` under the new results root.
These exploratory processes span edits; they are not frozen-source final evidence.
Both raw logs from the early reporting failure were retained and their summaries
recovered as NEW `metrics_recovered.json`, without replacing the failed run files.

Evidence and decisions:

- A reproduces the main 53.104 N error/contact loss and historical 10.753 s stop.
  B preserves every executed prefix but stops at 10.286 s (main) and 10.753 s
  (historical). Its first-prefix row is torque insensitive: current point velocity
  cannot be instantaneously changed by that interval's held command. Main B still
  produces an 18.439 N force error. Prefix enforcement alone is insufficient.
- Preliminary D completes main and historical with maximum errors 1.243 and
  1.047 N, no contact loss, positive physics energy. Main measured progress is
  87.5% of UF; pause 0.482 s and alpha recovery 0.041 s. These are preliminary
  measurements; all final conclusions require the fresh frozen-source reruns.
- An early C attempt stopped at 10.185 s after retarget overshoot and reserve
  rejection. Exact cache validation and preserving its verified suffix matter.
  Fixed selection to retain an admissible suffix after an expired progress plan;
  also offer the actual damped backup immediately rather than inserting an
  unnecessary nominal first tick. Tests explicitly force replanning rejection,
  verify suffix execution, invalidate on changed force, and check no rejected
  candidate mutates the live hybrid state. Revised main C completes with
  1.2415 N maximum error and zero absence. No future-force timing is used.
- Zero-human/zero-budget A/B/C/D and ample C are BITWISE identical to UF in
  q, qdot, actuator torque and physics normal force. Alpha remains one. This is
  stronger than merely equal task summaries; final tests repeat the comparison.
- Boundary backup checks at the same main snapshot settle at both human points;
  required reserves are 0.0298721 J (link5) and 0.0173450 J (link3). A 0.001 J
  account cannot cover either. Reduced rate factor 0.02 is explicitly a simulation
  challenge, not a hardware claim. Full matched boundary trials remain required.
- Independent 20 ms closed-loop execution matches plant state exactly. The
  within-tick varying-force shadow challenge differs by 2.47628e-5 J; numerical
  replay margins do NOT cover unknown future force changes.
- Central .01/.02/.04 stencils agree locally, but mixed 20%-trust-box commands
  show force affine errors up to 56.7 N and three contact-mode changes near
  6.701 s. No uniform affine task-error certificate is claimed. Fresh nonlinear
  validation is mandatory. Raw validation errors are retained.
- Complete failed-forecast terminal states and four-stage traces are now retained
  in rejection diagnostics. Terminal horizon summaries include measured speed,
  power, terminal time and persistence. Backup and execution check original trust
  bounds as well as torque/rate bounds. Unbounded QP diagnostics use explicit
  Infinity strings, so a strict-JSON failure record cannot lose the trial.

The source/settings freeze follows this round. The 10 N target, original plant,
path/gains and 8.71 s window are unchanged. Targets stay MAE <=0.5 N, max error
<=3 N, cross <=3 mm, orientation <=2 degrees, >=80% UF progress, alpha >=0.95
within 2 s of release, force settling within 0.5 N for 0.5 s, no permanent pause,
and every physics energy margin >=-1e-10 J. Defaults and the NEW seed-20261015
held-out manifest are frozen before any new held-out evaluation. No tuning after
held-out outcomes is permitted without relabeling and a fresh held-out set.

Reserve remains a held-current-force finite-horizon policy estimate. Both C and D
can have nonbinding task rows; if so, the improvement belongs to reserve/governor,
not evidence that extra task constraints independently improved those cases.

## Round 3 — final reproducibility and acceptance

Numerical/control freeze: `81dffa64109752a9a027975e7b831daa4ff91e86`, package SHA256
`3662a23fbcf913f4047e66361072808c75f98465dacf763d3136b13700cec80b`.
Every final case's start/end package hashes match that Git tree. Assets,
configuration and the predeclared held-out manifest match their frozen hashes.
No production numerical/control/configuration edits followed the freeze. Later
commits add an independent physical regression witness and reporting/audit code.

Fresh-process evidence:

- Full suite: **70 passed in 167.94 s** (`final_tests_02.txt`); the prior
  69-test frozen-source transcript remains preserved. Review found the task rows
  were nonbinding in successful primary runs. Added an explicitly labeled 20 N
  downward nominal-command-fault test, with all default settings unchanged. It
  exercises all 32 energy/task rows, corrects the force-band violation, and
  independently executes the accepted complete torque with `mj_applyFT` and split
  stepping. All four work/force samples and the full plant state match. This is
  coverage, not tuning against held-out outcomes.
- Required 13 fixtures x UF/A/B/C/D = **65 final runs**, with no process errors.
  C and D each complete 12/13; the 0.001 J stress trial stops explicitly at
  9.533 s. A completes 9/13 and B 5/13. A has three observed physics-energy
  violations; B/C/D have zero. Completion is separate from contact/energy success.
- Main D: max error 1.241493 N, MAE 0.063038 N, zero absence, minimum physics
  account 0.000409180 J, measured progress 87.526% UF, alpha recovery 0.040 s,
  force settling 0.062 s (0.5 s sustained window). First intervention 10.140 s
  with 0.031941223 J, versus A 10.285 s / 0.000161905 J. Total torque difference
  from the isolated ungoverned controller reaches 65.546 N m; the governor's
  effect is not hidden by reporting only correction from governed nominal.
- Historical D completes 18 s, max error 1.045290 N, MAE 0.052591 N, zero
  absence, min physics account 0.000087668 J, progress 89.363% UF, alpha recovery
  0.058 s and force settling 0.078 s. Historical A stops at 10.753 s.
- All three untouched held-outs pass D completion, energy, contact, progress and
  recovery: measured progress ratios 91.061%, 86.928%, 95.457%. Historical remains
  a known regression. No held-out-driven retuning occurred.
- No-human with 1 J and zero budget, both ample-budget contact locations, and
  assisting force are bitwise transparent in q/qdot/applied torque, with alpha=1.
  D's original force target remains exactly 10 N over every fixed task window.
  The 0.02-rate-authority challenge passes D; it is a simulation choice only.
- Independent raw-log recomputation over all 65 cases: maximum account error
  4.33e-15 J; all held-force work-prefix, stage-force and orientation prediction
  errors are zero; applied-torque log differences are zero. General multi-row
  residuals are independently recomputed from saved affine matrices: worst
  normalized violation 2.71e-10 (allowed <=1). Both account reserve inequalities
  pass independently. No predictions were substituted for executed energy.
- Fresh main D and stress D repeats reproduce the recorded physical metrics and
  failure reason exactly. From the stress snapshot, an independent original
  Simulation loop and separately coded backup/force application reproduce the
  forecast plant state exactly and all cumulative work within 5.42e-20 J. The
  rejected backup would reach -2.72705e-5 J in an ISOLATED SHADOW ONLY. It was
  correctly not executed by the online trial; this is not a completed stress task.
- Continuation to 20 s starts from complete final hybrid snapshots without
  resetting clock, controller or energy. Main and historical reach their matched
  UF 18 s amount after 0.668 and 0.546 additional physical seconds; both complete
  three measured strokes by 20 s. There is no permanent pause or clock catch-up.

Final review decisions and limits:

- The early-intervention hypothesis is supported for these fixtures. Timing alone
  did not meet the force band in the frozen diagnostic. Reserve/governor plus
  point-aware braking contribute; C and D coincide in successful primary trials,
  so the primary data do not demonstrate additional performance benefit from
  D's task rows. The active-row physical witness tests their implementation.
- The zero-force shortcut means zero reserve under the held-zero FORECAST. It
  neither predicts a person will remain inactive nor proves a settled terminal
  backup. D still freshly checks current task/energy prefixes. All finite-horizon
  results remain conditional on the current-held-force assumption. The 100 Hz
  mismatch challenge differs by 2.47628e-5 J; no robust future-force guarantee,
  invariant set, recursive feasibility, or hardware fallback is claimed.
- Reviewed full-resolution main/historical, recovery, energy and feasibility
  plots. Colors are consistent across panels; the 10 N target and [7,13] band
  remain fixed; stopped curves end at failure; path simplification is disabled.
  Symlog energy detail exposes A's -1.11334e-6 J substep deficit. A clipped heatmap
  caption was repaired in reporting only, then all figures regenerated into a
  fresh `final_02` directory. The first 326-file reporting attempt is retained
  byte-for-byte under `results/.../report_attempt_01`; see reporting_attempts.json.
- Per-component and full-tick timing includes allocation/serialization/logging;
  failed-tick time is separate and included in total wall time. Forecast ticks
  take far longer than 1 ms. Matrix jobs can share CPU. No 1 kHz claim is made.
- Final preservation audit passes all 428 tracked files, 540 ignored/untracked
  files and their independently hashed 5,238,703,332-byte backup. Original
  worktree is clean at `32c4866`; protected local/remote branch tips are unchanged.
  Old reports/plots/results have not been regenerated or replaced.

Final status: **PASS for the declared primary engineering and validation gates**,
with an explicit incomplete low-budget stress trial. Full per-case statuses and
separate correctness/completion/energy/contact/progress/preservation/generalization
categories are in `final_02/acceptance.json`. Production source stays frozen;
the next research question is robustness to changing human force and usable
braking viability beyond this finite held-force oracle setup.
