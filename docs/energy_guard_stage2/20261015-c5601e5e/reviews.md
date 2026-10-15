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

Pending frozen-source fresh-process matrix, tests, repeats and independent audit.
