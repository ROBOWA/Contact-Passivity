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

Pending development comparisons and settings freeze.

## Round 3 — final reproducibility and acceptance

Pending frozen-source fresh-process matrix, tests, repeats and independent audit.
