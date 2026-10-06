# Panda selective-passivation implementation report

## Outcome

Both requested milestones are implemented and reproducible. Milestone 1
provides stable straight 5 N sliding, a known EE human port, C0--C3, exact
spatial power mappings, sampled-versus-physical work logs, plots, and videos.
Milestone 2 adds bounded 2-D wiping, a forearm port, friction and normal-force
prediction mismatch, release/recovery, a governor ablation, and a redundancy
comparison, including a controlled 2% controller-side inertia-model error.
The original planar backend was not rewritten.

The nominal straight run met the original targets. The nominal bounded-wipe
run also met them. The final four-element pad remained continuously active in
the nominal tuning run; the raw constraint count and force remain logged in
addition to the documented compliant-band contact metric.

| validation | force RMSE | planar RMSE | orientation RMSE | contact | joint violation |
|---|---:|---:|---:|---:|---:|
| straight, C0 | 0.302 N | 3.85 mm | 0.96 deg | 100% | 0 |
| bounded wipe, C0 | 0.378 N | 2.91 mm | 0.67 deg | 100% | 0 |

The contact/motion retuning replaces the single sphere with a four-element
finite pad, uses a soft activation boundary (`solref = 0.10 1`,
`solimp = 0.05 0.95 0.005`), a 70 ms force filter, revised normal gains, and a
600 ms cosine slide ramp. Against the previous setting, the raw-force standard
deviation fell by 98%, normal-velocity variation by 88%, slide-onset
acceleration p99 by 92%, and torque-rate p99 by 93%. Raw contact occupancy
increased from 48.3% to 100%. The complete measurements and a rejected
spring-pad trial are documented in `SMOOTHING.md`.

## Focused evaluation

All evaluation runs used the bound frozen after separate calibration. The
calibration maximum was 0.002180 rad/s; the frozen evaluation assumption was
0.003 rad/s. The C3 evaluation maximum was 0.002292 rad/s, with zero C3 bound
violations. C1's fallback regime exceeded the bound 228 times across A--E;
those violations and its failed certificate are retained in the logs. This is
empirical coverage for the calibrated C3 conditions, not a universal bound.

Key C3 results are:

| scenario | human input/output/net | peak output | residual net | QP fallback | protected-floor violation |
|---|---:|---:|---:|---:|---:|
| A accurate, no human | 0 / 0 / 0 J | 0 W | 0 J | 0 | 0 |
| B mismatch, no human | 0 / 0 / 0 J | 0 W | -0.103 J | 0 | 0 |
| C oblique EE block | 0 / 0.049 / -0.049 J | 0.148 W | -0.049 J | 0 | 0 |
| D 13.5 N forearm wipe, predictor + 2% inertia mismatch | 0.010 / 0.060 / -0.049 J | 0.138 W | -0.103 J | 0 | 0 |
| E release/recovery | 0 / 0.049 / -0.049 J | 0.152 W | -0.049 J | 0 | 0 |

In scenario C, nominal C0 output 0.263 J through the human port; C3 reduced
that to 0.049 J. It did so by slowing/retreating rather than preserving the
task: over the sliding phase C3 force, planar, and orientation RMSE rose to
2.87 N, 15.7 mm, and 3.02 degrees, and maintained-contact fraction fell to
89.6%. This is the intended safety/task tradeoff, not a claim that redundancy
always permits continued wiping. By contrast, forearm scenario D provided a
usable redundant response: C3 maintained contact and stayed at 4.26 mm planar
RMSE while limiting net human extraction to the available charged budget.
After the human release in E, C3 re-established full contact and achieved
1.68 mm post-release planar RMSE over the final 1.5 s.

C2 and C3 use identical residual settings. With the oracle task model, C2's
larger residual account allowed scenario C's 0.263 J extraction; C3's smaller
nested human account isolated and limited it. Under mismatch, B and D show
the prediction error in residual work without fitting away the human channel.

## Failed and limiting cases

C1 is not reliable at small whole-port budgets. Its account includes normal
task/contact work, and in the focused blocking run it entered the documented
bounded fallback 1,535 times, crossed its protected floor 1,472 times, and
exceeded the calibrated prediction bound 120 times.
The sweep showed the tradeoff clearly: at usable budgets 0.04, 0.10, and
0.28 J, C1 had 2,077, 1,269, and 1,535 fallback steps respectively; at
0.58 J it had none, but then allowed the same 0.263 J human output as C0.
Those fallback segments are not certified.

The deliberately stronger 15 N forearm stress run remained feasible with no
protected-floor or prediction-bound violation. It is retained separately so
the 13.5 N focused evaluation was not selected to force a controller ranking.
The actual infeasible behavior in this evidence set is C1's small-budget
whole-port case; every such fallback is rate/torque-bounded and uncertified.

The governor ablation used C3 on both sides. With the governor off, the QP
alone limited human output to 0.0495 J. Enabling it did not change that energy
result materially, but reduced planar RMSE from 12.1 to 6.04 mm by stopping
the moving reference after the QP became active; maximum reference progress
lag was 44.79 mm and returned to zero after release. Thus the filter's effect
is visible before reference stopping, rather than being credited to the
governor.

## Work audit and claim boundary

Every ledger update stores the exact held generalized-wrench sample paired
with realized next-step arm velocity. In parallel, task contact work is
trapezoid-integrated from each MuJoCo contact point's wrench and twist. Across
the focused C3 runs their absolute final difference ranged from 0.000037 to
0.00649 J. The remaining discrepancy comes from held-sample timing, changing
Jacobians, and implicit integration rather than the former contact pulse train;
these effects appear in the logged prediction error. Human held-sample versus trapezoidal physical-work
discrepancy stayed below 0.000085 J in the focused C3 interactions. The
0.003 rad/s margin covers the measured one-step joint-velocity discrepancy in
calibration and evaluation, but does not turn
the sampled certificate into a continuous-work guarantee.

The supported claim is selective discrete port-energy regulation under the
documented known-wrench/location, predictor, one-step model, numerical
feasibility, and empirical error-bound assumptions. The results do not prove
unconditional global passivity, universal continuous-time work bounds,
human-force estimation, or complete human safety.

## Verification

The new focused suite covers actuator mapping and limits, spatial
wrench/twist and generalized-power identities, individual contact-point
power summation, oracle residual identities, predictor channel isolation,
affine dynamics versus an actual MuJoCo transition, C3 constraints, and
sampled/physical work accounting. Final test commands and exact generated
artifacts are listed in `README.md`. The final full run was 86 passed and one
documented pre-existing planar C1 expected failure in 44.53 s.
