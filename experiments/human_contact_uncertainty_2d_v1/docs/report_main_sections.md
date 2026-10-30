## 5. Closed-loop main comparison (tables T1-T5; figures `violations_main.png`, `completion_main.png`, `tradeoff_main.png`, `accounts_*.png`)

Setup: 8 scenarios × 5 controllers × 100 paired held-out seeds = 4000 episodes, 0 run errors.
Infeasible initialisation is reported separately (T6).

**5.1 The energy layer does what it is designed to do (tier C evidence for the tier A construction).**
* **C0 (unprotected).** It violates the true accounts in every episode of every human scenario. Net
  robot→human work is 1.1-2.1 J per episode (mean), against `BH0 = 0.15 J`, and peak human force is
  20-29 N.
* **C1-C4 (all with the Ec + BH tick-work filter).** Mean net work is 0.001-0.12 J and peak human force
  is 2-16 N.
* **Domain.** No body corner left the domain, except 2 low-BH episodes under C2.
* **Ec.** No Ec violation occurred in any of the 7050 filtered held-out episodes (all tags except smoke). Some of those runs used deliberately invalid credit, so this alone says nothing about the theorem's preconditions.
* **Energy audit.** The worst residual of `ΔS − W_a + W_H + ∫D_R + ∫D_T` over all 4000 episodes is
  given in T5. Booked credit never exceeded the true `∫D_T` (max excess in T5).

**5.2 Covariance-aware past-work accounting removes the residual violations of the deterministic ledger.**
Pooled over the five in-distribution scenarios (no human, block/release, assist/reverse,
normal/oblique, moment arm), 500 episodes per controller:

| | episodes with any true-account violation | one-sided 95% upper bound |
|---|---|---|
| C1 (no margin) | **7/500** (Wilson 0.7-2.9%): all `h_H` = BH − S, in normal/oblique 4 and moment-arm 3 | 2.6% |
| C2 | **0/500** | 0.60% |
| C3 | **0/500** | 0.60% |
| C4 | **0/500** | 0.60% |

C1's failures occur in the scenarios where the human pushes normally at an off-centre, uncertain
location. The booked BH then uses `ŝ`, and the true BH is lower by `G·(s − ŝ)`. C2-C4 carry that
error in their belief (`e_BH`, formulation §4.2) and tighten by `κσ_e`.

**This is the most clearly supported upgrade: an explicit, location-correlated past-work error
belief.** All three margin-carrying controllers deliver it, whether or not the structured GP is used.

**5.3 The structured GP does not improve closed-loop outcomes here; the simpler Gauss–Markov predictor is better.**
Paired differences (T3; s.e. over 100 paired seeds) show C2 beats C3 in every interaction scenario:
* lower positive robot→human work: −0.003 to −0.033 J (low BH −0.003, others −0.011 to −0.033; except OOD intent, +0.003 ± 0.003 J, i.e. equal within noise);
* lower normal-force RMSE: −0.06 to −0.30 N in distribution; −1.8 N in OOD intent; −1.5 N in low BH;
* equal or larger true minimum backup margin.

C3 hands about 2.5× more ticks to the servo filter (block/release: 20% vs 8%), with a higher
command-increment RMS.

The mechanism is visible in the logged predictions (seeds 100-101, all engaged calls). C3's predicted `h_H` s.d. is 10-35%
and its MC work s.d. 32-51% smaller than C2's (e.g. block/release: 4.4 vs 5.8 mJ for `h_H`; 4.1 vs 8.4 mJ for work). The
planner therefore operates closer to the backup boundary, without any gain in progress or coverage.
Sharper (and, in distribution, calibrated: §4, §7) predictions **spend** the human budget under this
objective, because only the constraints, not the objective, value BH. The reactive filter then
intervenes more.

A planner that should *exploit* better prediction would need an objective that values interaction
energy or margin (for example a larger `w_res`, or a work cost). That is a design choice, not
something the evidence here supports by default.

**5.4 Out-of-distribution intent (held-out oscillating family).**
* No account violations for C2-C4; C1 had 1/100.
* C3/C4 lose task quality: completion 95/100 and 93/100 (vs 100/100 for C1/C2); normal-force RMSE
  2.4 N (vs 0.6 N for C2); wiping coverage 0.96/0.94.
* The structured GP is over-confident on this family (offline NLPD 8.7). Its wrong extrapolations
  make most active candidates look infeasible, so it selects the brake (7.5%) or passive backup (2%)
  candidates during contact.
* **This is a regression of the proposed predictor**, and the main argument for an online
  calibration/innovation monitor (not implemented).

**5.5 Persistent sensor bias outside the calibrated prior (failure, all filtered controllers).**
In this scenario the force sensor is biased by (0.8, −0.5) N and the location sensor by 6 cm. The
priors assume 0.1 N and 2 cm.

The booked BH is optimistic by up to ≈ 81-84 mJ: the bias times the human-driven contact-point
displacement, which the force data cannot reveal because it is confounded with an intention
offset. After the human releases, the robot resumes wiping with `S ≈ 16-25 mJ` while the true BH is
far lower than booked.

| | true `h_H` violated | true BH violated |
|---|---|---|
| C1 | 95/100 | 9/100 |
| C2 | 75/100 | 0/100 |
| C3 | 98/100 | 26/100 |
| C4 | 98/100 | 17/100 |

C2 is least bad, most plausibly because its predicted uncertainty is wider (median predicted `h_H` s.d. 20 vs 13 mJ for
C3 in this scenario). None of the margins is calibrated for this error (`accounts_sensor_bias_100.png`). Section 6.7 tests a robust interval bound as a remedy
(post-hoc).

**5.6 Deliberately insufficient BH and infeasible initialisation.**

*Low BH* (`BH0 = 0.02 J`, while wiping needs `S ≈ 0.016 J`):
* All filtered controllers keep the accounts (only C1 has 1 `h_H` violation).
* Almost none can finish: 1-2/100 completed.
* Coverage: C2 0.86, C4 0.52, C3 0.40, C1 0.32.
* C2 had 2 domain exits (body corner below the wall).

*Infeasible initialisation* (preloaded 10 N contact, `BH0 = 0.004 J < S0 = 0.010 J`, so
`h_H(0) = −6 mJ`):
* This lies **outside the theorem's initial-condition requirement**. The 30/30 `h_H` "violations"
  are this initial state.
* The filter immediately extracts energy: it unloads the contact, so `S` drops from 10 to 0.9 mJ
  within 0.1 s and `h_H` becomes positive.
* After that the robot can only hold about 5 N.
* True BH and Ec never went negative. The task is infeasible under `BH ≥ S` (0/30 completed).

C0 shows the unprotected alternative (1.8 J of human work).

**5.7 Risk-sensitive objective (C4, β = 2) vs risk-neutral (C3).**
* Paired differences are negligible in distribution: |Δ W_H⁺| ≤ 0.002 J, Δ force RMSE within
  ±0.01 N, identical violation counts.
* C4 completes 2 fewer OOD episodes, and has 9 fewer true-BH violations under sensor bias (17 vs 26),
  but **neither difference is significant**: Wilson intervals overlap.
* In low BH, C4 improves coverage by +0.12 ± 0.03.

The β sweep (§6.2) shows the direction of the effect on stiffness and speed.

**5.8 Timing.** Wall-clock times come from 9 parallel workers on a 4P+6E-core M4, with another
agent's job running concurrently, so they are contention-affected.
* Planner p95 over all calls: C1 ≈ 14 ms, C2 ≈ 54 ms, C3/C4 ≈ 95-120 ms.
* A single-process measurement is in §6.8.
* None of the stochastic planners meets the 20 ms period in Python. **No real-time claim.**
