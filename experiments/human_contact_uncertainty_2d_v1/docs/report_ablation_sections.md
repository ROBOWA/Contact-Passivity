## 6. Targeted ablations (50 paired held-out seeds each unless stated; tables T7-T13)

**6.1 Temporal covariance and constraint moments (T7).**
Variants: C3 (full cross-time covariance, first-order constraint moments), C3_indep (deliberately
independent-step latent covariance) and C3_mc (Monte Carlo constraint moments), on block/release,
assist/reverse and OOD intent.
* **Safety:** none of the 450 episodes violated any account.
* **Task metrics:** differences are small and not consistently in one direction.
* **MC constraint moments:** these remove the finite-difference linearisation (planner p95 ≈ 57 vs
  98 ms under load). They completed fewer OOD episodes (43/50 vs 48/50).

Offline, ignoring temporal covariance makes work predictions **≈ 3× over-confident** (T15). In this
closed loop that error does not surface, because safety is enforced by the servo filter and the
planner's tightening is rarely the binding constraint. **Retaining temporal covariance is necessary
for calibrated prediction (tier B claims) but was not decisive for closed-loop outcomes here.**

**6.2 Risk sensitivity (β = −1, 0, 2, 5; T8).** No violations for any β.
* **Selected stiffness rises monotonically with β.** Pooled σ_K = 0.662 / 0.667 / 0.671 / 0.673.
  Paired, β = 5 vs 0: +0.011 ± 0.004 (block/release), +0.008 ± 0.003 (normal/oblique).
* **Selected speed falls with β.** β = 5 vs 0: −0.004 ± 0.001.
* **Other metrics:** positive human work and force RMSE fall marginally (W_H⁺ 0.123 → 0.120 J;
  force RMSE 0.85 → 0.81 N).
* So risk aversion (β > 0, i.e. P3/P4 θ < 0) made the policy **slightly stiffer and slower, not more
  compliant**. This matches P3/P4's report that their risk-averse controllers raise feedback gains.
  The effect is ≈ 1-2% of the candidate range and **practically negligible**.
* **The empirical entropic estimator degenerates at large β.** The minimum effective sample size of
  the exponential weights fell to 2 (β = 2) and 1 (β = 5) of 48 samples in some calls. Those
  objective values carry large sampling error (delta-method s.e. logged; mean ≈ 0.015-0.017).

**6.3 Servo filter only, no planner (block/release, T9).**

| variant | worst true `h_H` | worst true BH | episodes flagged | interpretation |
|---|---|---|---|---|
| ORACLE (§5.2 filter with **true** BH) | −21 µJ | −13 µJ | 28/50 `h_H`, 19/50 BH at the strict −1 µJ threshold; **none below −0.1 mJ** | sampled-data numerical deficit of the tick-work filter, about 1e-4 of BH0 |
| filter on booked ledger, no margin | −2.4 mJ | −2.3 mJ | 32/50 | the location-induced ledger error |
| instantaneous-power filter | −80 mJ | −79 mJ | 50/50 | energy leak inside the ZOH hold |

All three complete **0/50** wipes. Without anticipation, BH is spent during the block, and afterwards
`BH ≥ S` forbids the stored energy that wiping needs. The ORACLE run validates the continuous-time
reference only numerically: it is close, not exact. The planner is what preserves the task.

**6.4 Residual on/off (T10).** No difference in safety, and negligible task differences: C3 without
residual has a slightly lower force RMSE in block/release, 0.70 vs 0.79 N. The residual's value is
only offline and in-distribution (§4).

**6.5 Task-dissipation credit (T11).**
* **Exact credit:** `min Ec ≈ 0.40-0.52 J`.
* **Conservative credit** (true μ = 0.35, booked at the interval's lower end 0.25):
  * credit stays valid (excess ≤ 2.4e-16 J);
  * about 29% of the frictional dissipation goes **unreimbursed**, and the Ec budget drains to
    `min Ec ≈ 0.006-0.05 J` (from 0.6 J);
  * block/release completion falls to 13/50 (C1) and 5/50 (C3);
  * no violations.
* **Invalid over-credit diagnostic** (true μ = 0.25 booked as 0.35):
  * the auditor flags credit exceeding true `∫D_T` by **0.50 J**;
  * Ec looks healthy (0.58 J);
  * `h_H` is violated in 37-50/50 block/release episodes, because the filter's `D̂` overstates the
    real dissipation.

The guarantees rest on a **valid** dissipation bound. Conservative bounds cost task capability.

**6.6 Moment arm × location uncertainty (T12, `moment_arm_sweep.png`).** Setup:
* the same nominal push: a 4 cm intention into the wall, ≈ 6 N normal;
* fixed `s ∈ {−0.15, 0, +0.15}` m;
* location-sensor noise = bias s.d. ∈ {0.01, 0.03, 0.06} m, consistent between truth and model;
* 30 seeds per cell, comparing C1 vs C3 vs C3_forceonly (location treated as known in the belief).

Findings:
* **The moment arm matters physically.** The same push gives max |θ| = 0.050 / 0.038 / 0.016 rad at
  `s = −0.15 / 0 / +0.15`. The push's moment combines with the moment of the off-centre task contact.
* **The true past-work error grows with location uncertainty.** At `s = −0.15`: 1.7 → 5.2 → 10.4 mJ.
  * C3's belief σ_e tracks it (≈ 4.4 → 6.5 → 9.6 mJ).
  * The force-only variant's stays ≈ 3.5 mJ.
* **`h_H` violations:**
  * C1: **20/270**.
  * C3 force-only: **12/270**, including 9/30 (and 5/30 incomplete tasks) in the worst cell,
    `s = +0.15`, σ = 6 cm.
  * C3 with force + location uncertainty: **0/270**.

Modelling force uncertainty alone is not enough; the location term is what removes the violations.

**6.7 Post-hoc: robust interval bound for sensor biases (designed after observing §5.5; fresh seeds).** **Status: post-hoc.** This variant was designed after the frozen run revealed §5.5. It is
evaluated only on fresh seeds 300-349, paired with C2/C3 on the same seeds (T16).

**Assumed calibration bounds.** `B_f = 1.0 N` and `B_s = 0.08 m` were chosen to **cover** the test
biases (0.8 N, 6 cm). This is the optimistic case in which the bounds are right; real bounds must
come from sensor calibration.

**Sensor bias** (the original failure):

| | `h_H` violations | true-BH violations | worst true `h_H` | positive work on the human |
|---|---|---|---|---|
| C2 | 33/50 | 0/50 | −52 mJ | 0.134 J |
| C2 + bound | **0/50** | 0/50 | +32 mJ | 0.072 J |
| C3 | 48/50 | 11/50 | −82 mJ | 0.156 J |
| C3 + bound | **0/50** | 0/50 | +27 mJ | 0.078 J |

All bounded runs completed the wipe.

**Cost in distribution** (block/release, assist/reverse, moment arm):
* no violations, either way;
* completion time +0.05 to +0.5 s;
* block/release coverage 0.98 → 0.94 (C3) and 1.00 → 0.98 (C2);
* force RMSE 0.37 → 0.77 N (C2) and 0.67 → 1.16 N (C3);
* minimum true backup margin rises from 3-14 mJ to 31-70 mJ (much more conservative);
* positive robot→human work drops by about 20-60%;
* one C3_robust moment-arm episode was incomplete (49/50).

**Conclusion.** A correct interval bound removes the bias failure, and the price is task capability.
This supports making calibrated bias bounds part of the ledger. It does not show what happens when
the bound itself is wrong.

**6.8 Single-process timing.** Measured with one process and one BLAS thread on block/release, seeds 300-302; system load average 9 at the start (other activity on the machine).

| controller | p50 during human contact | p95 during contact | max | p95 over all calls | wall time per 9 s episode |
|---|---|---|---|---|---|
| C1 | 5.2 ms | 5.3 ms | 5.4 ms | 5.4 ms | 3.2 s |
| C2 | 20.3 ms | 20.8 ms | 24.5 ms | 20.6 ms | 4.8 s |
| C3 | 34.0 ms | 34.5 ms | 36.6 ms | 34.5 ms | 6.9 s |
| C4 | 33.8 ms | 34.5 ms | 45.6 ms | 34.5 ms | 6.9 s |

The first call includes import overhead (max 361 ms). The control period is 20 ms. C2 is at the period; C3/C4 need about 1.7× it. All of this is Python/numpy on an Apple M4 and is **not a real-time claim**.

## 7. Forked-truth calibration of the predicted margins (T14, `calibration_forks.png`)

Forks were taken at every third engaged planning call (seeds 200-219, five scenarios). At each,
the **true** simulator was deep-copied and the selected candidate executed open-loop for 0.3 s, with
no filter and no re-planning. The realised `h_H`, true BH, Ec and human work were compared with the
planner's predicted distributions: 4244 forks per predictor.

* **In-distribution pool** (block/release, assist/reverse, normal/oblique, moment arm; 3303 forks):

  | | `h_H` z s.d. | coverage 50/80/90/95/99% | lower-tail exceedance (nominal 1%) | work above predicted q99.5 |
  |---|---|---|---|---|
  | C3, first order | 2.25 | 0.51 / 0.76 / 0.84 / 0.88 / 0.93 | 1.6% | 5.5% |
  | C3, MC moments | 1.64 | 0.85 at 90% | – | – |
  | C2, first order | 1.39 | 0.59 / 0.82 / 0.87 / 0.91 / 0.94 | 0.0% | 2.5% |

  C3's work PIT is U-shaped, i.e. over-confident.
* **By scenario.** Block/release is well calibrated, even conservative, for both (z s.d. 0.5-0.6).
  C3 is over-confident for the step-like intents of normal/oblique and moment arm (z s.d. ≈ 3).
* **Misspecified** (OOD intent + sensor bias, 941 forks): C3 90% coverage 0.18, lower tail 54%;
  C2 0.32 and 37%.
* **Why offline calibration does not carry over.** The offline evaluation conditions on the *true*
  robot trajectory. The closed-loop covariance omits predictor-model error (5 ms semi-implicit Euler,
  stage-Taylor intention, feedback-motion error) and the extrapolation error of step-like intents.
* **First-order vs MC mean work** over 0.3 s differs by +1.2 mJ (C3) and +4.7 mJ (C2) on average,
  against a realised |W_H| median of 7-12 mJ. First-order mean propagation is materially biased. Here
  it over-predicts work, so the expected `−tr Cov(f, v)` correction is not the dominant nonlinearity.
* **Consequence.** Gaussian tightening at κ = 2.33 does **not** deliver its nominal 1% per-call tail
  in this closed loop. It is a heuristic margin (tier B); the realised safety comes from the servo
  filter (§6.3).

## 7b. Physics-step refinement (T13, `refinement.png`)

Setup: physics step h ∈ {1, 0.5, 0.25} ms with the servo fixed at 1 kHz; 10 seeds ×
{block/release, normal/oblique} × {C0, C1, C3}.

* **Plant energy channels.** Max |ΔS − W_a + W_H + ∫D_R + ∫D_T| falls 4.6e-7 → 3.8e-8 → 3.5e-10 J
  (C0) and 1.2e-6 → 7.2e-8 → 3.9e-9 J (C3), ≈ 4th order.
* **Independent auditor** (trapezoid on the 1 kHz log): 2.3-3.7e-4 J, independent of h because the
  log rate is fixed. This is the auditor's resolution, ≈ 0.1-0.3% of the energy flows.
* **Closed-loop account minima.**
  * C0/C1 converge (C0: |Δ min h_H| 0.07 → 0.02 mJ).
  * C3's are not pointwise convergent (|Δ| up to 0.9 mJ): tiny physics differences flip discrete
    candidate decisions.
  * Violation counts are unchanged, 0 at every h for C1/C3.
* **Credit.** The max booked-credit excess over the true `∫D_T` is ≤ 2.4e-16 J (round-off), except in
  the deliberately invalid diagnostic.
