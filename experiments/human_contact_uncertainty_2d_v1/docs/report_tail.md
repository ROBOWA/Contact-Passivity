## 8. Failures and regressions (all retained, none hidden)

1. **Sensor bias outside the calibrated prior** (§5.5).
   * `h_H` is violated in 75-98/100 episodes for every filtered controller.
   * True BH is violated in C3 26/100, C4 17/100, C1 9/100.
   * The booked BH is optimistic by ≈ 80 mJ (bias × human-driven displacement), and force data cannot
     reveal it.
2. **Structured GP out of distribution** (oscillating intent).
   * Offline it is over-confident: NLPD 8.7, 90% coverage 0.54.
   * In closed loop C3/C4 complete 95/100 and 93/100 (vs 100/100 for C1/C2), with normal-force RMSE
     2.4 N.
3. **Gain mismatch ×2.** The GP is over-confident offline: 90% coverage 0.74, intention coverage 0.59,
   response slope 0.46.
4. **Closed-loop margin calibration** (forked truth).
   * C3's 0.3 s `h_H` predictions are over-confident in distribution: z s.d. 2.25, 1% lower tail
     exceeded in 1.6% of forks.
   * Under misspecification the tail is exceeded in 37-54% of forks.
   * First-order and MC mean work differ by 1-5 mJ.
5. **C2 beats C3 in closed loop** on interaction energy, force tracking and margins in every
   interaction scenario (§5.3). The structured GP's offline advantage did not transfer.
6. **C1's past-work error** produced 7/500 in-distribution `h_H` violations (worst −7 mJ) and 20/270 in
   the location sweep.
7. **Risk-sensitive objective.**
   * The effect is negligible (1-2% stiffer, slower).
   * The empirical entropic estimator degenerates at β ≥ 2: minimum ESS 1-2 of 48.
8. **The servo filter alone (no planner) fails the task** (0/50 completed): BH is spent during the
   block, and `BH ≥ S` then forbids the stored energy that wiping needs.
   * With the booked ledger and no margin it violates true BH by up to −2.3 mJ.
   * The instantaneous variant leaks −80 mJ.
   * The ORACLE shows µJ-level sampled-data deficits (worst −21 µJ), so there is **no exact
     intersample invariance**.
9. **`BH ≥ S` is conservative.**
   * Low BH (0.02 J): 1-2/100 completed, and C2 had 2 domain exits (body corner below the wall).
   * Infeasible initialisation (`h_H(0) < 0`): 0/30 completed (by construction).
10. **Conservative credit** (true μ 0.35, booked 0.25) drains Ec from 0.6 J to ≈ 0.01-0.05 J and cuts
    block/release completion to 13/50 (C1) and 5/50 (C3). The **invalid over-credit diagnostic** is
    caught by the auditor (+0.50 J excess) and produces 37-50/50 `h_H` violations.
11. **Refinement.** Closed-loop account minima of the planners are not pointwise convergent under
    step refinement (discrete decision flips). The independent auditor's resolution is about 2-4e-4 J
    (1 kHz log).
12. **Residual identifiability.** The tangential-axis residual is not identified: it is confounded
    with intention offset and OU noise.
13. **Not real time.** The planners take ≈ 30-120 ms per 20 ms period.
14. **Design defects found and fixed during the smoke stage, before freezing** (`decision_log.md`):
    * the sampled-data energy leak;
    * the "backup trap" from a violent re-approach;
    * free-body drift under the passive backup, which motivated the dissipative brake candidate.
    These would have invalidated the comparison had they not been fixed. The fixes apply to all
    controllers equally.

## 10. Recommendation: which upgrade is supported by evidence

**Supported (adopt):**
1. **A sampled-data, tick-work energy filter.** It constrains the predicted actuator work over each
   ZOH tick instead of the instantaneous power. Without it, the Ec + BH filter leaks (−80 mJ). With
   true accounts it holds to ≈ 2e-5 J. An intersample proof is still missing.
2. **An explicit past-work (BH) error belief.** It must be correlated with the location posterior
   (and the force-bias belief) and never reset at a planning call. It removed every in-distribution
   violation of the deterministic ledger (7/500 → 0/500) and every location-sweep violation
   (20/270 → 0/270). Force-only uncertainty is not enough (12/270).
3. **Calibrated bias bounds for the BH ledger.** A Gaussian prior that the data cannot falsify is not
   a safety margin. §6.7 quantifies the post-hoc interval bound and its cost.

**Not supported in the current form (do not adopt yet):**
4. **The structured latent-intention GP as the planner's predictor.** It is the better *predictor*
   offline, and it correctly predicts how the human force responds to candidate motion. In closed
   loop, however:
   * its sharper, closed-loop over-confident margins make the planner spend more of the human budget
     for no task gain;
   * it regresses on OOD intents.

   Before reuse it needs:
   * online calibration (an innovation/NLPD monitor with variance inflation or fallback);
   * covariance that includes predictor-model error;
   * an objective that values the human budget, if sharper prediction is meant to buy anything.
5. **The entropic (risk-sensitive) objective.** Its effect is negligible here and in the P3/P4
   direction (stiffer). The empirical estimator needs far more samples at β ≥ 2. Risk sensitivity
   must not be presented as a route to compliance.

**Negative results are part of the recommendation.** The four-paper synthesis (impedance-informed
latent intention + uncertain contact location + correlated energy prediction + independent energy
filter) is coherent and implementable. In this PoC, though, the safety benefit came from the energy
filter and the past-work error belief, not from the more expressive human model.

## 11. Reproduction, files and runtime

* Smallest reproduction (≈ 1 min) and the full suite (≈ 2.5 h on an Apple M4; another agent's job
  shared the machine): see `README.md`.
* Every reported closed-loop run used frozen code: `checkpoints/freeze_main_v1.json`, and for §6.7,
  `freeze_posthoc_v1.json`. Seeds were held out (≥ 100; post-hoc ≥ 300). Smoke seeds 0-29 were used
  only for debugging.
* Machine-readable outputs:
  * per-episode metrics (including the hidden truth parameters): `outputs/runs/<tag>/metrics.jsonl`;
  * summaries: `summary.json`;
  * offline predictor results: `outputs/offline/`;
  * forked-truth calibration: `outputs/calibration/`;
  * frozen models: `outputs/models/gp_frozen.json`;
  * all tables: `outputs/report_tables.md`;
  * provenance: `manifest.json`.
* Figures (`outputs/figures/`):
  * `geometry.png`, `gp_inference_example.png`, `offline_calibration.png`,
    `offline_response_to_motion.png`, `offline_residual.png`;
  * `violations_main.png`, `completion_main.png`, `tradeoff_main.png`, `runtime_main.png`;
  * `accounts_block_release_100.png`, `accounts_sensor_bias_100.png`;
  * `moment_arm_sweep.png`, `calibration_forks.png`, `refinement.png`.
* Animation: `outputs/animation/block_release_seed100_C0_C1_C2_C3.mp4` (and `.gif`). The panels show
  the same hidden human (same intention, gains, location and noise) across C0 (marked UNPROTECTED),
  C1, C2 and C3. Each panel shows:
  * the task contact force;
  * the true human contact point and the location belief band;
  * force arrows;
  * live Ec / true BH / booked BH / `h_H` bars.
* Hardware: Apple M4 (4 performance + 6 efficiency cores), 16 GB RAM, macOS 26.2, Python 3.13.5,
  numpy 2.5.4, scipy 1.18.1.
