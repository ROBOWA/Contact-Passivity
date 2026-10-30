## 1. What was built

The task was an isolated planar PoC. The robot is a rigid tool with `q = (x, y, θ)` and
`M = diag(m, m, I)`. It wipes along a compliant wall with an off-centre tip, under three
constraints:
* a 10 N normal-force target;
* tangential progress at 8 cm/s;
* orientation regulation.

A simulated human grasps a known patch (the top edge) at a coordinate `s` that is **uncertain to
the robot**. The human force follows a hidden impedance model with a latent intended contact-point
trajectory, a velocity-dependent residual, and physical OU force variability. Everything below lives
in `experiments/human_contact_uncertainty_2d_v1/`.

| component | file | status |
|---|---|---|
| plant, passive unilateral contact (S_T, D_T), human truth | `hcu2d/truth.py` | implemented, tested |
| Ec + net BH accounts, exact actuator work from encoder increments (ZOH), conservative credit | `hcu2d/servo.py`, `episode.py` | implemented, audited |
| past-work (BH) error model, correlated with location/bias posterior | `hcu2d/servo.py::Ledger` | implemented; first-order |
| servo energy filter (sampled-data, tick-work projection) | `hcu2d/servo.py` | implemented; an instantaneous version leaks energy (kept as ablation) |
| structured latent-intention GP (P1 specialisation), joint conditioning over intention, location, residual weights, force bias and OU state | `hcu2d/gp.py`, `kernels.py` | implemented; brute-force-verified |
| naive force GP (P1 Alg. 1 step 1 analogue), Gauss–Markov force predictor | `hcu2d/gp.py` | implemented (naive GP offline only) |
| Monte Carlo and first-order closed-loop belief propagation (feedback Jacobians, full cross-time covariance) | `hcu2d/belief.py` | implemented |
| finite-candidate planners C1-C4, entropic objective, Gaussian-tightened `h_H`/Ec prefix constraints | `hcu2d/planner.py` | implemented |
| independent truth auditor | `hcu2d/auditor.py` | implemented (no shared code with the plant) |
| forked-truth calibration of predicted margins | `scripts/calibration_fork.py` | implemented |

**Deferred (not done):**
* P4-inspired reference adaptation;
* joint online gain inference;
* human inertia;
* unknown-link mixtures and multiple human contacts;
* KUKA/VLA;
* sigma-point propagation (Monte Carlo is the nonlinear check);
* a closed-loop planner using the naive GP (the naive GP is compared offline only, under matched data);
* a less conservative braking reserve;
* any real-time implementation;
* any certified probabilistic claim.

## 2. Guarantee tiers used in this report

* **(A) Proven** for the stated model (`formulation.md` §2-§5, §7.3):
  * the contact identity `p_T = Ṡ_T + D_T` (including unloading and transitions);
  * the composite balance;
  * `V̇ ≤ −p_H` for `V = S + Ec − E_min`;
  * the zero-input backup lemma `∫p_H ≤ S(t₀)`, which makes `BH ≥ S` sufficient;
  * the continuous-time reference-filter proposition with exact accounts;
  * the exactness of the conditional Gaussian under fixed gains and a conditioned trajectory.
* **(B) Approximate:**
  * belief propagation through feedback (Monte Carlo / first order);
  * Gaussian tightening `mean − κ σ`;
  * the finite-feature residual;
  * the stage-Taylor intention;
  * the local Gaussian location prior on a bounded patch.
* **(C) Empirical:**
  * per-episode violation frequencies (Wilson / exact Clopper–Pearson bounds over independent
    episodes, never time steps);
  * energy audits and step refinement;
  * forked-truth calibration.

Nothing in (C) is presented as a deterministic safety proof.

## 3. Deterministic physics and energy checks (tier A identities, tier C numerics)

`python -m pytest tests -q`: **14 passed**. The tests that check non-trivial mathematics:

* **Contact power identity.** It holds pointwise for 4000 random states: compression, unloading and
  separation, with `|p_T − Ṡ_T − D_T| ≤ 1e-10 W`. The force is never adhesive.
* **Composite balance under refinement.** The residual of `ΔS − W_a + W_H + ∫D_R + ∫D_T` falls from
  5e-4 J (h = 2 ms) to 3e-8 J (h = 0.125 ms) on an aggressive scripted run with lift-off and
  re-contact (`W_a ≈ 5 J`). Convergence is non-monotone near contact kinks.
* **Location Jacobian.** `J_H(q, s)` matches central differences. `∂τ_H/∂s = (0, 0, (R e₂)ᵀ f)`.
  At fixed q, s changes the moment for normal and oblique forces but not for forces along the edge,
  and it changes human power when `θ̇ ≠ 0`.
* **Kernel derivative blocks.** They match finite differences, and they transform correctly under a
  s→ms unit change (×10⁻³ for ∂, ×10⁻⁶ for ∂₁∂₂).
* **Joint covariance.** The joint prior covariance `[K_YY, C; Cᵀ, Σ_ζ]` is symmetric and PSD, and the
  posterior covariance is ≼ the prior.
* **Independent brute-force construction.** A dense time grid with finite-difference
  `L_t = K + D d/dt` reproduces the posterior mean and covariance of the intention, its derivative,
  and location (rtol ≤ 5e-3). This verifies the P1 Eq. (25)-(26) signs and transposes that P2
  prints inconsistently.
* **Zero-covariance limit.** With all prior variances → 0, the posterior force equals the
  deterministic impedance model, with s.d. < 1e-4 N.
* **Temporal work covariance.** `Var(ΣW_i) = Σ_ij Cov(W_i, W_j)` matches 40 000 MC samples within
  3%. It exceeds the independent-step sum by more than 1.5×.
* **Risk conventions.**
  * `ρ_0 = E[J]`;
  * `ρ_β ≈ E[J] + βVar(J)/2` for small β;
  * `ρ_β > E[J]` for β > 0 and `ρ_β < E[J]` for β < 0;
  * `ρ_{−θ/2} = γ(θ)` of P3/P4.
* **Zero-input backup lemma.** It is satisfied for 6 adversarial humans: random gains 50-600 N/m,
  moving and oscillating intentions, preloaded contact, initial velocity. `max_t ∫p_H − S(t₀) ≤ 1e-7 J`.
* **ORACLE reference filter (true BH).** It keeps `h_H`, `BH` and `Ec ≥ −1e-5 J` on block-release
  episodes. This is a numerical validation of §5.2, not an intersample proof.
* **Filter projection.** The tick-work projection is feasible and matches an SLSQP optimum.
* **Exact booked work and BH error model.** Booked actuator work equals the plant's RK4 work to 1e-9 J.
  The past-work error model `e_BH = G s − Γ − b_fᵀΔp̂ − n_e`, evaluated with the true `s`, explains
  the auditor's true-minus-booked BH.

## 4. Offline predictor evaluation (held-out by trajectory seed, intent family, location and gains)

Splits:
* training: 30 episodes with `g = 0` plus 50 with the residual;
* validation: 25;
* tests: in-distribution 30, `g = 0` 25, held-out families (oscillating, fast assist) 20,
  gain mismatch ×2 20.

All splits use the unprotected explorer controller, so robot motion is exogenous to the intention.
Hyperparameters were fitted by maximum likelihood on the training splits and **frozen before any test
evaluation**. They were not retuned after seeing test results, including the failures below.

Fitted population gains are K = 135 N/m and D = 13.4 N s/m (truth: lognormal around 150 / 15). The
profile likelihood on a 3×3 grid peaks at (150, 15), so the gains are identifiable **only because the
training data contain exogenous robot excitation**.

Main findings (table T15, figures `offline_calibration.png`, `offline_response_to_motion.png`,
`gp_inference_example.png`):

* **In distribution**, the structured GP is calibrated:
  * 0.3 s force RMSE 0.83-1.0 N, versus 1.5-1.9 N for the naive GP, Gauss–Markov and hold-last;
  * best NLPD;
  * 90% intervals cover 90-93%;
  * integrated-work RMSE 15-18 mJ versus 47-67 mJ, with work z s.d. 1.0-1.2;
  * intention inference: posterior s.d. ≈ 15 mm against a 102 mm prior, z s.d. 0.88-1.0, 95%
    coverage 0.92-0.96.
* **It responds correctly to candidate robot motion.** For a counterfactual +2 cm tangential ramp, or a
  +0.05 rad rotation, the predicted force change follows the true counterfactual change: slope 0.84-0.95
  translational, 0.84-0.89 rotational. The rotational response depends on `s`. The naive GP's slope is
  0.02-0.17, and the Gauss–Markov slope is 0 by construction.
* **Temporal covariance is essential.** Dropping cross-time covariance (independent-step approximation)
  makes work predictions about **3× over-confident**: z s.d. 3.0-3.5, 90% work coverage 0.43-0.49.
* **Location is not learned from force.** The posterior error of `s` equals the location-sensor-only
  error (≈ 21-24 mm), which is the bias floor. This matches the identifiability analysis:
  * `s` and an intention offset are confounded except through rotation;
  * the persistent sensor bias does not average out.
* **Residual confounding.**
  * Normal axis: the learned residual has the right shape.
  * Tangential axis: it does not. During wiping the tangential patch velocity is nearly constant, so
    the residual is confounded with the intention offset and OU noise (`offline_residual.png`).
  * Within an episode the posterior intention-residual correlation is moderate (median max |corr|
    ≈ 0.19).
* **Failures:**
  * **Held-out oscillating / fast intentions:** the structured GP is badly **over-confident**:
    NLPD 8.7, 90% coverage 0.54, work z s.d. ≈ 5. It is worse than the motion-blind Gauss–Markov
    predictor (NLPD 3.2). The impedance structure still predicts the correct response to motion
    (slope 0.88). Only the intention extrapolation is wrong.
  * **Gain mismatch ×2:** the structured GP stays best in RMSE (1.77 N) but is over-confident:
    90% coverage 0.74, intention 95% coverage 0.59, response slope 0.46 instead of 1.

## 9. Remaining theoretical gaps

1. **No intersample proof.** The tick-work filter uses a second-order Taylor model with held external
   force. The remainder and the within-tick variation of the human force are not bounded, so continuous
   invariance is validated numerically only. A certified version needs:
   * a bound on `q̈` that includes the human force rate, or
   * a discrete passivity-preserving construction that books **exact** discrete work and constrains it
     a priori.
2. **Gaussian tightening is not a probabilistic certificate.** It covers neither calibration error
   (OOD intent, gain mismatch, mis-specified bias priors) nor nonlinear propagation error. A certified
   claim needs:
   * calibrated joint input uncertainty;
   * verification over a joint confidence region, or a remainder bound;
   * mission-level risk allocation over executed constraint events.
   The per-call `α = 1%` does not translate into a mission level.
3. **The planner has no recursive-feasibility argument.** Safety in this PoC rests on the servo
   filter (tier A conditions with exact accounts). The planner is a performance and anticipation layer.
4. **The past-work error model is first-order.** It:
   * drops `ε_m·(s − ŝ)`;
   * assumes the 1 kHz ledger noise is independent of the 25 Hz GP samples;
   * relies on a **force-sensor bias prior** that the force data cannot falsify (bias is confounded
     with an intention offset).
   A mis-specified bias prior, as in the `sensor_bias` scenario, therefore defeats the margin.
5. **Credit validity is audited, not proven.** The min-endpoint quadrature lower-bounds `∫D_T` only
   when there is no interior dip within the 1 ms tick.
6. **Human model limits.** The model has:
   * no inertia;
   * fixed gains (integrating over gain uncertainty multiplies the intention and destroys
     Gaussianity);
   * no grasp/release prediction (the predictor assumes contact persists over the horizon);
   * no unknown-link or multi-contact inference.
7. **`BH ≥ S` is conservative.** It reserves BH for all stored energy, including kinetic energy and
   task-wall elastic energy that the human could only receive through the robot. Less conservative
   reserves need sound reachable-set bounds, not sampled backup rollouts.
8. **The bounded patch is approximated by a Gaussian location prior.** The posterior mass outside the
   patch is logged but not otherwise handled.
9. **Prediction-model error is not bounded.** The predictor uses 5 ms semi-implicit Euler, the truth
   1 ms RK4.
10. **Not real time.** The Python planner takes about 30-80 ms per 20 ms control period during human
    contact (p95).

## 5a. Closed-loop design (applies to §5-§8)

* **Controllers.** All share the nominal hybrid policy, the actuator box, the observation information,
  the budgets (`Ec0 = 0.6 J`, `BH0 = 0.15 J`) and the 1 kHz servo.
  * **C0:** nominal, **no energy filter (unprotected reference)**.
  * **C1:** deterministic held-force predictor, mean-only constraints, booked-ledger servo filter
    without margin.
  * **C2:** Gauss–Markov (motion-blind) force with Gaussian location; Gaussian-tightened constraints;
    servo filter with the past-work-error margin.
  * **C3:** structured latent-intention GP, otherwise as C2, risk-neutral.
  * **C4:** C3 with the entropic objective, β = 2 (risk-averse in our convention; θ = −4 in P3/P4).

  C1-C4 share the 18 candidates, N = 15 stages × 20 ms, the constraints and the energy-margin cost.
* **Seeds.**
  * Hidden human intention, gains, location, sensor biases and noise are **paired** across
    controllers.
  * Forces are regenerated from the hidden intention and the actual state; force traces are never
    replayed.
  * Reported runs use seeds ≥ 100. Debugging used seeds 0-29.
* **Statistics.**
  * Rates are per **episode**, with Wilson 95% intervals; pooled rates also get exact one-sided
    Clopper–Pearson 95% upper bounds.
  * Time steps are never counted as trials.
  * A violation is a true-account minimum below −1e-6 J. Raw minima are also reported.
* **Truth side.** The auditor recomputes the true BH, `S`, `h_H = BH − S`, the identities and credit
  validity independently. The controllers never see these values.

## Completion checklist (task's "before calling the PoC complete")

| requirement | evidence |
|---|---|
| location uncertainty is non-degenerate | posterior `s` s.d. ≈ 20 mm (bias floor, does not average out); moment-arm tests; closed-loop rotation and past-work error grow with `s` and location uncertainty (sweep, T12) |
| GP predictions respond to candidate robot motion | counterfactual response slope 0.84-0.95 (structured) vs ≈ 0.05 (naive) and 0 (Gauss–Markov), T15 and `offline_response_to_motion.png` |
| temporal covariance is retained | MC uses one joint function sample per rollout; first order carries `Φ_k` with cross-time covariance; offline work z s.d. 1.0-1.2 full vs 3.0-3.5 independent-step; closed-loop ablation T7 |
| the risk objective is separately ablated | C3 vs C4 (main) and the β sweep −1/0/2/5 with the measured σ_K/σ_v selection (T8) |
| accounts use executed work | booked `P_a` = `uᵀΔq` of the **applied** (filtered, saturated) command; matches RK4 to 1e-9 J; true BH from true force at the material point |
| passive-contact energy balance converges | pointwise identity tests; refinement (T13, `refinement.png`) |
| empirical results are not advertised as proofs | tiers A/B/C throughout; failures reported |
