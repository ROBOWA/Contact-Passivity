# Literature mapping: sources vs. our adaptations

Local copies were used. Their SHA-256 hashes match the `SOURCE_NOTES.txt` beside them in
`/Users/carl/Documents/Contact passivity/Nadia_Papers_2026-10-26/`. Page numbers refer to the local
PDF pages of these author versions. They are **not** the page numbers of the published proceedings or
journal. Equations were read from the text layer (`pdftotext`) and checked by hand where the
extraction was garbled.

| ID | Version used (local) | Pages | SHA-256 (prefix) |
|---|---|---|---|
| P1 | Medina, Börner, Endo, Hirche, *Impedance-Based Gaussian Processes for Modeling Human Motor Behavior in Physical and Non-Physical Interaction*, IEEE TBME 66(9):2499-2511, 2019. Author manuscript `TBME-01274-2018-R1` (received 13 Aug 2018, revised 25 Oct 2018, accepted 24 Dec 2018). The header states that the publisher changed the final version. | 12 | `2c94f26e…` |
| P2 | Medina, Endo, Hirche, *Impedance-based Gaussian Processes for Predicting Human Behavior during Physical Interaction*, ICRA 2016, 3055-3061. Author repository PDF (`impedance_gps_ICRA2016.dvi`, created Sep 2015). | 7 | `090353ab…` |
| P3 | Medina, Lee, Hirche, *Risk-Sensitive Optimal Feedback Control for Haptic Assistance*, ICRA 2012, 1025-1031. PaperPlaza proceedings PDF. | 7 | `65be2ca6…` |
| P4 | Medina, Sieber, Hirche, *Risk-sensitive Interaction Control in Uncertain Manipulation Tasks*, ICRA 2013, 502-507. Author repository PDF (`icra13_manip.dvi`). | 6 | `14d525b3…` |

All four sources were accessible, so no specialisation was forced by missing material.

## P1 (main implementation reference)

| P1 element (local page) | What it says | How we use it / what we changed |
|---|---|---|
| Eq. (1)-(6), p. 3 | Human arm rigid-body dynamics. Interaction torque `τ_int = −J(q)ᵀ u_int`, where `u_int` is the force/torque **measurement** at the robot end-effector. Feedback term `τ_FB = D_q ė_q + K_q e_q` with `e_q = q_d − q`. In Eq. (5), `−τ_int = M ë + … + D ė + K e + ε`. | Sign check. `u_int` is the wrench the human applies **to** the robot, and it points toward the human's desired position (`e = x_d − x`). Our `f_H` (force by the human on the robot) has the same sign. |
| Eq. (7)-(8), p. 3 | `u_int(ξ) = u_imp(ξ) + û_int,ε(ξ) + ε_N`, `u_imp = M ë + D(ξ) ė + K(ξ) e`, `e = x_d(θ) − x`. | Adopted with **M = 0** (no human inertia), **fixed diagonal** `K, D`, and Cartesian **contact-point** coordinates instead of the hand configuration. Our `û_int,ε` analogue is the residual `g(χ)`. Our `ε_N` analogue is split into a physical OU force variability (acts on the plant) and separate white **measurement** noise (does not act on the plant). P1 uses a single i.i.d. term σ_N². |
| Eq. (10)-(12), p. 4 | Per-axis model. Priors: `m_i ~ N`, `d_i, k_i ~ GP`, `x_d,i ~ GP(µ_xd, k_xd)`, `û_int,ε,i ~ GP(0, ·)`, all independent. Diagonal approximation with the reasons given. | Same per-axis diagonal structure. We take the deterministic-gain special case (`σ_d = σ_k = σ_m = 0`). The latent intention is `a(t) = p_Hc(t_on) + ã(t)`. Here `ã` has an SE + constant + linear-trend kernel in **time** (θ = t), anchored at the grasp point. P1's θ = t variant uses the equilibrium mean (21) instead. |
| Eq. (13)-(14), p. 5 | GP approximation of `u_imp` (products of Gaussians made Gaussian by moment matching). Mean (14). | Not needed. With fixed gains the operator is linear, so the conditional force is **exactly** Gaussian (formulation §7.3). |
| Eq. (16)-(18), p. 5 | PD kernel (17): `µ_kµ_d(∂k/∂t′ + ∂k/∂t) + (µ_d² + k_d)∂²k/∂t∂t′ + (µ_k² + k_k)k + …`. Impedance kernel (18) adds inertia terms up to ∂⁴. | Our `K_f = K² k_a + KD(∂₁k_a + ∂₂k_a) + D² ∂₁∂₂k_a` is (17) with `k_d = k_k = 0`. We drop the inertia terms of (18). Our addition: OU, residual-feature, bias and location-coupling terms. |
| Eq. (19)-(21), p. 5 | `u_int ~ GP(E[u_imp], k_int)`, `k_int = k_imp + k_c + σ_N²`. Equilibrium prior `x_d ~ GP(x_i, k_xd)` gives zero mean force. | We do **not** use the equilibrium mean for prediction: under it, the predicted force does not depend on a candidate robot motion. Our prior mean is the grasp point, plus a learned trend. |
| Eq. (23), p. 6 | SE kernel `σ² exp{−(z−z′)ᵀΛ⁻¹(z−z′)}` (no factor ½). | Equivalent reparameterisation `σ² exp(−r²/(2ℓ²))`. |
| Eq. (24)-(25), p. 6 | Joint prior of forces and `x_d(θ*)`. Cross-covariance `C_int = µ_m ∂²k/∂t² + µ_d ∂k/∂t + µ_k k`, with derivatives taken in the **observation** argument. | Our `Cov(y_j, ã(t*)) = K k + D ∂₁k` is (25) with `µ_m = 0`. We added `Cov(y, ã̇(t*))` for the derivative target, plus location, residual-weight, bias and OU cross-covariances. These were independently checked against a brute-force grid construction (`tests/test_math.py::test_conditioning_matches_independent_bruteforce`). |
| Eq. (26), p. 6 | Posterior `N(µ + Cᵀ K⁻¹(u − E[u]), k** − Cᵀ K⁻¹ C)`. | Used, with the transpose and minus sign as printed in P1. We condition **one joint** Gaussian over intention, location, residual weights, force bias and OU state, and keep all posterior cross-covariances. |
| Algorithm 1, p. 6 | Fit a naive GP, copy its hyperparameters to the residual prior, set EMVs from Table I, infer `x_d` by (26), predict by (19). | Adapted. (1) The naive GP is fitted by ML **as the baseline** (inputs `[t, Δp, v]`). (2) Our residual is not given the naive GP's hyperparameters: it is a finite-feature function of the patch velocity, learned **across training episodes** by exact Bayesian regression. (3) Population gains are fitted by ML on training data with exogenous robot excitation. They give 135 N/m and 13.4 N s/m; P1 Table I, p. 7, lists 10-1000 N/m and 5-100 N s/m. |
| Sec. V-B, pp. 7-8 | Lengthscale guidance: `Λ_xd` should exceed the naive-GP lengthscale, and small signal variances are advised. | The fitted `ℓ_a = 0.75 s` exceeds the naive-GP time lengthscale of 0.84 s only marginally, and `σ_a = 0.018 m` is small. This matches the guidance and also explains the over-confidence on held-out fast or oscillating intents (report §4). |

**What P1 does not provide:** robot contact localisation (`s_H`), any energy or passivity reasoning,
uncertainty propagation through robot dynamics, and control. Our structural assumption that the
human's intention starts at the grasp point is also ours.

## P2 (earlier conference version)

| P2 element (local page) | Observation |
|---|---|
| Eq. (5)-(6), p. 2 | The text layer prints `τ_int = M ë + D ė + K e + τ̂_int + …` and `−u_int(ξ) = M(x)ë + D ė + K e + û_int,ε + ε_τ`. The sign of `u_int` is **inconsistent** with P2's own Eq. (8)-(9), `u_int = u_imp + û_int,ε + ε_τ`, `u_imp = m ë + d ė + k e`. P1 Eq. (5)/(7) resolves this (`−τ_int = …`, `u_int = u_imp + …`). |
| Eq. (13)-(14), p. 3 | PD and impedance kernels. These are the same as P1 (17)-(18). |
| Eq. (15)-(17), pp. 3-4 | The posterior mean is printed as `µ + Cov[u(Ξ), x_d(θ*)] K⁻¹ (y − E[u])`, an untransposed h×1 cross-covariance. The variance is printed as `k** **+** Cov[u, x_d] K⁻¹ Cov[x_d, u]` with a **plus** sign. Gaussian conditioning requires `−` and `Cᵀ` (as in P1 Eq. (26)). We use P1's form, and the brute-force test confirms it numerically. |
| Eq. (18), p. 4 | Equilibrium prior. Same as P1 (21). |

## P3 (risk-sensitive haptic assistance)

| P3 element (local page) | What it says | Our use |
|---|---|---|
| Eq. (1)-(3), p. 2 | Admittance `u = u_h + u_r = M_r ẍ + D_r ẋ`. The human contribution is modelled as **process noise** `u_h = ε`. | Not adopted as a human model. Our human force is state-dependent impedance, not exogenous noise. The C2 Gauss–Markov baseline is the closest analogue. |
| Eq. (4)-(9), p. 3 | Finite-horizon LQR with a Mahalanobis-weighted tracking cost `Q_k = Σ̂^{-1/2} Q Σ̂^{-1/2}`, where **demonstration variability** encodes task flexibility (Fig. 2). | Not adopted. Our uncertainty is **state-estimation/prediction** uncertainty about the human, which is different from demonstration variability. We do not reweight tracking by a variance. |
| Eq. (10)-(11), p. 4 | `γ(θ) = −2θ⁻¹ ln E[exp(−θJ/2)]`, `γ ≈ E[J] − (θ/4)Var[J]`. θ < 0 is risk-averse, θ > 0 risk-seeking. | Mapped to `ρ_β = (1/β) log E[exp(βJ)]` with **β = −θ/2** (β > 0 risk-averse). Verified numerically in `test_entropic_risk_sign_and_limits`. |
| Eq. (12)-(13), p. 4 | Modified Riccati with `θΣ̂_u`. Text: risk-averse (θ < 0) gives **higher** feedback gain; risk-seeking gives lower gain. | Not used (our dynamics are nonlinear and the decision set is finite). We **measure** the selected stiffness instead of assuming a direction. |
| Sec. IV, p. 4 | HMM/GMR task model from demonstrations. | Not used. |

## P4 (risk-sensitive interaction control)

| P4 element (local page) | What it says | Our use |
|---|---|---|
| Eq. (1)-(2), p. 2 | Admittance with motion input `u_p` and force-control input `u_f = K_f(f_d − f)`. | Our task force PI is a separate channel. The P4 force correction is **not** the human force, and we keep the two signals separate. |
| Eq. (3)-(4), p. 3 | The force-controller correction is modelled as process noise `u_f = ε ~ N(u0, Σ_u)`, estimated from the last W samples. | Not adopted. |
| Eq. (5)-(7), p. 3 | Reference adaptation `ξ_a,k+1 = A ξ_a,k + B u0,k` added to the reference. | Optional P4-inspired adaptation is **deferred**. P4 reports drift and over-compensation (Sec. IV-C/D, pp. 4-5), which we list as relevant risks. |
| p. 3 (unnumbered, between (8) and (9)) | Same `γ(θ)` as P3. Risk-averse gives a higher feedback gain; **risk-seeking** (θ = 10⁻⁵ in the experiments, p. 4) gives lower stiffness under force corrections, which P4 calls "the most intuitive solution" for contact. | Our C4 uses β > 0 (risk-averse), with a β < 0 diagnostic. We report the measured compliance direction. Risk aversion is **not** presumed to produce compliance. |
| Sec. IV-A, p. 4 (text) | States that controllers b) and d) adapt their reference, while Sec. IV's list defines c) and d) as the reference-adapting ones. | This minor textual inconsistency in P4 is noted; we use the list definitions. |

## Our synthesis (proposed combination; not established by P1-P4)

The following elements come from none of the four papers:
* the impedance-informed latent-intention GP of P1, specialised to fixed gains and an exactly
  Gaussian conditional;
* an uncertain **robot-side** contact coordinate, coupled linearly into the same joint posterior;
* correlated prediction of integrated human work and of the Ec/BH/backup-margin accounts through
  closed-loop feedback rollouts;
* an entropic objective (P3/P4 sign convention) kept separate from Gaussian-tightened energy
  constraints;
* an independent energy/viability filter, the Ec + BH accounts with the `BH ≥ S` backup.

The account definitions come from this repository's existing Ec + BH mainline (workflow
`human-port-ecbh-20261029`, read-only). Comparing against the four papers alone does not
establish novelty.
