# Formulation — planar PoC of impedance-informed human uncertainty, risk-sensitive selection and Ec + BH

Folder: `experiments/human_contact_uncertainty_2d_v1/`. Research simulation only; not a hardware or
deployment claim. Numerical values are the frozen defaults in `configs/default.json` (they're
reproduced in §13). Every statement is tagged with the guarantee tier from §12:

* **(A)** a proven identity or continuous-time conditional theorem for the stated model;
* **(B)** approximate Gaussian/GP chance-constrained planning (approximation, not a guarantee);
* **(C)** empirical finite-sample frequencies and numerical audits.

What comes from the papers and what is our adaptation is mapped in `literature_mapping.md`. The
combination "latent-intention/impedance human prediction + uncertain robot contact coordinate +
correlated energy prediction + an independent energy/viability filter" is **our proposed
combination**. None of the four papers establishes it, and the comparison with them alone does
not establish novelty.

---------------------------------------------------------------------------------------------------

## 1. Plant: a planar rigid tool where contact location matters

Generalised coordinates `q = (x, y, θ)`, horizontal plane, **no gravity**. The body is a rectangle,
`L × H = 0.40 × 0.10 m`, mass `m`, inertia `I` about the centre of mass (COM).
`M = diag(m, m, I)` is constant.

Rotation `R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]`, `Ω = [[0, −1], [1, 0]]`, and `dR/dθ = R Ω`.
A body-fixed point `r` (body frame) has

    p(q, r) = [x, y]ᵀ + R(θ) r,
    ṗ       = J(q, r) q̇,         J(q, r) = [ I₂ , R(θ) Ω r ]   (2×3).

A world force `F` applied at `p` produces the generalised force `τ = Jᵀ F = [F; ρ × F]`, where
`ρ = R r` and `ρ × F = ρ_x F_y − ρ_y F_x` (because `(R Ω r)ᵀ F = (Ω ρ)ᵀ F`).

* **Task contact (tool tip):** a fixed body point `r_T = (0.12, −0.10)`. It is off-centre, so the
  wall reaction produces a moment.
* **Human contact patch:** the top edge `r_H(s) = (s, h_e)`, `h_e = +0.05`, with patch coordinate
  `s ∈ [−0.18, 0.18]` m. `s` is a **coordinate on the robot body**. It is not the human's
  intended trajectory.

Because `r_H` is affine in `s`, the contact point and its Jacobian are affine in `s` at fixed `q`:

    p_Hc(q, s) = c(q) + s u(θ),   c(q) = [x, y]ᵀ + h_e R(θ) e₂,   u(θ) = R(θ) e₁,
    v_H(q, q̇, s) = ċ + s u̇,       ċ = [ẋ, ẏ]ᵀ − θ̇ h_e R e₁,       u̇ = θ̇ R e₂,
    ∂J_Hᵀ/∂s · f = [0, 0, (R e₂)ᵀ f]ᵀ .

So `s` changes the human generalised torque `τ_H = J_Hᵀ f_H` only through the moment component.
It also changes the robot-side velocity `v_H` whenever `θ̇ ≠ 0`, and so it changes human work.
For a force along the edge (`(R e₂)ᵀ f = 0`) the moment does not depend on `s`. Normal and oblique
human forces make `s` matter. The scenarios therefore include tangential, normal and oblique
interaction at several moment arms.

Dynamics (all forces in world frame, actuation at the COM as generalised force `u_act`):

    M q̈ = u_act + J_Tᵀ F_T + J_Hᵀ f_H − B_R q̇ ,

`B_R = diag(b_x, b_y, b_θ) ⪰ 0` is the **physical** robot dissipation (bearing/support losses),
`D_R = q̇ᵀ B_R q̇ ≥ 0`. Controller damping is part of `u_act` and therefore of `P_a`. It is never
counted again in `D_R`. Actuation is bounded componentwise, `|u_act,i| ≤ u_max,i`. `u_act`
(called `u_actual` in the task text) is the command **after** every saturation and filter, i.e.
exactly what is applied. Zero-order hold at the servo rate (1 kHz).

## 2. Constitutively passive unilateral task contact

A fixed compliant wall occupies `y < 0` with outward normal `n = e_y`. Let `p_T = p(q, r_T)`,
`v_T = J_T q̇`, `gap = p_T,y`, `v_n = v_T,y`, `v_t = v_T,x`.

    δ = max(0, −gap) ,   δ̇ = −v_n   (used only when δ > 0),
    χ(δ) = δ / (δ + δ_s) ,   χ(0) = 0 ,  0 ≤ χ < 1,
    f_n = k_n δ + b_n χ(δ) [δ̇]₊          (normal force ON the tool, along +n),
    f_t = −μ f_n tanh(v_t / v_ε)          (tangential force ON the tool),
    F_T = f_t e_x + f_n e_y ,
    S_T = ½ k_n δ² ,
    D_T = b_n χ(δ) [δ̇]₊² + μ f_n v_t tanh(v_t / v_ε) .

**Identity (A).** The power flowing from tool into the environment is
`p_T = −F_T · v_T = −f_n v_n − f_t v_t`. In contact (δ > 0), `−f_n v_n = f_n δ̇ = k_n δ δ̇ + b_n χ [δ̇]₊ δ̇`, and `[δ̇]₊ δ̇ = [δ̇]₊²`. Also `−f_t v_t = μ f_n v_t tanh(v_t/v_ε)`. Hence

    p_T = Ṡ_T + D_T ,  with  D_T ≥ 0  (since f_n ≥ 0 and v tanh(v/v_ε) ≥ 0).

*Unloading* (`δ̇ < 0`): the damping term is switched off, `f_n = k_n δ ≥ 0`, and the identity holds
with only frictional dissipation. Nothing is credited for elastic release. *Separation* (`δ = 0`):
`f_n = f_t = S_T = D_T = p_T = 0`. *Transitions:* `S_T = ½ k_n max(0, −gap)²` is C¹ in `gap`, and
`f_n → 0` as `δ → 0` because `χ(0) = 0`. Forces stay bounded, there are no impulses, and the
identity holds almost everywhere. The force is never adhesive (`f_n ≥ 0`). The damping term needs
no assumption that clipped Kelvin–Voigt forces dissipate `b_n δ̇²`. The dissipation actually
produced is `b_n χ [δ̇]₊²`. Contact parameters are fixed and known in the main experiments. A
separate stress test (§10.6) uses a known friction interval and a conservative lower credit.

Only the tip interacts with the wall. The auditor checks that no other body corner penetrates the
wall (domain check, reported).

## 3. Composite storage and power balance (A)

Robot kinetic storage `S_R = ½ q̇ᵀ M q̇` (no gravity), task storage `S_T`, composite `S = S_R + S_T ≥ 0`.
With `P_a = u_actᵀ q̇`, `p_H = −f_Hᵀ v_H = −τ_Hᵀ q̇`, `D_R = q̇ᵀ B_R q̇`:

    Ṡ_R = P_a − p_T − p_H − D_R ,     Ṡ_T = p_T − D_T ,
    Ṡ   = P_a − p_H − D_R − D_T .

`p_H > 0` means robot-to-human output. `f_H` is the force exerted **by the human on the robot**
at the robot material point `p_Hc(q, s)`. The human power uses the material-point velocity
`v_H = J_H(q, s) q̇`. If a location estimate migrates, its migration velocity `ṡ` is **not** added
to `v_H`.

## 4. Ec + BH accounts (preserved, A)

Account definitions and sign conventions are taken from the existing Ec + BH mainline (workflow
`human-port-ecbh-20261029`, `FORMULATION.md`, read-only). They are re-implemented here for a
different plant.

    Ėc  = c_T + c_R − P_a ,    0 ≤ c_T ≤ D_T ,  0 ≤ c_R ≤ D_R ,  Ec ≥ E_min ,
    ḂH  = −p_H ,               BH ≥ 0   (net human-output account; human input replenishes it).

`c_R = 0` throughout (`D_R` is physical but not reimbursed). `c_T` is the **realised** credit from
measured state and the known contact law (§4.1). A predicted or mean `D_T` is never booked as
credit. There is no BU account, no task-authorisation coefficient, no residual-energy subtraction,
and no implicit reset or time refill of BH. Release does not refill BH.

With `V = S + Ec − E_min`:

    V̇ = −p_H − (D_T − c_T) − (D_R − c_R) ≤ −p_H .

If every prefix satisfies `Ec ≥ E_min`, `BH ≥ 0`, and the credit is valid, then
`W_H,net(t) = ∫₀ᵗ p_H ≤ min(V(0), BH(0))`. BH is a constraint ledger and is **not** added to `V`.
GP confidence, likelihood or risk preference never authorises credit above the valid physical
bound.

### 4.1 Discrete booking (what the software does)

Servo period `h_s = 1 ms`, zero-order-hold command `u_k`, exact encoder sampling of `q, q̇` (stated
assumption, as in the existing mainline).

* **Actuator work:** within a tick `u` is constant, so `∫ P_a dt = u_kᵀ (q_{k+1} − q_k)` **exactly**.
  The software books this value. Booked actuator work is exact under exact encoders.
* **Credit:** `C_k = h_s · min(D_T(x_k), D_T(x_{k+1}))` ("min-endpoint" conservative quadrature).
  This lower-bounds `∫ D_T` over the tick when `D_T` has no interior dip below both endpoints
  within 1 ms. That is not proven. The independent auditor checks the prefix condition
  `Σ_{j≤k} C_j ≤ ∫₀^{t_{k+1}} D_T` on the fine integration grid and reports any violation (C).
* **Ec ledger:** `Ec_{k+1} = Ec_k + C_k − u_kᵀ(q_{k+1} − q_k)`. It is exactly the account at tick
  boundaries. Intra-tick minima are audited (C).
* **BH ledger (booked):** `BH^book_{k+1} = BH^book_k + ½ h_s (y_kᵀ v̂_k + y_{k+1}ᵀ v̂_{k+1})`,
  `v̂ = J_H(q, ŝ) q̇`, `y` = measured force (§6). This is the **known software ledger**. It is
  not the true BH.

### 4.2 True-account estimation error (both accounts)

**Ec:** under exact encoders, booked actuator work equals true actuator work, so
`Ec_true = Ec_book` at tick boundaries. The only Ec-related estimation error is credit validity
(previous bullet). It is audited, not assumed. In the parameter-mismatch stress test, the
uncertainty about `D_T` is handled by a conservative lower credit (§10.6). The **future** Ec is
uncertain through uncertain robot motion and is propagated (§9).

**BH:** with `y = f + b_f + ε_m` and `v_H(s) − v_H(ŝ) = θ̇ R e₂ (s − ŝ)`, to first order (dropping
the second-order product of measurement and location errors):

    e_BH := BH_true − BH_book = Σ_k h_s [G_k (s − ŝ_k)] − b_fᵀ Δp̂ − n_e ,
    G_k = θ̇_k y_kᵀ R_k e₂ ,   Δp̂ = Σ_k h_s v̂_k ,   n_e ~ N(0, Σ_k h_s² v̂_kᵀ Σ_m v̂_k).

`e_BH` is affine in the latent `(s, b_f)` plus independent white-noise accumulation `n_e`. It is
therefore **correlated with the location/force posterior** of §7. That is how initially independent
errors become correlated through inference. The software tracks `G_tot = Σ h_s G_k`,
`Γ = Σ h_s G_k ŝ_k`, `Δp̂`, and `var(n_e)` per contact event. Completed events are frozen with
their posterior error. Uncertainty about past work is **never** reset to zero at a planning call.
(Independence of the 1 kHz ledger noise from the 25 Hz GP samples is an approximation. The shared
samples carry 1/40 of the ledger-noise variance.)

## 5. Backup reserve and the continuous-time reference filter

### 5.1 Lemma (passive zero-input backup, A)

Horizontal plane, no gravity, `S ≥ 0`, and the identities of §3. Under ideal immediate `u_act ≡ 0`
on `[t₀, t]`, `P_a = 0`, so `Ṡ = −p_H − D` with `D = D_R + D_T ≥ 0`. Integrating,

    ∫_{t₀}^{t} p_H dτ = S(t₀) − S(t) − ∫ D ≤ S(t₀)    for every prefix t ≥ t₀,

**whatever the human does.** Hence, with `R_H(x) := S(x)` and `h_H := BH − S`: if
`h_H(t₀) ≥ 0`, then after switching to zero input, `BH(t) = BH(t₀) − ∫ p_H ≥ BH(t₀) − S(t₀) ≥ 0`
for all future prefixes. Under zero input `Ėc = c_T + c_R ≥ 0`, so `Ec ≥ E_min` is also preserved.
`BH ≥ S` is a **sufficient** reserve for this passive backup's energy bound. It is conservative,
not the minimum reserve. Moreover,

    ḣ_H = ḂH − Ṡ = −p_H − (P_a − p_H − D) = D − P_a     (p_H cancels exactly).

**What the lemma does not say.** Zero input is **not** a safe physical stop. The tool may keep
moving or drift. The human can still be pushed by stored elastic energy released from the contact
(bounded by `S(t₀)`). Contact force can make excursions. Stopping at zero velocity does not remove
elastic storage. The lemma certifies only the stated energy accounts. It does not bound force,
peak power or injury.

### 5.2 Proposition (continuous-time reference filter, A — conditional)

Assume:
* exact current `Ec`, `BH`, `S` and a lower bound `0 ≤ D_lo ≤ D`;
* `u_act = 0` is admissible (it is in the box), and these are the only hard constraints;
* the closed loop has absolutely continuous solutions.

Choose `u_act(t)` so that, with rates `λ_H, λ_E > 0`,

    (i)  P_a ≤ D_lo + λ_H h_H ,      (ii) P_a ≤ c_T + c_R + λ_E (Ec − E_min).

Then (i) gives `ḣ_H = D − P_a ≥ −λ_H h_H`, and by the comparison lemma
`h_H(t) ≥ h_H(0) e^{−λ_H t}`. (ii) gives `Ec(t) − E_min ≥ (Ec(0) − E_min) e^{−λ_E t}`.

**Initial-condition requirement:** `h_H(0) = BH(0) − S(0) ≥ 0` and `Ec(0) ≥ E_min`. Then
`h_H ≥ 0`, `BH ≥ S ≥ 0` and `Ec ≥ E_min` for all t. **Feasibility:** `u = 0` satisfies (i)–(ii)
whenever the accounts are in the set, so the constraint set is non-empty inside the box.
If the initial condition fails (`h_H(0) < 0`, §10 "infeasible initialisation"), (i) forces
`P_a ≤ D − λ_H|h_H|`. This is active braking or energy extraction. Whether it is achievable within
the box is **not** guaranteed, and such runs are labelled infeasible-initialisation, not as covered
by the proposition.

**Scope.** Adding slew limits, force caps, workspace limits or gravity compensation changes the
claim, because `u = 0` may no longer be admissible. The implementation (§11) is sampled-data (ZOH,
1 ms) and uses estimated accounts. It is therefore a **numerical validation (C)** of this
continuous-time reference, not an intersample proof. An intersample bound would need
`|P_a(t) − P_a(t_k)| ≤ |u_k| ‖q̈‖_∞ h_s` with a bound on `q̈` that includes the human force; this
PoC does not establish one. Observed intersample deficits are reported.

## 6. Human truth model (simulator only; inaccessible to the controller)

Following P1 Eqs. (7)-(8) with diagonal impedance and **no human inertia** (we omit P1's inertia
term `M ë`). We model `f_H`, the force exerted **by the human on the robot**:

    f_H(t) = w(t) · [ K_H (a_H(t) − p_Hc(t)) + D_H (ȧ_H(t) − v_H(t)) + g_true(χ_t) + ε_phys(t) ] ,

* `a_H(t)` is the human's latent intended contact-point trajectory (world frame). It is generated
  from **intent families** (§10), not sampled from the fitted GP.
* `K_H, D_H` are positive diagonal. They're drawn per episode, lognormal around the nominal values,
  plus held-out gain-mismatch cases.
* `g_true(χ) = −c_g tanh(χ / v_g)` per axis is a residual nonlinearity in the patch velocity `χ = ċ`.
  It is switched off in the first validation stage.
* `ε_phys` is physical force variability, an OU process (`σ_p`, `τ_p`), sampled once per episode on
  a fine grid. It acts on the plant.
* `w(t) ∈ [0, 1]` is the engagement. At grasp, `a_H(t_on) = p_Hc(t_on)` (no pre-load), so the force
  starts continuously. Release is a 20 ms linear ramp to zero at `t_off`. It is triggered by the
  scenario script or by `|f_H| > F_rel` (the human lets go).
* The true location `s` is fixed for each contact event and drawn from the scenario.

Positive impedance gains with a moving intention do **not** make the human passive. The human
injects energy whenever `a_H` moves. We exclude human inertia. Adding `M_H(ä_H − p̈_Hc)` would couple
the robot and human dynamics implicitly and needs a new derivation. No future ground-truth
acceleration is fed to the predictor.

## 7. Structured (impedance-informed) latent-intention GP — our specialisation of P1

### 7.1 Observation model

These quantities are available to the controller:
* exact robot state;
* the measured human force `y_j = f_H(t_j) + b_f + ε_m,j`, white `ε_m ~ N(0, σ_m² I)` at 1 kHz;
  this measurement noise is **not** applied to the plant; `b_f` is a persistent sensor bias,
  zero in nominal runs;
* location readings `s_obs = s + b_s + ε_s` at 25 Hz, where `b_s` is a persistent per-episode
  error and `ε_s` is white;
* a binary contact-presence flag.

The truth intention `a_H`, the true `s`, gains, the residual, the OU path and future data are
inaccessible. Oracle runs are labelled.

### 7.2 Prior and its linear structure

For one contact event starting at `t_on`, and per axis `i ∈ {x, y}`:

    a(t) = p_Hc(t_on) + ã(t) = c_on + s u_on + ã(t) ,   ã_i ~ GP(0, k_a) ,
    k_a(t,t') = σ_a² exp(−(t−t')²/(2ℓ_a²)) + σ_c² + σ_l² ((t−t_on)(t'−t_on))/T_l² ,

We use two more latent groups:
* residual `g_i(χ) = φ(χ)ᵀ w_i` with Gaussian RBF features `φ`, weights `w_i ~ N(μ_w,i, Σ_w,i)`
  (the frozen offline posterior); this is an explicitly labelled finite-feature approximation;
* physical noise `ε_p,i ~ GP(0, σ_p² e^{−|t−t'|/τ_p})`.

Location, force-bias and location-bias priors are `s ~ N(μ_s0, σ_s0²)`, `b_f ~ N(0, σ_bf² I)`,
`b_s ~ N(0, σ_bs²)`. All latent groups are a-priori independent.

For fixed gains `K, D` and the **conditioned, deterministic** (measured) past robot trajectory,
`L_t = K + D d/dt` gives

    f(t) = K(c_on − c(t)) − D ċ(t) + s h(t) + K ã(t) + D ã̇(t) + Φ(χ_t) w + ε_p(t),
    h(t) = K(u_on − u(t)) − D u̇(t)                                      (2-vector),

All terms are **linear** in `(ã, s, w, b_f, ε_p)`. The residual input `χ_t = ċ(t)` (patch-reference
velocity) does not depend on `s`. Prior force mean and covariance per axis:

    m_f(t)     = K(c_on − c(t)) − D ċ(t) + μ_s0 h(t) + Φ(χ_t) μ_w ,
    K_f(t,t')  = K² k_a + K D (∂₁k_a + ∂₂k_a) + D² ∂₁∂₂k_a + k_p + Φ_t Σ_w Φ_{t'}ᵀ
                 + σ_s0² h(t) h(t')ᵀ (couples axes) .

This equals the task text's `L_t K_a L_t'ᵀ + K_g` (here `K_g = Φ Σ_w Φᵀ`) plus the OU and
location terms. It is P1 Eq. (17)-(18) with the inertia set to zero and deterministic gains
(`σ_d = σ_k = σ_m = 0`). For the stationary SE part the `K D` cross-term cancels on the
diagonal. The observation adds `σ_bf² + σ_m² δ_jl`. The location summary `s̄` (mean of `n_s`
readings) has `Var(s̄) = σ_s0² + σ_bs² + σ_εs²/n_s` and `Cov(y_j, s̄) = σ_s0² h(t_j)`.

**Intention–force cross-covariance** (P1 Eq. (25) with `μ_m = 0`; `∂₁` = derivative in the first
argument):

    Cov(y_j,i, ã_i(t*))  = K_i k_a(t_j,t*) + D_i ∂₁k_a(t_j,t*),
    Cov(y_j,i, ã̇_i(t*))  = K_i ∂₂k_a(t_j,t*) + D_i ∂₁∂₂k_a(t_j,t*),
    Cov(y_j, s) = σ_s0² h(t_j),  Cov(y_j,i, w_i) = Φ_j Σ_w,i,  Cov(y_j,i, b_f,i) = σ_bf²,
    Cov(y_j,i, ε_p,i(t_now)) = k_p(t_j, t_now).

**Units:** `k_a` [m²], `∂k_a` [m²/s], `∂₁∂₂k_a` [m²/s²]. Every force-covariance block is [N²].
This is tested, together with finite-difference derivatives.

### 7.3 Joint conditioning (exact Gaussian under the stated conditions)

The targets are

    ζ = [ã_x, ã̇_x, ã_y, ã̇_y at the N horizon nodes; s; w_x; w_y; b_f; ε_p(t_now)] .

We stack the window observations `Y = [y (2n); s̄]` and apply Gaussian conditioning
(P1 Eq. (26), with the minus sign and transpose verified):

    E[ζ | Y]   = μ_ζ + Cᵀ K_YY⁻¹ (Y − m_Y),     Cov[ζ | Y] = Σ_ζ − Cᵀ K_YY⁻¹ C .

This is **one** joint conditioning. After observing common force data, the intention, residual
weights, location, bias and OU state become correlated. We keep all of those cross-covariances.
We never condition two GPs separately and add their variances. The posterior is exact Gaussian
because:
* the gains are fixed;
* the past robot trajectory is conditioned (known);
* `χ` is deterministic;
* `p_Hc` is affine in `s`.

**Approximations, stated.** Integrating over uncertain gains, uncertain future robot motion
(feedback), or `s` inside nonlinear functions (e.g. future rotations) destroys exact Gaussianity.
Future propagation (§9) is therefore approximate (B): it is either Monte Carlo over consistent
joint samples or first-order linearisation. The Gaussian prior on `s` is a local approximation on
a bounded patch. Samples are **not** clamped. The predicted probability mass outside the patch is
logged; when it exceeds 1% the run is flagged.

### 7.4 Identifiability and confounding

* `s` and a constant intention offset are confounded. `K(ã − s u)` is invariant to
  `(ã, s) → (ã + Δ u, s + Δ)` when `u` is constant. Force data inform `s` only through **rotation**
  (`u(t) ≠ u_on`, `u̇ ≠ 0`). The independent location sensor is therefore essential. Its persistent
  bias `b_s` sets a floor on location uncertainty that does not average out.
* A constant force bias `b_f` is confounded with an intention offset: `K Δã = b_f`.
* A flexible residual `Φ w` and the intention `ã` can explain the same force. The residual is
  pinned by **cross-episode** training data, because its input is the patch velocity rather than
  time. Within one episode the decomposition is not unique. We report the posterior correlation
  between `w` and `ã`. We do not claim unique recovery of both.
* Gains, intention and location are **not** jointly inferred online from a single force
  observation. Gains are fixed: nominal values are chosen offline on training trajectories with
  exogenous robot excitation (marginal likelihood). Joint gain inference is deferred.

### 7.5 Baselines (matched data/window)

* **Naive force GP** (P1 Alg. 1 step 1 analogue): `f_i ~ GP(0, k_SE([t, χ])) + white`,
  hyperparameters by maximum likelihood on the same training split. It responds to candidate
  motion only through learned correlation with `χ`.
* **Gauss–Markov (C2):** per axis, `f = μ_f + η`, η an OU (`τ_f, σ_f` fitted offline),
  `μ_f` = window mean with its standard error. It is an exogenous force forecast and does **not**
  respond to candidate robot motion.
* **Deterministic (C1):** hold the last (window-averaged) measured force; location `ŝ`; no covariance.

## 8. Generalised human torque uncertainty (illustration, first order)

At known `q`, with `[s, f]` jointly uncertain:

    δτ_H ≈ G_s δs + J_Hᵀ δf ,   G_s = ∂J_Hᵀ/∂s · f̄ = [0, 0, (R e₂)ᵀ f̄]ᵀ ,
    Cov(τ_H) ≈ [G_s, J_Hᵀ] Cov([s; f]) [G_s, J_Hᵀ]ᵀ .

This is verified against Monte Carlo (tests and figure `moment_arm`).

## 9. Belief propagation over a short horizon (B)

Candidate policies are closed-loop feedback laws (§10.1), so the predicted human force depends on
the predicted robot motion. The predictor evaluates `f_H(x, s, ζ)` **inside** the rollout. It is
never an exogenous force forecast after conditioning at the current state.

**Random inputs.** These are drawn jointly:
* the GP posterior vector `ζ` from §7.3, which includes `s`, `b_f`, `w` and the current OU state,
  with all cross-covariances;
* OU innovations per stage, `ξ_k ~ N(0, I)`;
* the past-work error `e_BH` from §4.2, affine in the same `s`, `b_f`, plus `n_e`.

The augmented random vector is `η = [ζ; ξ; n_e]`. The stage force uses
`a(t_k + τ) ≈ a_k + τ ȧ_k` within a stage. This is first-order in τ ≤ 20 ms, and both `a_k` and
`ȧ_k` are jointly sampled.

**State.** `z = [q, q̇, Ec, BH_true, W_H, controller states (x_ref, ∫ force error), J]`.
The latent function values are static, so the augmented state of the task text,
`[x, s_H, human latents, Ec, BH, ctrl]`, is represented as `z` plus `η`. Their covariance is carried
through the sensitivity `Φ_k = ∂z_k/∂η`.

**Monte Carlo (nonlinear check / objective).** `N_s` joint samples of `η` (common random numbers
across candidates):
* **one** posterior function sample per rollout;
* one location per rollout;
* OU innovations evolve per stage.

Each sample is rolled out through the deterministic closed-loop step `z_{k+1} = G_Δ(z_k, η)`
(semi-implicit Euler, 5 ms substeps, known contact law, saturation). GP epistemic error is **not**
redrawn per step.

**First-order propagation.** The mean trajectory is `z̄_{k+1} = G_Δ(z̄_k, η̄)`, and
`Φ_{k+1} = A_k Φ_k + B_k E_k`, where:
* `A_k = ∂G/∂z` **includes the feedback-law derivatives**;
* `B_k = ∂G/∂(local η)`;
* `E_k` selects the components of `η` used in stage k.

Both `A_k` and `B_k` come from central finite differences on the vectorised step. Then
`Cov(z_k, z_l) = Φ_k Σ_η Φ_lᵀ`. This retains the full cross-time covariance, and it is equivalent
to `P_{k+1} = A P Aᵀ + L Q Lᵀ` on the augmented state.

**Integrated work.** Predicted human work uses the same predicted trajectory:
`W_H = ∫ −f_Hᵀ J_H q̇`, `Var(Σ_i W_i) = Σ_{i,j} Cov(W_i, W_j)` (cross-time terms kept), and
`Var(BH_next) = Var(BH) + Var(W_H) − 2 Cov(BH, W_H)`. The past-work error is correlated with `s`,
hence with future work.

**Mean-power correction.** `E[p_H] = −E[f]ᵀE[v] − tr Cov(f, v)`. First-order mean propagation
evaluates `−f̄ᵀ v̄` and misses the trace term. We report it from MC. With human damping,
`Cov(f, v) ≈ −D Var(v)`, so first order *under*-predicts expected robot-to-human power.

Monte Carlo and the first-order method are compared on mean, covariance and **work tails**
(empirical quantiles), not only on force RMSE. Sigma points or MC moments do not certify tail
probabilities.

## 10. Planning: risk-neutral vs risk-sensitive selection with separate constraints (B)

### 10.1 Decision space: finite candidate feedback policies

The nominal hybrid controller (all controllers) acts at the tool tip:
* **tangential impedance** to a reference `x_ref` advancing at `σ_v v_nom`, with stiffness
  `σ_K K_t`, damping `D_t`, and friction feed-forward from the measured contact force;
* **normal force PI** to `σ_F f_d`, plus normal damping;
* **orientation PD** to `θ_ref = 0`.

The tip force is mapped by `J_Tᵀ`, the orientation torque is added, and the result is saturated.
The normal channel also has an **approach-speed limiter**: extra damping `D_brake·[−v_app − v_n]₊`,
active only when the tip moves toward the wall faster than `v_app`. The force integrator also
leaks out of contact. Both were added after a smoke-test failure; see `decision_log.md`.

The candidate set is:
* `σ_v ∈ {0, 0.5, 1, 1.5}` × `σ_K ∈ {1, 0.35}` × `σ_F ∈ {1, 0.5}` (16 policies);
* the **passive backup** `u ≡ 0` (the policy of §5.1);
* a strictly dissipative **brake** `u = −D_brk q̇`, for which `P_a = −q̇ᵀD_brk q̇ ≤ 0`, so
  `ḣ_H ≥ 0` and `Ėc ≥ 0`.

The brake was added because, under the passive backup, the free tool drifted and rotated until
re-engaging the task needed more energy than the remaining BH. That is the "zero input is not a
safe physical stop" caveat in practice.

The selected policy runs at the 1 kHz servo rate for one 20 ms control period (receding horizon,
N = 15 stages). The solver is **finite enumeration** of these 18 candidates. It is not a QP or NLP,
and there is no claim of optimality over continuous inputs.

### 10.2 Objective (separate from safety)

The stage cost `J` is dimensionless, with each term normalised by a stated scale:
* progress shortfall against the **nominal task schedule**, so slowing down costs progress;
* normal-force error;
* orientation error;
* effort `‖u‖²_W`;
* control increments.

`J` is averaged over the horizon. A separately described **energy-margin (task-viability) cost**,
`w_res·max(0, (R_task − h_H,N)/R_task)²` at the horizon end, discourages spending the human budget
below what continued wiping needs (`R_task = 0.03 J`, `w_res = 2`). This term is a performance
term. It is identical in C1-C4, it is not a constraint, and it is never a substitute for one.
The safety constraints (§10.3) are separate and are never traded off inside `J`.

* risk-neutral (C3): `ρ₀(J) = E[J]` (MC mean);
* risk-sensitive (C4): `ρ_β(J) = (1/β) log E[exp(β J)]`, evaluated with stable log-sum-exp over
  MC samples with equal, non-negative weights. The delta-method sampling standard error and the
  effective sample size of the exponential weights are logged.

**Risk mapping:** P3/P4 use `γ(θ) = −(2/θ) ln E[exp(−θJ/2)]`. With `β = −θ/2`, `ρ_β = γ(θ)`. So
**β > 0 is risk-averse**, and `ρ_β ≈ E[J] + (β/2) Var(J)` for small β (P3 Eq. (11):
`γ ≈ E[J] − (θ/4) Var[J]`). This mean–variance expansion is an approximation and is not used as
the objective.

**MGF domain:** the empirical estimator is always finite. An analytic exponential-quadratic
evaluation for Gaussian `z` and quadratic `J = zᵀQz` would need `I − 2βΣ^{1/2}QΣ^{1/2} ≻ 0`. We
make no analytic claim.

Compliance direction is **measured**, not assumed: we log the selected `σ_K`, `σ_v`, `σ_F`. P4
reports that its risk-averse controller stiffens.

### 10.3 Chance constraints (approximate, B)

The constraints are evaluated on first-order moments, with an MC-moment ablation. For every
candidate and every stage prefix `k = 1..N`:

    E[Ec_k] − κ(α_E) std(Ec_k) ≥ E_min ,
    E[h_H,k] − κ(α_H) std(h_H,k) ≥ 0 ,   h_H,k = BH_true,k − S_k ,

* `h_H` is tightened **as a single quantity**, so the covariance between `BH_true` and `S` is kept.
  We never tighten BH and S separately and subtract.
* `BH_true,k = BH_book + e_BH − W_H(0..k)` includes the uncertain past-work error.
* `κ(α) = Φ⁻¹(1 − α)` with `α_E = α_H = 0.01` per planning call.

This per-call level is **not** a mission guarantee. The mission-level violation frequency over
executed episodes is measured (C), with a stated budget. A horizon-level budget reset at each
replan is not a mission guarantee.

**Selection:** among candidates satisfying all constraints, minimise `ρ_β`. If none satisfies
them, select the candidate with the largest minimum normalised margin and log
`planner_infeasible`. Then the servo filter acts.

### 10.4 Baselines with the same candidates, horizon, constraints and budgets

| ID | predictor | constraints | objective | servo filter |
|---|---|---|---|---|
| C0 | none (nominal policy σ=(1,1,1)) | none | — | **none (unprotected reference)** |
| C1 | deterministic hold-force, `ŝ` | mean only (κ = 0) | E[J] (1 sample) | ledger, no margin |
| C2 | Gauss–Markov force + Gaussian `s` | tightened (κ) | E[J] | ledger + belief margin |
| C3 | structured latent-intention GP | tightened (κ) | E[J] | ledger + belief margin |
| C4 | as C3 | as C3 | ρ_β, β > 0 | as C3 |

The diagnostic **ORACLE-BH≥S** run is a separate row. It runs the nominal policy with the
reference filter of §5.2 using **true** accounts. It has truth access and is reported separately.
All comparisons pair hidden intent, parameters and noise seeds across controllers. Human forces are
regenerated from the same hidden intention and the actual state; force traces are not replayed.

### 10.5 Optional P4-inspired reference adaptation

This is deferred, and the deferral is listed in `report.md`.

### 10.6 Exact vs conservative credit

In the stress test, the true friction is `μ ∈ [μ_lo, μ_hi]`. The software books
`c_T = D_T(μ_lo)`, which is valid for every μ in the interval. The unreimbursed loss and the
budget drain are quantified.

### 10.7 Post-hoc robust interval bound on the past-work error (added after the frozen run)

This bound was designed after the frozen main comparison showed the sensor-bias failure. Assume
known calibration intervals `|b_f,i| ≤ B_f` and `|s − ŝ_k| ≤ B_s`. Then, per contact event, the
error model of §4.2 satisfies

    e_BH ≥ (Gaussian part) − B_f ‖Δp̂‖₁ − B_s Σ_k |G_k| h_s ,

The planner starts its rollouts from `BH_book − R`, with `R = B_f‖Δp̂‖₁ + B_s Σ|G_k|h_s` summed over
events. The servo margin mean is shifted by `−R`. The Gaussian margin is kept, which double-counts
and is therefore conservative.

The bound holds only if the intervals hold. The second-order term `ε_m·(s − ŝ)` is still dropped, so
the bound is not a certificate. It is evaluated on fresh seeds and reported separately from the
frozen comparison.

## 11. Servo-rate energy filter (implementation of §5.2 with estimated accounts)

At every 1 ms tick the software uses:
* the measured state;
* the booked ledgers;
* the belief margin of `e_BH` (mean `m_e`, std `σ_e`, refreshed at 50 Hz);
* `Ŝ` (exact model and state).

From these it forms

    b_H = D̂_T + λ_H (BH_book + m_e − κ_H σ_e − Ŝ) ,    b_E = ĉ_T + λ_E (Ec_book − E_min) .

**Sampled-data constraint on tick work.** The naive instantaneous condition `uᵀq̇(t_k) ≤ b` was
implemented first. It **leaks energy**: when the tool is nearly stalled against the human, any
large force passes the test at `t_k`, then accelerates the tool and injects energy within the
1 ms hold. The result was bang-bang chattering and true BH < 0 in the first smoke test; this
version is kept as the ablation `Finst`. The implemented filter instead constrains the
**predicted actuator work over the tick**. It uses a second-order Taylor model of the ZOH response
with the measured external generalised force `τ_e` (known contact law, measured human force at
`ŝ`, known robot friction) held over the tick:

    Δq(u) ≈ q̇ h + ½h² M⁻¹(u + τ_e),
    g(u) = uᵀΔq(u)/h = uᵀq̇ + (h/2) uᵀM⁻¹u + (h/2) uᵀM⁻¹τ_e ≤ min(b_H, b_E),
    u_act = argmin_{u ∈ box} ‖u − u_nom‖²_W   s.t.  g(u) ≤ min(b_H, b_E) .

The feasible set is convex (an ellipsoid intersected with the box) and contains `u = 0` whenever
the bound is ≥ 0. The separable KKT system gives
`u_i(ν) = clip((2W_i u_nom,i − ν a_i)/(2W_i + ν c_i))`, where `a = q̇ + (h/2)M⁻¹τ_e`, `c = h·diag(M⁻¹)`.
`g(u(ν))` is monotone in ν, so ν is found by bisection. Optimality is checked against SLSQP in the
tests.

If even the box minimiser of `g` violates the bound, the most energy-extracting admissible input is
applied and the tick is logged as `infeasible_best_effort`. C1 uses `κ_H = 0` and `m_e = 0`. Any
filter modification is part of `u_act` and of booked `P_a`. The physical robot dissipation is not
used in `D̂` (`D_R,lo = 0`, conservative).

This remains tier C. The Taylor remainder and the variation of the human force within a tick are
not bounded, so there is no intersample proof.

## 12. Guarantee tiers and claim boundaries

* **(A) Proven:**
  * the contact identity `p_T = Ṡ_T + D_T` (§2);
  * the composite balance (§3);
  * the accounts inequality `V̇ ≤ −p_H` (§4);
  * the zero-input backup lemma (§5.1);
  * the continuous-time reference filter proposition under exact accounts (§5.2);
  * the exactness of the conditional Gaussian posterior under fixed gains and a conditioned
    trajectory (§7.3).
* **(B) Approximate:**
  * propagation through uncertain feedback motion, MC or first order (§9);
  * Gaussian tightening `mean − κ std` (§10.3);
  * the finite-feature residual;
  * the stage-Taylor intention;
  * the treatment of the Gaussian location prior on a bounded patch.

  Gaussian output tightening is **not** a probabilistic certificate. A certified claim would need:
  * calibrated joint input uncertainty;
  * conservative nonlinear verification over a joint confidence region, or a remainder bound;
  * coverage of model and calibration error;
  * coverage of intersample effects.

  None of these is established here.
* **(C) Empirical:** violation frequencies with Wilson/Clopper–Pearson intervals over independent
  episodes (never over correlated time steps), energy audits, and step-refinement convergence.

Not claimed:
* safe physical stopping, force/peak-power/injury bounds;
* recursive feasibility of the planner;
* real-time performance;
* hardware validity;
* that uncertainty or risk aversion produces compliance;
* that the four papers establish this combination.

## 13. Default parameters (frozen in `configs/default.json`)

| group | values |
|---|---|
| body | m = 2.0 kg, I = 0.03 kg m², L × H = 0.40 × 0.10 m, r_T = (0.12, −0.10) m, h_e = 0.05 m, s ∈ [−0.18, 0.18] m |
| robot loss | B_R = diag(1.0 N s/m, 1.0 N s/m, 0.005 N m s), c_R = 0 |
| actuation | |u| ≤ (40 N, 40 N, 4 N m), ZOH 1 ms |
| contact | k_n = 5000 N/m, b_n = 150 N s/m, δ_s = 0.5 mm, μ = 0.3, v_ε = 5 mm/s |
| task | f_d = 10 N, v_nom = 0.08 m/s, wipe x_tip ∈ [−0.25, 0.25] m, episode 9 s |
| controller | K_t = 600 N/m, D_t = 50 N s/m, k_f = 0.6, k_i = 8 s⁻¹, D_n = 40 N s/m, K_θ = 40 N m/rad, D_θ = 2 N m s/rad |
| human truth | K_H = 150 N/m, D_H = 15 N s/m nominal (lognormal σ = 0.2 per episode), OU σ_p = 0.4 N, τ_p = 0.15 s, residual c_g = 1 N, v_g = 0.05 m/s, F_rel = 35 N |
| sensors | σ_m = 0.25 N (1 kHz), location 25 Hz, σ_εs = 0.03 m, b_s ~ N(0, 0.02²) m |
| accounts | Ec0 = 0.6 J, BH0 = 0.15 J, E_min = 0, λ_E = λ_H = 5 s⁻¹ |
| planner | Δ = 20 ms, N = 15, substep 5 ms, N_s = 48, α_E = α_H = 0.01, β(C4) = 2, 18 candidates (16 + backup + brake) |
| added after smoke debugging | v_app = 3 cm/s, D_brake = 400 N s/m, integrator leak 5 s⁻¹, D_brk = diag(40, 40, 1.5), w_res = 2, R_task = 0.03 J |
| physics | RK4, 1 ms (refinement 0.5, 0.25 ms); servo 1 kHz in all cases |
| scenarios | no_human, block_release, assist_reverse, normal_oblique (two events), moment_arm (identical normal push, s varied), ood_intent (held-out oscillating intent), sensor_bias (b_f = (0.8, −0.5) N, b_s = 6 cm, both outside the calibrated priors), low_BH (BH0 = 0.02 J), infeasible_init (preloaded contact, BH0 = 0.004 J < S0) |

GP hyperparameters are fitted offline and frozen in `outputs/models/gp_frozen.json`.
