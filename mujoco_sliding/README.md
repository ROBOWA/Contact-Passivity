# mujoco_sliding — planar contact-sliding proof of concept

MuJoCo-based 2D contact-sliding simulation: a proof of concept for
contact-model-aware **selective passivation**. A planar two-link robot
approaches a horizontal surface, regulates a desired normal contact force,
and slides tangentially, while ground-truth **task-contact** and
**human-interaction** force channels are logged separately. On top of the
validated nominal force/motion controller sits a selective-passivation layer
(residual energy ledger + nested human energy ledger + task-weighted OSQP
filter, modes C0–C4) — see *Selective-passivation layer* below.

Tested with MuJoCo 3.3.1 **and** 3.11.0 (official `mujoco` Python bindings),
NumPy, matplotlib, SciPy, OSQP ≥ 1.0. MuJoCo renamed `MjData.qM` → `MjData.M`
and changed the `mj_fullM` signature after 3.3.x; `passivity_qp.mass_matrix`
feature-detects both, and the two versions reproduce every published metric to
≈1e-12.

## Running

From the `contactSimulation/` directory:

```bash
python -m mujoco_sliding.simulation                 # headless default run
python -m mujoco_sliding.simulation --human-force   # with the known human push
python -m mujoco_sliding.simulation --duration 8 --output-dir results --no-animation
python -m mujoco_sliding.viewer                     # interactive viewer (needs display)
python -m mujoco_sliding.viewer --human-force
python -m mujoco_sliding.viewer --scenario C --mode C3_dual_ledger_qp  # live passivation demo
python -m pytest mujoco_sliding/tests               # test suite

# Selective-passivation experiments (headless, deterministic):
python -m mujoco_sliding.experiments --all          # scenarios A-F + summary + acceptance
python -m mujoco_sliding.experiments --scenario C   # one scenario, its default modes
python -m mujoco_sliding.experiments --scenario C --modes C3_dual_ledger_qp C4_dual_ledger_scalar
python -m mujoco_sliding.experiments --sweep        # mu_hat sweep of the friction predictor
python -m mujoco_sliding.experiments --all --animate  # + GIF for the C3 runs
```

The headless run writes to `results/` (configurable): `log.npz` (full log),
`log.csv` (flat columns), `results.png` (3×3 results figure), and
`animation.gif` (x-z plane animation; skipped gracefully if no GIF writer is
available). The viewer is never required by tests or the headless pipeline.

## Model and coordinate convention

`models/planar_two_link.xml`:

* World `z` is **up**; gravity is −z. The surface (`surface` box geom) has its
  top face exactly at `z = 0`.
* The robot base sits at `(0, 0, 0.7)`; two links of length 0.6 m; a spherical
  `end_effector_pad` (radius 0.05 m) with the `ee_site` at its center.
* **Planar motion is enforced structurally**: both joints (`shoulder`,
  `elbow`) are revolute hinges about the y axis, so the mechanism has no y
  DOF at all (a test asserts the site Jacobian's y row is identically zero).
* Two torque-controlled `motor` actuators, ctrlrange ±60 N·m.
* Timestep 1 ms, `implicitfast` integrator (stable implicit handling of the
  joint damping), energy computation enabled.
* **Collision filtering**: only `surface` and `end_effector_pad` have
  `contype/conaffinity = 1`; every other geom has 0/0, so the *only* possible
  contact pair is pad–surface.
* `condim="3"`: one normal + two tangential friction directions,
  friction coefficient 0.3.
* Contact softness `solref="0.08 1.5"`, `solimp="0.2 0.7 0.02"` — a
  deliberately soft, overdamped contact. With stiff defaults the pad-contact
  resonance (~100 Hz) plus the one-step-delayed force feedback produced a
  micro-bouncing limit cycle (contact active only ~50–90 % of steps, F_n
  oscillating 0–16 N). With the soft setting, sliding contact stays active
  100 % of steps, F_n holds 5.00 ± 0.01 N, and steady penetration is ≈1.3 mm.

## Directional decomposition (C/U)

With tangent `t = [1,0,0]ᵀ` and upward normal `n = [0,0,1]ᵀ` (`config.py`):

* `J_t = tᵀ J_p`, `J_n = nᵀ J_p` from the site translational Jacobian `J_p`
  (`mj_jacSite`).
* Projectors `U = t tᵀ` (tangential) and `C = n nᵀ` (normal) satisfy
  `C + U = diag(1, 0, 1)` (identity on the task plane) and `CU = 0` (tested).
* The normal-force controller acts in C-space, the tangential motion
  controller in U-space; logged torque splits are `τ_C = J_pᵀ C F_cmd` and
  `τ_U = J_pᵀ U F_cmd`.

## Contact extraction and force sign convention

`contact_extraction.py` reads MuJoCo's own contact solution — no separate
plant-side contact model exists. Per step it scans `data.contact`, keeps only
pad–surface pairs (unrelated contacts are ignored; tested with a debris body),
and calls `mj_contactForce`.

Transformation: `contact.frame` stores the contact frame as **rows** (row 0 =
normal pointing from `geom1` into `geom2`); `mj_contactForce` returns
`[f_n, f_t1, f_t2, torques]` in that frame, acting on `geom2`. With
`R = frame.reshape(3,3)`:

```
f_world_on_geom2 = Rᵀ f_contact[:3]
f_on_pad = +f_world_on_geom2  if pad is geom2   (the usual case here)
           −f_world_on_geom2  if pad is geom1
```

so `f_task` is **always the force exerted on the robot end-effector** in the
world frame. `F_n = nᵀ f_task`, `F_t = tᵀ f_task`. Penetration is defined from
contact distance: `penetration = max(0, −dist)`.

**Experimentally verified** (tests + full runs): during contact `F_n > 0`
(upward on the EE), and kinetic friction opposes the sliding velocity — in the
default run, sliding at +0.05 m/s gives `F_t ≈ −1.45 N ≈ −μ F_n`. The
friction-sign unit test steps the dynamics briefly rather than using a single
static `mj_forward`, because the soft-friction constraint reference can
transiently point along the motion in a bare static solve.

Contact-point velocity uses `mj_jac` at the actual contact position
(`v_c = J_c q̇`); the surface is static, so task-contact power is
`P_task = f_taskᵀ v_c` (negative during steady sliding — friction dissipates).

**Timing**: the loop calls `mj_forward` once after reset, then per step:
extract contact → compute control → clear `qfrc_applied` and apply the human
force → log → `mj_step`. Contact forces read at the top of step *k* were
solved during the previous dynamics evaluation, i.e. the controller and log
use the contact force with a documented **one-step (1 ms) delay**; no staler
reads are possible.

## Controller phases

`controller.py`, a three-phase state machine:

1. **APPROACH** — Cartesian PD holds the initial tangential position and
   descends at 0.15 m/s; transitions on contact detection or gap < 2 mm.
2. **RAMP** — desired normal force ramps 0 → F_d (cosine, 1 s) and is
   regulated by PI on the *measured* MuJoCo force with `e_F = F_d − F_n`,
   `F_push = F_d + k_F e_F + k_I ∫e_F dt ≥ 0`, `F_z^cmd = −F_push − d_F v_z`
   (too little measured force ⇒ push down harder; sign verified in tests).
   Integral clamping + conditional integration provide anti-windup. The
   tangential position is held. After ramp + 0.5 s settle → SLIDE.
3. **SLIDE** — same force regulation plus tangential tracking of
   `x_d(t) = x_start + v_d (t − t_0)` with PD.

Torques: `τ_nom = τ_bias + J_pᵀ F_cmd − D_q q̇`, where `τ_bias =
data.qfrc_bias` (gravity + Coriolis; it does **not** include the model's
passive joint damping, so nothing is double-counted — the MJCF damping
(0.1 N·m·s/rad, integrated implicitly) and the small controller `D_q`
(0.5 N·m·s/rad) are both explicit and intentionally small). Torque magnitude
(±60 N·m) and rate (5000 N·m/s) limits are applied last.

Defaults: `F_d = 5 N`, `v_d = 0.05 m/s`, μ = 0.3, dt = 1 ms (`config.py`).

## Task contact vs. human interaction

Two strictly separate ground-truth channels:

* **Task contact** `f_task`: MuJoCo's pad–surface contact wrench (above).
* **Human interaction** `f_h(t)`: a smooth cosine-edged trapezoid force
  (configurable magnitude/direction/start/rise/hold/fall) applied directly at
  the EE site via `mj_applyFT` into `qfrc_applied` — no human collision object
  exists. The exact applied value is logged.

`f_measured = f_task + f_h` is logged as a *synthetic* combined channel, but
both ground truths are always retained separately. Later, an explicit or
learned contact model will predict `f_task` and the residual of `f_measured`
will estimate `f_h`; MuJoCo stays the simulation ground truth.

**Power convention** (v_ee = EE velocity): `P_{h→r} = f_hᵀ v_ee` is the power
the human injects into the robot; `P_{r→h} = −P_{h→r}`. Pushing along the
sliding direction (+x, default profile) during sliding gives
`P_{h→r} ≈ 3 N × 0.05 m/s = +0.15 W` (verified).

## Logging

`log.npz` / `log.csv` contain per step: time, phase, `qpos/qvel`, commanded
and applied torques, `τ_bias`, controller damping, `τ_C`, `τ_U`, EE
position/velocity, desired tangential position/velocity, desired + measured
normal force, PI internals (`F_push`, `e_F`, integral), contact flag/count/
position/distance/penetration/normal/tangent, `F_n`, `F_t`, full `f_task`,
`f_h`, `f_measured`, `P_task`, `P_{h→r}`, `P_{r→h}`, controller power
(`τᵀq̇`), passive-damping power, kinetic + potential energy, and a
saturation flag. `meta_*` entries record F_d, v_d, timestep, and the
geometry used by the animation.

## Selective-passivation layer

`contact_model.py` + `energy_ledgers.py` + `passivity_qp.py` +
`experiments.py`, enabled by setting `SimulationConfig.passivation`
(a `PassivationConfig`). The simulation loop, extraction and base logging
are unchanged; per step the layer wraps the nominal controller:

1. **Predict** the task contact: `oracle` (`F̂_T = F_T`) or `friction`
   (`F̂_{T,n} = F_{T,n}`, `F̂_{T,t} = −μ̂ F_{T,n} tanh(v_t/v_s)`,
   `μ̂ ∈ {0.20…0.35}` vs. true `μ = 0.30`). The predictor only sees the task
   channel and `v_ee` — it can never explain away the human force.
2. **Decompose**: `F_meas = F_T + F_H`, residual `F_R = F_meas − F̂_T`,
   mismatch `F_Δ = F_R − F_H`. Identities (tested): oracle ⇒ `F_R = F_H`,
   `F_Δ = 0`; no human ⇒ oracle `F_R = 0`, imperfect `F_R = F_Δ`.
3. **Port powers** use the *physical* EE velocity, never the tracking
   error: `p_H = F_H·v_ee`, `p_R = F_R·v_ee = p_H + p_Δ` (p > 0 = into the
   robot). The predicted nominal task power `F̂_T·v_ee` is deliberately
   excluded — this is selective **port** passivation, not global passivity
   of the active robot.
4. **Ledgers** (nested constraints, not additive tanks): residual
   `E_R ∈ [0.02, 0.50] J` (init 0.30) and human `E_H ∈ [0.005, 0.08] J`
   (init 0.05); charge is capped at `E_max`, discharge is *never* clamped —
   raw below-floor values stay logged. A conventional whole-port ledger
   `E_W` (on `F_meas·v_ee`) is observed everywhere and constrained in C1.
5. **One-step affine prediction** `v_tn⁺(u) = A u + b` from
   `τ(u) = τ₀ + J_eeᵀ B_tn u`, `τ₀ = qfrc_bias − D_q q̇`, with the measured
   force sample held over the 1 ms step (`J̇q̇` ignored; validated against
   explicit dynamics *and* real MuJoCo transitions; per-step prediction
   error logged, median ≈ 1e-4 m/s in closed loop).
6. **QP** (OSQP, warm-started, 2 vars × 7 rows): minimize
   `½(u−u_nom)ᵀ diag(1,20) (u−u_nom) + 1e-6‖u‖²` s.t. hard one-step ledger
   floors (`ε = 1e-4 J`), human instantaneous power `−p_H⁺ ≤ 0.10 W`,
   torque `|τ| ≤ 60 N·m` and torque rate `|Δτ| ≤ 5 N·m/step` — all
   *inside* the QP (no post-hoc clipping of QP solutions). One documented
   addition beyond the formulation's QP keeps the hard set feasible:
   CBF-style smoothing rows `−p⁺ ≤ k_cbf (E − E_min)` (`k_cbf = 50 s⁻¹`),
   which make depletion approach the floor exponentially so the torque-rate
   limit is never overwhelmed. These are a *tightening*, so the certificate
   is preserved — and they are **not optional**: with `k_cbf = 0` scenario B
   takes 788 infeasible steps and drives `E_R` to 0.0194 below its 0.02
   floor, and C/D/E violate the human bound. The one-step rows alone are not
   realizable at 1 kHz under the 5 N·m/step torque-rate limit.
   A `δ_p` feasibility tolerance on ledger rows also exists but now
   **defaults to 0**: it was the only term that weakened the certificate
   (permitting `δ_p · t` of extraction beyond the bound), and measurement
   showed it buys nothing — at `δ_p = 0` every scenario is feasible, the
   floors hold strictly, and `W_H` = 0.0449 J against its 0.045 J bound
   instead of overshooting to 0.045013 J.
   Certified ledger updates use the SAME held force sample with the actual
   next velocity: `p_k = F_kᵀ v_{ee,k+1}`. On infeasibility: full state +
   margins logged, bounded emergency damping applied, event counted.
7. **Anti-windup**: the force-PI integrator is frozen on steps where the
   filter modified the normal command.

Controller modes: `C0_nominal` (ledgers observed only),
`C1_whole_port_scalar` (single whole-port ledger, `u = γ u_nom` — depletes
on ordinary task friction), `C2_residual_qp` (residual constraint only),
`C3_dual_ledger_qp` (**proposed**: residual + human + power), and
`C4_dual_ledger_scalar` (C3's constraints, γ-only — task-preservation
ablation). Scalar modes solve the consistent 1-D problem with the same
one-step prediction and constraint rows.

Scenarios (`experiments.py`, deterministic — fixed keyframe, scripted
forces, no RNG): A exact/no human, B mismatch (`μ̂ = 0.20`, 17 s), C oracle
+ blocking human (−3 N·t, 6.5–8.5 s, cosine ramps), D mismatch + blocking,
E oblique human (3/√2·(−t−n), C3 vs C4), F helping-then-blocking
(charging → cap → discharge), plus a `μ̂` sweep. Artifacts: per-run
`log.npz/csv` + 9-panel `run.png`, per-scenario `comparison.png`,
`summary.{json,csv}`, `acceptance.json`.

**Scenario sizing.** The residual tank absorbs `E_R(0) − E_R,min = 0.28 J`
at rate `Δμ · F_N · v_t`, so the sliding *distance* needed to reach the floor
is `0.28 / (Δμ · F_N)` — independent of `v_slide`, since rate and travel both
scale with `v_t`. At `Δμ = 0.10` that is 0.56 m against ≈0.60 m of runway
before the two 0.6 m links reach their limit: the floor is reached at 15.55 s
and contact degrades shortly after. B ran 20 s, so contact was active only
80.4 % of the sliding phase and every tracking RMSE in the scenario was really
a workspace measurement. B is now 17 s (96.8 % contact, still 1.5 s of the
residual constraint holding the floor).

Enlarging `Δμ` to buy margin does **not** work, and the reason is worth
recording: depleting the tank faster also demands a larger correction to hold
the floor. At `μ̂ = 0.18` the run takes 86 infeasible steps; at `μ̂ = 0.15`,
1227, and `E_R` is driven to −0.039 J through the uncertified emergency
fallback. Past `|p_Δ| ≈ 0.035 W` the correction is simply not reachable under
the 5 N·m/step torque-rate limit while contact is maintained — the hard
certificate has a finite model-error operating envelope, visible in the sweep
at `μ̂ = 0.15`.

`compute_metrics` reports `contact_fraction_slide`, and `check_acceptance`
fails any scenario whose C0 run drops below 95 %.

## Known limitations

* Contact is deliberately soft (≈1.3 mm steady penetration at 5 N) to keep
  the 1 kHz delayed force loop free of micro-bouncing; a stiffer surface
  would need force-measurement filtering or a faster/impedance inner loop.
* The controller uses the contact force with a one-step (1 ms) delay.
* The tangential friction model is MuJoCo's regularized cone, not exact
  Coulomb friction; near zero sliding speed the friction force is
  velocity-dependent.
* Single contact geometry (sphere on plane); multi-contact summation is
  implemented but untested beyond one contact point.
* No sensor noise; both channels are perfect ground truth by design.
* **Finite reach bounds every run.** Two 0.6 m links give ≈0.6 m of sliding
  runway from the keyframe before the pad leaves the surface; at
  `v_slide = 0.05 m/s` that is ≈12 s of sliding. Any scenario longer than
  that measures the workspace limit rather than the controller. The
  `contact_fraction_slide` metric and its acceptance criterion exist to catch
  this, but the underlying constraint is structural — a longer experiment
  needs a longer arm, a slower slide, or a repositioned keyframe.
