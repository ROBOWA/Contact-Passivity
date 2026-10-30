# Decision log — human_contact_uncertainty_2d_v1

Chronological log of routine research/implementation choices made without asking the user
(as instructed). Each entry: decision, reason, consequence. Times are UTC.

## 2026-10-30 — inspection

- **Repository root** confirmed: `/Users/carl/code/contactPassivity` (branch `main`, base commit
  `5e792cf7157449a9806c9c3aeb6e3b5464780b3a`). No `AGENTS.md` or `CLAUDE.md` exists anywhere in the
  repository (searched recursively, excluding `.venv`/`.git`).
- **Initial git status** (before this task wrote anything): modified `.gitignore`, `README.md`,
  `pyproject.toml`, `scripts/kuka_poc_figures.py`; untracked `docs/consolidation/`,
  `human_port_ecbh_workflow_20261029.json`, `scripts/verify_consolidation.py`,
  `taskref_ecb_workflow_20261029.json`. These belong to other work and are left untouched/unstaged.
  `checkpoints/initial_git_status.txt` was captured just after creating the empty folder, so it
  additionally lists `?? experiments/`.
- **Other agents' work**: two active worktrees exist under `results/worktrees/`
  (`human-port-ecbh-20261029`, `taskref-ecb-20261029`). The `human-port-ecbh` worktree HEAD moved
  from `b05a994` to `a127f83` during this task's inspection, i.e. another agent is active there.
  It was inspected **read-only**. A read-only `cd` into it caused the harness to treat it as the
  session's working directory; to avoid touching it, every file this task writes uses an absolute
  path inside `experiments/human_contact_uncertainty_2d_v1/`, and commands are run in subshells.
- **Existing Ec + BH code** (`human_port_ecbh_mpc_p1/core.py`, `filter.py`, `FORMULATION.md` in that
  worktree) is a gravity point mass with a held (state-independent) human force, a Hunt–Crossley-like
  contact `Fn = k δ + c δ [δ̇]+`, and accounts `Ėc = cT + η DR − Pa`, `ḂH = −pH` (net). It cannot be
  reused as a plant for this task (no rotation, no contact location, gravity). **Decision:** write a
  new planar rigid-body plant; reuse only the account definitions/sign conventions (provenance recorded
  in `formulation.md` §4 and `literature_mapping.md`). No shared module is imported or edited.
- **Folder naming:** the repository `.gitignore` contains the pattern `results/`, which matches a
  directory of that name at any depth. Outputs therefore go to `outputs/` (not `results/`).
- **Python environment:** the repository `.venv` (Python 3.13.5, numpy 2.5.4, scipy 1.18.1,
  matplotlib 3.11.2, pillow 12.3.0, pytest 9.1.1) is used read-only; nothing is installed into it.
  No new dependency is needed. An existing ffmpeg binary
  (`/opt/anaconda3/envs/517_drake/bin/ffmpeg`, 6.1.1) is invoked by absolute path only for MP4
  encoding; a GIF fallback is produced with pillow.
- **Literature:** local copies found at
  `/Users/carl/Documents/Contact passivity/Nadia_Papers_2026-10-26/0{1,2,3,4}_*` with SHA-256 matching
  their `SOURCE_NOTES.txt`. P1 is the author manuscript `TBME-01274-2018-R1` (12 pp.), not the
  publisher version; page/equation numbers refer to this manuscript.
- **Sign conventions verified in the sources** (details in `literature_mapping.md`):
  P1 Eq. (7)-(8) `u_int = M ë + D ė + K e + …`, `e = x_d − x`, `τ_int = −Jᵀ u_int` ⇒ `u_int` is the
  wrench the human applies to the robot, pointing towards the human's desired position. P2 Eq. (6)
  prints `−u_int = …`, inconsistent with its own Eq. (8)-(9); P2 Eq. (17) prints the posterior
  variance with `+` and an untransposed cross-covariance. We follow P1 Eqs. (7)-(8), (25)-(26), which
  are dimensionally and sign consistent, and we re-derive and test them independently.
- **Risk-sign mapping:** P3 Eq. (10)/(11) and P4 (unnumbered, p. 3) use
  `γ(θ) = −(2/θ) ln E[exp(−θJ/2)]`, θ<0 risk-averse. We use `ρ_β = (1/β) log E[exp(βJ)]` with
  `β = −θ/2` (β>0 risk-averse). P4 reports that its **risk-seeking** (θ>0) controller gives lower
  stiffness under force corrections and its risk-averse controller becomes stiffer.
- **Naming clash resolved:** the task text uses `alpha_H, alpha_E` both for the filter decay rates
  and for violation probabilities. In this folder, filter rates are `λ_H, λ_E` [1/s] and violation
  probabilities are `α_H, α_E`; `κ(α) = Φ⁻¹(1−α)`.
- **PDF rendering:** no pandoc/LaTeX available; Markdown is converted with a small in-folder converter
  and printed with headless Google Chrome. Math in documents is written in plain Unicode.

## 2026-10-30 — smoke-suite debugging (30 paired seeds 0-29; not used for final claims)

- **Sampled-data filter leak (fixed before smoke):** the instantaneous condition `uᵀq̇(t_k) ≤ b` leaked
  energy within the 1 ms ZOH tick (bang-bang chattering when nearly stalled against the human; true BH
  went negative). Replaced by the tick-work constraint `uᵀΔq_pred(u) ≤ h·b` (second-order Taylor, exact
  convex projection). The instantaneous version is kept as the ablation `Finst`.
- **"Backup trap" (found in smoke `bias_ood`, fixed):** after the planner selected the zero-input backup
  and the tool lifted off, re-approach used the full force set-point (plus a stale integral), predicting
  a ~0.4 m/s impact whose kinetic+elastic energy exceeded the remaining BH, so every active candidate
  stayed infeasible for the rest of the episode. Fix, shared by ALL controllers through the nominal policy:
  an approach-speed limiter (extra normal damping `D_brake=400 N s/m` only when the tip moves toward the
  wall faster than `v_app=3 cm/s`) and integrator leakage `I_leak=5 s⁻¹` out of contact. GP data were
  regenerated and predictors refitted with the unchanged procedure for pipeline consistency.
- Smoke observations kept as context, not claims: in-distribution scenarios showed no account violations for
  any filtered controller; `bias_ood` violations are driven by the force-sensor bias (0.8 N) lying far
  outside the calibrated prior (σ_bf = 0.1 N).
- **Backup drift (found in smoke2 `bias_ood`, fixed):** the approach limiter did not remove the trap. Under the
  passive backup the free body kept drifting and rotating (θ up to 1.8 rad, corners through the wall: the
  contact model's domain is left), after which every active candidate's predicted re-orientation energy
  exceeded the remaining BH. This is the "zero input is not a safe physical stop" caveat made concrete.
  Fix for C1-C4 alike: add a strictly dissipative **brake** candidate `u = −D_brk q̇`
  (`D_brk = diag(40, 40, 1.5)`), for which `P_a ≤ 0`, hence `ḣ_H ≥ 0` and `Ėc ≥ 0`. The passive `u=0`
  backup is kept (the §5.1 lemma is about it). A `domain_violation` metric (any body corner below the
  wall) is now reported.

## 2026-10-30 — held-out runs (frozen `checkpoints/freeze_main_v1.json`)

- All reported closed-loop numbers use seeds ≥ 100 and the frozen code/models (`scripts/freeze.py --check main_v1` at
  the start of `scripts/run_all.sh`); smoke seeds 0-29 are debugging only.
- Another agent's job (`results/worktrees/human-port-ecbh-20261029/scripts/human_port_ecbh/quality_loop02/demos.py`
  plus video encoding) ran concurrently on the same machine (load average up to ~108 on 10 cores). It was not touched.
  Planner wall-times from the parallel runs are therefore contention-affected; a separate single-process timing run is
  reported as the representative timing.

## 2026-10-30 — post-hoc robust past-work bound (NOT part of the frozen main comparison)

- Designed **after** observing the sensor-bias failure in the frozen main run (§5.5 of the report). Patch
  `scripts/posthoc_robust_patch.py` adds an optional interval bound to the planner's past-work error:
  `R = B_f‖Δp̂‖₁ + B_s Σ_k|G_k|h` (per contact event, summed), subtracted from the booked BH in the rollouts and from
  the servo margin mean, on top of the Gaussian margin (conservative double counting). Assumed calibration bounds:
  `B_f = 1.0 N` (covers the 0.8 N test bias), `B_s = 0.08 m` (covers the 6 cm test bias).
- Existing controllers are unchanged (the option defaults to off); tests re-run (14 passed); recorded as freeze
  `posthoc_v1`. Evaluated on FRESH seeds 300-349 only, together with C2/C3 on the same seeds for pairing.

## 2026-10-30 — closing

- Code-state audit: relative to freeze `main_v1`, only `hcu2d/planner.py`, `hcu2d/servo.py` (post-hoc option, off by
  default) and `hcu2d/mdpdf.py` (documentation converter: nested lists) changed. Re-running main-run episodes with the
  final code (block_release 100/C1, 101/C3; sensor_bias 100/C2; moment_arm 102/C4) reproduces the frozen metrics
  **bit-for-bit** (max difference 0.0).
- Counting: reported held-out closed-loop episodes = main 4000 + infeasible 150 + ablations/sweep/refinement 3040 +
  calibration 200 (8488 forks) + post-hoc 800; debugging smoke runs (1470) are excluded from all claims.
- Bulk regenerable artefacts (GP data, fork records, episode traces) are excluded from the commit by the folder
  `.gitignore` but kept on disk.
- Commit policy: the main worktree stays on `main` and is not touched (no checkout/reset/stash/staging of others'
  changes). A single local commit with only this folder is created with git plumbing (temporary index) on a new local
  branch `research/human-contact-uncertainty-2d-v1`; nothing is pushed or merged.
