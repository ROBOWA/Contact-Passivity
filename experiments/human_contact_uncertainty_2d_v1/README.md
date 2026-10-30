# human_contact_uncertainty_2d_v1

This is an isolated planar proof of concept. It tests whether impedance-informed human-force prediction, risk-sensitive
selection and the existing Ec + BH energy accounts work together. The setting has:
* an uncertain **human** contact location and force on one known robot patch;
* a deterministic passive **task** contact (sustained wiping against a compliant wall);
* the Ec + BH accounts, with the `BH ≥ S` passive backup reserve.

**Research simulation only.** This is not authorisation for, or evidence about, hardware execution or deployment.

* Mathematics, assumptions and guarantee tiers: [formulation.md](formulation.md) (PDF: `formulation.pdf`)
* Which source says what, and what we changed: [literature_mapping.md](literature_mapping.md), [references.bib](references.bib)
* Results, failures and remaining gaps (Chinese executive summary first): [report.md](report.md) (PDF: `report.pdf`)
* Routine decisions made without asking: [decision_log.md](decision_log.md)
* Provenance (git, environment, files created): [manifest.json](manifest.json)

## Results at a glance (details, failures and tiers in `report.md`)

All numbers come from held-out seeds and frozen code. Tier C means empirical simulation evidence, not a proof.

* **Ec + BH tick-work filter.**
  * Unprotected C0 violated the true accounts in 100% of human scenarios, with 1.1-2.1 J of net work on the human.
  * The filtered controllers stayed at ≤ 0.12 J, with 0 Ec violations in 7050 episodes.
  * An instantaneous-power filter leaks (−80 mJ). The truth-access reference shows µJ-level sampled-data deficits.
* **Location-correlated past-work error belief** (C2-C4). It removed the deterministic ledger's violations:
  * in distribution, 7/500 → 0/500;
  * in the moment-arm × location-uncertainty sweep, C1 had 20/270, force-only uncertainty 12/270, and C3 0/270.
* **Structured (P1-style) GP.**
  * Offline it predicts better: about 2× lower force RMSE, calibrated work, and the correct response to candidate motion.
  * In closed loop it does **not** beat the motion-blind Gauss–Markov predictor. It is over-confident in forked-truth
    calibration and regresses on held-out oscillating intents.
* **Risk sensitivity (β > 0).** Selections become 1-2% stiffer and slower, consistent with P3/P4. This is not
  compliance, and the effect is negligible.
* **Failure.** A persistent sensor bias outside the prior gives 75-98% `h_H` violations. A post-hoc calibrated
  interval bound removes them on fresh seeds, at a task cost.

Main artefacts:
* `report.md` / `report.pdf`;
* `formulation.md` / `formulation.pdf`;
* `outputs/figures/`;
* `outputs/animation/block_release_seed100_C0_C1_C2_C3.mp4` (C0 is marked UNPROTECTED);
* `outputs/report_tables.md`.

Large regenerable artefacts are listed in `.gitignore` (GP data, fork records, traces). They are kept on disk and
can be regenerated with the full-suite commands.

## Folder isolation

Everything this task created is inside this folder. No shared module, root config, lockfile, earlier report or other
agent's worktree was edited. The existing Ec + BH code (worktree `results/worktrees/human-port-ecbh-20261029`)
was only **read**. Its account definitions were re-implemented here for a different plant, and the provenance is
recorded in `formulation.md` §4. Outputs live in `outputs/`. The repository `.gitignore` pattern `results/` would
hide a folder named `results`, which is why the name is not used.

## Setup

Use the repository's existing virtual environment read-only (nothing was installed into it):

```
PY=/Users/carl/code/contactPassivity/.venv/bin/python     # Python 3.13.5, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2
cd /Users/carl/code/contactPassivity/experiments/human_contact_uncertainty_2d_v1
```

Requirements: numpy, scipy, matplotlib, pillow, pytest. For MP4 output, an existing ffmpeg binary is called by absolute
path (`/opt/anaconda3/envs/517_drake/bin/ffmpeg`); without it only the GIF is written. PDFs use headless Google Chrome.

## Smallest reproduction (≈ 1 min)

```
$PY -m pytest tests -q                                   # 14 mathematical checks (~15 s)
$PY scripts/gen_gp_data.py --workers 8                   # synthetic GP data splits (~15 s)
$PY scripts/fit_gp.py                                    # frozen predictors (~6 s)
$PY scripts/run_batch.py --tag tiny --scenarios block_release --controllers C0 C1 C3 --seeds 100 103 --workers 3
$PY scripts/analyze.py tiny
```

## Full suite (≈ 1.5-2 h on a 10-core Apple M4)

```
$PY scripts/gen_gp_data.py && $PY scripts/fit_gp.py && $PY scripts/eval_gp_offline.py && $PY scripts/summarize_offline.py
$PY scripts/freeze.py main_v1
bash scripts/run_all.sh          # main matrix, ablations, sweeps, calibration forks, refinement (see file)
$PY scripts/posthoc_robust_patch.py && $PY scripts/freeze.py posthoc_v1   # post-hoc bias bound (already applied)
$PY scripts/run_batch.py --tag posthoc_robust --seeds 300 350 --scenarios sensor_bias block_release assist_reverse moment_arm --controllers C2 C3 C2_robust C3_robust
$PY scripts/timing_single.py     # single-process timing
$PY scripts/analyze.py main ...  # summaries
$PY scripts/fig_offline.py && $PY scripts/fig_closed_loop.py --tag main && $PY scripts/fig_extra.py
$PY scripts/make_animation.py --tag main --scenario block_release --seed 100 --controllers C0 C1 C3 C4
$PY scripts/build_docs.py --tables && $PY scripts/assemble_report.py && $PY scripts/build_docs.py --pdf --manifest
```

Runtime expectations, per episode of 9 simulated seconds with one process:
* C0: about 0.6 s;
* C1: about 3 s;
* C2: about 4.5 s;
* C3/C4: about 6 s.

During human contact the planner takes about 30 ms per 20 ms control period at p95. That is **not real time**, and
no real-time claim is made.

## Layout

| path | content |
|---|---|
| `hcu2d/truth.py` | ground-truth plant, impedance human (hidden), RK4 |
| `hcu2d/model.py` | known model, Jacobians, contact law, nominal policy and candidates |
| `hcu2d/servo.py` | sampled-data energy filter (tick-energy projection) and ledgers, including the BH-error belief |
| `hcu2d/gp.py`, `hcu2d/kernels.py` | structured latent-intention GP, naive GP, Gauss–Markov predictor |
| `hcu2d/belief.py` | Monte Carlo and first-order closed-loop belief rollouts |
| `hcu2d/planner.py` | C1-C4 finite-candidate planners; entropic risk; chance constraints |
| `hcu2d/auditor.py` | independent truth auditor (does not import the plant or model) |
| `hcu2d/episode.py`, `scenarios.py`, `metrics.py` | closed loop, paired hidden truth, metrics |
| `scripts/` | data generation, fitting, evaluation, batch runs, figures, animation, documentation |
| `tests/test_math.py` | mathematical checks |
| `outputs/` | data, frozen models, run metrics, figures, animation |
| `checkpoints/` | initial git/environment snapshot, progress and freeze records |
