#!/usr/bin/env bash
# Full experiment suite on HELD-OUT seeds (>= 100), after `scripts/freeze.py main_v1`.
# Smoke/debug seeds 0-29 are never used for reported claims.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/Users/carl/code/contactPassivity/.venv/bin/python}
W=${WORKERS:-9}
$PY scripts/freeze.py --check main_v1

# 1) Main comparison: 8 scenarios x C0-C4 x 100 paired seeds
$PY scripts/run_batch.py --tag main --workers $W --seeds 100 200 --trace-seeds 2 \
  --scenarios no_human block_release assist_reverse normal_oblique moment_arm ood_intent sensor_bias low_BH \
  --controllers C0 C1 C2 C3 C4
# infeasible initialisation (outside the backup set), reported separately
$PY scripts/run_batch.py --tag infeasible --workers $W --seeds 100 130 --trace-seeds 1 \
  --scenarios infeasible_init --controllers C0 C1 C2 C3 C4

# 2) Targeted ablations (50 paired seeds)
$PY scripts/run_batch.py --tag abl_time --workers $W --seeds 100 150 --trace-seeds 0 \
  --scenarios block_release assist_reverse ood_intent --controllers C3 C3_indep C3_mc
$PY scripts/run_batch.py --tag abl_beta --workers $W --seeds 100 150 --trace-seeds 0 \
  --scenarios block_release assist_reverse normal_oblique --controllers C4neg C3 C4 C4b5
$PY scripts/run_batch.py --tag abl_filter --workers $W --seeds 100 150 --trace-seeds 1 \
  --scenarios block_release --controllers C0 F Finst ORACLE
$PY scripts/run_batch.py --tag abl_residual --workers $W --seeds 100 150 --trace-seeds 0 \
  --scenarios block_release assist_reverse --controllers C3 C3_nores
for CR in exact conservative invalid; do
  case $CR in
    exact) O='{"contact":{"mu":0.3,"mu_true":0.3,"mu_credit":0.3}}';;
    conservative) O='{"contact":{"mu":0.3,"mu_true":0.35,"mu_credit":0.25}}';;
    invalid) O='{"contact":{"mu":0.3,"mu_true":0.25,"mu_credit":0.35}}';;
  esac
  $PY scripts/run_batch.py --tag abl_credit_$CR --workers $W --seeds 100 150 --trace-seeds 0 \
    --scenarios no_human block_release --controllers C1 C3 --override "$O"
done

# 3) Moment-arm x location-uncertainty sweep (identical nominal force, fixed s)
for S in -0.15 0.0 0.15; do for U in 0.01 0.03 0.06; do
  O="{\"sensors\":{\"sigma_bs\":$U,\"sigma_es\":$U},\"scenario_override\":{\"s\":$S}}"
  $PY scripts/run_batch.py --tag sweep_s${S}_u${U} --workers $W --seeds 100 130 --trace-seeds 0 \
    --scenarios moment_arm --controllers C1 C3 C3_forceonly --override "$O"
done; done

# 4) Belief calibration with forked truth (seeds 200-219)
$PY scripts/calibration_fork.py --tag calib --controllers C3 C2 --seeds 200 220 --workers $W

# 5) Physics-step refinement (servo stays 1 kHz)
for H in 0.001 0.0005 0.00025; do
  $PY scripts/run_batch.py --tag refine_h$H --workers $W --seeds 100 110 --trace-seeds 0 --h-phys $H \
    --scenarios block_release normal_oblique --controllers C0 C1 C3
done
