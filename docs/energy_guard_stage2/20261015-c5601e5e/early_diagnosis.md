# Offline early-intervention diagnosis (frozen before online reserve design)

Hindsight experiment only. Forked complete plant/controller/FSM/observer/integrator/clock/account snapshots at 10.180 s (main) and 10.648 s (historical). Same quintic alpha slowdown, 100 ms transition, target 0 while measured human force remains nonzero; recovery targets 1 when the current measured force is zero. Stage 1 endpoint QP and task weight 1 unchanged, same torque/rate/trust bounds, 0.12 J budgets, force scripts and spatial path. Only the onset shifts. No future-force sampling is part of an online trigger.

| case | completion | net output J | max error from 10 N | absence s | deceleration net cost J | reference progress m |
|---|---|---:|---:|---:|---:|---:|
| resistance_0ms | PASS | 0.119999951 | 53.104 | 0.93550 | 0.00013887334669921285 | 0.2000 |
| resistance_20ms | FAIL | 0.119975034 | 1.618 | 0.00000 | 0.004643432070669587 | 0.0355 |
| resistance_50ms | PASS | 0.119999985 | 51.663 | 0.00000 | 0.011337628447926228 | 0.2164 |
| resistance_100ms | PASS | 0.119999281 | 6.945 | 0.00000 | 0.022291981374817096 | 0.2147 |
| held-out_0ms | FAIL | 0.119977500 | 0.353 | 0.00000 | None | 0.0586 |
| held-out_20ms | FAIL | 0.119929723 | 0.685 | 0.00000 | 0.0042252027138160175 | 0.0585 |
| held-out_50ms | PASS | 0.119999991 | 14.746 | 0.00000 | 0.010720168650413524 | 0.2229 |
| held-out_100ms | PASS | 0.119999936 | 10.753 | 0.00000 | 0.021519556632183648 | 0.2209 |

Evidence supports a limited early-intervention benefit: 100 ms earlier removes main-case contact loss and reduces max force error from the frozen 53.104 N to 6.945 N. Historical 50/100 ms cases complete where 0/20 ms fail. However early action alone does not satisfy the 7–13 N band; TCP slowdown does not remove all forearm-port output and the endpoint filter still exhausts the budget. It is not a universal causal proof or a certified braking reserve.

Decision before implementing the online trigger: evaluate a held-current-force, full closed-loop backup which smoothly slows progress AND damps measured human-point motion. Estimate its worst physics-work prefix, require terminal settling/persistence and contact admissibility, and choose actions before reserve is exhausted. No policy may use the hindsight onsets above. Preserve unsuccessful variants.

Exact run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m kuka_table_sim.energy_diagnostic --out results/energy_guard_stage2/NEW_UNIQUE_ID/offline_timing` (directory must not exist). Original full evidence remains in `results/energy_guard_stage2/20261015-c5601e5e/offline_timing_01/`.
