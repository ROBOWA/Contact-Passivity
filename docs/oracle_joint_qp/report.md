# KUKA oracle joint-torque QP — PARTIAL

The unchanged nominal baseline and shadow prediction are validated. The main active
fixture completes and limits cumulative NET human output, but strict task preservation
and held-out active feasibility fail. This is a partial research result, not a completed
task-preserving passivity controller. No robot hardware was accessed.

## Reproduction and provenance

Starting checkout `dbff1c1` / `main`, clean. Work branch
`feature/kuka-oracle-joint-qp`. Source revision for these runs: `43a6661c7d4a916311d732ce31f8abf21618c4de`;
source SHA256 `023460012a05710dffd75a3e146a66659b94d67f4ad64d1cf94385f69bccba41`. Artifact/reporting commits may be later;
the provenance gate verifies that all numerical/control files still match this
recorded Git tree. Seed 0 (deterministic fixtures); validation scan seed 7341.
Hardware `{'platform': 'macOS-26.2-arm64-arm-64bit-Mach-O', 'machine': 'arm64', 'cpu': 'Apple M4'}`; software `{'mujoco': '3.15.0', 'numpy': '2.5.4', 'osqp': '1.1.3', 'scipy': '1.18.1', 'python': '3.13.5'}`.
XML hashes and unchanged nominal gains/trajectory are in `baseline_final.txt`
and full `results/oracle/final/*/baseline_log.npz` metadata.

```
.venv/bin/python -m pytest -q
.venv/bin/python -m kuka_table_sim.scenarios nominal --log results/oracle/baseline_final.npz
.venv/bin/python -m kuka_table_sim.oracle_validation --out results/oracle/validation_final
.venv/bin/python -m kuka_table_sim.oracle_experiments matrix --out results/oracle/final
.venv/bin/python -m kuka_table_sim.oracle_report --root results/oracle/final --validation results/oracle/validation_final --out docs/oracle_joint_qp
```

The matrix deliberately returns exit 2 when held-out/optional active trials fail;
all logs are still saved. A single failed case also exits 2. Unsupported inputs
fail closed. `results/` holds full logs and is ignored; `raw/` contains compact
full-precision account/work data for nine representative cases, force/power physics samples around interaction,
strided task recovery traces preserving global force extrema and contact transitions,
rejected candidates and failure snapshots. `compaction_json` declares retained
samples; omitted human powers are zero. No
float32 conversion is applied to work/accounts. Regenerate figures with the report
command after reproducing logs; do not treat a missing test as passing.

## Timing, ports and accounts

Control dt=0.001 s, physics h=0.00025 s, four split physics steps per held COMPLETE
torque command. Nominal bias/gravity is computed once by the existing controller.
`ArmIndex` checks gear=1, fixed gain=1, no bias, joint transmission; the oracle
additionally excludes actuator dynamics, extra force clamps, plugins, RK4, disabled
actuation and global dynamics callbacks. Every rollout and executed physics sample
checks actual `qfrc_actuator` against the candidate. No downstream command clipping
is hidden. Rate/trust bounds are simulation choices, not hardware specifications:
rho=(80,80,60,60,40,20,20) N m; rate=rho/dt N m/s. Torque limits are unchanged.

Known human point: link5 local (0,.04,.12) m; second point link3 local
(.02,.04,.16) m. The main force is −x, 5 N, cosine ramp .2 s / hold 1 s /
ramp .2 s, starting 9.5 s. Unfiltered point velocity is +x in this interval;
the measured negative power confirms resistance. Assisting case reverses force.
Cartesian force is sampled at the control boundary, then held WORLD-fixed while
point/J are recomputed at every physics sample. The predictor receives only this
current force. The varying-force check samples the plant at each physics step
while the predictor continues holding the current value; it runs in shadow.

Human power uses actual link point J qdot; applied input uses mj_applyFT after
restoring base qfrc_applied every substep. Force is applied exactly once. Task
work restricts contact to tool_tip/tabletop, with world-frame force ON the tool,
material-point velocity including rotation, and any contact moment. The baseline
meter was extracted without changing its arithmetic and is shared with rollouts.
Independent tests use constraint power, a force+moment at body COM, and a condim=6
fixture to check the sign, frame and moment term.

Sampling: step1 evaluates kinematics at x_k; human point/J/velocity are saved
before integration. step2 solves contact at x_k and advances q/qdot. The task
meter uses still-current x_k cvel/contact forces; it never pairs solved forces
with the new qdot. Work is signed left rectangle P_k*h. Tests independently
check these stage semantics, generalized constraint power and timestep halving.

Full `mjSTATE_INTEGRATION` snapshots include time, q/qdot, act/history, warmstart,
ctrl, external forces, equality flags, mocap, user and plugin fields exposed by
installed MuJoCo. Each candidate restores the SAME snapshot in a separate MjData,
overrides the torque and human force, and advances exactly one interval. The
shared model is immutable during prediction. No integrator/reference/FSM/RNG/
recorder callbacks are called. Replay and candidate-order invariance are tested.

W_T_nominal=W_T_actual; W_R=W_H EXACTLY. Both accounts update only from actual
executed signed work, with explicit E_init and E_min=0. No recharge except
positive work, no account clipping or per-cycle resets. Off/shadow violations
are retained. Gross outgoing energy is separately logged and is NOT bounded by
the net-work constraint. Control-boundary minima and observed substep minima
are separate; negative substep margins can occur even when endpoints pass.
This is not a continuous-time passivity certificate.

## Gradients, margin and QP

Seven-variable delta=tau−ORIGINAL tau_nom, full joint space. Central double
precision stencil uses 15 rollouts with eta=.02 N m; .01/.02/.04 checks identify
noise. Stencils crossing actuator bounds are rejected explicitly. Validation
uses 15 snapshots and 1155 single/mixed-direction commands over the
declared trust box. Calibration and held-out trajectories are separate.
Normalized-to-5-N max one-sided affine overprediction: calibration
1.83695e-05 J, held-out 2.16785e-05 J.
Margin epsilon_linearization=2.5e-5 J * |f|/5 N; deterministic held-force
epsilon_prediction=1e-11 J. These are empirical resolutions/envelopes on the
logged validation set, not uniform certified bounds. At accepted candidates a
fresh nonlinear rollout checks E+W_hat−epsilon_prediction >= E_min, so the
actual accepted point does not rely on its affine remainder bound. No-human
work/margin is exactly zero only under the guaranteed held-zero-input fixture.

Cost H=diag(1/torque_scale²)+S_nom^T W_y S_nom. Scale=(20,20,10,10,5,2,2)
N m; task outputs are average normal force, terminal linear/angular velocity,
normalized by (1 N,.01 m/s x3,.1 rad/s x3). Main weight=1; weight=0 and 100
are reported as explicit ablations. Nominal task sensitivity and H freeze
during retries. Objective whitening preserves the optimum; exactly equal
energy rows consolidate to the tighter budget, then both rows are checked in
original units. OSQP eps_abs/eps_rel=1e-9, max_iter=20000, polishing on. A small
bound-active KKT recovery handles solver nonconvergence without changing bounds
or using slack; an independent dense-H analytic optimum test exercises it.

Torque/rate/trust and affine energy residuals are checked in N m/J, regardless
of solver status. Fresh nonlinear check follows every solved candidate. Up to
three relinearizations shrink the trust region and shift the center from the
original snapshot; the objective always remains anchored to tau_nom. Nominal
zero correction is skipped only if it already satisfies bounds and the same
energy reserve. Rejecting a candidate or an empty box never executes it.
An impossible affine halfspace/box intersection stops the trial with state and
diagnostics; it is NOT a proof of global nonlinear infeasibility or a hardware
fallback. Integrators use their existing clamps and freeze after filter
intervention, extending the existing previous-saturation condition. No inverse
Cartesian interpretation of general joint correction is invented.

## Quantitative comparison

Every row has the same nominal initial state, plant, trajectory, and controller.
Matched cases differ only by explicitly logged mode/budget/task cost/force fixture.
Force max is relative to the ORIGINAL 10 N target; lowering f_d during SEARCH
does not hide contact loss. Whole steady/contact/recovery window starts 8.71 s.

| case | trial | completed s | net outgoing J | min endpoint margin J | max 10 N error N | contact absent s | along max mm | intervention ticks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no-human-off | PASS | 18.000 | -0.000000000 | 1 | 0.050 | 0.0000 | 3.899 | 0 |
| no-human-shadow | PASS | 18.000 | -0.000000000 | 1 | 0.050 | 0.0000 | 3.899 | 0 |
| no-human-active | PASS | 18.000 | -0.000000000 | 1 | 0.050 | 0.0000 | 3.899 | 0 |
| ample-off | PASS | 18.000 | 0.241623266 | 0.758 | 0.397 | 0.0000 | 3.929 | 0 |
| ample-active | PASS | 18.000 | 0.241623266 | 0.758 | 0.397 | 0.0000 | 3.929 | 0 |
| tight-off | PASS | 18.000 | 0.241623266 | -0.122 | 0.397 | 0.0000 | 3.929 | 0 |
| tight-shadow | PASS | 18.000 | 0.241623266 | -0.122 | 0.397 | 0.0000 | 3.929 | 0 |
| tight-active | PASS | 18.000 | 0.119998983 | 1.86e-07 | 53.104 | 1.1498 | 5.976 | 334 |
| assist | PASS | 18.000 | -0.242016041 | 0.12 | 0.332 | 0.0000 | 3.967 | 0 |
| stress | PASS | 18.000 | 0.000999680 | 2.6e-07 | 18.068 | 1.4685 | 4.478 | 575 |
| elbow | PASS | 18.000 | 0.141353740 | 0.859 | 1.130 | 0.0000 | 3.916 | 0 |
| elbow-tight | FAIL | 10.884 | 0.119999555 | 4.45e-07 | 188.191 | 0.2198 | 2.604 | 106 |
| varying | PASS | 18.000 | 0.241578691 | -0.122 | 0.398 | 0.0000 | 3.929 | 0 |
| held-out-off | PASS | 18.000 | 0.194653917 | -0.0747 | 0.353 | 0.0000 | 3.927 | 0 |
| held-out-active | FAIL | 10.753 | 0.119977500 | 2.25e-05 | 0.353 | 0.0000 | 2.729 | 1 |
| task-cost | FAIL | 10.814 | 0.119975714 | 1.17e-05 | 10.000 | 0.4965 | 4.669 | 225 |
| lower-budget | PASS | 18.000 | 0.099999992 | 8.16e-09 | 24.599 | 0.9640 | 5.639 | 358 |
| torque-only | FAIL | 10.443 | 0.119975000 | 2.38e-05 | 43.485 | 0.1303 | 24.209 | 83 |

Primary net-output reduction: 50.336%.
Primary gross outgoing=0.124647681 J (separate from net), minimum
observed substep margin=-1.11334e-06 J. Endpoint
largest violation=0 J. Both accounts have
identical work/gradients by design. Full per-scenario force mean/max, along/cross/
orientation, power/speed, signed task/human/residual work, failure/retry categories,
constraint activity, integrator bounds and final-second recovery metrics are
in `metrics.json`; event histories preserve accepted and rejected candidates.

The active energy demonstration has a material task cost: force transients and
temporary contact interruption. It must not be described as constant-force
sliding during intervention. Recovery is exercised through the existing FSM.
Held-out and elbow tight-budget failures expose a local one-tick feasibility
limit; budget feasibility is not monotonic and no universal boundary is claimed.
The report intentionally remains PARTIAL when these/task-quality gates fail.

## Runtime

Main full tick wall seconds (median/p95/p99/max): `{'median': 0.0006400830461643636, 'p95': 0.0006837919354438782, 'p99': 0.0030134579865261912, 'max': 0.009496500017121434}`.
Gradient-only active ticks: `{'median': 0.0020618330454453826, 'p95': 0.00308239979785867, 'p99': 0.00317109814612195, 'max': 0.00602012500166893}`;
QP solve active ticks: `{'median': 0.00023647950729355216, 'p95': 0.0002784356998745351, 'p99': 0.00035161191248334945, 'max': 0.0014405000256374478}`.
All snapshot/rollout/gradient/QP/nonlinear/retry quantiles are in `metrics.json`.
Total 11.540 wall seconds for 18.000 simulation
seconds (factor 1.560). Benchmarks have no visualization or
synchronous per-step disk output; arrays/events are buffered, plots/files written
after timing. Memory allocation and Python logging remain in the measured path.
Median speed or a 1 ms simulated interval does not establish 1 kHz wall-clock
performance; tail latency and intervening-gradient ticks exceed the deadline.

## Audit, final gates and research decision

Three evidence-backed rounds and repairs are in `reviews.md`; exact source/test
and run provenance are in `acceptance.json`. Final figures are regenerated from
the matching final logs, including the failed trials. Numerical/control sources
must match the recorded Git revision and SHA; reporting-only repairs are tracked
separately and do not change the physics logs. Final manifest:

| gate | status |
|---|---|
| final_source_provenance | PASS |
| physical_QP_and_regression_suite | PASS |
| unchanged_25s_baseline | PASS |
| no_human_transparency | PASS |
| ample_budget_transparency | PASS |
| human_power_and_oracle_identities | PASS |
| held_force_execution_prediction | PASS |
| gradients_and_empirical_envelope | PASS |
| main_active_energy | PASS |
| logged_QP_physical_residuals | PASS |
| main_active_task_quality | FAIL |
| release_recovery | PASS |
| stress_outcome_is_explicit | PASS |
| held_out_active | FAIL |
| varying_force_error_measured | PASS |

Next decision: whether the research scope should add a viability/braking-reserve
policy and stronger task constraints, rather than expecting this greedy one-tick
hard-energy QP to preserve contact under depleted budgets. Those would be explicit
extensions; this implementation does not silently add them or enlarge budgets.

References: [MuJoCo simulation/state](https://mujoco.readthedocs.io/en/stable/programming/simulation.html),
[MuJoCo API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html),
[OSQP solver](https://osqp.org/docs/solver/).
