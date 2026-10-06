# Torque-controlled Panda selective-passivation PoC

This backend extends the planar benchmark to a seven-joint Franka Panda. It
uses a direct-torque, no-hand MuJoCo Menagerie model, a fixed finite-area tool
pad with four soft contact elements, a compliant horizontal table, a
three-phase force/motion controller, spatial task and human ports, and a
seven-variable absolute-torque OSQP filter.

## Reproduce

Install the root requirements, then run from the repository root:

```bash
python -m pip install -r requirements.txt
pytest -q mujoco_panda/tests
python -m mujoco_panda.cli --scenario C_ee_oblique_blocking \
  --mode C3_dual_ledger_qp --output results_panda/single
python -m mujoco_panda.experiments --output results_panda
python -m mujoco_panda.smoothing --output results_panda/smoothing
python -m mujoco_panda.viewer \
  results_panda/D_forearm_wiping/C3_dual_ledger_qp/log.npz --location forearm
python -m mujoco_panda.video \
  results_panda/D_forearm_wiping/C3_dual_ledger_qp/log.npz \
  results_panda/videos/forearm_wiping_c3.mp4 --location forearm
```

On macOS, use `mjpython -m mujoco_panda.viewer ...` if MuJoCo reports that
its interactive window must run through `mjpython`.

The viewer is an exact real-time replay of the saved dynamics log. The red
arrow is the simulator-known human force at the configured physical site;
the green arrow marks the governed setpoint. Neither marker affects physics.

## What is modeled

The arm coordinates and actuators are explicitly resolved by names
`joint1`–`joint7` and `actuator1`–`actuator7`; code does not treat arbitrary
MuJoCo state entries as arm states. The hand-free Menagerie model avoids
uncontrolled finger dynamics. Direct motors have unit gain and zero bias,
with limits `[87, 87, 87, 87, 12, 12, 12]` N m.

The nominal command is complete: MuJoCo bias compensation, Cartesian
tangential motion, normal-force PI with ramp and anti-windup, fixed
orientation regulation, joint damping, and weak null-space posture torque.
The QP decides the complete seven-vector applied afterward; no hidden torque
is added downstream.
The QP correction objective combines joint-torque deviation with predicted
tool-twist deviation; normal motion is weighted five times more than
tangential or orientation motion so contact is preferentially retained when
the protected port constraints still leave that choice feasible.

Task work is `sum_i (f_i dot v_i + m_i dot omega_i)` at individual contact
points. Human forces at either the tool or `forearm_site` use that site's own
Jacobian and velocity. Perfect human wrench and location are simulator ground
truth supplied to this PoC controller; this is not a force-estimation result.

The friction predictor spans world x-y, uses only the task-contact normal
channel, and never fits task-plus-human force. Oracle, coefficient mismatch,
normal-force scale mismatch, and a controlled controller-side robot-inertia
scale error are configurable; the MuJoCo plant remains unchanged.

## Certificate scope and work audit

Positive external power injects energy into the robot. Residual and human
ledgers are nested constraints, not additive physical tanks. Charging is
capped and discharge is deliberately not clamped, so violations remain in
the log.

The full-resolution compressed NPZ contains every 1 ms physics sample; CSV
is a 100 Hz review view (including the final sample). The controller
certificate is discrete and conditional on the frozen
one-step arm-velocity prediction-error bound. `p_*_sample` is the held wrench
sample paired with the realized next velocity and is exactly what updates a
ledger. `p_task_physical` independently trapezoid-integrates per-contact
MuJoCo wrench/twist samples at physics resolution; the human site's physical
trapezoidal work is audited the same way. Their discrepancies are
reported rather than treating sampled ledger work as continuous contact
work. The calibration/evaluation split is saved in `calibration.json`; its
maximum is evidence only for tested configurations.

This demonstrates selective regulation of stated ports under a known human
wrench/location and an empirical prediction-error assumption. It does not
establish unconditional global passivity, universal prediction bounds,
force estimation, or complete human safety. A fallback is rate/torque-bounded
but is logged as uncertified. Solver candidates beyond `5e-6` row feasibility
tolerance are rejected; candidate and applied minimum margins are both logged.

## Contact acceptance convention

The former single sphere alternated raw impulses at the 1 ms solver scale
near its 1 mm activation margin. The final four-element pad and low boundary
impedance keep the nominal contact continuously active. "Contact maintained"
still uses residence in the compliant activation band plus causal filtered
normal force above 0.5 N so interaction cases remain well defined. The raw
constraint flag, number of active contact points, and raw normal force are
also logged. Normal-force RMSE uses the same 70 ms causal signal consumed by
the controller.

See `REPORT.md` for measured results and failures, `BASELINE.md` for the
planar audit, `SMOOTHING.md` for the contact-motion retuning, `ENVIRONMENT.md`
for recorded dependency versions, and
`models/UPSTREAM.md` plus the vendored `LICENSE` for model provenance.
