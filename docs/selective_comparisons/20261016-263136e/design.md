# Selective KUKA comparisons: preregistered design

Frozen before physical evaluation, 2026-10-16. Starting revision:
263136e22d721e1e72ec7325c71647ce4167a6d4. All earlier worktrees and results
remain unchanged. This stage implements the comparisons missing from the prior
reference-paper mapping; it does not retroactively relabel the old H=R results.

Five controllers share the unchanged 10 N force/motion nominal controller,
plant, initial state, gains, trajectory and joint torque limits. K0 is nominal;
K1 protects whole monitored power pW=pH+pT; K2 protects pR=pH+pDelta;
K3 protects H and R and bounds human outgoing peak power, governor OFF;
K4 uses exactly the K3 QP with only the reference governor ON. All modes log all
three accounts. pDelta=pT-pT_hat, with pT_hat evaluated in an independent
nominal contact model at the SAME actual state and material-point velocities.
The explicit tip/table pair's tangential friction is 1 in the actual plant;
nominal friction is either 1 (exact) or .20 (mismatch). Contact moments and
world-frame material-point work use the existing validated task meter.

Initial (H,R,W)=(.12,.30,.30) J, upper caps=(.15,.30,.30) J, floors=0.
Updates at each 0.25 ms sample discard only charging above the cap; negative
accounts are retained in monitored/unprotected modes. No lower clipping.
Separate gross outgoing work is reported. Human outgoing peak bound=.15 W;
CBF rate=5/s, safe-energy offset=25 microjoules. These are new KUKA design
parameters, not the original 2-DOF paper's numerical parameters.

The causal human-force class is ||F[k+1]-F[k]|| <= 60 N/s * 1 ms. Force is
measured at the control boundary and held world-fixed for four physics steps.
Every protected account has four energy and four CBF rows; K3/K4 add four
human peak rows and terminal human CBF/peak rows robust to that next-boundary
force increment. This is a conditional sampled-data check, not a continuous
time passivity proof or a viability certificate. Rollouts use the true plant
dynamics as an explicitly disclosed simulation oracle, and the current force
only. Nominal task-model mismatch is tested; unknown plant dynamics are not.

Seven joint torques replace the original paper's two Cartesian command
variables. Central FD=.02 Nm, original torque/rate/trust bounds, task-cost
weight=1, hard constraints, at most three shrinking relinearizations; a fresh
nonlinear rollout checks every accepted command. Failures stop before executing
a rejected action and are included in plots/tables. No slack or reserve policy.
K4 pauses the virtual reference clock after an actual H-triggered intervention,
then resumes after measured force falls below .02 N; transition=.1 s. Spatial
endpoints stay fixed. This is a reference-clock governor adaptation, not an
exact reproduction of the paper's spatial-anchor governor. K3/K4 constraints,
model, objective, accounts and bounds are identical.

Primary matrix: six 18 s fixtures x K0..K4. A: exact/no human. B: mismatch/no
human. C: exact/TCP blocking. D: mismatch/TCP blocking. E: exact/link5 blocking
at local (0,.04,.12) m. F: exact/TCP helping then blocking. Blocking is -x 5 N,
onset 9.5 s, cosine ramp .2 s, hold 1 s, ramp .2 s. F first helps +x 5 N at
9.5 s with .2 s ramps and .2 s hold, then blocks at 10.3 s with .2 s ramps
and 1 s hold. Human TCP point is the model's actual tcp site, resolved before
simulation. Two additional starved-H=.001 J C fixtures use K3/K4. No retuning
after outcomes; implementation fixes and failed exploratory runs are retained.

Report completion, protected and unprotected minima, peak power, QP attempts
(including failed attempts), contact absence, original 10 N force error,
measured progress and recovery. Hypotheses may fail: K1 should be susceptible
to task-work depletion; mismatch should separate R/H; H protection should
matter when R has a different budget; governor benefit requires actual distinct
closed-loop outcomes. Identical or stopped curves cannot establish superiority.
