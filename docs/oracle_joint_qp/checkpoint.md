# Oracle joint-torque QP checkpoint

Starting revision: `dbff1c1`, branch `main`, clean working tree. Work branch:
`feature/kuka-oracle-joint-qp`. No AGENTS.md found in the repository or its ancestors.
No hardware access. Preserve plant XML, nominal controller gains, trajectory, and 10 N target.

Plan:
1. Reproduce baseline; inspect stage-consistent work meter and actuator mapping.
2. Add known link point port, full-state isolated rollouts, central gradients; validate shadow.
3. Add scaled seven-variable QP, physical residual checks, nonlinear acceptance and finite retries.
4. Run matched demonstration matrix, gradient/hold-out checks, runtime measurements.
5. Complete three evidence-backed review rounds, final fresh-process reproduction, compact artifacts and report.

Confirmed timing: h=0.00025 s, dt=0.001 s. Installed Python 3.13.5, MuJoCo 3.15.0,
NumPy 2.5.4 on macOS 26.2 arm64. OSQP/Scipy initially missing; installing locally.
Full integration state includes time/history/warmstart/inputs/mocap/user/plugin state.
Global control callback absent. Existing controller has three clamped integrators and
conditional integration on previous saturation. The oracle will extend that condition
to actual filter intervention, without a fictitious Cartesian inverse of joint torque.

Initial commands:
```
.venv/bin/python -m pytest -q
.venv/bin/python -m kuka_table_sim.scenarios nominal --log results/oracle/baseline.npz
```

Status: baseline 50/50 passed; new physics/recorder 10/10 passed; QP/mode checks
6/6 passed. Baseline before/after physics and control arrays reproduce bitwise.
Shadow matched off: W_H=-0.241623266070888 J, prediction error exactly zero.
Ample 1 J active is transparent. Initial 0.12 J active trial stops at 10.662 s
with `energy_infeasible` (best affine box margin -4.6812e-6 J); 188 intervention
ticks, 5 nonlinear retries, smallest preceding boundary margin +7.7641e-8 J.
This is a preserved failed fixture, not an accepted protected demonstration.
Next: validate 0.20/0.22 J feasible budget candidates, gradients over the declared
trust domain, then freeze final configuration and reproduce full matrix.
The initial attempted `scenarios nominal --out ...` was rejected by argparse without
running; corrected to the supported `--log` option above.
