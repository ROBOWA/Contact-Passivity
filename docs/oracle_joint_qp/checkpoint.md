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

Status: baseline and regression processes started; implementation not yet validated.
The initial attempted `scenarios nominal --out ...` was rejected by argparse without
running; corrected to the supported `--log` option above.
