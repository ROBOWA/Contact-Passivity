# Oracle joint-torque QP checkpoint

Starting revision: `dbff1c1`, branch `main`, clean working tree. Work branch:
`feature/kuka-oracle-joint-qp`. No AGENTS.md found in the repository or its ancestors.
No hardware access. Preserve plant XML, nominal controller gains, trajectory, and 10 N target.

Completed workflow:
1. Reproduce baseline; inspect stage-consistent work meter and actuator mapping.
2. Add known link point port, full-state isolated rollouts, central gradients; validate shadow.
3. Add scaled seven-variable QP, physical residual checks, nonlinear acceptance and finite retries.
4. Run matched demonstration matrix, gradient/hold-out checks, runtime measurements.
5. Complete three evidence-backed review rounds, final fresh-process reproduction, compact artifacts and report.

Confirmed timing: h=0.00025 s, dt=0.001 s. Installed Python 3.13.5, MuJoCo 3.15.0,
NumPy 2.5.4 on macOS 26.2 arm64 / Apple M4. OSQP 1.1.3 and SciPy 1.18.1 installed locally.
Full integration state includes time/history/warmstart/inputs/mocap/user/plugin state.
Global control callback absent. Existing controller has three clamped integrators and
conditional integration on previous saturation. The oracle extends that condition
to actual filter intervention, without a fictitious Cartesian inverse of joint torque.

Initial commands:
```
.venv/bin/python -m pytest -q
.venv/bin/python -m kuka_table_sim.scenarios nominal --log results/oracle/baseline.npz
```

Final status: **PARTIAL**. Numerical/control revision `43a6661`; later commits
repair only reporting/provenance/compaction. Fresh full suite: 61 passed in 70.03 s.
All 18 final matrix cases ran; matrix exits 2 for explicitly failed active cases.
1155 directional scans across 15 snapshots pass the calibrated empirical envelope.
Full 25 s baseline and no-human/ample mode comparisons reproduce bitwise.

Primary .12 J / task weight 1 active completes 18 s: net output .1199989829 J
vs .2416232661 J off (50.34% reduction), minimum boundary margin +1.86166e-7 J.
However task force error peaks at 53.104 N and contact absence totals 1.14975 s.
Minimum observed substep margin -1.11334e-6 J; only endpoints are enforced.
Held-out timing (.12 J) fails at 10.753 s and elbow tight at 10.884 s; snapshots
are preserved. The .001 J / weight 1 stress case completes, confirming that
this greedy controller's feasibility is nonmonotonic, not a universal threshold.

Read `report.md`, `acceptance.json`, `reviews.md`, `morning_summary_zh.md` here.
Full logs: `results/oracle/final/`; compact evidence: `raw/`. Report commands are
in the report and Chinese summary. Final figures include all force extrema.
No plant/gain/trajectory changes, hardware access, merge or push.

Remaining blocking work: preserving table contact/force quality and robust
feasibility across held-out timing. Next research decision is whether to add
explicit viability/braking reserve and stronger task constraints in a new stage.
