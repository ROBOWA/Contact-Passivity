# contactSimulation

Simulations for the contact-model-aware **selective-passivation** project.

- [`mujoco_sliding/`](mujoco_sliding/README.md) — MuJoCo-based planar
  contact-sliding proof of concept: a two-link robot approaches a surface,
  regulates a 5 N normal force, and slides tangentially, with ground-truth
  task-contact and human-disturbance force channels, C0--C4 selective-
  passivation controllers, experiments, plots, and tests.
- [`mujoco_panda/`](mujoco_panda/README.md) — seven-torque Franka Panda
  extension with spatial task/human ports, straight and bounded wiping
  trajectories, one-step OSQP filter, work audit, focused A--E comparisons,
  interactive replay, and video rendering.

## Quick start

```bash
pip install -r requirements.txt
python -m mujoco_sliding.simulation                 # headless run -> results/
python -m mujoco_sliding.simulation --human-force   # with the known human push
python -m mujoco_sliding.viewer                     # interactive MuJoCo viewer
python -m pytest mujoco_sliding/tests               # test suite
python -m mujoco_panda.cli --scenario C_ee_oblique_blocking --mode C3_dual_ledger_qp
python -m pytest mujoco_panda/tests
```

See each backend README for exact reproduction commands and the scope of its
energy/passivity claims.
