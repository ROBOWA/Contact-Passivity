# contactSimulation

Simulations for the contact-model-aware **selective-passivation** project.

- [`mujoco_sliding/`](mujoco_sliding/README.md) — MuJoCo-based planar
  contact-sliding proof of concept: a two-link robot approaches a surface,
  regulates a 5 N normal force, and slides tangentially, with ground-truth
  task-contact and human-disturbance force channels logged separately for a
  future selective-passivation QP.

## Quick start

```bash
pip install -r requirements.txt
python -m mujoco_sliding.simulation                 # headless run -> results/
python -m mujoco_sliding.simulation --human-force   # with the known human push
python -m mujoco_sliding.viewer                     # interactive MuJoCo viewer
python -m pytest mujoco_sliding/tests               # test suite
```

See [mujoco_sliding/README.md](mujoco_sliding/README.md) for the model,
contact-extraction, controller, and sign-convention documentation.
