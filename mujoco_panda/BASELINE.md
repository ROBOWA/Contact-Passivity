# Baseline and dimensional audit

Baseline recorded on 2026-09-10 before the Panda backend was added.

- Checkout: `6cd9b057b4b3233a6e8fe0489a4c2d9394e161b1` (the reviewed
  `6cd9b05`), with pre-existing uncommitted paper/video changes left intact.
- Command: `pytest -q -rxX`
- Result: 76 passed, 1 expected failure in 33.65 s.
- Existing expected failure:
  `mujoco_sliding/tests/test_passivation.py::test_c1_whole_port_qp_solver_feasible`.
  It records the already documented whole-port/contact-sample limit cycle and
  bounded fallback; it is an unmet acceptance criterion, not a hidden pass.

The planar implementation's dimensional assumptions were inventoried as:

- `config.py`: fixed `TANGENT`, `NORMAL`, and 3-by-2 `B_TN` for x-z motion.
- `passivity_qp.py`: a two-variable Cartesian command, fixed seven-row QP,
  and exactly two joint torque rows.
- `contact_extraction.py`: a single pad/site/body and summed contact force
  paired with an arithmetic mean contact-point velocity.
- `simulation.py`: two-entry command/log arrays and one EE human location.
- Model tests: structural assertions `nq == nv == nu == 2`.

The Panda extension is isolated in `mujoco_panda/`; no planar dimensional
assumption was changed. Its spatial port implementation instead sums every
contact point's wrench/twist power and generalized force.
