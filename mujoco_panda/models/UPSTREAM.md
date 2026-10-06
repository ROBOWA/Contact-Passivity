# Vendored Franka Panda model

`menagerie/franka_emika_panda/` is copied from the MuJoCo Menagerie
`franka_emika_panda` model at commit
`8161bba264d7fa7c99ca301e91e7fb44737676ad` (2026-09-04).

Upstream: <https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda>

The vendored files retain the upstream Apache-2.0 `LICENSE`. The original
`panda_nohand.xml` and meshes are byte-for-byte unmodified. `panda_table.xml`
is this project's derived model: its arm position servos are replaced by
direct torque actuators, robot collision meshes are disabled, and a fixed
finite four-element tool pad, table, interaction sites, camera, and keyframe
are added. A disabled single sphere is retained only for the documented
contact-layout ablation.
No Franka mesh geometry was modified.
