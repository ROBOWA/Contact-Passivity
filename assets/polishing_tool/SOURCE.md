# Polishing tool meshes

| Field | Value |
| --- | --- |
| Origin | MEAM 6230 polishing project (`MEAM_6230_Final_Example`, commit 7be747c), file `assets/franka_panda/franka_emika_panda/panda_demo_2.xml` and its `assets/` meshes |
| Authors | project members (ROBOWA, Cindy Wu, Yifan Wu); project-internal assets, no separate license file |
| Files | `ft_sensor.stl` (force/torque sensor housing, visual only), `sphere_tool.stl` (tool shaft, visual only); copied verbatim, extensions lower-cased |

In the Panda demo the stack hangs off `link7` at z = 0.142 m (35 mm above the
Panda flange at 0.107 m): the FT-sensor body, a `sphere_tool` body carrying
the shaft mesh, and an `attachment` body 0.107 m further along z whose 9 mm
`tool_sphere` tip sits at +0.03 m, i.e. the tip centre is 0.279 m from the
Panda `link7` origin and 0.172 m beyond the flange.

`assets/polishing_tool.xml` mounts the same stack on the iiwa14 flange with
the same offsets; see its header comment for what was changed.
