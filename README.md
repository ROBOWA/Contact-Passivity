# KUKA iiwa14 table scene (MuJoCo)

Milestone 1 of the contact-passivity simulation rebuild: a KUKA LBR iiwa 14
standing above a stationary table in MuJoCo, driven by seven joint-torque
actuators and held in place by a gravity-compensated joint PD controller. The
spherical polishing tool from the MEAM 6230 Panda project is mounted on the
flange. Nothing else is implemented yet (no object, contact task, Cartesian
impedance, force control, or hardware interface).

![KUKA iiwa14 above the table](docs/kuka_table_scene.png)

**Previous research work** (planar sliding PoC, Franka Panda selective
passivation, paper sources, results) lives on the `proposal` branch. It is still
present in this branch's history and tree (`mujoco_sliding/`, `mujoco_panda/`,
`paper*/`, `results*/`), but nothing here imports it and its original
`requirements.txt` is the one on `proposal`.

## Installation

Tested on macOS 26.2 (arm64) with Python 3.13.5, `mujoco==3.15.0`,
`numpy==2.5.4` (project `.venv`), and also with Python 3.12.2,
`mujoco==3.3.7`, `numpy==2.3.4` (conda base: headless validation passes,
viewer launches and exits cleanly). `mj_fullM` changed signature between
those releases; `holding.full_mass_matrix` handles both.

```bash
git clone git@github.com:ROBOWA/Contact-Passivity.git && cd Contact-Passivity
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pinned, tested versions
```

Robot meshes and the MJCF are vendored under `assets/kuka_iiwa_14/`, so no
download happens at launch. The pinned upstream commit and the re-vendoring
recipe are in [assets/kuka_iiwa_14/SOURCE.md](assets/kuka_iiwa_14/SOURCE.md).
The tool meshes are under `assets/polishing_tool/` with their own
[SOURCE.md](assets/polishing_tool/SOURCE.md).

## Running

```bash
# Headless validation: 10 s of simulated time, prints a report, exit code 0 on PASS.
python -m kuka_table_sim.run headless --duration 10

# Same, plus an offscreen PNG from the "overview" camera (needs a GL backend).
python -m kuka_table_sim.run headless --duration 10 --render docs/kuka_table_scene.png

# Interactive viewer. macOS needs the mjpython launcher shipped with the mujoco wheel:
mjpython -m kuka_table_sim.run viewer
# Linux / Windows:
python -m kuka_table_sim.run viewer
```

Viewer controls: the viewer starts on MuJoCo's free camera, so the mouse
works immediately (left drag orbits, right drag pans, scroll zooms, double
click selects a body). `R` or `Backspace` resets the scene (home pose, zero
velocity, `t = 0`, holding target re-armed). `Ctrl-Q` or closing the window
quits, and `Ctrl-C` in the terminal also quits cleanly. `--camera overview`
starts on the fixed scene camera instead; a fixed camera ignores mouse orbiting
until `Esc` switches back to the free camera (`[` and `]` cycle cameras).

All timing parameters are command-line options (`--timestep`,
`--control-period`, `--render-hz`, `--hold-freq-hz`, `--hold-zeta`); see
`python -m kuka_table_sim.run --help`. Run the commands from the repository
root so that `kuka_table_sim` is importable; the package locates `assets/`
relative to its own files.

On macOS the passive viewer must run under `mjpython` (see
<https://mujoco.readthedocs.io/en/stable/python.html>). The headless path never
imports `mujoco.viewer`.

## Scene definition

| Item | Value |
| --- | --- |
| World frame | right-handed, z up, origin at the centre of the robot mounting plate, +x toward the table, SI units |
| Robot base pose | `base` body at the world origin, identity orientation, mounted on a 0.30 x 0.30 x 0.55 m pedestal box |
| Floor | plane at z = -0.55 m |
| Tabletop | box 0.60 (x) x 0.80 (y) x 0.04 m, centred at (0.65, 0, 0.13) m, **top surface z = +0.15 m**, spans x in [0.35, 0.95] m, y in [-0.40, 0.40] m |
| Table legs | four 0.04 x 0.04 m boxes from the floor to the tabletop underside, at (0.65 +/- 0.27, +/- 0.37) m |
| Collision | floor, pedestal and all table geoms collide with the robot's collision spheres; the seven upstream adjacent-link `<exclude>` pairs are kept, nothing else is disabled |
| Initial pose `q_home` (rad) | `[0, 0.43, 0, -1.51, 0, 1.2, 0]` (keyframe `home`) |
| Flange at `q_home` | position (0.5483, 0, 0.4714) m, z-axis (0, 0, -1) within 2e-3, i.e. tool pointing straight down |
| TCP (tool tip centre) at `q_home` | (0.5486, 0, 0.2994) m; tip surface 0.140 m above the tabletop, 0.20 m inside the near table edge |
| Joint-limit margin at `q_home` | min 0.584 rad (joint 4) |
| Contacts at `q_home` | none |
| `flange` site | on `link7` at (0, 0, 0.045) m, same frame as the upstream `attachment_site`: origin at the flange face centre, z-axis along the joint-7 axis pointing out of the flange, x/y parallel to the `link7` frame |
| `tcp` site | centre of the 9 mm tip sphere, 0.172 m from the flange along its z-axis (see Tool below); contact point with a flat surface is 0.009 m further along the surface normal |
| Cameras | free camera framed by `<statistic center="0.35 0 0.05" extent="1.6">` and `<visual><global azimuth="140" elevation="-20">` (viewer default); fixed `overview` camera at (1.75, -1.65, 0.80) m looking at (0.35, 0, 0.05) m (used for offscreen renders and `--camera overview`) |

Files: `assets/kuka_table_scene.xml` (world, table, lights, camera),
`assets/iiwa14_torque.xml` (robot, derived from upstream, changes listed in its
header comment), `assets/polishing_tool.xml` (tool stack, included inside
`link7`), `assets/kuka_iiwa_14/` (verbatim upstream copy plus license).

## Tool

The end-effector is the spherical polishing tool from the MEAM 6230 Panda
polishing project, re-mounted on the iiwa14 flange with the same offsets it had
beyond the Panda flange ([assets/polishing_tool/SOURCE.md](assets/polishing_tool/SOURCE.md)):

| Body / element | Position along flange z | Notes |
| --- | --- | --- |
| `tool_mount` | 0 | flange frame |
| `ft_sensor`, site `ft_site` | +0.035 m (sensor top face) | housing mesh (visual), collision cylinder r = 0.0375 m, h = 0.035 m below the top face; welded to `link7`, so it never self-collides with it |
| `sphere_tool` | +0.035 m | shaft mesh, visual only (as in the original) |
| `tool_tip` geom, site `tcp` | +0.172 m | sphere r = 0.009 m, friction `0.8 0.005 0.0001` from the original, collides with the table like every other geom |

Masses are assumed (0.30 kg sensor, 0.20 kg shaft; the original used 0) so
gravity compensation and the mass matrix include the tool. The Panda demo's
intermediate `attachment` body with its 45 degree z-rotation was flattened into
a plain offset; a sphere tip has no orientation. No force sensor element is
defined yet; `ft_site` is there so one can be added later.

## Model source

[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
`kuka_iiwa_14`, commit `0059d4335f8156206f63a35662313385f7ad6d74` (2026-10-07),
BSD-3-Clause (Drake-derived). License and attribution files are kept in
`assets/kuka_iiwa_14/`.

## Actuators

The upstream model ships position servos (`<general>` with
`gainprm="2000" biasprm="0 -2000 -200"`, angle-valued `ctrlrange`, and a
keyframe `ctrl` holding target angles). `assets/iiwa14_torque.xml` replaces them
with seven `<motor>` actuators:

| Actuator | Joint | `gear` | `ctrlrange` (N m) |
| --- | --- | --- | --- |
| `joint1_torque` | `joint1` | 1 | +/- 320 |
| `joint2_torque` | `joint2` | 1 | +/- 320 |
| `joint3_torque` | `joint3` | 1 | +/- 176 |
| `joint4_torque` | `joint4` | 1 | +/- 176 |
| `joint5_torque` | `joint5` | 1 | +/- 110 |
| `joint6_torque` | `joint6` | 1 | +/- 40 |
| `joint7_torque` | `joint7` | 1 | +/- 40 |

`data.ctrl[i]` is the torque applied to joint `i+1` in N m. The limits are the
KUKA LBR iiwa 14 R820 joint torque figures from the spec sheet the upstream
README cites; here they are **simulation settings only**, with no claim of
hardware certification. MuJoCo clamps `ctrl` to `ctrlrange`; the controller also
clips to the same values so saturation can be reported. The servo gains, the
angle `ctrlrange` defaults and the keyframe `ctrl` were removed, so the
keyframe stores only `qpos`. `ArmIndex.from_model` verifies all of this at load
time (hinge joints, one-to-one joint mapping, gear 1, fixed unit gain, no bias,
symmetric limits).

## Holding controller (scene infrastructure only)

```
tau = qfrc_bias + Kp * (q_home - q) - Kd * qvel
```

* `qfrc_bias` is MuJoCo's `C(q, qvel) qvel + g(q)` for the state about to be
  integrated. The control step calls `mj_step1` (kinematics, bias forces,
  contacts), computes `tau` from those quantities, writes `ctrl`, then calls
  `mj_step2`. Compensation is applied once, here, and nowhere else.
* Default gains are scaled by the mass-matrix diagonal at `q_home`:
  `Kp_i = M_ii wn^2`, `Kd_i = 2 zeta M_ii wn` with `wn = 2 pi * 2 Hz`,
  `zeta = 1` (`--hold-freq-hz`, `--hold-zeta`). This matters because joint 7's
  modelled inertia is 0.001 kg m^2 while joint 2's is about 2.9; a fixed gain
  vector either chatters on the wrist or is sluggish on the shoulder. Explicit
  `Kp`/`Kd` vectors can be passed to `HoldingController` from Python.
* Torques are clipped to the actuator limits.
* Gravity stays on, joints are free, nothing writes `qpos` during stepping.
  `Simulation.reset()` restores the `home` keyframe, zero velocity, `t = 0`, the
  step counters and the holding target.

This is only there to keep the arm still while the scene is checked. It is not
the hybrid controller of the later milestones.

## Timing

| Setting | Default | Note |
| --- | --- | --- |
| Physics timestep | 0.00025 s | `--timestep`; integrator `implicitfast` |
| Holding update period | 0.001 s simulated | `--control-period`; torque recomputed every 4 physics steps and held constant in between (1 kHz of **simulated** time, not a real-time claim) |
| Viewer frame | 67 physics steps (~59.7 Hz simulated) | `--render-hz`; `viewer.sync()` once per frame, independent of the control schedule |

Scheduling uses integer step counts (`n_steps % control_interval`). The viewer
sleeps the remainder of each frame's wall-clock budget purely for viewing
comfort; sleeping never changes how many physics steps are taken or the
timestep. Headless runs do not sleep at all.

## Validation

`python -m kuka_table_sim.run headless --duration 10` checks and prints:

* model structure: 7 hinge joints, 7 unit-gear torque motors mapped one-to-one,
  torque limits present;
* initial pose: no contacts, >= 0.1 rad from every joint limit, tool tip at
  least 0.05 m above the tabletop and inside its footprint;
* 10 s unperturbed hold with gravity on: finite state, no new MuJoCo warnings,
  no joint-limit hits, no robot contacts with anything, no sustained torque
  saturation, max `|q - q_home| < 0.01 rad`;
* physics sanity: with `ctrl = 0` the arm falls (> 0.05 rad in 0.5 s);
  starting 0.05 rad off `q_home` the controller returns it within 0.01 rad;
* reset: restores `q_home`, zero velocity, `t = 0`, zero contacts, and a replay
  after reset reproduces a fresh run exactly;
* optional offscreen render of the `overview` camera.

Results on the tested platform (2026-10-12):

| Check | Result |
| --- | --- |
| 10 s hold, max `|q - q_home|` | 0.0 rad (exact compensation from the same state gives zero acceleration) |
| 10 s hold, max `|tau|` | 47.6 N m on joint 2 (gravity), no saturation |
| Contacts / warnings / limit hits | none |
| Controller off, 0.5 s | max deviation 1.68 rad (arm falls) |
| Start +0.05 rad on all joints | within 0.01 rad after 0.330 s, final deviation 9e-9 rad, no contacts |
| Reset | `t = 0`, `q == q_home`, replay identical |
| Wall-clock | 10 s simulated in ~0.7 s |
| Viewer | launches under `mjpython` (mujoco 3.15.0 and 3.3.7), free camera, Ctrl-C exits cleanly |

## Layout

```
assets/kuka_table_scene.xml   world: floor, pedestal, table, lights, camera (includes the robot)
assets/iiwa14_torque.xml      iiwa14 with torque motors, flange site, home keyframe, tool include
assets/polishing_tool.xml     FT-sensor housing + shaft + 9 mm tip sphere (tcp site)
assets/kuka_iiwa_14/          vendored Menagerie model + LICENSE + SOURCE.md
assets/polishing_tool/        tool meshes + SOURCE.md
kuka_table_sim/scene.py       paths, TimingConfig, ArmIndex (joint/actuator mapping checks)
kuka_table_sim/holding.py     HoldingController, Simulation (reset, mj_step1/step2 scheduling)
kuka_table_sim/run.py         CLI: headless validation / render, interactive viewer
docs/kuka_table_scene.png     render from the headless validation
pyproject.toml, requirements.txt
```
