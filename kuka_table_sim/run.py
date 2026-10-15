"""Command-line entry point.

    python   -m kuka_table_sim.run headless --duration 10 [--render out.png]
    python   -m kuka_table_sim.run --controller force headless --duration 25 [--log run.npz] [--plot run.png]
    mjpython -m kuka_table_sim.run [--controller force] viewer   # macOS; plain python elsewhere
    python   -m kuka_table_sim.run ik --site tcp --pos 0.6 0.15 0.3 --rpy 0 180 0 [--render pose.png]
    python   -m kuka_table_sim.run --init-site tcp --init-pos 0.6 0.15 0.3 --init-rpy 0 180 0 headless

Any command accepts --init-site/--init-pos with --init-quat or --init-rpy: the
initial state is then the IK solution for that site pose instead of the keyframe.

`headless` runs the scene without any GUI, prints a validation report and exits
non-zero if a check fails. `viewer` opens the passive MuJoCo viewer on the free camera (mouse orbits,
pans and zooms); press R (or Backspace) to reset, Ctrl-Q or close the window
to quit, Ctrl-C in the terminal also exits cleanly. `--camera overview` starts
on the fixed scene camera instead (Esc returns to the free camera).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import signal
import struct
import sys
import time
import zlib
from pathlib import Path

import mujoco
import numpy as np

from .force_control import ForceApproachController, ForceControlConfig
from .holding import DEFAULT_HOLD_FREQ_HZ, DEFAULT_HOLD_ZETA, HoldingController, Simulation
from .ik import quat_from_rpy, site_pose, solve_site_ik
from .scene import (FLANGE_SITE, OVERVIEW_CAMERA, ROBOT_BODIES, SCENE_XML, TCP_SITE, TOOL_TIP_GEOM, ArmIndex,
                    TimingConfig, home_qpos, load_model)


# --------------------------------------------------------------------------- helpers

N_WARNINGS = int(mujoco.mjtWarning.mjNWARNING)


def _warning_names() -> dict[int, str]:
    return {int(v): k for k, v in mujoco.mjtWarning.__members__.items() if k != "mjNWARNING"}


def _warning_counts(d: mujoco.MjData) -> np.ndarray:
    return np.array([d.warning[i].number for i in range(N_WARNINGS)])


def _contact_pairs(sim: Simulation) -> list[tuple[str, str, float]]:
    m, d = sim.model, sim.data
    return [(m.geom(c.geom1).name or m.body(m.geom_bodyid[c.geom1]).name,
             m.geom(c.geom2).name or m.body(m.geom_bodyid[c.geom2]).name,
             float(c.dist)) for c in d.contact[: d.ncon]]


def _robot_contact_pairs(sim: Simulation) -> list[tuple[str, str, float]]:
    """Contacts in which at least one geom belongs to a robot body (any partner)."""
    m, d = sim.model, sim.data
    robot = {m.body(n).id for n in ROBOT_BODIES}
    out = []
    for c in d.contact[: d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in robot or b2 in robot:
            out.append((m.body(b1).name + "/" + (m.geom(c.geom1).name or str(c.geom1)),
                        m.body(b2).name + "/" + (m.geom(c.geom2).name or str(c.geom2)),
                        float(c.dist)))
    return out


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal RGB8 PNG writer (stdlib only)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def render_overview(sim: Simulation, path: Path, width: int = 1280, height: int = 720) -> None:
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    try:
        renderer.update_scene(sim.data, camera=OVERVIEW_CAMERA)
        write_png(path, renderer.render())
    finally:
        renderer.close()


# --------------------------------------------------------------------------- headless

def run_headless(sim: Simulation, duration: float, render: Path | None, verbose: bool = True) -> bool:
    m, d, arm = sim.model, sim.data, sim.arm
    tcp = m.site(TCP_SITE).id
    ok = True
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        if not cond:
            ok = False
            failures.append(msg)

    # --- structure -------------------------------------------------------------------
    n_arm = int(np.sum(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
    check(m.nq == 7 and m.nv == 7 and n_arm == 7, f"expected 7 hinge joints, got nq={m.nq} nv={m.nv}")
    check(m.nu == 7, f"expected 7 actuators, got {m.nu}")
    check(d.ncon == 0, f"{d.ncon} contact(s) at the initial pose")
    jr = m.jnt_range[arm.joint_ids]
    margin = np.minimum(sim.q - jr[:, 0], jr[:, 1] - sim.q)
    check(np.all(margin > 0.1), f"initial pose is within 0.1 rad of a joint limit: margins {margin}")
    tcp_pos0 = d.site_xpos[tcp].copy()
    tcp_z0 = d.site_xmat[tcp].reshape(3, 3)[:, 2].copy()
    flange = m.site(FLANGE_SITE).id
    flange_pos0 = d.site_xpos[flange].copy()
    tip_radius = float(m.geom_size[m.geom(TOOL_TIP_GEOM).id][0])
    tabletop = m.geom("tabletop").id
    top_z = d.geom_xpos[tabletop][2] + m.geom_size[tabletop][2]
    tip_clearance = tcp_pos0[2] - tip_radius - top_z
    check(tip_clearance > 0.05, f"tool tip is only {tip_clearance:.3f} m above the tabletop")
    half = m.geom_size[tabletop]
    centre = d.geom_xpos[tabletop]
    check(abs(tcp_pos0[0] - centre[0]) < half[0] and abs(tcp_pos0[1] - centre[1]) < half[1],
          "TCP is not over the tabletop footprint")

    if verbose:
        print("== scene ==")
        print(f"model file          : {SCENE_XML}")
        print(f"joints / actuators  : {m.nq} hinge joints, {m.nu} torque motors (gear 1, N*m)")
        for i in range(7):
            print(f"  {m.actuator(arm.actuator_ids[i]).name:14s} -> {m.joint(arm.joint_ids[i]).name:7s} "
                  f"|tau| <= {arm.torque_limit[i]:6.1f} N*m   range [{jr[i,0]:+.3f}, {jr[i,1]:+.3f}] rad")
        print(f"timestep / control  : {sim.timing.timestep} s / {sim.timing.control_period} s "
              f"(every {sim.timing.control_interval} steps), integrator "
              f"{mujoco.mjtIntegrator(m.opt.integrator).name}, gravity {m.opt.gravity}")
        print(f"q_home (rad)        : {np.array2string(sim.q_home, precision=3)}")
        print(f"joint-limit margin  : min {margin.min():.3f} rad")
        print(f"flange position (m) : {np.array2string(flange_pos0, precision=4)}  z-axis "
              f"{np.array2string(tcp_z0, precision=3)}")
        print(f"TCP (tip centre, m) : {np.array2string(tcp_pos0, precision=4)}  tip radius {tip_radius:.3f}")
        print(f"tabletop top z (m)  : {top_z:.3f}  -> tip surface clearance {tip_clearance:.3f} m")
        print(f"initial contacts    : {d.ncon}")
        print(f"gravity torque (N*m): {np.array2string(d.qfrc_bias[arm.dof_adr], precision=2)}")
        print(f"Kp (N*m/rad)        : {np.array2string(sim.controller.kp, precision=3)}")
        print(f"Kd (N*m*s/rad)      : {np.array2string(sim.controller.kd, precision=3)}")

    # --- run -------------------------------------------------------------------------
    n = sim.steps_for(duration)
    ci = sim.timing.control_interval
    max_dev = np.zeros(7)
    max_vel = np.zeros(7)
    max_tau = np.zeros(7)
    sat_updates = 0
    sat_run = sat_run_max = 0
    contact_events: dict[tuple[str, str], int] = {}
    limit_hits = 0
    warn_before = _warning_counts(d)
    t0 = time.perf_counter()
    for k in range(n):
        sim.step()
        if sim.n_steps % ci == 0:  # sample once per control update
            dev = np.abs(sim.q - sim.q_home)
            max_dev = np.maximum(max_dev, dev)
            max_vel = np.maximum(max_vel, np.abs(sim.qvel))
            max_tau = np.maximum(max_tau, np.abs(sim.tau))
            if sim.controller.saturated.any():
                sat_updates += 1
                sat_run += 1
                sat_run_max = max(sat_run_max, sat_run)
            else:
                sat_run = 0
            if np.any(sim.q <= jr[:, 0]) or np.any(sim.q >= jr[:, 1]):
                limit_hits += 1
            for a, b, _ in _robot_contact_pairs(sim):
                contact_events[(a, b)] = contact_events.get((a, b), 0) + 1
            if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                check(False, f"non-finite state at t={d.time:.4f}")
                break
    wall = time.perf_counter() - t0
    warn_after = _warning_counts(d)
    new_warnings = {_warning_names()[i]: int(warn_after[i] - warn_before[i])
                    for i in range(N_WARNINGS) if warn_after[i] != warn_before[i]}

    check(abs(d.time - n * sim.timing.timestep) < 1e-9, "simulated time does not match step count")
    check(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)), "non-finite final state")
    check(not new_warnings, f"MuJoCo warnings raised: {new_warnings}")
    check(limit_hits == 0, f"joint limit reached in {limit_hits} control updates")
    check(not contact_events, f"robot contacts during hold: {contact_events}")
    check(sat_run_max * sim.timing.control_period < 0.05, f"torque saturated for {sat_run_max} consecutive updates")
    check(max_dev.max() < 0.01, f"max joint deviation {max_dev.max():.4f} rad exceeds 0.01 rad")
    tcp_drift = float(np.linalg.norm(d.site_xpos[tcp] - tcp_pos0))

    if verbose:
        print("== hold ==")
        print(f"simulated / wall    : {d.time:.3f} s in {n} steps, {sim.n_control_updates} control updates; "
              f"wall {wall:.2f} s ({d.time / wall:.1f}x)")
        print(f"max |q - q_home|    : {np.array2string(max_dev, precision=5)}  -> max {max_dev.max():.2e} rad")
        print(f"max |qvel| (rad/s)  : {np.array2string(max_vel, precision=4)}")
        print(f"max |tau| (N*m)     : {np.array2string(max_tau, precision=2)}  of limits {arm.torque_limit}")
        print(f"saturated updates   : {sat_updates} (longest run {sat_run_max})")
        print(f"joint-limit hits    : {limit_hits}")
        print(f"robot contacts      : {dict(contact_events) if contact_events else 'none'}")
        print(f"MuJoCo warnings     : {new_warnings if new_warnings else 'none'}")
        print(f"TCP drift (m)       : {tcp_drift:.2e}")

    # --- physics sanity: controller off, gravity must move the arm -----------------------
    q_before, t_before = sim.q.copy(), d.time
    sim.reset()
    d.ctrl[:] = 0.0
    for _ in range(sim.steps_for(0.5)):
        mujoco.mj_step(m, d)
    drop = float(np.abs(sim.q - sim.q_home).max())
    check(drop > 0.05, f"arm barely moved with zero torque ({drop:.4f} rad); is gravity on?")

    # --- perturbed hold: start 0.05 rad off home, controller must pull it back ------------
    sim.reset()
    offset = 0.05
    d.qpos[arm.qpos_adr] = sim.q_home + offset
    mujoco.mj_forward(m, d)
    check(d.ncon == 0, "perturbed start pose is in contact")
    settle_time = None
    pert_contacts = 0
    for _ in range(sim.steps_for(3.0)):
        sim.step()
        if sim.n_steps % ci == 0:
            pert_contacts += len(_robot_contact_pairs(sim))
            if settle_time is None and np.abs(sim.q - sim.q_home).max() < 0.01:
                settle_time = d.time
    pert_dev = float(np.abs(sim.q - sim.q_home).max())
    check(settle_time is not None and pert_dev < 0.01, f"perturbed hold did not settle: final deviation {pert_dev:.4f} rad")
    check(pert_contacts == 0, f"{pert_contacts} robot contact samples during perturbed hold")
    if verbose:
        print("== sanity ==")
        print(f"controller off 0.5 s: max |q - q_home| = {drop:.3f} rad (arm falls under gravity)")
        print(f"start +{offset} rad off : back within 0.01 rad at t={settle_time:.3f} s, final deviation {pert_dev:.2e} rad, "
              f"contacts {pert_contacts}")

    # --- reset -----------------------------------------------------------------------
    sim.reset()
    check(d.time == 0.0 and sim.n_steps == 0, "reset did not zero time/step count")
    check(np.array_equal(sim.q, sim.q_home) and not np.any(sim.qvel), "reset did not restore q_home / zero velocity")
    check(np.array_equal(sim.controller.q_target, sim.q_home), "reset did not restore the holding target")
    check(d.ncon == 0, "contacts present after reset")
    # Determinism: the first 0.5 s after reset must reproduce the first 0.5 s of a fresh simulation.
    sim.advance(sim.steps_for(0.5))
    q_a, t_a = sim.q.copy(), d.time
    sim.reset()
    sim.advance(sim.steps_for(0.5))
    check(np.array_equal(q_a, sim.q) and t_a == d.time, "trajectory after reset is not reproducible")
    sim.reset()
    if verbose:
        print("== reset ==")
        print(f"before reset        : t={t_before:.3f} s, |q - q_home| max {np.abs(q_before - sim.q_home).max():.2e}")
        print(f"after reset         : t={d.time} s, steps={sim.n_steps}, q == q_home: "
              f"{np.array_equal(sim.q, sim.q_home)}, contacts {d.ncon}")
        print("replay after reset  : identical to fresh run")

    # --- render ----------------------------------------------------------------------
    if render is not None:
        try:
            render_overview(sim, render)
            if verbose:
                print(f"== render ==\nwrote {render} ({OVERVIEW_CAMERA} camera, 1280x720)")
        except Exception as exc:  # offscreen GL may be unavailable on headless hosts
            check(False, f"offscreen render failed: {exc!r}")

    if verbose:
        print("== result ==")
        print("PASS" if ok else "FAIL:\n  - " + "\n  - ".join(failures))
    return ok


def run_force_headless(sim: Simulation, duration: float, render: Path | None, log: Path | None, plot: Path | None = None,
                       verbose: bool = True) -> bool:
    """Force-controller validation through the stage-coherent recorder (see recorder.py and scenarios.py):
    the controller under test is the one in `sim`; the report, gates and logs use time-aligned samples."""
    from . import scenarios as sc
    from .recorder import Recorder
    sim.recorder = Recorder(sim, duration)
    sim.reset()
    run_log = sc.run_for(sim, duration)
    res = sc.ScenarioResult("nominal", sim, run_log)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if not verbose else sys.stdout):
        ok = sc.report_nominal(res)
    if log is not None:
        run_log.save(log, float32=True)
        if verbose:
            print(f"== log ==\nwrote {log} (compressed RunLog, float32 except time/work; load with RunLog.load)")
    if plot is not None:
        from .plot import plot_scenario
        plot_scenario(res, plot)
        if verbose:
            print(f"== plot ==\nwrote {plot}")
    if render is not None:
        try:
            render_overview(sim, render)
            if verbose:
                print(f"== render ==\nwrote {render} (final state, {OVERVIEW_CAMERA} camera)")
        except Exception as exc:  # offscreen GL may be unavailable on headless hosts
            print(f"offscreen render failed: {exc!r}")
            ok = False
    return ok


# --------------------------------------------------------------------------- viewer

def run_viewer(sim: Simulation, realtime: bool = True, camera: str = "free") -> None:
    import mujoco.viewer  # imported lazily so headless runs never touch GUI code

    reset_requested = False
    stop_requested = False

    def on_key(keycode: int) -> None:
        nonlocal reset_requested
        if keycode in (ord("R"), ord("r"), 259):  # 259 = GLFW_KEY_BACKSPACE
            reset_requested = True

    def on_sigint(signum, frame) -> None:  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True

    # Under mjpython the Cocoa event loop owns the OS main thread, so a pending
    # KeyboardInterrupt is never raised inside this loop. An explicit handler that
    # only sets a flag works on every platform; the loop polls it once per frame.
    try:
        previous_handler = signal.signal(signal.SIGINT, on_sigint)
    except ValueError:  # not the main thread: fall back to KeyboardInterrupt
        previous_handler = None

    m, d = sim.model, sim.data
    ri = sim.timing.render_interval
    frame_dt = ri * sim.timing.timestep
    last_status_second = -1
    print(f"viewer: R / Backspace = reset; Ctrl-Q, window close or Ctrl-C in the terminal = quit; "
          f"mouse drag = orbit/pan/zoom (free camera), Esc = back to the free camera, [ ] = cycle cameras. "
          f"{sim.timing.control_interval} physics steps per control update, "
          f"{ri} per frame ({1 / frame_dt:.1f} Hz of simulated time).", flush=True)
    try:
        with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
            if camera == "overview":
                # Fixed camera: framing is locked and the mouse does not orbit until Esc.
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = m.camera(OVERVIEW_CAMERA).id
            # Otherwise keep the default free camera, framed by <statistic> and <visual><global>.
            while viewer.is_running() and not stop_requested:
                frame_start = time.perf_counter()
                with viewer.lock():
                    if reset_requested:
                        sim.reset()
                        reset_requested = False
                        last_status_second = -1
                        print("reset -> t=0, q=q_home", flush=True)
                    sim.advance(ri)
                    if int(d.time) != last_status_second:
                        last_status_second = int(d.time)
                        print(f"t={d.time:6.2f} s  {sim.controller.status()}", flush=True)
                viewer.sync()
                if realtime:
                    # Wall-clock pacing only; never changes how many physics steps are taken.
                    remaining = frame_dt - (time.perf_counter() - frame_start)
                    if remaining > 0:
                        time.sleep(remaining)
    except KeyboardInterrupt:
        stop_requested = True
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)
    why = "interrupted" if stop_requested else "window closed"
    print(f"viewer {why} at t={d.time:.3f} s after {sim.n_steps} steps", flush=True)


# --------------------------------------------------------------------------- CLI

def build_simulation(args: argparse.Namespace) -> Simulation:
    timing = TimingConfig(timestep=args.timestep, control_period=args.control_period, render_hz=args.render_hz)
    model = load_model(timing)
    arm = ArmIndex.from_model(model)
    if args.controller == "force":
        cfg = ForceControlConfig(force_desired=args.force, approach_speed=args.approach_speed,
                                 creep_speed=args.creep_speed, creep_distance=args.creep_distance,
                                 kp_force=args.kp_force, ki_force=args.ki_force,
                                 slide_distance=args.slide_distance, slide_direction=tuple(args.slide_direction),
                                 slide_speed=args.slide_speed, slide_cycles=args.slide_cycles, slide_dwell=args.slide_dwell,
                                 motion_freq_hz=args.motion_freq, rotation_freq_hz=args.rotation_freq,
                                 ki_tangential=args.ki_tangential, ki_rotation=args.ki_rotation,
                                 friction_ff=args.friction_ff, friction_viscous_ff=args.friction_viscous_ff)
        controller = ForceApproachController(model, arm, timing.control_period, cfg)
    else:
        controller = HoldingController.inertia_scaled(model, arm, home_qpos(model, arm), args.hold_freq_hz, args.hold_zeta)
    sim = Simulation(model, timing, controller)
    if args.init_pos is not None:
        rpy = args.init_rpy
        quat = args.init_quat
        result = sim.set_initial_pose(args.init_site, args.init_pos, quat, rpy_deg=rpy)
        print(f"initial state from IK ({args.init_site} at {args.init_pos}): {result.summary()}", flush=True)
    return sim


def run_ik(args: argparse.Namespace) -> int:
    model = load_model()
    sim = Simulation(model)
    if args.self_test:
        return 0 if ik_self_test(sim) else 1
    if args.pos is None:
        print("ik: --pos is required (or --self-test)")
        return 2
    quat = quat_from_rpy(*args.rpy) if args.rpy is not None else args.quat
    result = solve_site_ik(model, args.site, args.pos, quat, arm=sim.arm, q_seed=sim.q_home, q_pref=sim.q_home)
    print(result.summary())
    if not result.converged:
        return 1
    sim.set_home(result.q)
    pos, q_now = site_pose(model, sim.data, args.site)
    print(f"{args.site}: pos {np.array2string(pos, precision=4)} quat(wxyz) {np.array2string(q_now, precision=4)}; "
          f"contacts at this pose: {sim.data.ncon}")
    for pair in [(model.geom(c.geom1).name, model.geom(c.geom2).name) for c in sim.data.contact[:sim.data.ncon]]:
        print("  contact:", pair)
    print(f"keyframe line: <key name=\"home\" qpos=\"{' '.join(f'{v:.6g}' for v in result.q)}\"/>")
    if args.render is not None:
        render_overview(sim, args.render)
        print(f"wrote {args.render}")
    return 0 if sim.data.ncon == 0 else 1


def ik_self_test(sim: Simulation, n_random: int = 50) -> bool:
    """Round-trip, random reachable targets, position-only, unreachable target, joint limits."""
    model, arm = sim.model, sim.arm
    lo, hi = model.jnt_range[arm.joint_ids].T
    ok = True
    rng = np.random.default_rng(1)
    data = mujoco.MjData(model)

    def fk(q, site):
        data.qpos[arm.qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        return site_pose(model, data, site)

    # 1. round trip: home pose recovered from a seed perturbed by up to 0.3 rad per joint
    pos, quat = fk(sim.q_home, TCP_SITE)
    res = solve_site_ik(model, TCP_SITE, pos, quat, arm=arm, q_seed=sim.q_home + rng.uniform(-0.3, 0.3, 7), q_pref=sim.q_home)
    p2, _ = fk(res.q, TCP_SITE)
    print(f"round trip          : {res.summary()}\n                      |q - q_home| max {np.abs(res.q - sim.q_home).max():.2e} rad, "
          f"FK pos err {np.linalg.norm(p2 - pos):.2e} m")
    ok &= res.converged and np.abs(res.q - sim.q_home).max() < 1e-3
    # 2. random reachable 6-DoF targets for several sites, seeded from home
    for site in (TCP_SITE, FLANGE_SITE):
        n_ok, worst_p, worst_o, iters = 0, 0.0, 0.0, []
        for _ in range(n_random):
            q_rand = lo + 0.1 * (hi - lo) + rng.random(7) * 0.8 * (hi - lo)
            pos, quat = fk(q_rand, site)
            res = solve_site_ik(model, site, pos, quat, arm=arm, q_seed=sim.q_home, q_pref=sim.q_home)
            n_ok += res.converged
            worst_p, worst_o = max(worst_p, res.pos_err), max(worst_o, res.ori_err)
            iters.append(res.iterations)
            ok &= np.all(res.q >= lo) and np.all(res.q <= hi)
        print(f"random targets      : site {site:7s} {n_ok}/{n_random} converged, worst residual {worst_p:.1e} m / {worst_o:.1e} rad, "
              f"iterations median {int(np.median(iters))} max {max(iters)}")
        ok &= n_ok >= 0.95 * n_random
    # 3. position-only
    res = solve_site_ik(model, TCP_SITE, [0.45, 0.2, 0.35], None, arm=arm, q_seed=sim.q_home)
    print(f"position only       : {res.summary()}")
    ok &= res.converged
    # 4. unreachable
    res = solve_site_ik(model, TCP_SITE, [1.6, 0.0, 0.3], quat_from_rpy(0, 180, 0), arm=arm, q_seed=sim.q_home)
    print(f"unreachable (1.6 m) : {res.summary()}")
    ok &= not res.converged
    # 5. set_initial_pose end to end, then restore
    q_before = sim.q_home.copy()
    res = sim.set_initial_pose(TCP_SITE, [0.60, 0.15, 0.30], rpy_deg=(0, 180, 0))
    pos, quat = site_pose(model, sim.data, TCP_SITE)
    print(f"set_initial_pose    : {res.summary()}\n                      tcp now at {np.array2string(pos, precision=4)}, "
          f"contacts {sim.data.ncon}, t={sim.data.time}, keyframe updated: "
          f"{np.allclose(model.key_qpos[model.key('home').id, arm.qpos_adr], res.q)}")
    ok &= bool(np.linalg.norm(pos - [0.60, 0.15, 0.30]) < 1e-4 and sim.data.ncon == 0)
    try:
        sim.set_initial_pose(TCP_SITE, [0.60, 0.15, 0.145], rpy_deg=(0, 180, 0))  # tip 5 mm into the tabletop
        print("contact rejection   : FAILED (pose below the table was accepted)")
        ok = False
    except ValueError as exc:
        print(f"contact rejection   : OK ({str(exc)[:80]}...)")
        ok &= bool(np.allclose(sim.q_home, res.q))   # restored to the previous valid home
    sim.set_home(q_before)
    print("== result ==\n" + ("PASS" if ok else "FAIL"))
    return bool(ok)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kuka_table_sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timestep", type=float, default=TimingConfig.timestep, help="physics step, s")
    p.add_argument("--control-period", type=float, default=TimingConfig.control_period, help="holding update period, simulated s")
    p.add_argument("--render-hz", type=float, default=TimingConfig.render_hz, help="viewer frame rate target")
    p.add_argument("--hold-freq-hz", type=float, default=DEFAULT_HOLD_FREQ_HZ,
                   help="holding-loop natural frequency per joint (Kp_i = M_ii wn^2)")
    p.add_argument("--hold-zeta", type=float, default=DEFAULT_HOLD_ZETA,
                   help="holding-loop damping ratio per joint (Kd_i = 2 zeta M_ii wn)")
    p.add_argument("--controller", choices=("hold", "force"), default="hold",
                   help="hold: joint-space posture hold; force: approach the table and hold a normal force")
    fc = ForceControlConfig()
    p.add_argument("--force", type=float, default=fc.force_desired, help="[force] desired normal force, N")
    p.add_argument("--approach-speed", type=float, default=fc.approach_speed, help="[force] peak descent speed, m/s")
    p.add_argument("--creep-speed", type=float, default=fc.creep_speed, help="[force] final approach speed until contact, m/s")
    p.add_argument("--creep-distance", type=float, default=fc.creep_distance,
                   help="[force] height above the expected surface where the creep starts, m")
    p.add_argument("--slide-distance", type=float, default=fc.slide_distance, help="[force] stroke length, m (0 = no sliding)")
    p.add_argument("--slide-direction", type=float, nargs=3, metavar=("X", "Y", "Z"), default=list(fc.slide_direction),
                   help="[force] world direction of the stroke (projected onto the table plane)")
    p.add_argument("--slide-speed", type=float, default=fc.slide_speed, help="[force] peak tangential speed, m/s")
    p.add_argument("--slide-cycles", type=int, default=fc.slide_cycles, help="[force] out-and-back cycles, 0 = unlimited")
    p.add_argument("--slide-dwell", type=float, default=fc.slide_dwell, help="[force] pause at each stroke end, s")
    p.add_argument("--motion-freq", type=float, default=fc.motion_freq_hz, help="[force] translation impedance natural frequency, Hz")
    p.add_argument("--rotation-freq", type=float, default=fc.rotation_freq_hz, help="[force] orientation impedance natural frequency, Hz")
    p.add_argument("--ki-tangential", type=float, default=fc.ki_tangential, help="[force] tangential integral gain while in contact, 1/s^3")
    p.add_argument("--ki-rotation", type=float, default=fc.ki_rotation, help="[force] orientation integral gain while in contact, 1/s^3")
    p.add_argument("--friction-ff", type=float, default=fc.friction_ff, help="[force] nominal Coulomb coefficient for the tangential feedforward")
    p.add_argument("--friction-viscous-ff", type=float, default=fc.friction_viscous_ff,
                   help="[force] nominal viscous friction coefficient (N*s/m) for the tangential feedforward")
    p.add_argument("--kp-force", type=float, default=fc.kp_force, help="[force] proportional force gain")
    p.add_argument("--ki-force", type=float, default=fc.ki_force, help="[force] integral force gain, 1/s")
    p.add_argument("--init-site", default=TCP_SITE, help="site whose pose defines the initial state (with --init-pos)")
    p.add_argument("--init-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                   help="initial state: solve IK so --init-site is at this world position (m)")
    p.add_argument("--init-quat", type=float, nargs=4, metavar=("W", "X", "Y", "Z"), default=None,
                   help="orientation for --init-pos as a quaternion (wxyz); omit both --init-quat/--init-rpy to leave it free")
    p.add_argument("--init-rpy", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"), default=None,
                   help="orientation for --init-pos as roll/pitch/yaw in degrees, R = Rz(yaw) Ry(pitch) Rx(roll)")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("headless", help="run without GUI and print the validation report")
    h.add_argument("--duration", type=float, default=10.0, help="simulated seconds")
    h.add_argument("--render", type=Path, default=None, help="write an offscreen PNG of the final state")
    h.add_argument("--log", type=Path, default=None, help="[force] write the compressed RunLog (.npz, float32)")
    h.add_argument("--plot", type=Path, default=None, help="[force] write a force/position PNG (needs matplotlib)")
    h.add_argument("--quiet", action="store_true")
    v = sub.add_parser("viewer", help="interactive passive viewer (mjpython on macOS)")
    v.add_argument("--no-realtime", action="store_true", help="do not pace frames to wall-clock time")
    v.add_argument("--camera", choices=("free", "overview"), default="free",
                   help="start on the mouse-controllable free camera (default) or the fixed overview camera")
    k = sub.add_parser("ik", help="solve IK for a site pose and print the joint configuration")
    k.add_argument("--site", default=TCP_SITE)
    k.add_argument("--pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None)
    k.add_argument("--quat", type=float, nargs=4, metavar=("W", "X", "Y", "Z"), default=None)
    k.add_argument("--rpy", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"), default=None, help="degrees")
    k.add_argument("--render", type=Path, default=None, help="render the solved pose to a PNG")
    k.add_argument("--self-test", action="store_true", help="run the IK consistency checks")
    args = p.parse_args(argv)
    if args.init_pos is not None and args.init_quat is not None and args.init_rpy is not None:
        p.error("use either --init-quat or --init-rpy")
    if args.cmd == "ik":
        return run_ik(args)

    sim = build_simulation(args)
    if args.cmd == "headless":
        if args.controller == "force":
            ok = run_force_headless(sim, args.duration, args.render, args.log, args.plot, verbose=not args.quiet)
            return 0 if ok else 1
        return 0 if run_headless(sim, args.duration, args.render, verbose=not args.quiet) else 1
    run_viewer(sim, realtime=not args.no_realtime, camera=args.camera)
    return 0


if __name__ == "__main__":
    sys.exit(main())
