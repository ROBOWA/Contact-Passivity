"""Interactive MuJoCo viewer running the same controller in real time.

Usage:
    python -m mujoco_sliding.viewer
    python -m mujoco_sliding.viewer --human-force
    python -m mujoco_sliding.viewer --scenario C --mode C3_dual_ledger_qp

With --scenario, the viewer replays one of the scripted passivation
experiments (mujoco_sliding.experiments) in real time, including the
selective-passivation layer selected by --mode; stdout additionally reports
the ledger states.

Requires a display; the headless pipeline and the tests never import the
viewer. Contact points and contact-force arrows are enabled in the
visualization options, and the window title area (via stdout) reports the
controller phase and human-force status.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from .config import PASSIVATION_MODES, SimulationConfig
from .contact_extraction import ModelHandles, extract_task_contact
from .controller import SlidingForceController
from .human_interaction import HumanForce, HumanForceSequence
from .simulation import load_model, reset_to_keyframe


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Interactive contact-sliding viewer")
    parser.add_argument("--human-force", action="store_true")
    parser.add_argument("--duration", type=float, default=float("inf"))
    parser.add_argument("--scenario", default=None,
                        help="passivation scenario key (A-F); see experiments")
    parser.add_argument("--mode", default="C3_dual_ledger_qp",
                        choices=PASSIVATION_MODES,
                        help="controller mode used with --scenario")
    args = parser.parse_args(argv)

    if args.scenario is not None:
        from .experiments import SCENARIOS, build_config

        spec = SCENARIOS[args.scenario]
        cfg = build_config(spec, args.mode)
        if args.duration == float("inf"):
            args.duration = spec.duration
    else:
        cfg = SimulationConfig()
        if args.human_force:
            cfg = cfg.with_human_force()

    model = load_model()
    data = mujoco.MjData(model)
    handles = ModelHandles.from_model(model)
    controller = SlidingForceController(model, handles, cfg.controller)
    if cfg.human_pulses:
        human = HumanForceSequence([cfg.human, *cfg.human_pulses])
    else:
        human = HumanForce(cfg.human)
    runtime = None
    if cfg.passivation is not None:
        from .passivity_qp import PassivationRuntime

        runtime = PassivationRuntime(cfg.passivation, cfg.controller,
                                     handles, model.opt.timestep)
    reset_to_keyframe(model, data)

    last_status = ""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.cam.lookat[:] = [0.5, 0.0, 0.35]
            viewer.cam.distance = 2.2
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 90

        t_wall0 = time.time()
        while viewer.is_running() and data.time < args.duration:
            contact = extract_task_contact(model, data, handles)
            f_h = human.force(data.time)
            if runtime is None:
                out = controller.update(data, contact)
                tau = out.tau
            else:
                out, tau = runtime.control_step(model, data, contact, f_h,
                                                controller)
            data.ctrl[:] = tau
            data.qfrc_applied[:] = 0.0
            human.apply(model, data, handles, f_h)
            mujoco.mj_step(model, data)
            if runtime is not None:
                runtime.post_step(model, data)
            viewer.sync()

            status = (
                f"phase={out.phase.name:8s} F_n={contact.f_n:6.2f} N "
                f"human={'ON ' if np.any(f_h) else 'off'}"
            )
            if runtime is not None:
                # 3 decimals: coarse enough to keep stdout reporting sparse.
                status += (f" E_H={runtime.ledger_h.e:6.3f} J "
                           f"E_R={runtime.ledger_r.e:6.3f} J")
            if status != last_status:
                print(f"t={data.time:7.3f} s  {status}")
                last_status = status

            # Real-time pacing.
            lag = data.time - (time.time() - t_wall0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
