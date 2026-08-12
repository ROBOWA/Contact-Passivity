"""Interactive MuJoCo viewer running the same controller in real time.

Usage:
    python -m mujoco_sliding.viewer
    python -m mujoco_sliding.viewer --human-force

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

from .config import SimulationConfig
from .contact_extraction import ModelHandles, extract_task_contact
from .controller import SlidingForceController
from .human_interaction import HumanForce
from .simulation import load_model, reset_to_keyframe


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Interactive contact-sliding viewer")
    parser.add_argument("--human-force", action="store_true")
    parser.add_argument("--duration", type=float, default=float("inf"))
    args = parser.parse_args(argv)

    cfg = SimulationConfig()
    if args.human_force:
        cfg = cfg.with_human_force()

    model = load_model()
    data = mujoco.MjData(model)
    handles = ModelHandles.from_model(model)
    controller = SlidingForceController(model, handles, cfg.controller)
    human = HumanForce(cfg.human)
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
            out = controller.update(data, contact)
            data.ctrl[:] = out.tau
            data.qfrc_applied[:] = 0.0
            f_h = human.force(data.time)
            human.apply(model, data, handles, f_h)
            mujoco.mj_step(model, data)
            viewer.sync()

            status = (
                f"phase={out.phase.name:8s} F_n={contact.f_n:6.2f} N "
                f"human={'ON ' if np.any(f_h) else 'off'}"
            )
            if status != last_status:
                print(f"t={data.time:7.3f} s  {status}")
                last_status = status

            # Real-time pacing.
            lag = data.time - (time.time() - t_wall0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
