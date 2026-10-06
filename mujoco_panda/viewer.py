"""Interactive real-time replay of a saved Panda simulation log."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from .simulation import load_model
from .visualization import add_force_arrow, add_reference_marker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", help="log.npz produced by mujoco_panda.experiments")
    p.add_argument("--location", choices=("tool", "forearm"), default="tool")
    args = p.parse_args(); log = dict(np.load(args.log))
    model, data, h = load_model(); site = h.tool_site if args.location == "tool" else h.forearm_site
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [.52, 0, .48]; viewer.cam.distance = 1.55
            viewer.cam.azimuth = 135; viewer.cam.elevation = -20
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        wall = time.perf_counter()
        for k, sim_time in enumerate(log["time"]):
            if not viewer.is_running(): break
            data.qpos[h.qpos_ids] = [log[f"q{i}"][k] for i in range(1, 8)]
            data.qvel[h.dof_ids] = [log[f"qd{i}"][k] for i in range(1, 8)]
            data.ctrl[h.actuator_ids] = [log[f"tau{i}"][k] for i in range(1, 8)]
            mujoco.mj_forward(model, data)
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                force = np.array([log["human_fx"][k], log["human_fy"][k], log["human_fz"][k]])
                add_force_arrow(viewer.user_scn, data.site_xpos[site], force,
                                nominal=max(7., np.linalg.norm(force)))
                add_reference_marker(viewer.user_scn,
                                     [log["x_des"][k], log["y_des"][k], log["z"][k]])
            viewer.sync()
            delay = float(sim_time) - (time.perf_counter() - wall)
            if delay > 0: time.sleep(delay)


if __name__ == "__main__": main()
