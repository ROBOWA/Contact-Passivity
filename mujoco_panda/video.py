"""Render a saved Panda log to a PowerPoint-compatible H.264 video."""

from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

from .ports import PandaHandles
from .simulation import load_model
from .visualization import add_force_arrow, add_reference_marker


def render(log_path: str, output: str, location="tool", fps=30,
           width=960, height=720):
    log = dict(np.load(log_path))
    model, data, h = load_model()
    renderer = mujoco.Renderer(model, height=height, width=width)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    writer = imageio.get_writer(output, fps=fps, codec="libx264",
                                pixelformat="yuv420p", quality=8,
                                macro_block_size=None)
    stride = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    site = h.tool_site if location == "tool" else h.forearm_site
    try:
        for k in range(0, len(log["time"]), stride):
            data.qpos[h.qpos_ids] = [log[f"q{i}"][k] for i in range(1, 8)]
            data.qvel[h.dof_ids] = [log[f"qd{i}"][k] for i in range(1, 8)]
            data.ctrl[h.actuator_ids] = [log[f"tau{i}"][k] for i in range(1, 8)]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="overview", scene_option=option)
            force = np.array([log["human_fx"][k], log["human_fy"][k], log["human_fz"][k]])
            add_force_arrow(renderer.scene, data.site_xpos[site], force,
                            nominal=max(7., np.linalg.norm(force)))
            add_reference_marker(renderer.scene,
                                 [log["x_des"][k], log["y_des"][k], log["z"][k]])
            writer.append_data(renderer.render())
    finally:
        writer.close(); renderer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log"); p.add_argument("output")
    p.add_argument("--location", choices=("tool", "forearm"), default="tool")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    render(args.log, args.output, args.location, args.fps)


if __name__ == "__main__": main()
