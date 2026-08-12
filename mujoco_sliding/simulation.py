"""Headless simulation loop, logging, and CLI entry point.

Loop structure and contact-force timing
---------------------------------------
``mj_forward`` is called once after reset so the very first controller step
sees a dynamically consistent state. Inside the loop, the contact forces
read at the top of iteration k were solved during the previous dynamics
evaluation (the ``mj_step`` of iteration k-1, or the initial ``mj_forward``),
i.e. the controller and the log use the contact force with a ONE-STEP
(1 ms) delay relative to the logged state. This is documented, deliberate,
and irrelevant at 1 kHz; it avoids re-solving the dynamics twice per step
and guarantees no stale-force reads beyond that single step.

Per step:
  1. read state + extract pad-surface contact (ground-truth task channel)
  2. compute controller torques
  3. clear ``qfrc_applied``, apply the known human force f_h at the EE site
  4. log everything
  5. ``mj_step``

Usage:
    python -m mujoco_sliding.simulation
    python -m mujoco_sliding.simulation --human-force
    python -m mujoco_sliding.simulation --duration 8 --output-dir results
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import mujoco
import numpy as np

from .config import MODEL_XML_PATH, SimulationConfig
from .contact_extraction import ModelHandles, extract_task_contact, site_jacobian
from .controller import Phase, SlidingForceController
from .human_interaction import HumanForce


def load_model() -> mujoco.MjModel:
    """Load the packaged planar two-link MJCF model."""
    return mujoco.MjModel.from_xml_path(MODEL_XML_PATH)


def reset_to_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset to the 'init' keyframe and initialize dynamics with mj_forward."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)


def run_simulation(cfg: SimulationConfig | None = None) -> dict:
    """Run the simulation headlessly and return the full log as arrays."""
    cfg = cfg or SimulationConfig()
    model = load_model()
    data = mujoco.MjData(model)
    handles = ModelHandles.from_model(model)
    controller = SlidingForceController(model, handles, cfg.controller)
    human = HumanForce(cfg.human)

    reset_to_keyframe(model, data)

    nsteps = int(round(cfg.duration / model.opt.timestep))
    log: dict[str, list] = {k: [] for k in _LOG_KEYS}

    for _ in range(nsteps):
        contact = extract_task_contact(model, data, handles)
        out = controller.update(data, contact)

        data.ctrl[:] = out.tau
        data.qfrc_applied[:] = 0.0
        f_h = human.force(data.time)
        human.apply(model, data, handles, f_h)

        # --- log (state at time t; contact force one step delayed) ---
        p_ee = data.site_xpos[handles.ee_site].copy()
        jp = site_jacobian(model, data, handles.ee_site)
        v_ee = jp @ data.qvel
        f_measured = contact.f_task + f_h
        p_h_to_r = float(f_h @ v_ee)

        log["time"].append(data.time)
        log["phase"].append(int(out.phase))
        log["qpos"].append(data.qpos.copy())
        log["qvel"].append(data.qvel.copy())
        log["ctrl"].append(out.tau.copy())
        log["tau_applied"].append(data.qfrc_actuator.copy())
        log["tau_bias"].append(out.tau_bias)
        log["tau_damping"].append(out.tau_damping)
        log["tau_c"].append(out.tau_c)
        log["tau_u"].append(out.tau_u)
        log["ee_pos"].append(p_ee)
        log["ee_vel"].append(v_ee.copy())
        log["x_desired"].append(out.x_desired)
        log["vx_desired"].append(out.v_desired)
        log["f_n_desired"].append(out.f_desired)
        log["f_n"].append(contact.f_n)
        log["f_t"].append(contact.f_t)
        log["f_task"].append(contact.f_task.copy())
        log["f_push"].append(out.f_push)
        log["e_f"].append(out.e_f)
        log["integral"].append(out.integral)
        log["contact_active"].append(contact.active)
        log["ncontacts"].append(contact.ncontacts)
        log["contact_pos"].append(
            contact.positions.mean(axis=0) if contact.active else np.full(3, np.nan)
        )
        log["contact_dist"].append(contact.dist)
        log["penetration"].append(contact.penetration)
        log["contact_normal"].append(
            contact.normals.mean(axis=0) if contact.active else np.full(3, np.nan)
        )
        log["contact_tangent"].append(
            contact.tangents1.mean(axis=0) if contact.active else np.full(3, np.nan)
        )
        log["f_h"].append(f_h.copy())
        log["f_measured"].append(f_measured)
        log["p_task"].append(contact.p_task)
        log["p_h_to_r"].append(p_h_to_r)
        log["p_r_to_h"].append(-p_h_to_r)
        log["p_ctrl"].append(float(out.tau @ data.qvel))
        log["p_damping"].append(float(data.qfrc_passive @ data.qvel))
        log["energy_potential"].append(data.energy[0])
        log["energy_kinetic"].append(data.energy[1])
        log["saturated"].append(out.saturated)

        mujoco.mj_step(model, data)

    arrays = {k: np.asarray(v) for k, v in log.items()}
    arrays["meta_f_desired"] = np.array(cfg.controller.f_desired)
    arrays["meta_v_slide"] = np.array(cfg.controller.v_slide)
    arrays["meta_timestep"] = np.array(model.opt.timestep)
    arrays["meta_base_pos"] = np.array([0.0, 0.0, 0.7])
    arrays["meta_link_lengths"] = np.array([0.6, 0.6])
    arrays["meta_pad_radius"] = np.array(model.geom_size[handles.pad_geom][0])
    return arrays


_LOG_KEYS = [
    "time", "phase", "qpos", "qvel", "ctrl", "tau_applied", "tau_bias",
    "tau_damping", "tau_c", "tau_u", "ee_pos", "ee_vel", "x_desired",
    "vx_desired", "f_n_desired", "f_n", "f_t", "f_task", "f_push", "e_f",
    "integral", "contact_active", "ncontacts", "contact_pos", "contact_dist",
    "penetration", "contact_normal", "contact_tangent", "f_h", "f_measured",
    "p_task", "p_h_to_r", "p_r_to_h", "p_ctrl", "p_damping",
    "energy_potential", "energy_kinetic", "saturated",
]


def save_log(log: dict, output_dir: str, stem: str = "log") -> tuple[str, str]:
    """Save the log as NPZ (full fidelity) and CSV (flat columns)."""
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"{stem}.npz")
    np.savez_compressed(npz_path, **log)

    # Flatten vector channels into per-component CSV columns.
    columns: dict[str, np.ndarray] = {}
    for key, arr in log.items():
        if key.startswith("meta_"):
            continue
        arr = np.asarray(arr)
        if arr.ndim == 1:
            columns[key] = arr
        elif arr.ndim == 2:
            labels = "xyz" if arr.shape[1] == 3 else [str(i) for i in range(arr.shape[1])]
            for j in range(arr.shape[1]):
                columns[f"{key}_{labels[j]}"] = arr[:, j]
    header = ",".join(columns)
    mat = np.column_stack(list(columns.values()))
    csv_path = os.path.join(output_dir, f"{stem}.csv")
    np.savetxt(csv_path, mat, delimiter=",", header=header, comments="")
    return npz_path, csv_path


def steady_slide_metrics(log: dict, skip: float = 1.0) -> dict:
    """Tracking metrics over the sliding phase (skipping its first ``skip`` s)."""
    t = log["time"]
    slide = log["phase"] == int(Phase.SLIDE)
    if not slide.any():
        return {"valid": False}
    t0 = t[slide][0] + skip
    win = slide & (t >= t0)
    if not win.any():
        return {"valid": False}
    f_d = float(log["meta_f_desired"])
    v_d = float(log["meta_v_slide"])
    fn = log["f_n"][win]
    vx = log["ee_vel"][win][:, 0]
    return {
        "valid": True,
        "window": (float(t[win][0]), float(t[win][-1])),
        "mean_f_n": float(fn.mean()),
        "f_n_rel_err": float(abs(fn.mean() - f_d) / f_d),
        "mean_vx": float(vx.mean()),
        "vx_rel_err": float(abs(vx.mean() - v_d) / v_d),
        "contact_fraction": float(log["contact_active"][win].mean()),
        "max_penetration": float(log["penetration"][win].max()),
        "max_abs_tau": float(np.abs(log["ctrl"]).max()),
        "mean_p_task": float(log["p_task"][win].mean()),
        "max_abs_p_h": float(np.abs(log["p_h_to_r"]).max()),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Planar contact-sliding simulation")
    parser.add_argument("--human-force", action="store_true",
                        help="enable the known human disturbance force")
    parser.add_argument("--duration", type=float, default=None, help="simulated seconds")
    parser.add_argument("--output-dir", default=None, help="artifact directory")
    parser.add_argument("--no-plots", action="store_true", help="skip all figures")
    parser.add_argument("--no-animation", action="store_true", help="skip the GIF")
    args = parser.parse_args(argv)

    cfg = SimulationConfig()
    if args.human_force:
        cfg = cfg.with_human_force()
    if args.duration is not None:
        cfg = replace(cfg, duration=args.duration)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)

    print(f"MuJoCo {mujoco.__version__} | duration {cfg.duration:.1f} s | "
          f"human force {'ON' if cfg.human.enabled else 'off'}")
    log = run_simulation(cfg)
    npz_path, csv_path = save_log(log, cfg.output_dir)
    print(f"saved {npz_path}\nsaved {csv_path}")

    m = steady_slide_metrics(log)
    if m.get("valid"):
        print(
            f"steady sliding {m['window'][0]:.2f}-{m['window'][1]:.2f} s: "
            f"F_n {m['mean_f_n']:.3f} N (err {100 * m['f_n_rel_err']:.1f}%), "
            f"v_x {m['mean_vx']:.4f} m/s (err {100 * m['vx_rel_err']:.1f}%), "
            f"contact {100 * m['contact_fraction']:.1f}%, "
            f"max pen {1e3 * m['max_penetration']:.3f} mm, "
            f"max |tau| {m['max_abs_tau']:.2f} N·m"
        )
    else:
        print("WARNING: sliding phase never reached")

    if not args.no_plots:
        from . import plotting

        fig_path = plotting.plot_results(log, cfg.output_dir)
        print(f"saved {fig_path}")
        if not args.no_animation:
            anim_path = plotting.animate(log, cfg.output_dir)
            if anim_path:
                print(f"saved {anim_path}")
            else:
                print("animation export skipped (no GIF writer available)")


if __name__ == "__main__":
    main()
