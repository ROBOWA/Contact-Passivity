"""Headless Panda simulation, sampled ledgers, and physical-work audit."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from mujoco_sliding.energy_ledgers import EnergyLedger

from .config import MODEL_XML_PATH, PandaConfig
from .controller import PandaController, Phase, orientation_error
from .passivity import (TorquePassivityFilter, affine_dynamics,
                        predict_task_port)
from .ports import (PandaHandles, apply_spatial_port, assert_power_identity,
                    extract_task_contact, site_jacobian, spatial_site_port)


def human_wrench(t: float, cfg) -> np.ndarray:
    if not cfg.enabled:
        return np.zeros(6)
    direction = np.asarray(cfg.direction, dtype=float)
    direction /= max(np.linalg.norm(direction), 1e-12)
    u = t - cfg.start
    total = cfg.rise + cfg.hold + cfg.fall
    if u < 0.0 or u >= total:
        scale = 0.0
    elif u < cfg.rise:
        scale = 0.5 * (1.0 - np.cos(np.pi * u / cfg.rise))
    elif u < cfg.rise + cfg.hold:
        scale = 1.0
    else:
        scale = 0.5 * (1.0 + np.cos(np.pi * (u - cfg.rise - cfg.hold) / cfg.fall))
    return np.r_[cfg.magnitude * scale * direction, np.zeros(3)]


@dataclass
class RunMetrics:
    mode: str
    predictor: str
    trajectory: str
    location: str
    normal_force_rmse: float
    cartesian_rmse_m: float
    orientation_rmse_deg: float
    contact_fraction: float
    task_progress_m: float
    governed_progress_m: float
    max_progress_lag_m: float
    governor_min_scale: float
    human_input_J: float
    human_output_J: float
    human_net_J: float
    human_physical_work_J: float
    human_work_discrepancy_J: float
    peak_human_output_W: float
    residual_net_J: float
    sampled_task_work_J: float
    physical_task_work_J: float
    task_work_discrepancy_J: float
    max_qvel_prediction_error: float
    prediction_bound_violations: int
    ledger_floor_violations: int
    human_floor_violations: int
    residual_floor_violations: int
    whole_floor_violations: int
    qp_intervention_steps: int
    qp_fallback_steps: int
    torque_limit_steps: int
    torque_rate_limit_steps: int
    joint_limit_violations: int
    mean_solve_time_ms: float
    max_solve_time_ms: float
    slide_acceleration_p99_m_s2: float
    slide_jerk_p99_m_s3: float
    slide_torque_rate_p99_Nm_s: float
    nominal_targets_passed: bool


@dataclass
class SimulationResult:
    config: PandaConfig
    log: dict[str, np.ndarray]
    metrics: RunMetrics

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.savez_compressed(os.path.join(directory, "log.npz"), **self.log)
        keys = list(self.log)
        # NPZ retains every 1 ms physics sample. CSV is decimated to 100 Hz
        # for human review and repository size; its final row is included.
        csv_indices = np.unique(np.r_[np.arange(0, len(self.log["time"]), 10),
                                     len(self.log["time"]) - 1])
        with open(os.path.join(directory, "log.csv"), "w", newline="") as f:
            writer = csv.writer(f); writer.writerow(keys)
            for i in csv_indices:
                writer.writerow([self.log[k][i] for k in keys])
        with open(os.path.join(directory, "metrics.json"), "w") as f:
            json.dump(asdict(self.metrics), f, indent=2)
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)


def load_model():
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    handles = PandaHandles.from_model(model)
    # The XML compiles both layouts so the single-contact ablation can be
    # enabled at runtime. Multipoint is the public/default model state.
    model.geom_contype[handles.single_pad_geom] = 0
    model.geom_conaffinity[handles.single_pad_geom] = 0
    mujoco.mj_forward(model, data)
    return model, data, handles


def run(config: PandaConfig | None = None) -> SimulationResult:
    cfg = config or PandaConfig()
    model, data, h = load_model()
    model.opt.timestep = cfg.plant.timestep
    if cfg.plant.contact_layout not in ("multipoint", "single"):
        raise ValueError(f"unknown contact layout {cfg.plant.contact_layout!r}")
    active_pads = (h.multipoint_pad_geoms if cfg.plant.contact_layout == "multipoint"
                   else np.array([h.single_pad_geom]))
    model.geom_contype[h.pad_geoms] = 0
    model.geom_conaffinity[h.pad_geoms] = 0
    model.geom_contype[active_pads] = 1
    model.geom_conaffinity[active_pads] = 1
    for geom in (*h.pad_geoms, h.table_geom):
        model.geom_solref[geom] = [cfg.plant.contact_time_constant,
                                   cfg.plant.contact_damping_ratio]
        impedance = np.asarray(cfg.plant.contact_impedance, dtype=float)
        model.geom_solimp[geom, :len(impedance)] = impedance
        model.geom_margin[geom] = cfg.plant.contact_margin
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    controller = PandaController(model, data, h, cfg.controller)
    # Establish gravity compensation as the initialized previous command;
    # the simulated run begins from this explicit actuator state.
    initial = data.qfrc_bias[h.dof_ids].copy()
    data.ctrl[h.actuator_ids] = initial
    filt = TorquePassivityFilter(model, h, cfg.controller,
                                 cfg.passivation, initial)
    ledgers = {
        "residual": EnergyLedger(cfg.passivation.residual, "residual"),
        "human": EnergyLedger(cfg.passivation.human, "human"),
        "whole": EnergyLedger(cfg.passivation.whole, "whole"),
    }
    fields = [
        "time", "phase", "contact", "contact_solver", "contact_points", "gap", "normal_force", "normal_force_filtered", "force_desired",
        "x", "y", "z", "vx", "vy", "vz", "x_des", "y_des", "cart_error", "orientation_error_deg",
        "original_progress", "governed_progress", "governor_scale",
        "p_task_sample", "p_task_physical", "p_human", "p_human_physical", "p_residual", "p_whole",
        "human_fx", "human_fy", "human_fz",
        "w_task_sample", "w_task_physical", "w_human", "w_human_physical", "w_residual",
        "e_human", "e_residual", "e_whole", "human_input", "human_output",
        "peak_human_output", "qvel_prediction_error", "prediction_bound_violation",
        "qp_active", "qp_fallback", "qp_status", "energy_row_active", "minimum_margin", "candidate_minimum_margin",
        "torque_limited", "rate_limited", "solve_time_ms", "joint_margin_min",
    ] + [f"q{i+1}" for i in range(7)] + [f"qd{i+1}" for i in range(7)] \
      + [f"tau{i+1}" for i in range(7)] + [f"tau_nom{i+1}" for i in range(7)]
    log = {k: [] for k in fields}
    cumulative = dict(task_sample=0.0, task_physical=0.0, human=0.0,
                      human_physical=0.0,
                      residual=0.0, human_input=0.0, human_output=0.0,
                      peak_human_output=0.0)
    protect_prev = False
    nsteps = int(np.ceil(cfg.duration / dt))
    for _ in range(nsteps):
        t = float(data.time)
        data.qfrc_applied[:] = 0.0
        site_id = h.tool_site if cfg.human.location == "tool" else h.forearm_site
        human = spatial_site_port(model, data, site_id, human_wrench(t, cfg.human))
        assert_power_identity(human, data.qvel)
        apply_spatial_port(model, data, human)
        mujoco.mj_forward(model, data)
        contact = extract_task_contact(model, data, h)
        ctl = controller.update(data, contact,
                                human_present=np.linalg.norm(human.wrench[:3]) > 0.05,
                                protect_active=protect_prev)
        ports = predict_task_port(model, data, h, contact, human, cfg.passivation)
        dyn = affine_dynamics(model, data, h, ports.external,
                              cfg.passivation.dynamics_mass_scale)
        energies = {name: ledger.e for name, ledger in ledgers.items()}
        filtered = filt.solve(data, ctl.tau_nominal, dyn, ports, energies)
        data.ctrl[:] = 0.0
        data.ctrl[h.actuator_ids] = filtered.torque
        # qfrc_applied already contains precisely this held human wrench.
        p_task_before = contact.power
        mujoco.mj_step(model, data)
        qd_next = data.qvel.copy()
        qerr = float(np.linalg.norm(qd_next - filtered.qvel_predicted))
        sample = {
            "task": float(ports.task_actual @ qd_next),
            "human": float(ports.human @ qd_next),
            "residual": float(ports.residual @ qd_next),
            "whole": float(ports.external @ qd_next),
        }
        updates = {name: ledgers[name].update(sample[name], dt)
                   for name in ("human", "residual", "whole")}
        # Post-step power is recomputed from every actual contact wrench and
        # point velocity. Trapezoidal work is a diagnostic independent of the
        # held-sample controller ledger.
        # mj_step ends after integration; forward the new state so positions,
        # Jacobians, and constraint wrenches share the post-step timestamp.
        mujoco.mj_forward(model, data)
        contact_after = extract_task_contact(model, data, h)
        p_task_physical = 0.5 * (p_task_before + contact_after.power)
        human_after = spatial_site_port(model, data, site_id, human.wrench)
        p_human_physical = 0.5 * (human.power + human_after.power)
        cumulative["task_sample"] += dt * sample["task"]
        cumulative["task_physical"] += dt * p_task_physical
        cumulative["human"] += dt * sample["human"]
        cumulative["human_physical"] += dt * p_human_physical
        cumulative["residual"] += dt * sample["residual"]
        cumulative["human_input"] += dt * max(sample["human"], 0.0)
        cumulative["human_output"] += dt * max(-sample["human"], 0.0)
        cumulative["peak_human_output"] = max(cumulative["peak_human_output"],
                                               max(-sample["human"], 0.0))
        jp, _ = site_jacobian(model, data, h.tool_site)
        pnow = data.site_xpos[h.tool_site]
        vnow = jp @ data.qvel
        rnow = data.site_xmat[h.tool_site].reshape(3, 3)
        rotdeg = np.degrees(np.linalg.norm(orientation_error(rnow, controller.r_des)))
        cart_error = np.linalg.norm(pnow[:2] - ctl.desired_position[:2])
        q = data.qpos[h.qpos_ids]; qd = data.qvel[h.dof_ids]
        qlo = model.jnt_range[h.joint_ids, 0]; qhi = model.jnt_range[h.joint_ids, 1]
        jmargin = min(np.min(q - qlo), np.min(qhi - q))
        maintained = (contact_after.gap <= cfg.plant.contact_margin + 2e-5 and
                      controller.filtered_normal_force > 0.5)
        vals = {
            "time": data.time, "phase": int(ctl.phase), "contact": int(maintained),
            "contact_solver": int(contact_after.active),
            "contact_points": len(contact_after.points),
            "gap": contact_after.gap, "normal_force": contact_after.normal_force,
            "normal_force_filtered": controller.filtered_normal_force,
            "force_desired": ctl.desired_force, "x": pnow[0], "y": pnow[1], "z": pnow[2],
            "vx": vnow[0], "vy": vnow[1], "vz": vnow[2],
            "x_des": ctl.desired_position[0], "y_des": ctl.desired_position[1],
            "cart_error": cart_error, "orientation_error_deg": rotdeg,
            "original_progress": ctl.original_progress,
            "governed_progress": ctl.governed_progress, "governor_scale": ctl.governor_scale,
            "p_task_sample": sample["task"], "p_task_physical": p_task_physical,
            "p_human": sample["human"], "p_human_physical": p_human_physical,
            "p_residual": sample["residual"], "p_whole": sample["whole"],
            "human_fx": human.wrench[0], "human_fy": human.wrench[1], "human_fz": human.wrench[2],
            "w_task_sample": cumulative["task_sample"], "w_task_physical": cumulative["task_physical"],
            "w_human": cumulative["human"], "w_human_physical": cumulative["human_physical"],
            "w_residual": cumulative["residual"],
            "e_human": ledgers["human"].e, "e_residual": ledgers["residual"].e,
            "e_whole": ledgers["whole"].e, "human_input": cumulative["human_input"],
            "human_output": cumulative["human_output"],
            "peak_human_output": cumulative["peak_human_output"],
            "qvel_prediction_error": qerr,
            "prediction_bound_violation": int(qerr > cfg.passivation.qvel_prediction_bound),
            "qp_active": int(filtered.active), "qp_fallback": int(filtered.fallback),
            "qp_status": filtered.solver_status_val,
            "energy_row_active": int(filtered.energy_row_active),
            "minimum_margin": filtered.minimum_margin,
            "candidate_minimum_margin": filtered.candidate_minimum_margin,
            "torque_limited": int(filtered.torque_limited), "rate_limited": int(filtered.rate_limited),
            "solve_time_ms": 1e3 * filtered.solve_time, "joint_margin_min": jmargin,
        }
        vals.update({f"q{i+1}": q[i] for i in range(7)})
        vals.update({f"qd{i+1}": qd[i] for i in range(7)})
        vals.update({f"tau{i+1}": filtered.torque[i] for i in range(7)})
        vals.update({f"tau_nom{i+1}": ctl.tau_nominal[i] for i in range(7)})
        for key in fields:
            log[key].append(vals[key])
        protect_prev = filtered.energy_row_active

    arrays = {k: np.asarray(v) for k, v in log.items()}
    steady = (arrays["phase"] == int(Phase.SLIDE)) & (arrays["force_desired"] > 0.99 * cfg.controller.f_desired)
    if not np.any(steady):
        steady = arrays["time"] > 0.5 * cfg.duration
    fn_rmse = float(np.sqrt(np.mean((arrays["normal_force_filtered"][steady] - cfg.controller.f_desired) ** 2)))
    cart_rmse = float(np.sqrt(np.mean(arrays["cart_error"][steady] ** 2)))
    rot_rmse = float(np.sqrt(np.mean(arrays["orientation_error_deg"][steady] ** 2)))
    contact_fraction = float(np.mean(arrays["contact"][steady]))
    velocity = np.c_[arrays["vx"], arrays["vy"], arrays["vz"]]
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    torque = np.column_stack([arrays[f"tau{i}"] for i in range(1, 8)])
    torque_rate = np.diff(torque, axis=0) / dt
    slide_times = arrays["time"][arrays["phase"] == int(Phase.SLIDE)]
    established_time = ((slide_times[0] if len(slide_times) else cfg.duration / 2)
                        + cfg.controller.slide_ramp_duration + 0.2)
    established_2 = ((arrays["time"][:-2] >= established_time)
                     & (arrays["phase"][:-2] == int(Phase.SLIDE)))
    established_1 = ((arrays["time"][:-1] >= established_time)
                     & (arrays["phase"][:-1] == int(Phase.SLIDE)))
    accel_p99 = (float(np.percentile(np.linalg.norm(acceleration[:-1][established_2], axis=1), 99))
                  if np.any(established_2) else float("nan"))
    jerk_p99 = (float(np.percentile(np.linalg.norm(jerk[established_2], axis=1), 99))
                if np.any(established_2) else float("nan"))
    torque_rate_p99 = (float(np.percentile(np.linalg.norm(torque_rate[established_1], axis=1), 99))
                       if np.any(established_1) else float("nan"))
    human_floors = arrays["e_human"] < cfg.passivation.human.e_min - 1e-9
    residual_floors = arrays["e_residual"] < cfg.passivation.residual.e_min - 1e-9
    whole_floors = arrays["e_whole"] < cfg.passivation.whole.e_min - 1e-9
    if cfg.passivation.mode == "C1_whole_port_qp":
        protected_floors = whole_floors
    elif cfg.passivation.mode == "C2_residual_qp":
        protected_floors = residual_floors
    elif cfg.passivation.mode == "C3_dual_ledger_qp":
        protected_floors = residual_floors | human_floors
    else:
        protected_floors = residual_floors | human_floors
    metrics = RunMetrics(
        cfg.passivation.mode, cfg.passivation.predictor, cfg.controller.trajectory,
        cfg.human.location, fn_rmse, cart_rmse, rot_rmse, contact_fraction,
        float(arrays["original_progress"][-1]), float(arrays["governed_progress"][-1]),
        float(np.max(arrays["original_progress"] - arrays["governed_progress"])),
        float(np.min(arrays["governor_scale"])),
        cumulative["human_input"], cumulative["human_output"], cumulative["human"],
        cumulative["human_physical"], cumulative["human_physical"] - cumulative["human"],
        cumulative["peak_human_output"], cumulative["residual"],
        cumulative["task_sample"], cumulative["task_physical"],
        cumulative["task_physical"] - cumulative["task_sample"],
        float(np.max(arrays["qvel_prediction_error"])),
        int(np.sum(arrays["prediction_bound_violation"])), int(np.sum(protected_floors)),
        int(np.sum(human_floors)), int(np.sum(residual_floors)), int(np.sum(whole_floors)),
        int(np.sum(arrays["qp_active"])), int(np.sum(arrays["qp_fallback"])),
        int(np.sum(arrays["torque_limited"])), int(np.sum(arrays["rate_limited"])),
        int(np.sum(arrays["joint_margin_min"] < 0.0)),
        float(np.mean(arrays["solve_time_ms"])), float(np.max(arrays["solve_time_ms"])),
        accel_p99, jerk_p99, torque_rate_p99,
        bool(fn_rmse <= 0.5 and cart_rmse <= 0.005 and rot_rmse <= 3.0 and
             contact_fraction >= 0.99 and not np.any(arrays["joint_margin_min"] < 0.0)))
    return SimulationResult(cfg, arrays, metrics)
