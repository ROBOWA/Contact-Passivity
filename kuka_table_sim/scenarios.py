"""Reproducible scenarios for the contact-baseline repairs.

    python -m kuka_table_sim.scenarios nominal                    # 25 s, 10 N, 0.10 m strokes at 0.05 m/s
    python -m kuka_table_sim.scenarios free-air-static
    python -m kuka_table_sim.scenarios free-air-dynamic           # joint excitation, no contact
    python -m kuka_table_sim.scenarios fast-approach              # 1 m/s free-air descent, then genuine touchdown
    python -m kuka_table_sim.scenarios loss-short                 # brief interruption (UNCERTAIN, no LOST)
    python -m kuka_table_sim.scenarios loss-reacquire             # loss -> bounded SEARCH -> reacquisition -> sliding resumes
    python -m kuka_table_sim.scenarios loss-sustained             # loss -> SEARCH -> HOLD, then resume -> reacquisition
    python -m kuka_table_sim.scenarios removal-stress             # STRESS TEST, not a benchmark: table moved away
    python -m kuka_table_sim.scenarios timestep-study             # nominal at dt and dt/2, same control period

Each command prints a report, and writes a compressed RunLog (--log) and a plot (--plot) if asked.

Loss fixture (declared before any result is judged). An operator pulls the WRIST upward with a smooth
force on link7, i.e. above the FT sensor, so the sensor sees only what a real sensor would: the contact force
and the inertial/gravity loads of the tool. The table is never moved or teleported. Pull: 60 N, 0.1 s
cosine ramps, applied while the first stroke is at about 55 mm (t = 12 s).
    (the pull envelopes are chosen by the PHYSICAL outcome they produce, see LOSS_FIXTURES)
The declared recovery limits are the ForceControlConfig defaults: reference speed <= 0.05 m/s, reference
acceleration <= 0.30 m/s^2, search time <= 4 s, search distance <= 0.03 m below the start, stop time 0.6 s.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import metrics as mx
from .force_control import CODE_PHASE, ForceApproachController, ForceControlConfig
from .holding import HoldingController, Simulation
from .recorder import Recorder, RunLog
from .scene import ArmIndex, TimingConfig, home_qpos, load_model


# --------------------------------------------------------------------------- fixtures

@dataclass
class WristPull:
    """Operator pulls the wrist (link7) with a smooth force: cosine ramp up, hold, cosine ramp down."""

    t_start: float
    hold: float
    force: float = 60.0
    ramp: float = 0.1
    direction: tuple = (0.0, 0.0, 1.0)
    body: str = "link7"

    def envelope(self, t: float) -> float:
        r = self.ramp
        if t < self.t_start or t > self.t_start + self.hold + 2 * r:
            return 0.0
        if t < self.t_start + r:
            return 0.5 - 0.5 * np.cos(np.pi * (t - self.t_start) / r)
        if t < self.t_start + r + self.hold:
            return 1.0
        return 0.5 + 0.5 * np.cos(np.pi * (t - self.t_start - r - self.hold) / r)

    @property
    def t_end(self) -> float:
        return self.t_start + self.hold + 2 * self.ramp

    def bind(self, model: mujoco.MjModel):
        bid = model.body(self.body).id
        d = np.asarray(self.direction, float)

        def apply(t_k: float, data: mujoco.MjData) -> None:
            data.xfrc_applied[bid, :3] = self.force * self.envelope(t_k) * d
        return apply


@dataclass
class ScenarioResult:
    name: str
    sim: Simulation
    log: RunLog
    info: dict = field(default_factory=dict)


def make_force_sim(config: ForceControlConfig | None = None, timestep: float = TimingConfig.timestep,
                   control_period: float = TimingConfig.control_period, duration: float = 25.0,
                   record: bool = True, legacy_mixed: bool = False) -> Simulation:
    timing = TimingConfig(timestep=timestep, control_period=control_period)
    model = load_model(timing)
    arm = ArmIndex.from_model(model)
    ctl = ForceApproachController(model, arm, timing.control_period, config)
    sim = Simulation(model, timing, ctl)
    if record:
        sim.recorder = Recorder(sim, duration, legacy_mixed=legacy_mixed)
        sim.reset()
    return sim


def run_for(sim: Simulation, duration: float, on_step=None) -> RunLog | None:
    n = sim.steps_for(duration)
    ci = sim.timing.control_interval
    for _ in range(n):
        if on_step is not None and sim.n_steps % ci == 0:
            on_step(sim)
        sim.step()
    return sim.recorder.finish() if sim.recorder is not None else None


def run_nominal(duration: float = 25.0, config: ForceControlConfig | None = None, timestep: float = TimingConfig.timestep,
                control_period: float = TimingConfig.control_period, record: bool = True, legacy_mixed: bool = False) -> ScenarioResult:
    sim = make_force_sim(config, timestep, control_period, duration, record, legacy_mixed)
    log = run_for(sim, duration)
    return ScenarioResult("nominal", sim, log)


def run_free_air(kind: str, duration: float | None = None) -> ScenarioResult:
    """static / dynamic: joint-space hold with (dynamic) sinusoidal targets and no table contact.
    fast: the force controller with a 1 m/s peak approach speed, run through the genuine touchdown."""
    if kind == "fast":
        cfg = ForceControlConfig(approach_speed=1.0, slide_distance=0.0)
        duration = duration or 12.0
        sim = make_force_sim(cfg, duration=duration)
        log = run_for(sim, duration)
        return ScenarioResult("fast-approach", sim, log, dict(approach_speed=1.0))
    timing = TimingConfig()
    model = load_model(timing)
    arm = ArmIndex.from_model(model)
    q_home = home_qpos(model, arm)
    ctl = HoldingController.inertia_scaled(model, arm, q_home)
    sim = Simulation(model, timing, ctl)
    duration = duration or (3.0 if kind == "static" else 6.0)
    sim.recorder = Recorder(sim, duration)
    sim.reset()
    amp = np.array([0.0, 0.10, 0.0, -0.12, 0.0, 0.25, 0.8])
    freq = np.array([0.0, 1.7, 0.0, 2.3, 0.0, 3.1, 2.2])

    def excite(s: Simulation) -> None:
        if kind == "dynamic":
            s.controller.q_target = q_home + amp * np.sin(2 * np.pi * freq * s.data.time)

    log = run_for(sim, duration, excite)
    return ScenarioResult(f"free-air-{kind}", sim, log, dict(amp=amp.tolist(), freq=freq.tolist()))


# Declared fixtures. The pull envelope is chosen by the PHYSICAL outcome it produces (calibrated once with the
# default controller and not tuned afterwards): `short` keeps the tip off the surface for less than the 0.3 s
# loss persistence, `reacquire` for about 0.7 s (LOST escalation, then reacquisition), `sustained` for longer
# than the 4 s search budget.
LOSS_FIXTURES = {
    "short": dict(force=11.0, ramp=0.05, hold=0.0, duration=22.0),
    "reacquire": dict(force=40.0, ramp=0.1, hold=0.5, duration=26.0),
    "sustained": dict(force=40.0, ramp=0.1, hold=8.0, duration=40.0, resume_after_pull=1.0),
}


def run_loss(kind: str, config: ForceControlConfig | None = None, t_pull: float = 12.0) -> ScenarioResult:
    fx = LOSS_FIXTURES[kind]
    duration = fx["duration"]
    sim = make_force_sim(config, duration=duration)
    pull = WristPull(t_start=t_pull, hold=fx["hold"], force=fx["force"], ramp=fx["ramp"])
    sim.disturbance = pull.bind(sim.model)
    t_resume = pull.t_end + fx["resume_after_pull"] if "resume_after_pull" in fx else None
    done = {"resumed": False}

    def hook(s: Simulation) -> None:
        if t_resume is not None and not done["resumed"] and s.data.time >= t_resume:
            s.controller.request_resume()
            done["resumed"] = True

    log = run_for(sim, duration, hook)
    return ScenarioResult(f"loss-{kind}", sim, log, dict(pull=pull.__dict__, t_pull_end=pull.t_end, t_resume=t_resume))


def run_removal_stress(config: ForceControlConfig | None = None, t_remove: float = 12.0, duration: float = 24.0) -> ScenarioResult:
    """STRESS TEST, not a reacquisition benchmark: the whole table body is moved 1 m down at t_remove."""
    sim = make_force_sim(config, duration=duration)
    tb = sim.model.body("table").id
    state = {"done": False}

    def hook(s: Simulation) -> None:
        if not state["done"] and s.data.time >= t_remove:
            s.model.body_pos[tb][2] -= 1.0
            state["done"] = True

    log = run_for(sim, duration, hook)
    return ScenarioResult("removal-stress", sim, log, dict(t_remove=t_remove, label="STRESS TEST: table teleported away"))


# --------------------------------------------------------------------------- reports

def _fmt(d: dict) -> str:
    return json.dumps(d, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def nominal_gates(res: ScenarioResult) -> tuple[dict, dict]:
    """Steady-contact gates. Sliding configuration (definitions kept from the baseline, evaluated over the whole
    SLIDE phase): mean |f_d - f| < 0.5 N, max < 3 N, along-path error < max(5 mm, 0.1 s * slide speed),
    cross-path < 3 mm, orientation < 2 deg. Hold-only configuration (slide_distance = 0): window from the end of
    the force ramp plus 1 s, mean < 0.2 N, max < 0.5 N, tangential drift < 3 mm, orientation < 1 deg. Both: no
    unexpected contacts, no actuator saturation, no recovery/loss events, sensor estimate vs truth < 0.3 N."""
    log = res.log
    cfg = res.sim.controller.cfg
    tt = mx.phase_times(log)
    t_end = float(log.phys["t"][-1])
    sliding = cfg.slide_distance > 0
    t0 = tt["slide"] if sliding and tt["slide"] is not None else (
        (tt["contact"] + cfg.force_ramp_time + 1.0) if tt["contact"] is not None else t_end)
    win = (t0, t_end)
    f = mx.force_stats(log, win)
    tr = mx.tracking_stats(log, win, cfg.slide_direction)
    tq = mx.torque_stats(log)
    ev = mx.fsm_events(log)
    along_tol = max(5.0, 100.0 * cfg.slide_speed)
    gates = {}
    if sliding:
        gates.update({
            "mean |f_d - f| < 0.5 N": f["truth_mean_abs"] < 0.5,
            "max |f_d - f| < 3 N": f["truth_max_abs"] < 3.0,
            f"along-path < {along_tol:.1f} mm": tr["along_max_mm"] < along_tol,
            "cross-path < 3 mm": tr["cross_max_mm"] < 3.0,
            "orientation < 2 deg": tr["ori_max_deg"] < 2.0})
    else:
        drift = float(np.max(np.linalg.norm(log.ctl["x"][mx._sel(log.ctl["t"], win)][:, :2] - log.ctl["x"][mx._sel(log.ctl["t"], win)][0, :2], axis=1)) * 1e3)
        gates.update({
            "mean |f_d - f| < 0.2 N": f["truth_mean_abs"] < 0.2,
            "max |f_d - f| < 0.5 N": f["truth_max_abs"] < 0.5,
            "tangential drift < 3 mm": drift < 3.0,
            "orientation < 1 deg": tr["ori_max_deg"] < 1.0})
    gates.update({
        "no unexpected contacts": int(log.phys["n_unexpected"].sum()) == 0,
        "no actuator saturation": tq["saturated_substeps"] == 0,
        "no recovery/loss events": ev.get("n_uncertain", 0) == 0 and ev.get("n_lost", 0) == 0,
        "sensor vs truth < 0.3 N": f["sensor_vs_truth_max"] < 0.3,
        "contact detected": tt["contact"] is not None})
    return gates, dict(force=f, tracking=tr, torque=tq, window=win)


def report_nominal(res: ScenarioResult) -> bool:
    log = res.log
    gates, st = nominal_gates(res)
    w = st["window"]
    print(f"== nominal: {log.meta['n_physics_steps']} physics steps, {log.meta['n_control_updates']} control updates, "
          f"dt {log.meta['timestep']} s, control period {log.meta['control_period']} s, mujoco {log.meta['mujoco']}, commit {log.meta['commit'][:10]}"
          f"{' (dirty)' if log.meta['dirty'] else ''}")
    print(f"transitions: {[(round(t, 3), a, b) for t, a, b in mx.transitions(log)]}")
    f, tr, tq = st["force"], st["tracking"], st["torque"]
    print(f"steady-sliding window t in [{w[0]:.2f}, {w[1]:.2f}] s")
    print(f"  force   : truth |f_d - f| mean {f['truth_mean_abs']:.4f} N max {f['truth_max_abs']:.4f} N; controller |f_d - f_n| mean "
          f"{f['controller_mean_abs']:.4f} max {f['controller_max_abs']:.4f}; f_true mean {f['f_true_mean']:.4f} std {f['f_true_std']:.4f}; "
          f"sensor vs truth max {f['sensor_vs_truth_max']:.4f} N")
    print(f"  tracking: along max {tr['along_max_mm']:.2f} mm (rms {tr['along_rms_mm']:.2f}), cross max {tr['cross_max_mm']:.3f} mm, orientation max {tr['ori_max_deg']:.3f} deg")
    print(f"  torque  : peak applied utilisation {tq['peak_applied_utilisation'] * 100:.0f} % of limit, saturated substeps {tq['saturated_substeps']}")
    a = mx.contact_absence(log, w)
    print(f"  contact : geometric absence {a['geometric_absence']['events']} events (longest {a['geometric_absence']['longest_s']:.4f} s, "
          f"fraction {a['geometric_absence']['fraction']:.4f}); low load (<{a['low_load_threshold_n']} N) {a['low_load']['events']} events "
          f"(longest {a['low_load']['longest_s']:.4f} s, fraction {a['low_load']['fraction']:.4f}); FSM {mx.fsm_events(log, w)}")
    wk = mx.work_stats(log, w)
    print(f"  work    : environment on robot in the window: rect {wk['rect_J']:+.5f} J, trapezoid {wk['trap_J']:+.5f} J, "
          f"force(k) x velocity(k+1) {wk['next_J']:+.5f} J (positive part {wk['positive_J']:+.5f}, negative part {wk['negative_J']:+.5f})")
    p = log.phys
    both = p["n_task"] > 0
    if both.any():
        print(f"  identity: P_task vs qvel . qfrc_constraint max |diff| {np.abs(p['P_task'][both] - p['P_constraint'][both]).max():.2e} W")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(gates.values())
    print("== result:", "PASS" if ok else "FAIL")
    return ok


def report_free_air(res: ScenarioResult, contact_threshold: float = 1.0, samples: int = 3) -> bool:
    """False-contact analysis. A 'false contact' is the estimate exceeding the contact threshold for `samples`
    consecutive control intervals while NO physical tip/table contact exists (before the first one for the
    fast approach)."""
    log = res.log
    p = log.phys
    t = p["t"]
    ci = log.meta["control_interval"]
    first_true = float(t[np.argmax(p["present"])]) if p["present"].any() else None
    free = t < first_true if first_true is not None else np.ones(len(t), bool)
    grav, corr, truth = p["ft_grav"][:, 2], p["ft_corr"][:, 2], p["F_task"][:, 2]

    def false_runs(sig):
        return [(a, b) for a, b in mx.runs((sig > contact_threshold) & free) if b - a >= samples * ci]

    acc = np.linalg.norm(p["a_com"][free], axis=1)
    print(f"== {res.name}: {len(t)} physics steps; first physical tip/table contact at t = {first_true}")
    print(f"  free-air part: tool COM acceleration |a| max {acc.max():.2f} m/s^2")
    print(f"  normal force estimate (N), free-air part: gravity-only max {grav[free].max():+.3f} min {grav[free].min():+.3f}; "
          f"corrected max {corr[free].max():+.3e} min {corr[free].min():+.3e}; ground truth max {truth[free].max():.3f}")
    gf, cf = false_runs(grav), false_runs(corr)
    print(f"  would-be false CONTACT (estimate > {contact_threshold} N for >= {samples} control intervals in free air): gravity-only {len(gf)} time(s), corrected {len(cf)} time(s)")
    ok = len(cf) == 0 and float(np.max(np.abs(corr[free]))) < 1e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] corrected estimate stays below 1 mN in free air, no false CONTACT")
    if res.name == "fast-approach":
        tt = mx.transitions(log)
        t_contact = next((a for a, _, b in tt if b == "contact"), None)
        genuine = t_contact is not None and first_true is not None and t_contact >= first_true
        lat = (t_contact - first_true) * 1e3 if genuine else float("nan")
        print(f"  transitions {[(round(a, 3), b, c) for a, b, c in tt]}")
        print(f"  genuine touchdown: CONTACT entered {lat:.2f} ms after the first physical contact; peak corrected force {corr.max():.2f} N, truth {truth.max():.2f} N")
        print(f"  [{'PASS' if genuine else 'FAIL'}] CONTACT entered only after genuine contact")
        ok &= genuine
    else:
        ok &= int(p["present"].sum()) == 0
    print("== result:", "PASS" if ok else "FAIL")
    return ok


def _loss_windows(res: ScenarioResult) -> dict:
    pull_end = res.info["t_pull_end"]
    t_pull = res.info["pull"]["t_start"]
    tt = mx.phase_times(res.log)
    t_slide = tt["slide"]
    t_end = float(res.log.phys["t"][-1])
    return dict(pre=(t_slide + 1.0, t_pull - 0.2), disturbance=(t_pull, pull_end), t_end=t_end)


def report_loss(res: ScenarioResult) -> bool:
    log = res.log
    cfg = res.sim.controller.cfg
    w = _loss_windows(res)
    t_pull = res.info["pull"]["t_start"]
    pe = res.info["t_pull_end"]
    tt = mx.transitions(log)
    print(f"== {res.name}: pull {res.info['pull']['force']:.0f} N hold {res.info['pull']['hold']} s from t = {t_pull} s to {pe:.2f} s"
          + (f"; resume requested at {res.info['t_resume']:.2f} s" if res.info.get("t_resume") else ""))
    print(f"  FSM transitions: {[(round(a, 3), b, c) for a, b, c in tt]}")
    print(f"  recovery (SEARCH/HOLD) intervals: {[(round(a, 3), round(b, 3)) for a, b in mx.recovery_intervals(log)]}")
    ev = mx.fsm_events(log, (t_pull - 0.5, w['t_end']))
    ab = mx.contact_absence(log, (t_pull - 0.5, w["t_end"]))
    print(f"  FSM events (decisions): {ev}")
    print(f"  physical, after the pull starts: geometric absence {ab['geometric_absence']['events']} event(s), longest {ab['geometric_absence']['longest_s']:.3f} s, "
          f"total {ab['geometric_absence']['total_s']:.3f} s, first at {ab['geometric_absence']['first_t']}")
    print(f"                                   low load (<{ab['low_load_threshold_n']} N) {ab['low_load']['events']} event(s), longest {ab['low_load']['longest_s']:.3f} s, total {ab['low_load']['total_s']:.3f} s")
    rc = mx.reference_continuity(log, (t_pull - 0.5, w["t_end"]))
    tg, nm = rc["tangential"], rc["normal"]
    print(f"  reference continuity (control updates): tangential step max {tg['pos_step_max_mm']:.4f} mm/ms (peak speed {tg['peak_speed_fd']:.4f} m/s, peak accel {tg['peak_accel_fd']:.3f} m/s^2, "
          f"pos residual {tg['pos_residual_max_mm']:.4f} mm, vel residual {tg['vel_residual_max_mm_s']:.3f} mm/s)")
    print(f"      normal      step max {nm['pos_step_max_mm']:.4f} mm/ms (peak speed {nm['peak_speed_fd']:.4f} m/s, peak accel {nm['peak_accel_fd']:.3f} m/s^2, pos residual {nm['pos_residual_max_mm']:.4f} mm)")
    print(f"      normal position-loop activations: {[(round(a['t'], 3), round(a['pos_err_mm'], 2), round(a['vel_err_mm_s'], 1)) for a in rc['normal_activation']]}  (t, pos err mm, vel err mm/s)")
    fp = mx.force_peaks(log, (t_pull - 0.5, w["t_end"]))
    print(f"  force peaks (true contact force, physics rate): overall {fp['peak_N']:.2f} N; first 0.5 s after each CONTACT entry: "
          f"{[(t, round(f, 2)) for t, f in fp['peak_after_contact_entry']]} (target {cfg.force_desired} N)")
    tq = mx.torque_stats(log, (t_pull - 0.5, w["t_end"]))
    print(f"  torque: peak applied utilisation {tq['peak_applied_utilisation'] * 100:.0f} %, commanded {tq['peak_commanded_utilisation'] * 100:.0f} %, saturated substeps {tq['saturated_substeps']}")
    c = log.ctl
    print(f"  integrators: max |f_int| {np.nanmax(np.abs(c['f_int'])):.2f} N (clamp {cfg.integral_limit / cfg.ki_force:.1f}), max |tan_int| {np.nanmax(np.linalg.norm(c['tan_int'], axis=1)):.5f}")
    after = log.phys["t"] >= t_pull
    exc = (np.max(log.phys["tcp"][after, 2]) - res.sim.controller.z_surface) * 1e3
    print(f"  tip height after the pull starts: max {exc:.1f} mm above the expected contact height, min {np.min(log.phys['tcp'][after, 2]):.4f} m "
          f"(expected contact height {res.sim.controller.z_surface:.4f}, search floor limit {res.sim.controller.z_floor:.4f})")
    # gates (declared before the result was judged)
    gates: dict[str, bool] = {}
    gates["no actuator saturation"] = tq["saturated_substeps"] == 0
    gates["tangential reference step <= declared speed * dt + 0.01 mm"] = tg["pos_step_max_mm"] <= cfg.recovery_speed_max * log.meta["control_period"] * 1e3 * 1.05 + 0.01
    gates["tangential reference accel <= 1.1 * declared a_max"] = tg["peak_accel_fd"] <= 1.1 * cfg.recovery_accel_max
    gates["normal reference accel <= 1.1 * declared a_max"] = nm["peak_accel_fd"] <= 1.1 * cfg.recovery_accel_max or nm["peak_accel_fd"] == 0.0
    gates["normal position-loop activation error < 2 mm"] = all(abs(a["pos_err_mm"]) < 2.0 for a in rc["normal_activation"])
    gates["no unexpected contacts"] = int(log.phys["n_unexpected"].sum()) == 0
    kind = res.name
    final = CODE_PHASE[int(c["phase"][-1])]
    if kind == "loss-short":
        gates["no LOST escalation"] = ev.get("n_lost", 0) == 0
        gates["interruption handled by one SEARCH that reacquired"] = ev.get("n_uncertain", 0) == 1 and ev.get("n_reacquired", 0) == 1
        gates["sliding resumed"] = final == "slide"
    elif kind == "loss-reacquire":
        gates["one LOST escalation, one reacquisition"] = ev.get("n_lost", 0) == 1 and ev.get("n_reacquired", 0) == 1
        gates["no HOLD"] = ev.get("n_hold", 0) == 0
        gates["sliding resumed"] = final == "slide"
    elif kind == "loss-sustained":
        gates["LOST then HOLD"] = ev.get("n_lost", 0) >= 1 and ev.get("n_hold", 0) >= 1
        gates["reacquired after resume"] = ev.get("n_reacquired", 0) >= 1 and final == "slide"
    if final == "slide":
        t_final_slide = max((a for a, _, b in tt if b == "slide"), default=None)
        if t_final_slide is not None and w["t_end"] - t_final_slide > 4.0:
            post = (t_final_slide + 3.0, w["t_end"])
            f = mx.force_stats(log, post)
            gates["post-recovery steady force error < 0.5 N (mean)"] = f["truth_mean_abs"] < 0.5
            print(f"  post-recovery window [{post[0]:.2f}, {post[1]:.2f}] s: mean |f_d - f| {f['truth_mean_abs']:.4f} N, max {f['truth_max_abs']:.4f} N")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(gates.values())
    print("== result:", "PASS" if ok else "FAIL")
    return ok


def report_removal(res: ScenarioResult) -> bool:
    log = res.log
    tt = mx.transitions(log)
    ctl = res.sim.controller
    t_rm = res.info["t_remove"]
    rc = mx.reference_continuity(log, (t_rm - 0.5, float(log.phys["t"][-1])))
    tq = mx.torque_stats(log, (t_rm - 0.5, float(log.phys["t"][-1])))
    print(f"== removal-stress ({res.info['label']}) at t = {t_rm} s")
    print(f"  FSM transitions: {[(round(a, 3), b, c) for a, b, c in tt]}; events {mx.fsm_events(log, (t_rm - 0.5, float(log.phys['t'][-1])))}")
    print(f"  tip z: min {np.min(log.phys['tcp'][:, 2]):.4f} m, final {log.phys['tcp'][-1, 2]:.4f} m (expected contact height {ctl.z_surface:.4f}, floor {ctl.z_floor:.4f})")
    print(f"  reference: tangential step max {rc['tangential']['pos_step_max_mm']:.4f} mm/ms; normal step max {rc['normal']['pos_step_max_mm']:.4f} mm/ms, "
          f"activations {[(round(a['t'], 3), round(a['pos_err_mm'], 2)) for a in rc['normal_activation']]}")
    print(f"  torque: peak applied {tq['peak_applied_utilisation'] * 100:.0f} %, saturated substeps {tq['saturated_substeps']}; max tip height after removal {np.max(log.phys['tcp'][log.phys['t'] > t_rm, 2]):.4f} m")
    ok = CODE_PHASE[int(log.ctl["phase"][-1])] == "hold" and tq["saturated_substeps"] == 0
    print("== result (bounded search then controlled HOLD, no saturation):", "PASS" if ok else "FAIL")
    return ok


def run_timestep_study(duration: float = 25.0) -> dict:
    """Nominal run at dt and dt/2 with the SAME control period; both evaluated on the same window (the later of
    the two slide starts to the end) so a one-update difference in the slide start does not enter the comparison."""
    runs_ = {}
    for dt in (TimingConfig.timestep, TimingConfig.timestep / 2):
        runs_[dt] = run_nominal(duration, timestep=dt, control_period=TimingConfig.control_period)
    t0 = max(mx.phase_times(r.log)["slide"] for r in runs_.values())
    out = {}
    for dt, res in runs_.items():
        log = res.log
        win = (t0, float(log.phys["t"][-1]) - 1e-9)
        out[dt] = dict(work=mx.work_stats(log, win), force=mx.force_stats(log, win),
                       tracking=mx.tracking_stats(log, win, res.sim.controller.cfg.slide_direction), t_slide=mx.phase_times(log)["slide"],
                       t_contact=mx.phase_times(log)["contact"], window=win, res=res)
    return out


def report_timestep(study: dict) -> None:
    (dt1, a), (dt2, b) = list(study.items())
    print(f"== timestep study: dt {dt1} s vs {dt2} s, control period {TimingConfig.control_period} s fixed (same control interval in time); "
          f"common window [{a['window'][0]:.3f}, {a['window'][1]:.3f}] s")
    for key, lab in (("rect_J", "rectangle"), ("trap_J", "trapezoid"), ("next_J", "force(k) x velocity(k+1)")):
        x1, x2 = a["work"][key], b["work"][key]
        print(f"  contact work, sliding window ({lab:22s}): {x1:+.5f} J  vs  {x2:+.5f} J   diff {x2 - x1:+.2e} J ({abs(x2 - x1) / abs(x1) * 100:.3f} %)")
    for dt, r in study.items():
        w = r["work"]
        print(f"  dt {dt}: quadrature spread at fixed dt (max-min of three rules) {max(w['rect_J'], w['trap_J'], w['next_J']) - min(w['rect_J'], w['trap_J'], w['next_J']):.2e} J")
    for key in ("along_max_mm", "along_rms_mm", "cross_max_mm", "ori_max_deg"):
        print(f"  tracking {key:13s}: {a['tracking'][key]:.4f} vs {b['tracking'][key]:.4f}")
    for key in ("truth_mean_abs", "truth_max_abs", "f_true_std"):
        print(f"  force    {key:14s}: {a['force'][key]:.5f} vs {b['force'][key]:.5f} N")
    print(f"  contact time: {a['t_contact']:.4f} vs {b['t_contact']:.4f} s; slide start {a['t_slide']:.4f} vs {b['t_slide']:.4f} s")


# --------------------------------------------------------------------------- CLI

NAMES = ["nominal", "free-air-static", "free-air-dynamic", "fast-approach", "loss-short", "loss-reacquire", "loss-sustained",
         "removal-stress", "timestep-study"]


def run_by_name(name: str) -> tuple[ScenarioResult | dict, bool]:
    if name == "nominal":
        r = run_nominal(legacy_mixed=True)
        return r, report_nominal(r)
    if name == "free-air-static":
        r = run_free_air("static")
        return r, report_free_air(r)
    if name == "free-air-dynamic":
        r = run_free_air("dynamic")
        return r, report_free_air(r)
    if name == "fast-approach":
        r = run_free_air("fast")
        return r, report_free_air(r)
    if name.startswith("loss-"):
        r = run_loss(name[5:])
        return r, report_loss(r)
    if name == "removal-stress":
        r = run_removal_stress()
        return r, report_removal(r)
    if name == "timestep-study":
        s = run_timestep_study()
        report_timestep(s)
        return s, True
    raise SystemExit(f"unknown scenario {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=NAMES)
    ap.add_argument("--log", type=Path, default=None, help="write the RunLog (.npz, compressed)")
    ap.add_argument("--log-stride", type=int, default=1, help="keep every n-th physics row when saving")
    ap.add_argument("--log-float32", action="store_true", help="save float32 (except time and work)")
    ap.add_argument("--log-window", type=float, nargs=2, default=None, metavar=("T0", "T1"), help="save only this time window")
    ap.add_argument("--plot", type=Path, default=None, help="write a PNG (needs matplotlib)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    res, ok = run_by_name(args.scenario)
    print(f"(wall {time.perf_counter() - t0:.1f} s)")
    if isinstance(res, ScenarioResult):
        if args.log is not None:
            res.log.save(args.log, args.log_stride, tuple(args.log_window) if args.log_window else None, args.log_float32)
            print(f"wrote {args.log}")
        if args.plot is not None:
            from .plot import plot_scenario
            plot_scenario(res, args.plot)
            print(f"wrote {args.plot}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
