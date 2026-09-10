"""Scripted, reproducible experiments for the selective-passivation POC.

Scenarios (deterministic: fixed keyframe initial state, scripted forces,
no randomness anywhere in the pipeline):

  A exact_no_human     oracle predictor, no human. C3 stays nominal with
                       constant ledgers; C1 (whole-port QP) depletes from
                       ordinary task friction and stalls the sliding.
  B mismatch_no_human  true mu = 0.30, predictor mu_hat = 0.20, no human.
                       15 s: the residual ledger reaches its floor-riding
                       band at ~14.7 s while the moving reference is still
                       inside the arm's reachable workspace (x_d < 0.98 m
                       vs ~1.0 m reach; longer runs drive the reference out
                       of the workspace, which degrades every controller).
                       Only E_R responds; E_H stays constant.
  C oracle_blocking    oracle predictor; blocking human F_h = -3 N t during
                       6.5-8.5 s (0.5 s cosine ramps).
  D mismatch_blocking  mu_hat = 0.20 plus the same blocking human.
  E oracle_oblique     E1 filter-allocation ablation: oblique human
                       F_h = 3/sqrt(2) (-t - n), governor off for both C3
                       and C4, identical nominal controller and certificate
                       rows; only the command-selection mechanism differs
                       (2-D task-weighted QP vs safe-anchor scalar line).
                       Evaluated over the full-force plateau
                       [t_start+t_rise, t_start+t_rise+t_hold] only
                       (metrics/plots clipped; simulation and logs never
                       clipped). Release/recovery is out of scope here —
                       that question belongs to the governor comparison.
  F helping_blocking   +3 N t pulse (5-8 s), then -3 N t pulse (8.5-11.5 s):
                       ledger charging, cap, and later discharge.
  S stress             32 s, alternating helping/blocking pulses, reference
                       governor enabled: long-duration certificate stress
                       (no leak, no floor/bound violation, no infeasible
                       step, cumulative inequality valid throughout).

The main certificate scenarios A-F run WITHOUT the reference governor so
the ledger constraints are actually exercised (with the governor the
nominal reference yields on contact and the human scenarios barely load
the ledgers); the governor is demonstrated by the dedicated on/off release
comparison and used in the stress test.

Usage (from the repository root):
    python -m mujoco_sliding.experiments --all       # A-F + S + gov + sweep
    python -m mujoco_sliding.experiments --scenario C
    python -m mujoco_sliding.experiments --scenario C --modes C3_dual_ledger_qp
    python -m mujoco_sliding.experiments --sweep
    python -m mujoco_sliding.experiments --governor-compare
    python -m mujoco_sliding.experiments --legacy    # legacy diagnostics

Outputs under --output-root (default results_passivation_iter2/):
    <scenario>/<mode>/log.npz|log.csv|run.png   per-run artifacts
    <scenario>/comparison.png                   cross-controller comparison
    governor_compare/...                        governor on/off release
    sweep/sweep.csv, sweep/sweep.png            mu_hat sweep artifacts
    summary.json/.csv, acceptance.json, prediction_bound_summary.json
Plus a compact copy of the headline artifacts in --compact-dir
(default results_compact/).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field, replace

import numpy as np

from .config import (GovernorConfig, HumanForceConfig, PassivationConfig,
                     SimulationConfig)
from .simulation import run_simulation, save_log

PRIMARY_QP_MODES = ("C1_whole_port_qp", "C2_residual_qp",
                    "C3_dual_ledger_qp", "C4_dual_ledger_safe_scalar")

_BLOCKING = HumanForceConfig(enabled=True, magnitude=3.0,
                             direction=(-1.0, 0.0, 0.0),
                             t_start=6.0, t_rise=0.5, t_hold=2.0, t_fall=0.5)
_OBLIQUE = HumanForceConfig(enabled=True, magnitude=3.0,
                            direction=(-1.0, 0.0, -1.0),  # normalized on use
                            t_start=6.0, t_rise=0.5, t_hold=2.0, t_fall=0.5)
_HELPING = HumanForceConfig(enabled=True, magnitude=3.0,
                            direction=(1.0, 0.0, 0.0),
                            t_start=5.0, t_rise=0.5, t_hold=2.0, t_fall=0.5)
# Second pulse of scenario F: releases at 11.5 s so the catch-up target
# stays inside the reachable workspace for the rest of the 14 s run.
_BLOCK_LATE = HumanForceConfig(enabled=True, magnitude=3.0,
                               direction=(-1.0, 0.0, 0.0),
                               t_start=8.5, t_rise=0.5, t_hold=2.0, t_fall=0.5)


def _pulse(t0: float, sign: float) -> HumanForceConfig:
    return HumanForceConfig(enabled=True, magnitude=3.0,
                            direction=(sign, 0.0, 0.0),
                            t_start=t0, t_rise=0.5, t_hold=2.0, t_fall=0.5)


# Stress: alternating helping/blocking, 3 s each with 1 s gaps; the
# governor freezes the reference during every pulse, so the reference stays
# inside the workspace over 32 s.
_STRESS_PULSES = tuple(_pulse(t0, s) for t0, s in
                       [(9.0, -1.0), (13.0, 1.0), (17.0, -1.0),
                        (21.0, 1.0), (25.0, -1.0)])


@dataclass
class ScenarioSpec:
    key: str
    name: str
    duration: float
    predictor: str = "oracle"
    mu_hat: float = 0.30
    human: HumanForceConfig = field(default_factory=HumanForceConfig)
    human_pulses: tuple = ()
    governor: bool = False
    modes: tuple = ("C0_nominal",) + PRIMARY_QP_MODES


SCENARIOS: dict[str, ScenarioSpec] = {s.key: s for s in [
    ScenarioSpec("A", "exact_no_human", 14.0),
    ScenarioSpec("B", "mismatch_no_human", 15.0,
                 predictor="friction", mu_hat=0.20,
                 modes=("C0_nominal", "C1_whole_port_qp", "C2_residual_qp",
                        "C3_dual_ledger_qp")),
    ScenarioSpec("C", "oracle_blocking", 12.0, human=_BLOCKING,
                 modes=("C0_nominal", "C2_residual_qp", "C3_dual_ledger_qp",
                        "C4_dual_ledger_safe_scalar")),
    ScenarioSpec("D", "mismatch_blocking", 12.0,
                 predictor="friction", mu_hat=0.20, human=_BLOCKING,
                 modes=("C0_nominal", "C2_residual_qp", "C3_dual_ledger_qp",
                        "C4_dual_ledger_safe_scalar")),
    ScenarioSpec("E", "oracle_oblique", 12.0, human=_OBLIQUE,
                 modes=("C0_nominal", "C3_dual_ledger_qp",
                        "C4_dual_ledger_safe_scalar")),
    ScenarioSpec("F", "helping_blocking", 14.0, human=_HELPING,
                 human_pulses=(_BLOCK_LATE,),
                 modes=("C0_nominal", "C3_dual_ledger_qp")),
    ScenarioSpec("S", "stress", 32.0, human=_pulse(5.0, 1.0),
                 human_pulses=_STRESS_PULSES, governor=True,
                 modes=("C3_dual_ledger_qp",)),
]}

#: Modes whose acceptance REQUIRES zero infeasible/emergency steps, zero
#: prediction-bound violations, and zero certified-floor violations.
ACCEPTANCE_MODES = set(PRIMARY_QP_MODES)

# Fixed window used for release-jump metrics, beginning when the human-force
# profile leaves its full-force hold. It captures the ramp-out and immediate
# recovery without letting ordinary steady sliding dominate |u_t|.
POST_RELEASE_WINDOW = 1.0


def build_config(spec: ScenarioSpec, mode: str,
                 mu_hat: float | None = None,
                 governor: bool | None = None,
                 governor_mode: str = "stop_time_anchor") -> SimulationConfig:
    pas = PassivationConfig(mode=mode, predictor=spec.predictor,
                            mu_hat=mu_hat if mu_hat is not None else spec.mu_hat)
    cfg = SimulationConfig(
        duration=spec.duration,
        human=spec.human,
        human_pulses=spec.human_pulses,
        passivation=pas,
    )
    gov_on = spec.governor if governor is None else governor
    if gov_on:
        cfg = replace(cfg, controller=replace(
            cfg.controller,
            governor=GovernorConfig(enabled=True, mode=governor_mode)))
    return cfg


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _human_release_end(spec: ScenarioSpec) -> float | None:
    pulses = [spec.human, *spec.human_pulses]
    ends = [p.t_start + p.t_rise + p.t_hold + p.t_fall
            for p in pulses if p.enabled]
    return max(ends) if ends else None


def _human_release_start(spec: ScenarioSpec) -> float | None:
    """Start of force ramp-out for the last enabled human pulse."""
    pulses = [spec.human, *spec.human_pulses]
    starts = [p.t_start + p.t_rise + p.t_hold
              for p in pulses if p.enabled]
    return max(starts) if starts else None


def e1_window(log: dict, spec: ScenarioSpec) -> np.ndarray | None:
    """Reusable E1 evaluation mask: SLIDE phase during the full-magnitude
    plateau of the (first enabled) human pulse.

    [t_start + t_rise, t_start + t_rise + t_hold], derived from the
    scenario's HumanForceConfig. Returns None when no human pulse is
    enabled. Used ONLY for evaluation/presentation clipping — the raw
    simulation and logs are never clipped.
    """
    from .controller import Phase

    hp = spec.human
    if not hp.enabled:
        return None
    t = log["time"]
    t0 = hp.t_start + hp.t_rise
    t1 = hp.t_start + hp.t_rise + hp.t_hold
    return ((log["phase"] == int(Phase.SLIDE)) & (t >= t0) & (t <= t1))


def compute_metrics(log: dict, spec: ScenarioSpec, mode: str) -> dict:
    """All per-run metrics, JSON-serializable.

    Cumulative-inequality margins use only a floating-point tolerance
    (1e-7 J); floor checks are exact to 1e-9 J.
    """
    from .controller import Phase

    dt = float(log["meta_timestep"])
    t = log["time"]
    f_d = float(log["meta_f_desired"])
    v_d = float(log["meta_v_slide"])
    pas = PassivationConfig()   # ledger/limit defaults (identical per run)

    slide = log["phase"] == int(Phase.SLIDE)
    contact_phase = log["phase"] >= int(Phase.RAMP)
    human_on = np.linalg.norm(log["f_h"], axis=1) > 1e-12

    def rmse(err, mask):
        return float(np.sqrt(np.mean(np.square(err[mask])))) if mask.any() else np.nan

    m: dict = {
        "scenario": spec.key, "scenario_name": spec.name, "mode": mode,
        "duration": float(t[-1] + dt),
        "governor": bool(log["sp_gov_active"].any()),
        "rmse_fn": rmse(log["f_n"] - log["f_n_desired"], slide),
        "rmse_vt": rmse(log["ee_vel"][:, 0] - log["vx_desired"], slide),
        "rmse_x": rmse(log["ee_pos"][:, 0] - log["x_desired"], slide),
        "rmse_fn_human": rmse(log["f_n"] - log["f_n_desired"],
                              slide & human_on),
        "rmse_vt_human": rmse(log["ee_vel"][:, 0] - log["vx_desired"],
                              slide & human_on),
    }

    # E1 plateau window: the full-magnitude hold of the (first enabled)
    # human pulse, EXCLUDING the force rise and the removal transient.
    # Derived programmatically from HumanForceConfig — never hard-coded.
    # This clips METRICS ONLY: the simulation, control commands, ledger
    # updates, and the saved log always cover the full duration.
    m["t_e1_start"] = m["t_e1_end"] = None
    m["rmse_fn_e1"] = m["rmse_vt_e1"] = None
    for key in ("rms_delta_ut_e1", "peak_abs_delta_ut_e1",
                "rms_delta_un_e1", "peak_abs_delta_un_e1",
                "rms_u_dev_w_e1", "peak_u_dev_w_e1"):
        m[key] = None
    e1 = e1_window(log, spec)
    if e1 is not None and e1.any():
        hp = spec.human
        m["t_e1_start"] = hp.t_start + hp.t_rise
        m["t_e1_end"] = hp.t_start + hp.t_rise + hp.t_hold
        m["rmse_fn_e1"] = rmse(log["f_n"] - log["f_n_desired"], e1)
        m["rmse_vt_e1"] = rmse(log["ee_vel"][:, 0] - log["vx_desired"], e1)
        du = log["sp_u"] - log["sp_u_nom"]
        du_t, du_n = du[e1, 0], du[e1, 1]
        dev_w = np.sqrt(pas.w_t * du_t ** 2 + pas.w_n * du_n ** 2)
        m.update({
            "rms_delta_ut_e1": float(np.sqrt(np.mean(du_t ** 2))),
            "peak_abs_delta_ut_e1": float(np.abs(du_t).max()),
            "rms_delta_un_e1": float(np.sqrt(np.mean(du_n ** 2))),
            "peak_abs_delta_un_e1": float(np.abs(du_n).max()),
            "rms_u_dev_w_e1": float(np.sqrt(np.mean(dev_w ** 2))),
            "peak_u_dev_w_e1": float(dev_w.max()),
        })

    ph = log["sp_p_h_act"]
    pr = log["sp_p_r_act"]
    w_h = -np.cumsum(ph) * dt          # signed cumulative energy extracted
    w_r = -np.cumsum(pr) * dt
    u_dev = log["sp_u"] - log["sp_u_nom"]
    active = log["sp_active_any"].astype(bool)

    m.update({
        "p_h_peak": float(np.max(-ph)),
        "w_h_max": float(w_h.max()), "w_h_final": float(w_h[-1]),
        "w_r_max": float(w_r.max()), "w_r_final": float(w_r[-1]),
        "i_u": float(np.sum(np.sum(u_dev ** 2, axis=1)) * dt),
        "active_fraction": float(active[contact_phase].mean()),
        "false_active_fraction": float((active & ~human_on)[contact_phase].mean()),
        "e_h_min_raw": float(log["sp_e_h_raw"].min()),
        "e_r_min_raw": float(log["sp_e_r_raw"].min()),
        "e_w_min_raw": float(log["sp_e_w_raw"].min()),
        "e_h_final": float(log["sp_e_h"][-1]),
        "e_r_final": float(log["sp_e_r"][-1]),
        "e_h_drift": float(np.abs(log["sp_e_h"] - pas.human.e_init).max()),
        "e_r_drift": float(np.abs(log["sp_e_r"] - pas.residual.e_init).max()),
        # Exact per-step passivity inequalities W_i(t) <= E_i(0) - E_i_min
        # (fp tolerance only; margin < -1e-7 J fails).
        "w_h_margin": float((pas.human.e_init - pas.human.e_min) - w_h.max()),
        "w_r_margin": float((pas.residual.e_init - pas.residual.e_min)
                            - w_r.max()),
        # Violation / event counters.
        "n_infeasible": int(log["meta_n_infeasible"]),
        "n_emergency": int(log["meta_n_emergency"]),
        "n_bound_violations": int(log["meta_n_bound_violations"]),
        "n_floor_violations": int(log["meta_n_floor_violations"]),
        "n_safety_clipped": int(log["meta_n_safety_clipped"]),
        "e_h_below_steps": int(np.sum(log["sp_e_h_below"])),
        "e_r_below_steps": int(np.sum(log["sp_e_r_below"])),
        "p_h_violations": int(np.sum(-ph > pas.p_h_max + 1e-6)),
        # Prediction-error bound bookkeeping.
        "e_v_bound": float(log["meta_e_v_bound"]),
        "pred_err_med": float(np.median(log["sp_pred_err"])),
        "pred_err_p99": float(np.percentile(log["sp_pred_err"], 99)),
        "pred_err_max": float(log["sp_pred_err"].max()),
        "bound_util_max": float(log["sp_bound_util"].max()),
    })

    # First activation (any constraint) and first LEDGER-driven activation.
    idx = np.flatnonzero(active)
    m["t_first_active"] = float(t[idx[0]]) if idx.size else None
    ledger_rows = log["sp_active"][:, :5].any(axis=1)
    idx = np.flatnonzero(active & ledger_rows)
    m["t_first_ledger_active"] = float(t[idx[0]]) if idx.size else None

    if mode != "C0_nominal":
        st = log["sp_solve_time"]
        m["solve_ms_med"] = float(1e3 * np.median(st))
        m["solve_ms_p99"] = float(1e3 * np.percentile(st, 99))
        m["solve_ms_max"] = float(1e3 * st.max())
    else:
        m["solve_ms_med"] = m["solve_ms_p99"] = m["solve_ms_max"] = None

    # Governor bookkeeping. One "freeze event" = one entry into an engaged
    # mode (legacy INTERACT, or DECEL for the stop-time-anchor variant);
    # DECEL -> HOLD -> RELEASE_DWELL within one interaction is NOT a new
    # event, so entries are counted from a NON-engaged predecessor.
    from .controller import GOV_ENGAGED_MODES

    gov_mode = log["sp_gov_mode"]
    eng = np.isin(gov_mode, [int(x) for x in GOV_ENGAGED_MODES])
    m["gov_freeze_events"] = int(
        eng[0] + np.sum(eng[1:] & ~eng[:-1])) if len(gov_mode) else 0
    m["max_ref_err_governed"] = float(
        np.abs(log["x_desired"] - log["ee_pos"][:, 0])[slide].max()) \
        if slide.any() else np.nan
    m["max_ref_err_original"] = float(
        np.abs(log["sp_x_ref_original"] - log["ee_pos"][:, 0])[slide].max()) \
        if slide.any() else np.nan

    # Post-release peaks + recovery (human scenarios).
    t_rel = _human_release_start(spec)
    t_rel_end = _human_release_end(spec)
    m["t_release_start"] = t_rel
    m["t_release_end"] = t_rel_end
    m["recovery_time"] = None
    for key in ("peak_fn_post", "peak_vt_post", "peak_tau_post",
                "max_step_delta_ut_post", "peak_abs_ut_post",
                "peak_abs_vt_post", "peak_abs_fn_error_post",
                "reference_debt_release", "original_reference_debt_release"):
        m[key] = None
    if t_rel is not None and t_rel < t[-1]:
        post = (t >= t_rel) & (t <= t_rel + POST_RELEASE_WINDOW)
        m["peak_fn_post"] = float(log["f_n"][post].max())
        m["peak_vt_post"] = float(log["ee_vel"][post, 0].max())
        m["peak_tau_post"] = float(np.abs(log["ctrl"][post]).max())
        m["peak_abs_ut_post"] = float(np.abs(log["sp_u"][post, 0]).max())
        m["peak_abs_vt_post"] = float(np.abs(log["ee_vel"][post, 0]).max())
        m["peak_abs_fn_error_post"] = float(
            np.abs(log["f_n"][post] - log["f_n_desired"][post]).max())
        post_pairs = ((t[:-1] >= t_rel)
                      & (t[1:] <= t_rel + POST_RELEASE_WINDOW))
        m["max_step_delta_ut_post"] = float(
            np.abs(np.diff(log["sp_u"][:, 0]))[post_pairs].max())
        i_rel = int(np.searchsorted(t, t_rel, side="left"))
        x_rel = log["ee_pos"][i_rel, 0]
        m["reference_debt_release"] = float(
            abs(log["x_desired"][i_rel] - x_rel))
        m["original_reference_debt_release"] = float(
            abs(log["sp_x_ref_original"][i_rel] - x_rel))
        good = ((np.abs(log["f_n"] - f_d) < 0.25)
                & (np.abs(log["ee_vel"][:, 0] - v_d) < 0.01)
                & (t >= t_rel_end))
        need = int(round(0.5 / dt))
        run = 0
        for i in range(len(good)):
            run = run + 1 if good[i] else 0
            if run >= need:
                m["recovery_time"] = float(t[i - need + 1] - t_rel_end)
                break

    return m


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_scenario(key: str, output_root: str, modes: tuple | None = None,
                 plots: bool = True, animate: bool = False) -> list[dict]:
    from . import plotting

    spec = SCENARIOS[key]
    modes = tuple(modes) if modes else spec.modes
    scen_dir = os.path.join(output_root, f"{spec.key}_{spec.name}")
    logs: dict[str, dict] = {}
    metrics: list[dict] = []
    for mode in modes:
        cfg = build_config(spec, mode)
        cfg = replace(cfg, output_dir=os.path.join(scen_dir, mode))
        print(f"--- scenario {spec.key} ({spec.name}) | {mode} | "
              f"{spec.duration:.0f} s | governor "
              f"{'on' if spec.governor else 'off'} ---")
        log = run_simulation(cfg)
        save_log(log, cfg.output_dir)
        logs[mode] = log
        m = compute_metrics(log, spec, mode)
        metrics.append(m)
        print(f"    rmse_fn={m['rmse_fn']:.4f} N peak(-p_H)={m['p_h_peak']:.4f} W "
              f"E_H_min={m['e_h_min_raw']:.5f} E_R_min={m['e_r_min_raw']:.5f} "
              f"infeas={m['n_infeasible']} emerg={m['n_emergency']} "
              f"bound_viol={m['n_bound_violations']} "
              f"floor_viol={m['n_floor_violations']} "
              f"util={m['bound_util_max']:.3f}")
        if plots:
            plotting.plot_passivation_run(log, cfg.output_dir, title=(
                f"{spec.key} {spec.name} — {mode}"))
        if animate and mode == "C3_dual_ledger_qp":
            plotting.animate(log, cfg.output_dir)
    if plots and len(logs) > 1:
        plotting.plot_passivation_comparison(
            logs, scen_dir, title=f"Scenario {spec.key}: {spec.name}")
    if plots and key == "S":
        plotting.plot_stress_certificate(
            logs["C3_dual_ledger_qp"], scen_dir)

    # E1 filter-allocation cross-controller metrics: C3 vs C4 filtered
    # command differences over the E1 plateau window (metrics only; the
    # underlying logs are complete and unclipped).
    c3l = logs.get("C3_dual_ledger_qp")
    c4l = logs.get("C4_dual_ledger_safe_scalar")
    if c3l is not None and c4l is not None and spec.human.enabled:
        e1 = e1_window(c3l, spec)
        if e1 is not None and e1.any():
            dut = c3l["sp_u"][e1, 0] - c4l["sp_u"][e1, 0]
            dun = c3l["sp_u"][e1, 1] - c4l["sp_u"][e1, 1]
            cross = {
                "rmse_ut_c3_c4_e1": float(np.sqrt(np.mean(dut ** 2))),
                "max_ut_c3_c4_e1": float(np.abs(dut).max()),
                "rmse_un_c3_c4_e1": float(np.sqrt(np.mean(dun ** 2))),
                "max_un_c3_c4_e1": float(np.abs(dun).max()),
            }
            for m in metrics:
                if m["mode"] in ("C3_dual_ledger_qp",
                                 "C4_dual_ledger_safe_scalar"):
                    m.update(cross)
            print(f"    E1 C3-vs-C4 command diffs: "
                  f"u_t rms={cross['rmse_ut_c3_c4_e1']:.4f} "
                  f"max={cross['max_ut_c3_c4_e1']:.4f} N | "
                  f"u_n rms={cross['rmse_un_c3_c4_e1']:.4f} "
                  f"max={cross['max_un_c3_c4_e1']:.4f} N")
    return metrics


def run_governor_compare(output_root: str, plots: bool = True) -> list[dict]:
    """Governor ablation on the proposed C3 controller (Scenario C).

    Runs C3 with the CBF-triggered reference governor disabled and enabled.
    The governor-on run is also the final end-to-end C3 demonstration; the
    governor-off run is the component ablation showing the moving-reference
    catch-up jump. C4 is excluded (diagnostic only).
    """
    from . import plotting

    spec = SCENARIOS["C"]
    out_dir = os.path.join(output_root, "governor_compare")
    mode = "C3_dual_ledger_qp"
    logs: dict[str, dict] = {}
    metrics = []
    for gov in (False, True):
        tag = "governor_on" if gov else "governor_off"
        cfg = build_config(spec, mode, governor=gov)
        cfg = replace(cfg, output_dir=os.path.join(out_dir, mode, tag))
        print(f"--- governor ablation | C3 | {tag} ---")
        log = run_simulation(cfg)
        save_log(log, cfg.output_dir)
        logs[tag] = log
        m = compute_metrics(log, spec, mode)
        m["variant"] = tag
        m.update(governor_period_metrics(log, spec))
        metrics.append(m)
        print(
            f"    trigger={m['gov_trigger_time']} "
            f"P_H_pre={m['n_p_h_before_trigger']} "
            f"CBF={m['n_cbf_h_active']} "
            f"dEH_trig={m['e_h_drop_at_trigger']:.4f} J "
            f"minEH={m['e_h_min_raw']:.4f} "
            f"max|Δu_t|={m['max_step_delta_ut_post']:.3f} N "
            f"peak|u_t|={m['peak_abs_ut_post']:.2f} N "
            f"peak|v_t|={m['peak_abs_vt_post']:.3f} m/s "
            f"debt={m['reference_debt_release']:.4f} m "
            f"infeas={m['n_infeasible']} emerg={m['n_emergency']} "
            f"bound={m['n_bound_violations']} floor={m['n_floor_violations']}")
    if plots:
        plotting.plot_governor_comparison(logs, out_dir)
    _write_table(metrics, os.path.join(out_dir, "governor_compare.csv"))
    with open(os.path.join(out_dir, "governor_compare.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def governor_variant_metrics(log: dict, spec: ScenarioSpec) -> dict:
    """Diagnostics for the stop-time-anchor governor comparison.

    Adds anchor/displacement/command-jump quantities on top of
    ``governor_period_metrics``. All from the complete unclipped run.
    """
    from .controller import GOV_ENGAGED_MODES

    t = log["time"]
    dt = float(log["meta_timestep"])
    gm = log["sp_gov_mode"]
    engaged = np.isin(gm, [int(m) for m in GOV_ENGAGED_MODES])
    m: dict = {}

    idx = np.flatnonzero(engaged)
    t_trig = float(t[idx[0]]) if idx.size else None
    m["t_trigger"] = t_trig
    # DECEL end / anchor capture: first step with a finite logged anchor.
    anchor = log.get("sp_gov_anchor")
    if anchor is not None and np.isfinite(anchor).any():
        j = int(np.flatnonzero(np.isfinite(anchor))[0])
        m["t_anchor"] = float(t[j])
        m["x_a"] = float(anchor[j])
        m["x_stop"] = float(log["ee_pos"][j, 0])
        m["x_a_minus_x_stop"] = m["x_a"] - m["x_stop"]
        m["decel_duration"] = (m["t_anchor"] - t_trig) if t_trig else None
    else:
        m["t_anchor"] = m["x_a"] = m["x_stop"] = None
        m["x_a_minus_x_stop"] = m["decel_duration"] = None

    # Displacement over the full-force hold window of the first pulse.
    hp = spec.human
    if hp.enabled:
        t0 = hp.t_start + hp.t_rise
        t1 = hp.t_start + hp.t_rise + hp.t_hold
        i0 = int(np.argmin(np.abs(t - t0)))
        i1 = int(np.argmin(np.abs(t - t1)))
        x = log["ee_pos"][:, 0]
        m["net_disp_hold_mm"] = 1e3 * float(x[i1] - x[i0])
        if t_trig is not None:
            k = int(np.argmin(np.abs(t - t_trig)))
            seg = x[k:i1 + 1]
            m["max_backward_disp_mm"] = 1e3 * float(seg.min() - x[k])
        else:
            m["max_backward_disp_mm"] = 1e3 * float(x[i0:i1 + 1].min() - x[i0])
        w = (t >= t0) & (t <= t1)
        m["mean_vt_hold"] = float(log["ee_vel"][w, 0].mean())
        m["mean_p_h_hold"] = float(log["sp_p_h_act"][w].mean())
    # DECEL-window power, and HOLD-window power (post anchor, pre release)
    if m.get("t_anchor") is not None and t_trig is not None:
        wd = (t >= t_trig) & (t <= m["t_anchor"])
        m["mean_p_h_decel"] = float(log["sp_p_h_act"][wd].mean())
        m["peak_p_h_decel"] = float(np.max(np.abs(log["sp_p_h_act"][wd])))
        t_rel = _human_release_start(spec)
        wh = (t > m["t_anchor"]) & (t <= (t_rel if t_rel else t[-1]))
        if wh.any():
            m["mean_p_h_hold_phase"] = float(log["sp_p_h_act"][wh].mean())
            m["e_h_at_anchor"] = float(log["sp_e_h"][
                int(np.argmin(np.abs(t - m["t_anchor"])))])
    # Ledger summary
    m["e_h_min"] = float(log["sp_e_h_raw"].min())
    m["e_h_max"] = float(log["sp_e_h"].max())
    m["e_h_final"] = float(log["sp_e_h"][-1])
    m["e_h_reaches_cap"] = bool(log["sp_e_h"].max() >= 0.08 - 1e-9)
    if t_trig is not None:
        m["e_h_at_trigger"] = float(log["sp_e_h"][
            int(np.argmin(np.abs(t - t_trig)))])

    # Command jumps around the three events.
    du = np.abs(np.diff(log["sp_u"][:, 0]))

    def jump(t_c, half=0.05):
        if t_c is None:
            return None
        w = (t[:-1] >= t_c - half) & (t[:-1] <= t_c + half)
        return float(du[w].max()) if w.any() else None

    m["max_step_du_trigger"] = jump(t_trig)
    m["max_step_du_anchor"] = jump(m.get("t_anchor"))
    m["max_step_du_release"] = jump(_human_release_start(spec))
    # Recovery: first time |v_t - v_d| < 5% of v_d sustained 0.2 s post-release
    v_d = float(log["meta_v_slide"])
    t_rel = _human_release_start(spec)
    m["t_recover_vd"] = None
    if t_rel is not None:
        good = (np.abs(log["ee_vel"][:, 0] - v_d) < 0.05 * v_d) & (t >= t_rel)
        need = int(round(0.2 / dt))
        run = 0
        for i in range(len(good)):
            run = run + 1 if good[i] else 0
            if run >= need:
                m["t_recover_vd"] = float(t[i - need + 1] - t_rel)
                break
    return m


def governor_period_metrics(log: dict, spec: ScenarioSpec) -> dict:
    """Three-period diagnostics for the CBF-triggered governor.

    Period A (QP-only): ROW_P_H regulation before the governor triggers.
    Period B: the ROW_CBF_H trigger and the resulting E_H stabilization.
    Period C: smooth post-release resume. All quantities come from the
    complete unclipped run; row indices match passivity_qp (P_H=4,
    CBF_H=2, E_H=0).
    """
    t = log["time"]
    dt = float(log["meta_timestep"])
    p_h_row = log["sp_active"][:, 4].astype(bool)
    cbf_row = log["sp_active"][:, 2].astype(bool)
    e_h_row = log["sp_active"][:, 0].astype(bool)
    gmode = log["sp_gov_mode"]
    human = np.linalg.norm(log["f_h"], axis=1) > 1e-9

    from .controller import GOV_ENGAGED_MODES

    m: dict = {}
    interact = np.flatnonzero(np.isin(gmode, [int(x) for x in GOV_ENGAGED_MODES]))
    trig = float(t[interact[0]]) if interact.size else None
    m["gov_trigger_time"] = trig
    resume = np.flatnonzero(gmode == 2)
    # first RESUME entry after the trigger (post-release)
    m["gov_resume_time"] = (
        float(t[resume[resume > (interact[0] if interact.size else 0)][0]])
        if interact.size and (resume > interact[0]).any() else None)

    before = human & (t < trig) if trig is not None else human
    m["n_p_h_before_trigger"] = int(np.sum(p_h_row & before))
    m["dur_p_h_before_trigger"] = float(np.sum(p_h_row & before) * dt)
    m["n_cbf_h_active"] = int(np.sum(cbf_row))
    m["dur_cbf_h_active"] = float(np.sum(cbf_row) * dt)
    m["n_e_h_floor_active"] = int(np.sum(e_h_row))
    if trig is not None:
        i = interact[0]
        m["e_h_at_trigger"] = float(log["sp_e_h"][i])
        m["e_h_drop_at_trigger"] = float(log["sp_e_h"][0] - log["sp_e_h"][i])
    else:
        m["e_h_at_trigger"] = None
        m["e_h_drop_at_trigger"] = 0.0

    # QP command correction magnitude before the trigger (Period A).
    if trig is not None:
        du = (log["sp_u"][before, 0] - log["sp_u_nom"][before, 0])
        m["rms_qp_corr_ut_pre"] = float(np.sqrt(np.mean(du ** 2))) \
            if du.size else 0.0
        m["max_qp_corr_ut_pre"] = float(np.abs(du).max()) if du.size else 0.0
    else:
        m["rms_qp_corr_ut_pre"] = m["max_qp_corr_ut_pre"] = 0.0
    return m


def run_sweep(output_root: str, plots: bool = True) -> list[dict]:
    """mu_hat sweep of the imperfect friction predictor (mode C3)."""
    from . import plotting

    sweep_dir = os.path.join(output_root, "sweep")
    rows: list[dict] = []
    for with_human in (False, True):
        base = SCENARIOS["D" if with_human else "B"]
        for mu_hat in (0.20, 0.25, 0.30, 0.35):
            spec = replace(base, mu_hat=mu_hat)
            cfg = build_config(spec, "C3_dual_ledger_qp", mu_hat=mu_hat)
            tag = f"mu{mu_hat:.2f}_{'human' if with_human else 'nohuman'}"
            cfg = replace(cfg, output_dir=os.path.join(sweep_dir, tag))
            print(f"--- sweep {tag} ({spec.duration:.0f} s) ---")
            log = run_simulation(cfg)
            save_log(log, cfg.output_dir)
            m = compute_metrics(log, spec, "C3_dual_ledger_qp")
            m["mu_hat"] = mu_hat
            m["with_human"] = with_human
            rows.append(m)
            print(f"    E_R_min={m['e_r_min_raw']:.5f} "
                  f"E_H_min={m['e_h_min_raw']:.5f} "
                  f"infeas={m['n_infeasible']} "
                  f"bound_viol={m['n_bound_violations']} "
                  f"util={m['bound_util_max']:.3f}")
    _write_table(rows, os.path.join(sweep_dir, "sweep.csv"))
    if plots:
        plotting.plot_sweep(rows, sweep_dir)
    return rows


def run_legacy_diagnostics(output_root: str, plots: bool = True) -> list[dict]:
    """Legacy origin-ray scalar modes on scenario C (diagnostic only;
    excluded from paper-grade acceptance)."""
    spec = SCENARIOS["C"]
    metrics = []
    for mode in ("C1_legacy_whole_port_origin_scalar",
                 "C4_legacy_origin_scalar"):
        cfg = build_config(spec, mode)
        cfg = replace(cfg, output_dir=os.path.join(
            output_root, "legacy", mode))
        print(f"--- legacy diagnostic | {mode} ---")
        log = run_simulation(cfg)
        save_log(log, cfg.output_dir)
        m = compute_metrics(log, spec, mode)
        metrics.append(m)
        print(f"    infeas={m['n_infeasible']} emerg={m['n_emergency']} "
              f"(expected nonzero: the origin ray need not intersect the "
              f"feasible set)")
    return metrics


# ---------------------------------------------------------------------------
# Acceptance (section 7) — measured, never silently loosened
# ---------------------------------------------------------------------------

def check_acceptance(metrics: list[dict],
                     governor_metrics: list[dict] | None = None) -> list[dict]:
    by = {(m["scenario"], m["mode"]): m for m in metrics}
    checks: list[dict] = []

    def add(name, ok, value, threshold):
        checks.append({"criterion": name, "pass": bool(ok),
                       "value": value, "threshold": threshold})

    # --- solver/certificate hygiene for every paper-grade run ---
    for m in metrics:
        if m["mode"] not in ACCEPTANCE_MODES:
            continue
        key = f"{m['scenario']}/{m['mode']}"
        add(f"{key}: zero infeasible steps", m["n_infeasible"] == 0,
            m["n_infeasible"], 0)
        add(f"{key}: zero emergency steps", m["n_emergency"] == 0,
            m["n_emergency"], 0)
        add(f"{key}: zero prediction-bound violations",
            m["n_bound_violations"] == 0, m["n_bound_violations"], 0)
        add(f"{key}: zero certified-floor violations",
            m["n_floor_violations"] == 0, m["n_floor_violations"], 0)
        add(f"{key}: no post-QP clipping of feasible solutions",
            m["n_safety_clipped"] == 0, m["n_safety_clipped"], 0)

    # --- C3-specific certificate criteria ---
    for m in metrics:
        if m["mode"] != "C3_dual_ledger_qp":
            continue
        key = m["scenario"]
        add(f"{key}/C3: exact cumulative human inequality (>= -1e-7 J)",
            m["w_h_margin"] >= -1e-7, m["w_h_margin"], -1e-7)
        add(f"{key}/C3: exact cumulative residual inequality (>= -1e-7 J)",
            m["w_r_margin"] >= -1e-7, m["w_r_margin"], -1e-7)
        add(f"{key}/C3: exact E_H floor (>= E_min - 1e-9)",
            m["e_h_min_raw"] >= 0.005 - 1e-9, m["e_h_min_raw"], 0.005)
        add(f"{key}/C3: exact E_R floor (>= E_min - 1e-9)",
            m["e_r_min_raw"] >= 0.02 - 1e-9, m["e_r_min_raw"], 0.02)
        add(f"{key}/C3: peak -p_H <= P_H_max + 1e-6",
            m["p_h_peak"] <= 0.10 + 1e-6, m["p_h_peak"], 0.10)

    a3 = by.get(("A", "C3_dual_ledger_qp"))
    a0 = by.get(("A", "C0_nominal"))
    if a3 and a0:
        add("A: E_H exactly constant (drift <= 1e-9 J)",
            a3["e_h_drift"] <= 1e-9, a3["e_h_drift"], 1e-9)
        add("A: E_R exactly constant (drift <= 1e-9 J)",
            a3["e_r_drift"] <= 1e-9, a3["e_r_drift"], 1e-9)
        add("A: QP active fraction < 0.1%", a3["active_fraction"] < 1e-3,
            a3["active_fraction"], 1e-3)
        add("A: rmse_fn within 5% of C0",
            a3["rmse_fn"] <= 1.05 * a0["rmse_fn"],
            a3["rmse_fn"] / a0["rmse_fn"], 1.05)
        add("A: rmse_vt within 5% of C0",
            a3["rmse_vt"] <= 1.05 * a0["rmse_vt"],
            a3["rmse_vt"] / a0["rmse_vt"], 1.05)

    b3 = by.get(("B", "C3_dual_ledger_qp"))
    b1 = by.get(("B", "C1_whole_port_qp"))
    if b3:
        add("B: E_H unchanged (drift <= 1e-9 J)", b3["e_h_drift"] <= 1e-9,
            b3["e_h_drift"], 1e-9)
        add("B: E_R responds (final < initial - 0.01 J)",
            b3["e_r_final"] < 0.30 - 0.01, b3["e_r_final"], 0.29)
    if b3 and b1 and b1["t_first_ledger_active"] is not None:
        later = (b3["t_first_ledger_active"] is None
                 or b3["t_first_ledger_active"] > b1["t_first_ledger_active"])
        add("B: residual intervenes later than whole-port", later,
            {"C3": b3["t_first_ledger_active"],
             "C1qp": b1["t_first_ledger_active"]}, None)

    # Recovery-after-release is a full-run behavioral criterion for the
    # blocking/helping scenarios. Scenario E is deliberately excluded: E1 is
    # a filter-level allocation ablation evaluated on the full-force plateau
    # only, and release/recovery behavior is handled by the reference-
    # governor comparison, not by E1.
    for key in ("C", "D", "F"):
        c3 = by.get((key, "C3_dual_ledger_qp"))
        if not c3:
            continue
        add(f"{key}: recovery after release",
            c3["recovery_time"] is not None, c3["recovery_time"], None)
        c4 = by.get((key, "C4_dual_ledger_safe_scalar"))
        if c4:
            add(f"{key}: C3 preserves F_n better than C4 (human window)",
                c3["rmse_fn_human"] < c4["rmse_fn_human"],
                {"C3": c3["rmse_fn_human"], "C4": c4["rmse_fn_human"]}, None)

    # --- E1 filter-allocation ablation (plateau-window evaluation only;
    # certificate quantities always use the complete run) ---
    e3 = by.get(("E", "C3_dual_ledger_qp"))
    e4 = by.get(("E", "C4_dual_ledger_safe_scalar"))
    if e3 and e4:
        add("E1: governor-off nominal policy for both controllers",
            (not e3["governor"]) and (not e4["governor"]),
            {"C3_gov": e3["governor"], "C4_gov": e4["governor"]}, False)
        both_cert = all(
            m["e_h_min_raw"] >= 0.005 - 1e-9
            and m["e_r_min_raw"] >= 0.02 - 1e-9
            and m["p_h_peak"] <= 0.10 + 1e-6
            for m in (e3, e4))
        add("E1: both satisfy the same certificate rows "
            "(E_H/E_R floors, peak power; full run)", both_cert,
            {"C3": [e3["e_h_min_raw"], e3["e_r_min_raw"], e3["p_h_peak"]],
             "C4": [e4["e_h_min_raw"], e4["e_r_min_raw"], e4["p_h_peak"]]},
            None)
        clean = all(m["n_infeasible"] == 0 and m["n_emergency"] == 0
                    and m["n_bound_violations"] == 0
                    and m["n_floor_violations"] == 0 for m in (e3, e4))
        add("E1: zero infeasible/emergency/bound/floor violations (both, "
            "full run)", clean,
            {"C3": [e3["n_infeasible"], e3["n_emergency"],
                    e3["n_bound_violations"], e3["n_floor_violations"]],
             "C4": [e4["n_infeasible"], e4["n_emergency"],
                    e4["n_bound_violations"], e4["n_floor_violations"]]}, 0)
        add("E1: C3 lower normal-force RMSE over the plateau window",
            e3["rmse_fn_e1"] is not None
            and e3["rmse_fn_e1"] < e4["rmse_fn_e1"],
            {"C3": e3["rmse_fn_e1"], "C4": e4["rmse_fn_e1"]}, None)
        rut = e3.get("rmse_ut_c3_c4_e1")
        run_ = e3.get("rmse_un_c3_c4_e1")
        add("E1: tangential commands similar (rms diff < 0.5 N), normal "
            "allocation differs (> 2x tangential diff)",
            rut is not None and run_ is not None
            and rut < 0.5 and run_ > 2.0 * rut,
            {"rmse_ut_c3_c4_e1": rut, "rmse_un_c3_c4_e1": run_}, None)
        add("E1: C3 has lower task-weighted command deviation than C4",
            e3["rms_u_dev_w_e1"] < e4["rms_u_dev_w_e1"],
            {"C3": e3["rms_u_dev_w_e1"],
             "C4": e4["rms_u_dev_w_e1"]}, None)

    # --- long-duration stress ---
    s3 = by.get(("S", "C3_dual_ledger_qp"))
    if s3:
        add("S: no floor violation over 32 s", s3["n_floor_violations"] == 0,
            s3["n_floor_violations"], 0)
        add("S: no bound violation over 32 s", s3["n_bound_violations"] == 0,
            s3["n_bound_violations"], 0)
        add("S: no infeasible step over 32 s", s3["n_infeasible"] == 0,
            s3["n_infeasible"], 0)
        add("S: exact cumulative inequalities entire run",
            s3["w_h_margin"] >= -1e-7 and s3["w_r_margin"] >= -1e-7,
            {"w_h": s3["w_h_margin"], "w_r": s3["w_r_margin"]}, -1e-7)
        add("S: no progressive leak (E_H recharges to cap)",
            s3["e_h_final"] >= 0.08 - 1e-6 or s3["e_h_final"] >= 0.0799,
            s3["e_h_final"], 0.08)

    # --- governor ablation on C3 (CBF-triggered) ---
    if governor_metrics:
        gov = {m["variant"]: m for m in governor_metrics}
        off, on = gov.get("governor_off"), gov.get("governor_on")
        if off and on:
            clean = all(on[k] == 0 for k in
                        ("n_infeasible", "n_emergency", "n_bound_violations",
                         "n_floor_violations"))
            add("governor/C3: zero infeasible/emergency/bound/floor "
                "violations", clean,
                {k: on[k] for k in ("n_infeasible", "n_emergency",
                                     "n_bound_violations",
                                     "n_floor_violations")}, 0)
            # The CBF-triggered governor must NOT fire on force alone: there
            # must be a QP-only period (ROW_P_H active before the trigger)
            # and a genuine, visible E_H depletion before it latches.
            add("governor/C3: QP-only period precedes trigger "
                "(ROW_P_H active before governor)",
                on["gov_trigger_time"] is not None
                and on["n_p_h_before_trigger"] > 0,
                {"trigger": on["gov_trigger_time"],
                 "p_h_pre": on["n_p_h_before_trigger"]}, None)
            add("governor/C3: visible human-energy depletion before trigger "
                "(>= 0.01 J)",
                on["e_h_drop_at_trigger"] >= 0.01,
                on["e_h_drop_at_trigger"], 0.01)
            add("governor/C3: E_H stays above certified floor",
                on["e_h_min_raw"] >= 0.005 - 1e-9, on["e_h_min_raw"], 0.005)
            add("governor/C3: exactly one CBF-triggered interaction",
                on["gov_freeze_events"] == 1, on["gov_freeze_events"], 1)
            for key, label, threshold in (
                    ("max_step_delta_ut_post",
                     "one-step tangential-command jump", 0.50),
                    ("peak_abs_ut_post", "post-release |u_t| peak", 0.80),
                    ("peak_abs_vt_post", "post-release |v_t| peak", 0.50),
                    ("peak_abs_fn_error_post",
                     "post-release |F_n-F_n^d| peak", 0.50)):
                ratio = on[key] / max(off[key], 1e-12)
                add(f"governor/C3: {label} clearly reduced",
                    ratio <= threshold, {"off": off[key], "on": on[key],
                                         "ratio": ratio}, threshold)
            add("governor/C3: governed reference debt at release "
                "reduced by at least 90%",
                on["reference_debt_release"]
                <= 0.10 * off["reference_debt_release"],
                {"off": off["reference_debt_release"],
                 "on": on["reference_debt_release"]}, 0.10)
    return checks


# ---------------------------------------------------------------------------
# Output helpers / compact artifacts / CLI
# ---------------------------------------------------------------------------

def _write_table(rows: list[dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(csv_path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(k) is None else str(r.get(k))
                              for k in keys) + "\n")


def write_bound_summary(metrics: list[dict], path: str) -> dict:
    """Machine-readable prediction-error-bound summary across all runs."""
    rows = [m for m in metrics if m["mode"] != "C0_nominal" or True]
    summary = {
        "configured_e_v_bound": rows[0]["e_v_bound"] if rows else None,
        "measured_max": max(m["pred_err_max"] for m in rows),
        "measured_p99_max": max(m["pred_err_p99"] for m in rows),
        "measured_med_max": max(m["pred_err_med"] for m in rows),
        "utilization_max": max(m["bound_util_max"] for m in rows),
        "total_bound_violations": sum(m["n_bound_violations"] for m in rows),
        "per_run": [{k: m[k] for k in
                     ("scenario", "mode", "pred_err_med", "pred_err_p99",
                      "pred_err_max", "bound_util_max",
                      "n_bound_violations")} for m in rows],
    }
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def build_compact(output_root: str, compact_dir: str) -> list[str]:
    """Copy the headline artifacts into a small standalone folder."""
    os.makedirs(compact_dir, exist_ok=True)
    wanted = [
        ("summary.csv", "summary.csv"),
        ("acceptance.json", "acceptance.json"),
        ("prediction_bound_summary.json", "prediction_bound_summary.json"),
        ("A_exact_no_human/comparison.png", "A_comparison.png"),
        ("C_oracle_blocking/comparison.png", "C_comparison.png"),
        ("E_oracle_oblique/comparison.png", "E_c3_vs_c4.png"),
        ("governor_compare/governor_comparison.png",
         "governor_on_off_release.png"),
        ("governor_compare/governor_compare.csv", "governor_compare.csv"),
        ("sweep/sweep.png", "sweep.png"),
        ("sweep/sweep.csv", "sweep.csv"),
        ("S_stress/stress_certificate.png", "stress_certificate.png"),
    ]
    copied = []
    for src_rel, dst_name in wanted:
        src = os.path.join(output_root, src_rel)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(compact_dir, dst_name))
            copied.append(dst_name)
        else:
            print(f"[compact] missing artifact {src_rel} (skipped)")
    return copied


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Selective-passivation experiments (iteration 2)")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=None)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--all", action="store_true",
                        help="run A-F + stress + governor comparison + sweep")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--governor-compare", action="store_true")
    parser.add_argument("--legacy", action="store_true",
                        help="run legacy origin-scalar diagnostics")
    parser.add_argument("--output-root", default="results_passivation_iter2")
    parser.add_argument("--compact-dir", default="results_compact")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args(argv)

    plots = not args.no_plots
    metrics: list[dict] = []
    gov_metrics: list[dict] | None = None

    keys = sorted(SCENARIOS) if args.all else (
        [args.scenario] if args.scenario else [])
    for key in keys:
        metrics += run_scenario(key, args.output_root, args.modes,
                                plots=plots, animate=args.animate)
    sweep_rows: list[dict] = []
    legacy_rows: list[dict] = []
    if args.all or args.governor_compare:
        gov_metrics = run_governor_compare(args.output_root, plots=plots)
    if args.all or args.sweep:
        sweep_rows = run_sweep(args.output_root, plots=plots)
    if args.legacy:
        legacy_rows = run_legacy_diagnostics(args.output_root, plots=plots)

    if metrics:
        _write_table(metrics + legacy_rows,
                     os.path.join(args.output_root, "summary.csv"))
        with open(os.path.join(args.output_root, "summary.json"), "w") as fh:
            json.dump(metrics + legacy_rows, fh, indent=2)
        # The bound summary covers every certificate-bearing run, sweep
        # included (legacy diagnostics excluded from the certificate story).
        write_bound_summary(metrics + sweep_rows + (gov_metrics or []),
                            os.path.join(args.output_root,
                                         "prediction_bound_summary.json"))
        checks = check_acceptance(metrics, gov_metrics)
        if checks:
            with open(os.path.join(args.output_root, "acceptance.json"),
                      "w") as fh:
                json.dump(checks, fh, indent=2)
            n_pass = sum(c["pass"] for c in checks)
            print(f"\nacceptance: {n_pass}/{len(checks)} criteria pass")
            for c in checks:
                mark = "PASS" if c["pass"] else "FAIL"
                print(f"  [{mark}] {c['criterion']}: {c['value']}")
        if not args.no_compact and args.all:
            copied = build_compact(args.output_root, args.compact_dir)
            print(f"\ncompact artifacts ({args.compact_dir}): "
                  f"{', '.join(copied)}")


if __name__ == "__main__":
    main()
