"""Scripted, reproducible experiments for the selective-passivation POC.

Scenarios (deterministic: fixed keyframe initial state, scripted forces,
no randomness anywhere in the pipeline):

  A exact_no_human     oracle predictor, no human. C3 should stay nominal
                       with constant ledgers; C1 depletes from task friction.
  B mismatch_no_human  true mu = 0.30, predictor mu_hat = 0.20, no human
                       (20 s so the slower residual depletion is visible).
                       Only E_R responds; E_H stays constant.
  C oracle_blocking    oracle predictor; blocking human F_h = -3 N t during
                       6.5-8.5 s (0.5 s cosine ramps).
  D mismatch_blocking  mu_hat = 0.20 plus the same blocking human.
  E oracle_oblique     oblique human F_h = 3/sqrt(2) (-t - n): QP task
                       prioritization vs scalar scaling (C3 vs C4).
  F helping_blocking   +3 N t pulse (5-8 s), then -3 N t pulse (9.5-12.5 s):
                       ledger charging, cap, and later discharge.

Usage (from the repository root):
    python -m mujoco_sliding.experiments --all             # A-F + summary
    python -m mujoco_sliding.experiments --scenario C
    python -m mujoco_sliding.experiments --scenario C --modes C3_dual_ledger_qp
    python -m mujoco_sliding.experiments --sweep           # mu_hat sweep
    python -m mujoco_sliding.experiments --all --sweep --animate

Outputs under --output-root (default results_passivation/):
    <scenario>/<mode>/log.npz|log.csv|run.png   per-run artifacts
    <scenario>/comparison.png                   cross-controller comparison
    summary.json, summary.csv                   machine-readable metrics
    acceptance.json                             acceptance-criteria checks
    sweep/sweep.csv, sweep/sweep.png            mu_hat sweep artifacts
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, replace

import numpy as np

from .config import (HumanForceConfig, PassivationConfig, SimulationConfig)
from .simulation import run_simulation, save_log

MODES_ALL = ("C0_nominal", "C1_whole_port_scalar", "C2_residual_qp",
             "C3_dual_ledger_qp", "C4_dual_ledger_scalar")

_BLOCKING = HumanForceConfig(enabled=True, magnitude=3.0,
                             direction=(-1.0, 0.0, 0.0),
                             t_start=6.0, t_rise=0.5, t_hold=2.0, t_fall=0.5)
_OBLIQUE = HumanForceConfig(enabled=True, magnitude=3.0,
                            direction=(-1.0, 0.0, -1.0),  # normalized on use
                            t_start=6.0, t_rise=0.5, t_hold=2.0, t_fall=0.5)
_HELPING = HumanForceConfig(enabled=True, magnitude=3.0,
                            direction=(1.0, 0.0, 0.0),
                            t_start=5.0, t_rise=0.5, t_hold=2.0, t_fall=0.5)
# Second pulse of scenario F. Starts at 8.5 s so its release (11.5 s) leaves
# the catch-up target inside the arm's reachable workspace (x_d stays below
# the ~1.0 m reach limit for the rest of the 14 s run).
_BLOCK_LATE = HumanForceConfig(enabled=True, magnitude=3.0,
                               direction=(-1.0, 0.0, 0.0),
                               t_start=8.5, t_rise=0.5, t_hold=2.0, t_fall=0.5)


@dataclass
class ScenarioSpec:
    key: str
    name: str
    duration: float
    predictor: str = "oracle"
    mu_hat: float = 0.30
    human: HumanForceConfig = field(default_factory=HumanForceConfig)
    human_pulses: tuple = ()
    modes: tuple = ("C0_nominal", "C1_whole_port_scalar", "C2_residual_qp",
                    "C3_dual_ledger_qp")


SCENARIOS: dict[str, ScenarioSpec] = {s.key: s for s in [
    ScenarioSpec("A", "exact_no_human", 12.0),
    # 17 s, not the original 20 s. The residual tank absorbs
    # E_R(0) - E_R,min = 0.28 J at rate d_mu * F_N * v_t, so the sliding
    # DISTANCE to the floor is 0.28 / (d_mu * F_N) -- independent of v_slide,
    # since rate and travel both scale with v_t. At d_mu = 0.10 that is 0.56 m
    # against roughly 0.60 m of runway before the two 0.6 m links reach their
    # limit, so the floor is reached at 15.55 s and contact degrades shortly
    # after. 20 s left contact active only 80.4% of the sliding phase and
    # contaminated every tracking RMSE in the scenario; 17 s keeps 96.8% while
    # still showing 1.5 s of the residual constraint holding the floor.
    #
    # Enlarging d_mu to fit more margin does NOT work and is worth recording:
    # at mu_hat = 0.18 the run takes 86 infeasible steps and at mu_hat = 0.15
    # it takes 1227, driving E_R to -0.039 J through the uncertified emergency
    # fallback. Depleting the tank faster also demands a larger correction to
    # hold the floor, and past |p_delta| ~ 0.035 W that correction is not
    # reachable under the 5 N.m/step torque-rate limit while contact is kept.
    ScenarioSpec("B", "mismatch_no_human", 17.0,
                 predictor="friction", mu_hat=0.20),
    ScenarioSpec("C", "oracle_blocking", 12.0, human=_BLOCKING,
                 modes=MODES_ALL),
    ScenarioSpec("D", "mismatch_blocking", 12.0,
                 predictor="friction", mu_hat=0.20, human=_BLOCKING,
                 modes=MODES_ALL),
    ScenarioSpec("E", "oracle_oblique", 12.0, human=_OBLIQUE,
                 modes=("C0_nominal", "C3_dual_ledger_qp",
                        "C4_dual_ledger_scalar")),
    # 14 s: the second pulse releases at 11.5 s, leaving room for the
    # sustained-recovery window while x_d stays inside the workspace.
    ScenarioSpec("F", "helping_blocking", 14.0, human=_HELPING,
                 human_pulses=(_BLOCK_LATE,),
                 modes=("C0_nominal", "C3_dual_ledger_qp")),
]}


def build_config(spec: ScenarioSpec, mode: str,
                 mu_hat: float | None = None) -> SimulationConfig:
    pas = PassivationConfig(mode=mode, predictor=spec.predictor,
                            mu_hat=mu_hat if mu_hat is not None else spec.mu_hat)
    return SimulationConfig(
        duration=spec.duration,
        human=spec.human,
        human_pulses=spec.human_pulses,
        passivation=pas,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _human_release_end(spec: ScenarioSpec) -> float | None:
    pulses = [spec.human, *spec.human_pulses]
    ends = [p.t_start + p.t_rise + p.t_hold + p.t_fall
            for p in pulses if p.enabled]
    return max(ends) if ends else None


def compute_metrics(log: dict, spec: ScenarioSpec, mode: str) -> dict:
    """All per-run metrics of section 10/12, JSON-serializable."""
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
        "rmse_fn": rmse(log["f_n"] - log["f_n_desired"], slide),
        "rmse_vt": rmse(log["ee_vel"][:, 0] - log["vx_desired"], slide),
        "rmse_x": rmse(log["ee_pos"][:, 0] - log["x_desired"], slide),
        "rmse_fn_human": rmse(log["f_n"] - log["f_n_desired"],
                              slide & human_on),
        "rmse_vt_human": rmse(log["ee_vel"][:, 0] - log["vx_desired"],
                              slide & human_on),
    }

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
        "e_h_violations": int(np.sum(log["sp_e_h_below"])),
        "e_r_violations": int(np.sum(log["sp_e_r_below"])),
        "p_h_violations": int(np.sum(-ph > pas.p_h_max * 1.05)),
        "n_infeasible": int(log["meta_n_infeasible"]),
        "n_emergency": int(log["meta_n_emergency"]),
        # Passivity-inequality margins (>= ~-1e-3 J passes):
        # W_i(t) <= E_i(0) - E_i_min for all t.
        "w_h_margin": float((pas.human.e_init - pas.human.e_min) - w_h.max()),
        "w_r_margin": float((pas.residual.e_init - pas.residual.e_min)
                            - w_r.max()),
        "pred_err_med": float(np.median(log["sp_pred_err"])),
        "pred_err_p99": float(np.percentile(log["sp_pred_err"], 99)),
        "pred_err_max": float(log["sp_pred_err"].max()),
        # Fraction of the sliding phase with live pad-surface contact. A
        # scenario whose sliding target outruns the arm's reach loses contact
        # and silently contaminates every RMSE above; this makes that
        # visible instead (see the workspace acceptance criterion).
        "contact_fraction_slide": (
            float(log["contact_active"].astype(bool)[slide].mean())
            if slide.any() else float("nan")),
        # Release-surge diagnostics. A blocking human that stops the arm makes
        # the sliding reference wind up unless the governor caps it, and the
        # stored position error then discharges as one burst after release --
        # fast motion the port certificate cannot see, because f_H = 0 there
        # makes p_H = 0 at any speed. Guard both the cause and the effect.
        "track_lag_max": (float(np.abs(log["ee_pos"][:, 0]
                                       - log["x_desired"])[slide].max())
                          if slide.any() else float("nan")),
        "task_debt_max": (float(np.abs(log["x_desired_free"]
                                       - log["x_desired"])[slide].max())
                          if slide.any() else float("nan")),
        "vt_peak": (float(np.abs(log["ee_vel"][:, 0])[slide].max())
                    if slide.any() else float("nan")),
        "fn_peak": (float(log["f_n"][slide].max())
                    if slide.any() else float("nan")),
    })

    # First activation time (any constraint modifies the command), and the
    # first LEDGER-driven activation (energy/power rows 0-4 active while the
    # command is modified) — the latter excludes one-step torque-rate
    # transients at contact establishment.
    idx = np.flatnonzero(active)
    m["t_first_active"] = float(t[idx[0]]) if idx.size else None
    ledger_rows = log["sp_active"][:, :5].any(axis=1)
    idx = np.flatnonzero(active & ledger_rows)
    m["t_first_ledger_active"] = float(t[idx[0]]) if idx.size else None

    # Solver stats (QP/scalar modes only; C0 has no solve).
    if mode != "C0_nominal":
        st = log["sp_solve_time"]
        m["solve_ms_med"] = float(1e3 * np.median(st))
        m["solve_ms_p99"] = float(1e3 * np.percentile(st, 99))
        m["solve_ms_max"] = float(1e3 * st.max())
    else:
        m["solve_ms_med"] = m["solve_ms_p99"] = m["solve_ms_max"] = None

    # Recovery time after the last human release (human scenarios only):
    # first instant after release from which |F_n - F_d| < 0.25 N and
    # |v_t - v_d| < 0.01 m/s hold for a sustained 0.5 s.
    t_rel = _human_release_end(spec)
    m["t_release_end"] = t_rel
    m["recovery_time"] = None
    if t_rel is not None:
        good = ((np.abs(log["f_n"] - f_d) < 0.25)
                & (np.abs(log["ee_vel"][:, 0] - v_d) < 0.01)
                & (t >= t_rel))
        need = int(round(0.5 / dt))
        run = 0
        for i in range(len(good)):
            run = run + 1 if good[i] else 0
            if run >= need:
                m["recovery_time"] = float(t[i - need + 1] - t_rel)
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
              f"{spec.duration:.0f} s ---")
        log = run_simulation(cfg)
        save_log(log, cfg.output_dir)
        logs[mode] = log
        m = compute_metrics(log, spec, mode)
        metrics.append(m)
        print(f"    rmse_fn={m['rmse_fn']:.4f} N rmse_vt={m['rmse_vt']:.5f} "
              f"peak(-p_H)={m['p_h_peak']:.4f} W W_H_max={m['w_h_max']:.5f} J "
              f"E_H_min={m['e_h_min_raw']:.5f} E_R_min={m['e_r_min_raw']:.5f} "
              f"active={100 * m['active_fraction']:.2f}% "
              f"infeas={m['n_infeasible']}")
        if plots:
            plotting.plot_passivation_run(log, cfg.output_dir, title=(
                f"{spec.key} {spec.name} — {mode}"))
        if animate and mode in ("C3_dual_ledger_qp",):
            plotting.animate(log, cfg.output_dir)
    if plots and len(logs) > 1:
        plotting.plot_passivation_comparison(
            logs, scen_dir, title=f"Scenario {spec.key}: {spec.name}")
    return metrics


def run_sweep(output_root: str, plots: bool = True) -> list[dict]:
    """mu_hat sweep of the imperfect friction predictor (mode C3)."""
    from . import plotting

    sweep_dir = os.path.join(output_root, "sweep")
    rows: list[dict] = []
    for with_human in (False, True):
        base = SCENARIOS["D" if with_human else "B"]
        for mu_hat in (0.15, 0.20, 0.25, 0.30, 0.35):
            spec = replace(base, key=base.key, mu_hat=mu_hat)
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
                  f"t_first_ledger_active={m['t_first_ledger_active']} "
                  f"rmse_fn={m['rmse_fn']:.4f}")
    _write_table(rows, os.path.join(sweep_dir, "sweep.csv"))
    if plots:
        plotting.plot_sweep(rows, sweep_dir)
    return rows


# ---------------------------------------------------------------------------
# Acceptance criteria (section 12) — measured, never silently loosened
# ---------------------------------------------------------------------------

def check_acceptance(metrics: list[dict]) -> list[dict]:
    by = {(m["scenario"], m["mode"]): m for m in metrics}
    checks: list[dict] = []

    def add(name, ok, value, threshold, scope="method"):
        """Record one criterion.

        scope="method"   the proposed controller (C3) or the scenario setup
                         must satisfy this; a failure is a defect.
        scope="ablation" a claim about a baseline (C1/C2/C4). A failure is a
                         reportable finding about that baseline, not a defect
                         in the proposed method, so it is tallied separately
                         and must not be read as "the method failed".
        """
        checks.append({"criterion": name, "pass": bool(ok),
                       "value": value, "threshold": threshold,
                       "scope": scope})

    a3 = by.get(("A", "C3_dual_ledger_qp"))
    a0 = by.get(("A", "C0_nominal"))
    if a3 and a0:
        add("A: E_H drift < 1e-4 J", a3["e_h_drift"] < 1e-4,
            a3["e_h_drift"], 1e-4)
        add("A: E_R drift < 1e-4 J", a3["e_r_drift"] < 1e-4,
            a3["e_r_drift"], 1e-4)
        add("A: QP active fraction < 0.1%", a3["active_fraction"] < 1e-3,
            a3["active_fraction"], 1e-3)
        add("A: rmse_fn within 5% of C0",
            a3["rmse_fn"] <= 1.05 * a0["rmse_fn"],
            a3["rmse_fn"] / a0["rmse_fn"], 1.05)
        add("A: rmse_vt within 5% of C0",
            a3["rmse_vt"] <= 1.05 * a0["rmse_vt"],
            a3["rmse_vt"] / a0["rmse_vt"], 1.05)

    b3 = by.get(("B", "C3_dual_ledger_qp"))
    b1 = by.get(("B", "C1_whole_port_scalar"))
    if b3:
        add("B: E_H unchanged (drift < 1e-4 J)", b3["e_h_drift"] < 1e-4,
            b3["e_h_drift"], 1e-4)
        add("B: E_R responds (final < initial - 0.01 J)",
            b3["e_r_final"] < 0.30 - 0.01, b3["e_r_final"], 0.29)
    if b3 and b1 and b1["t_first_ledger_active"] is not None:
        later = (b3["t_first_ledger_active"] is None
                 or b3["t_first_ledger_active"] > b1["t_first_ledger_active"])
        add("B: residual intervenes later than whole-port", later,
            {"C3": b3["t_first_ledger_active"],
             "C1": b1["t_first_ledger_active"]}, None)

    # Scenario-sizing guard. If the sliding target outruns the arm's reach the
    # pad leaves the surface and every tracking RMSE in that scenario silently
    # becomes a workspace measurement instead of a controller one. Checked on
    # C0 so it reflects the SCENARIO, not the passivation layer (C3 legitimately
    # backs off contact while a constraint is active).
    for key in sorted(SCENARIOS):
        c0 = by.get((key, "C0_nominal"))
        if c0 and not np.isnan(c0.get("contact_fraction_slide", np.nan)):
            # 95%: the reach limit is structural to this two-link arm (see the
            # scenario-sizing note in the README), so the bar is set to catch a
            # scenario that has genuinely run out of workspace -- B at 20 s
            # scored 80.4% -- rather than the sub-1% contact transients every
            # scenario shows when a constraint first engages.
            add(f"{key}: scenario fits workspace (C0 contact >= 95% of slide)",
                c0["contact_fraction_slide"] >= 0.95,
                c0["contact_fraction_slide"], 0.95)

    # Cumulative residual inequality, Eq. (39): -sum dt p_R <= E_R(0) - E_R,min
    # = 0.28 J. The human counterpart Eq. (38) is checked per human scenario
    # below; this is the second, looser certificate and applies in EVERY
    # scenario, including the two with no human at all.
    for key in sorted(SCENARIOS):
        c3 = by.get((key, "C3_dual_ledger_qp"))
        if c3:
            add(f"{key}: cumulative residual inequality (margin >= -1e-3 J)",
                c3["w_r_margin"] >= -1e-3, c3["w_r_margin"], -1e-3)
        # Same bound on the residual-only ablation, which enforces exactly
        # this row and nothing else -- so Eq. (39) is precisely its claim.
        c2 = by.get((key, "C2_residual_qp"))
        if c2:
            add(f"{key}: C2 residual-only honours its own Eq. (39) bound",
                c2["w_r_margin"] >= -1e-3, c2["w_r_margin"], -1e-3,
                scope="ablation")

    for key in ("C", "D", "E", "F"):
        c3 = by.get((key, "C3_dual_ledger_qp"))
        if not c3:
            continue
        add(f"{key}: E_H_min >= E_H_min - 1e-4 J",
            c3["e_h_min_raw"] >= 0.005 - 1e-4, c3["e_h_min_raw"],
            0.005 - 1e-4)
        add(f"{key}: cumulative human inequality (margin >= -1e-3 J)",
            c3["w_h_margin"] >= -1e-3, c3["w_h_margin"], -1e-3)
        add(f"{key}: peak -p_H <= 1.05 * P_H_max",
            c3["p_h_peak"] <= 0.10 * 1.05, c3["p_h_peak"], 0.105)
        add(f"{key}: recovery after release",
            c3["recovery_time"] is not None, c3["recovery_time"], None)
        add(f"{key}: zero infeasible steps (C3)",
            c3["n_infeasible"] == 0, c3["n_infeasible"], 0)
        c4 = by.get((key, "C4_dual_ledger_scalar"))
        if c4:
            add(f"{key}: C3 preserves F_n better than C4 (human window)",
                c3["rmse_fn_human"] < c4["rmse_fn_human"],
                {"C3": c3["rmse_fn_human"], "C4": c4["rmse_fn_human"]}, None)
    return checks


# ---------------------------------------------------------------------------
# Output helpers / CLI
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Selective-passivation experiments")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=None)
    parser.add_argument("--modes", nargs="+", default=None,
                        help="controller modes (default: scenario's set)")
    parser.add_argument("--all", action="store_true", help="run scenarios A-F")
    parser.add_argument("--sweep", action="store_true",
                        help="run the mu_hat sweep")
    parser.add_argument("--output-root", default="results_passivation")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--animate", action="store_true",
                        help="also export a GIF for the C3 runs")
    args = parser.parse_args(argv)

    plots = not args.no_plots
    metrics: list[dict] = []
    keys = sorted(SCENARIOS) if args.all else (
        [args.scenario] if args.scenario else [])
    for key in keys:
        metrics += run_scenario(key, args.output_root, args.modes,
                                plots=plots, animate=args.animate)

    if metrics:
        _write_table(metrics, os.path.join(args.output_root, "summary.csv"))
        with open(os.path.join(args.output_root, "summary.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
        checks = check_acceptance(metrics)
        if checks:
            with open(os.path.join(args.output_root, "acceptance.json"),
                      "w") as fh:
                json.dump(checks, fh, indent=2)
            method = [c for c in checks if c.get("scope", "method") == "method"]
            ablat = [c for c in checks if c.get("scope") == "ablation"]
            n_pass = sum(c["pass"] for c in method)
            print(f"\nacceptance: {n_pass}/{len(method)} method criteria pass")
            for c in method:
                mark = "PASS" if c["pass"] else "FAIL"
                print(f"  [{mark}] {c['criterion']}: {c['value']}")
            if ablat:
                n_ab = sum(c["pass"] for c in ablat)
                print(f"\nablation baselines: {n_ab}/{len(ablat)} hold "
                      f"(a failure here is a finding about the baseline, "
                      f"not a defect in C3)")
                for c in ablat:
                    mark = "HOLDS" if c["pass"] else "BROKEN"
                    print(f"  [{mark}] {c['criterion']}: {c['value']}")

    if args.sweep:
        run_sweep(args.output_root, plots=plots)


if __name__ == "__main__":
    main()
