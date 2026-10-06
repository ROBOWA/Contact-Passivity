"""Focused A--E comparisons, calibration split, and whole-port sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, replace

from .config import MODES, PandaConfig
from .plotting import comparison, sweep_plot
from .simulation import run


def scenario(name: str) -> PandaConfig:
    c = PandaConfig(duration=7.0)
    if name == "A_accurate_no_human":
        pass
    elif name == "B_mismatch_no_human":
        c.passivation.predictor = "friction"; c.passivation.mu_hat = 0.18
    elif name == "C_ee_oblique_blocking":
        c.human.enabled = True; c.human.direction = (-1.0, -0.35, 0.0)
    elif name == "D_forearm_wiping":
        c.duration = 8.0; c.controller.trajectory = "wipe"
        c.passivation.predictor = "friction"; c.passivation.mu_hat = 0.18
        c.passivation.normal_scale = 0.94
        c.passivation.dynamics_mass_scale = 1.02
        c.human.enabled = True; c.human.location = "forearm"
        c.human.magnitude = 13.5; c.human.direction = (0.0, -1.0, 0.0)
    elif name == "E_release_recovery":
        c.duration = 8.0; c.human.enabled = True
        c.human.start = 3.5; c.human.hold = 1.0
    else:
        raise ValueError(name)
    return c


SCENARIOS = (
    "A_accurate_no_human", "B_mismatch_no_human", "C_ee_oblique_blocking",
    "D_forearm_wiping", "E_release_recovery",
)


def calibrate(output: str) -> dict:
    """Separate non-evaluation runs; the configured bound is never adapted."""
    configs = []
    for trajectory in ("straight", "wipe"):
        c = PandaConfig(duration=7.0); c.controller.trajectory = trajectory
        c.passivation.mode = "C3_dual_ledger_qp"; configs.append(c)
        h = PandaConfig(duration=7.0); h.controller.trajectory = trajectory
        h.passivation.mode = "C3_dual_ledger_qp"; h.human.enabled = True
        h.human.start = 3.0; h.human.hold = .6
        h.human.magnitude = 5.; h.human.direction = (-.8, -.6, 0)
        if trajectory == "wipe":
            h.passivation.dynamics_mass_scale = 1.02
        configs.append(h)
    maxima = [run(c).metrics.max_qvel_prediction_error for c in configs]
    record = {
        "calibration_scenarios": len(configs), "observed_max_rad_s": max(maxima),
        "frozen_bound_rad_s": configs[0].passivation.qvel_prediction_bound,
        "covered": max(maxima) <= configs[0].passivation.qvel_prediction_bound,
        "note": "Evidence for these tested conditions only; not a universal bound.",
    }
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "calibration.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record


def run_all(output: str) -> dict:
    os.makedirs(output, exist_ok=True)
    calibration = calibrate(output)
    nominal_wipe_cfg = PandaConfig(duration=8.0)
    nominal_wipe_cfg.controller.trajectory = "wipe"
    nominal_wipe_cfg.passivation.mode = "C0_nominal"
    nominal_wipe = run(nominal_wipe_cfg)
    nominal_wipe.save(os.path.join(output, "nominal_wipe_validation"))
    summary = []
    for name in SCENARIOS:
        results = {}
        for mode in MODES:
            c = scenario(name); c.passivation.mode = mode
            result = run(c); results[mode] = result
            dest = os.path.join(output, name, mode); result.save(dest)
            row = {"scenario": name, **asdict(result.metrics)}; summary.append(row)
        comparison(results, os.path.join(output, name, "comparison.png"), name.replace("_", " "))

    sweep = []
    for initial in (.06, .12, .30, .60):
        c = scenario("C_ee_oblique_blocking"); c.passivation.mode = "C1_whole_port_qp"
        c.passivation.whole = replace(c.passivation.whole, e_init=initial,
                                      e_max=max(initial, c.passivation.whole.e_max))
        result = run(c)
        sweep.append({"whole_budget_J": initial - c.passivation.whole.e_min,
                      **asdict(result.metrics)})
    with open(os.path.join(output, "whole_port_sweep.json"), "w") as f:
        json.dump(sweep, f, indent=2)
    sweep_plot(sweep, os.path.join(output, "whole_port_sweep.png"))

    # Deliberately stronger than evaluation D, kept as a separate stress
    # point rather than using it to tune the focused comparison.
    stress_cfg = scenario("D_forearm_wiping")
    stress_cfg.human.magnitude = 15.0
    stress_cfg.passivation.mode = "C3_dual_ledger_qp"
    stress = run(stress_cfg)
    stress.save(os.path.join(output, "stress_forearm_15N", "C3_dual_ledger_qp"))

    ablation = {}
    for enabled in (False, True):
        c = scenario("E_release_recovery"); c.passivation.mode = "C3_dual_ledger_qp"
        c.controller.governor.enabled = enabled
        result = run(c); key = "governor_on" if enabled else "governor_off"
        ablation[key] = result; result.save(os.path.join(output, "governor_ablation", key))
    comparison(ablation, os.path.join(output, "governor_ablation", "comparison.png"),
               "C3 governor ablation")

    keys = list(summary[0])
    with open(os.path.join(output, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(summary)
    report = {"calibration": calibration,
              "nominal_wipe_validation": asdict(nominal_wipe.metrics),
              "runs": summary, "whole_port_sweep": sweep,
              "stress_forearm_15N": asdict(stress.metrics),
              "governor_ablation": {k: asdict(v.metrics) for k, v in ablation.items()}}
    with open(os.path.join(output, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    c3_runs = [row for row in summary if row["mode"] == "C3_dual_ledger_qp"]
    straight_c0 = next(row for row in summary
                       if row["scenario"] == "A_accurate_no_human"
                       and row["mode"] == "C0_nominal")
    acceptance = {
        "nominal_straight_targets": straight_c0["nominal_targets_passed"],
        "nominal_wipe_targets": nominal_wipe.metrics.nominal_targets_passed,
        "c3_zero_protected_floor_violations": all(
            row["ledger_floor_violations"] == 0 for row in c3_runs),
        "c3_zero_fallbacks": all(row["qp_fallback_steps"] == 0 for row in c3_runs),
        "c3_evaluation_prediction_bound_violations": sum(
            row["prediction_bound_violations"] for row in c3_runs),
        "all_mode_evaluation_prediction_bound_violations": sum(
            row["prediction_bound_violations"] for row in summary),
        "whole_port_small_budget_acceptance": False,
        "whole_port_failure_is_reported": True,
    }
    with open(os.path.join(output, "acceptance.json"), "w") as f:
        json.dump(acceptance, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_panda")
    args = parser.parse_args(); run_all(args.output)


if __name__ == "__main__":
    main()
