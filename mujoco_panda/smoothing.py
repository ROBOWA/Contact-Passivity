"""Reproduce the original-versus-revised sliding-smoothness comparison."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import PandaConfig
from .simulation import run


def metrics(result):
    d = result.log; dt = result.config.plant.timestep; t = d["time"]
    velocity = np.c_[d["vx"], d["vy"], d["vz"]]
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    torque = np.column_stack([d[f"tau{i}"] for i in range(1, 8)])
    torque_rate = np.diff(torque, axis=0) / dt
    onset = (t[:-2] > 2.0) & (t[:-2] < 2.8)
    steady = (t[:-2] > 2.8) & (t[:-2] < 5.0)
    steady1 = (t[:-1] > 2.8) & (t[:-1] < 5.0)
    contact = (d["phase"] == 2) & (t > 2.8) & (t < 5.0)
    raw_force = d["normal_force"][contact]
    filtered_force = d["normal_force_filtered"][contact]
    return {
        "force_rmse_N": result.metrics.normal_force_rmse,
        "planar_rmse_mm": 1e3 * result.metrics.cartesian_rmse_m,
        "onset_acceleration_p99_m_s2": float(np.percentile(
            np.linalg.norm(acceleration[:-1][onset], axis=1), 99)),
        "steady_acceleration_p99_m_s2": float(np.percentile(
            np.linalg.norm(acceleration[:-1][steady], axis=1), 99)),
        "steady_jerk_p99_m_s3": float(np.percentile(
            np.linalg.norm(jerk[steady], axis=1), 99)),
        "torque_rate_norm_p99_Nm_s": float(np.percentile(
            np.linalg.norm(torque_rate[steady1], axis=1), 99)),
        "normal_velocity_std_mm_s": float(1e3 * np.std(velocity[:-2][steady, 2])),
        "raw_contact_occupancy": float(np.mean(d["contact_solver"][contact])),
        "mean_contact_points": float(np.mean(d["contact_points"][contact])),
        "multi_contact_fraction": float(np.mean(d["contact_points"][contact] >= 2)),
        "raw_force_std_N": float(np.std(raw_force)),
        "raw_force_peak_N": float(np.max(raw_force)),
        "raw_force_step_rms_N": float(np.sqrt(np.mean(np.diff(raw_force) ** 2))),
        "filtered_force_std_N": float(np.std(filtered_force)),
    }


def rolling_rms(values, window=50):
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(np.asarray(values) ** 2, kernel, mode="same"))


def compare(output="results_panda/smoothing"):
    old = PandaConfig(duration=5.2); old.passivation.mode = "C0_nominal"
    old.plant.contact_layout = "single"
    old.plant.contact_time_constant = .020
    old.plant.contact_impedance = (.90, .95, .005)
    old.controller.k_force = .35
    old.controller.ki_force = 4.0
    old.controller.d_force = 12.0
    old.controller.force_filter_time = .050
    old.controller.slide_ramp_duration = 0.0
    revised = PandaConfig(duration=5.2); revised.passivation.mode = "C0_nominal"
    runs = {"original": run(old), "revised": run(revised)}
    os.makedirs(output, exist_ok=True)
    for name, result in runs.items(): result.save(os.path.join(output, name))
    summary = {name: metrics(result) for name, result in runs.items()}
    with open(os.path.join(output, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for name, result in runs.items():
        d, dt = result.log, result.config.plant.timestep; t = d["time"]
        v = np.c_[d["vx"], d["vy"], d["vz"]]
        a = np.diff(v, axis=0) / dt
        tau = np.column_stack([d[f"tau{i}"] for i in range(1, 8)])
        torque_rate = np.linalg.norm(np.diff(tau, axis=0) / dt, axis=1)
        axes[0].plot(t, d["normal_force_filtered"], label=name, lw=1.1)
        axes[1].plot(t[:-1], rolling_rms(np.linalg.norm(a, axis=1)), lw=1.1)
        axes[2].plot(t[:-1], rolling_rms(torque_rate), lw=1.1)
    axes[0].axhline(5, color="black", ls=":", lw=.8)
    axes[0].set_ylabel("filtered normal force [N]")
    axes[1].set_ylabel("50 ms accel. RMS [m/s2]")
    axes[2].set_ylabel("50 ms torque-rate RMS [N m/s]")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(frameon=False)
    for ax in axes: ax.grid(alpha=.2); ax.set_xlim(1.7, 5.2)
    fig.tight_layout(); fig.savefig(os.path.join(output, "comparison.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7))
    for name, result in runs.items():
        d, t = result.log, result.log["time"]
        axes[0].plot(t, d["normal_force"], label=name, lw=.75, alpha=.85)
        axes[2].plot(t, 1e3 * d["vz"], label=name, lw=.85)
    d = runs["revised"].log; t = d["time"]
    zoom = (t >= 4.0) & (t <= 4.25)
    axes[1].plot(t[zoom], d["normal_force"][zoom], label="raw", lw=1.0,
                 drawstyle="steps-post")
    filter_ms = 1e3 * runs["revised"].config.controller.force_filter_time
    axes[1].plot(t[zoom], d["normal_force_filtered"][zoom],
                 label=f"{filter_ms:.0f} ms filtered",
                 lw=1.5)
    axes[0].axhline(5, color="black", ls=":", lw=.8)
    axes[0].set_ylabel("raw normal force [N]")
    axes[0].set_xlim(2.8, 5.0); axes[0].set_ylim(-.5, 14)
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("revised force [N]")
    axes[1].set_xlabel("time [s], 250 ms close-up")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].set_ylabel("normal velocity [mm/s]")
    axes[2].set_xlabel("time [s]")
    axes[2].set_xlim(2.8, 5.0)
    axes[2].set_ylim(-5, 5)
    axes[2].legend(frameon=False)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(os.path.join(output, "raw_contact.png"), dpi=180)
    plt.close(fig)
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", default="results_panda/smoothing")
    args = p.parse_args(); print(json.dumps(compare(args.output), indent=2))


if __name__ == "__main__": main()
