"""End-to-end simulation tests: stability, tracking, channel separation."""

import numpy as np

from mujoco_sliding.controller import Phase
from mujoco_sliding.simulation import steady_slide_metrics


def _finite(arr):
    return np.isfinite(np.asarray(arr, dtype=float)).all()


def test_default_run_stable(default_log):
    log = default_log
    for key in ("qpos", "qvel", "ctrl", "ee_pos", "f_n", "f_t", "p_task"):
        assert _finite(log[key]), f"NaN/inf in {key}"
    tau_limit = 60.0
    assert np.abs(log["ctrl"]).max() < tau_limit, "torque limit violated"
    # All three phases are visited, in order.
    phases = log["phase"]
    assert set(np.unique(phases)) == {0, 1, 2}
    assert (np.diff(phases) >= 0).all()


def test_steady_sliding_tracking(default_log):
    m = steady_slide_metrics(default_log)
    assert m["valid"]
    assert m["f_n_rel_err"] < 0.10, f"normal-force error {m['f_n_rel_err']:.2%}"
    assert m["vx_rel_err"] < 0.15, f"tangential-velocity error {m['vx_rel_err']:.2%}"
    assert m["contact_fraction"] > 0.99, "contact must stay active while sliding"
    assert m["max_penetration"] < 0.005, "penetration exceeds configured limit"


def test_friction_opposes_sliding_in_full_run(default_log):
    log = default_log
    t = log["time"]
    win = (log["phase"] == int(Phase.SLIDE)) & (t > t[log["phase"] == 2][0] + 1.0)
    vx = log["ee_vel"][win][:, 0]
    ft = log["f_t"][win]
    assert vx.mean() > 0
    assert ft.mean() < 0, "kinetic friction must oppose +x sliding"
    # Magnitude consistent with Coulomb friction mu * F_n (mu = 0.3).
    mu_eff = -ft.mean() / log["f_n"][win].mean()
    assert 0.2 < mu_eff < 0.35


def test_task_and_human_channels_separate(human_log):
    log = human_log
    t = log["time"]
    push = (t >= 6.6) & (t <= 8.4)  # hold window of the default profile
    quiet = t < 5.5
    np.testing.assert_allclose(log["f_h"][push][:, 0], 3.0, atol=1e-9)
    assert np.abs(log["f_h"][quiet]).max() == 0.0
    # The synthetic combined channel is exactly the sum; ground truths stay apart.
    np.testing.assert_allclose(
        log["f_measured"], log["f_task"] + log["f_h"], atol=1e-12
    )
    assert np.abs(log["f_task"][push][:, 2]).mean() > 1.0  # task contact still there


def test_human_power_sign_in_full_run(human_log):
    log = human_log
    t = log["time"]
    push = (t >= 6.6) & (t <= 8.4)
    # Push along +x while sliding along +x => human injects power.
    assert log["p_h_to_r"][push].mean() > 0.05
    np.testing.assert_allclose(log["p_r_to_h"], -log["p_h_to_r"], atol=1e-12)
    # No human force => zero human power.
    assert np.abs(log["p_h_to_r"][t < 5.5]).max() == 0.0


def test_human_run_still_tracks(human_log):
    m = steady_slide_metrics(human_log)
    assert m["valid"]
    assert m["f_n_rel_err"] < 0.15
    assert m["vx_rel_err"] < 0.15
    assert m["contact_fraction"] > 0.95


def test_energy_and_power_logged(default_log):
    log = default_log
    assert _finite(log["energy_kinetic"]) and _finite(log["energy_potential"])
    # Sliding against friction dissipates: P_task < 0 in steady sliding.
    t = log["time"]
    win = (log["phase"] == 2) & (t > t[log["phase"] == 2][0] + 1.0)
    assert log["p_task"][win].mean() < 0
