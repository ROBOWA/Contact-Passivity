"""Reference-trajectory planners: nominal equivalence, limits and continuity of suspend/resume."""

import numpy as np
import pytest

from kuka_table_sim.trajectories import (ApproachProfile, SlideProfile, StrokeSequencer, descent_trajectory,
                                         move_to_trajectory, stop_trajectory)


def test_descent_reproduces_the_nominal_approach_profile():
    old = ApproachProfile(0.2994, 0.169, 0.149, 0.05, 0.005)
    new = descent_trajectory(0.0, 0.2994, 0.0, 0.0, 0.169, 0.149, 0.05, 0.005)
    t = np.arange(0.0, 6.0, 0.001)
    diff = np.array([np.subtract(old(x), new(x)) for x in t])
    assert np.abs(diff).max() < 1e-8


def test_descent_stops_smoothly_at_the_floor():
    tr = descent_trajectory(0.0, 0.2994, 0.0, 0.0, 0.169, 0.149, 0.05, 0.005)
    z = np.array([tr(x) for x in np.arange(0, 14.0, 0.001)])
    assert z[:, 0].min() >= 0.149 - 1e-9 and z[-1, 1] == 0.0
    assert np.abs(np.diff(z[:, 2])).max() < 2e-4          # acceleration continuous (no abrupt clamp)


def test_sequencer_without_interruption_equals_slide_profile():
    sp, sq = SlideProfile(0.10, 0.05), StrokeSequencer(0.10, 0.05)
    sq.start(8.7)
    err = max(max(abs(a - b) for a, b in zip(sp(x - 8.7), sq.eval(x))) for x in np.arange(8.7, 28.7, 0.001))
    assert err < 1e-12


@pytest.mark.parametrize("t_stop", [0.4, 1.3, 2.2, 3.1, 3.6])
def test_suspend_resume_is_bounded_continuous_and_never_returns_to_the_origin(t_stop):
    a_max, v_max, dt = 0.3, 0.05, 0.001
    sq = StrokeSequencer(0.10, v_max, a_max=a_max, stop_time=0.6)
    sq.start(0.0)
    traj = []
    for x in np.arange(0.0, 12.0, dt):
        if abs(x - t_stop) < dt / 2:
            sq.suspend(x)
        if abs(x - (t_stop + 2.0)) < dt / 2:
            sq.resume(x)
        traj.append(sq.eval(x))
    s, v, a = np.array(traj).T
    assert np.abs(v).max() <= v_max * (1 + 1e-9)
    assert np.abs(a).max() <= a_max * (1 + 1e-4)
    assert np.abs(np.diff(s)).max() <= v_max * dt * (1 + 1e-4) + 1e-9      # no position jump at any switch
    assert np.abs(np.diff(v)).max() <= a_max * dt * 1.5                    # no velocity jump
    assert np.abs(np.diff(a)).max() < 3e-3                                 # acceleration continuous
    assert s.min() >= -5e-5 and s.max() <= 0.10 + 5e-5                     # stays inside the stroke range
    k_stop = int((t_stop + 1.5) / dt)                                      # at rest between stop and resume
    assert abs(v[k_stop]) < 1e-12 and abs(a[k_stop]) < 1e-12
    # after the resume the stroke continues toward its original endpoint instead of restarting from s = 0
    k_res = int((t_stop + 2.0) / dt)
    assert abs(s[k_res + 1] - s[k_res]) < 1e-4
    assert sq.strokes >= 2


def test_descent_from_a_moving_state_respects_the_declared_acceleration():
    tr = descent_trajectory(0.0, 0.1595, 0.02, 0.0, 0.1612, 0.149, 0.05, 0.005, a_max=0.3)
    z = np.array([tr(x) for x in np.arange(0, 8, 0.001)])
    assert np.abs(z[:, 2]).max() <= 0.3 * (1 + 1e-4)
    assert z[:, 0].min() >= 0.149 - 1e-9
    assert np.abs(np.diff(z[:, 0])).max() <= 0.06 * 0.001 + 1e-9


def test_move_to_and_stop_trajectories_are_rest_to_rest():
    tr = move_to_trajectory(0.0, 0.10, 0.03, 0.0, 0.12, v_max=0.05, a_max=0.3, T_min=0.6)
    end = tr(100.0)
    assert end == pytest.approx((0.12, 0.0, 0.0), abs=1e-12)
    st = stop_trajectory(0.0, 0.0, 0.3, 0.0, 0.6, a_max=0.5)
    assert st(100.0)[1] == 0.0 and max(abs(st(x)[2]) for x in np.arange(0, 3, 0.001)) <= 0.5 * (1 + 1e-4)
