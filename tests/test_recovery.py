"""Priority 3: controlled, continuous contact loss and reacquisition (fixtures declared in scenarios.py)."""

import numpy as np
import pytest

from kuka_table_sim import metrics as mx
from kuka_table_sim import scenarios as sc
from kuka_table_sim.force_control import CODE_PHASE, ForceApproachController, ForceControlConfig

CFG = ForceControlConfig()
DT_C = 0.001


def _events(res, t0=11.0):
    return mx.fsm_events(res.log, (t0, float(res.log.phys["t"][-1])))


def test_nominal_sliding_has_no_recovery_events_and_meets_the_gates(nominal):
    gates, _ = sc.nominal_gates(nominal)
    assert all(gates.values()), [k for k, v in gates.items() if not v]
    assert [b for _, _, b in mx.transitions(nominal.log)] == ["contact", "slide"]


def test_short_interruption_is_one_search_without_loss_and_sliding_resumes(loss_short):
    log, ev = loss_short.log, _events(loss_short)
    assert ev == dict(n_uncertain=1, n_lost=0, n_reacquired=1, n_hold=0)
    ab = mx.contact_absence(log, (11.5, 22.0))
    assert ab["geometric_absence"]["events"] == 1 and ab["geometric_absence"]["longest_s"] < CFG.release_time
    assert CODE_PHASE[int(log.ctl["phase"][-1])] == "slide"


def test_loss_escalates_once_and_reacquires_at_the_original_surface(loss_reacquire):
    log, ev = loss_reacquire.log, _events(loss_reacquire)
    assert ev == dict(n_uncertain=1, n_lost=1, n_reacquired=1, n_hold=0)
    ab = mx.contact_absence(log, (11.5, 26.0))
    assert ab["geometric_absence"]["events"] == 1 and ab["geometric_absence"]["longest_s"] > CFG.release_time
    assert ab["low_load"]["events"] == 1
    p = log.phys
    final = p["tcp"][-1]
    assert abs(final[2] - loss_reacquire.sim.controller.z_surface) < 2e-3
    assert CODE_PHASE[int(log.ctl["phase"][-1])] == "slide"


def test_sustained_loss_ends_in_a_bounded_hold_and_resumes_only_on_request(loss_sustained):
    log, info = loss_sustained.log, loss_sustained.info
    c = log.ctl
    tr = mx.transitions(log)
    t_hold = next(t for t, _, b in tr if b == "hold")
    t_search2 = next(t for t, a, b in tr if a == "hold")
    assert t_search2 == pytest.approx(info["t_resume"], abs=0.01)                 # HOLD persists until the request
    assert t_hold - next(t for t, _, b in tr if b == "search") <= CFG.search_time_max + 0.01     # bounded search time
    ph = c["phase"].astype(int)
    held = (c["t"] > t_hold + 0.1) & (c["t"] < t_search2 - 0.01)
    assert (ph[held] == 4).all()
    ev = _events(loss_sustained)
    assert ev["n_lost"] == 1 and ev["n_hold"] == 1 and ev["n_reacquired"] == 1       # one loss, however many searches
    # bounded search distance: the tip never goes below the search floor
    assert loss_sustained.log.phys["tcp"][:, 2].min() >= loss_sustained.sim.controller.z_floor - 1e-3
    # HOLD retreats with a bounded quintic and then stays at the standoff above the surface, not at a fixed far height
    standoff = loss_sustained.sim.controller.z_surface + CFG.creep_distance
    settled = (c["t"] > t_hold + 2.0) & (c["t"] < t_search2 - 0.01)
    assert settled.sum() > 1000 and np.allclose(c["x_d"][settled, 2], standoff, atol=1e-6)
    retreat = (c["t"] > t_hold) & (c["t"] <= t_hold + 2.0)
    assert np.nanmax(np.abs(np.diff(c["x_d"][retreat, 2]))) / DT_C <= CFG.recovery_speed_max * 1.02
    assert CODE_PHASE[int(c["phase"][-1])] == "slide"


@pytest.mark.parametrize("name", ["loss_short", "loss_reacquire", "loss_sustained", "removal_stress"])
def test_active_reference_is_continuous_within_the_declared_limits(name, request):
    res = request.getfixturevalue(name)
    log = res.log
    t_end = float(log.phys["t"][-1])
    rc = mx.reference_continuity(log, (11.0, t_end))
    tg, nm = rc["tangential"], rc["normal"]
    v_max, a_max = CFG.recovery_speed_max, CFG.recovery_accel_max
    # finite differences of the ACTUAL active reference, not the reported fields
    assert tg["pos_step_max_mm"] <= v_max * DT_C * 1e3 * 1.02 + 1e-3
    assert nm["pos_step_max_mm"] <= v_max * DT_C * 1e3 * 1.02 + 1e-3
    assert tg["peak_accel_fd"] <= a_max * 1.05 and nm["peak_accel_fd"] <= a_max * 1.05
    assert tg["pos_residual_max_mm"] < 5e-3 and nm["pos_residual_max_mm"] < 5e-3         # reported velocity agrees with positions
    assert tg["vel_residual_max_mm_s"] < 0.5 and nm["vel_residual_max_mm_s"] < 0.5          # reported acceleration agrees
    for act in rc["normal_activation"]:                                                  # activating a position loop: no command step
        assert abs(act["pos_err_mm"]) < 1.0 and abs(act["vel_err_mm_s"]) < 5.0
    # no snap back to the stroke origin: the in-plane reference never moves by more than the stroke
    xd = log.ctl["x_d"][:, 0]
    assert np.nanmax(xd) - np.nanmin(xd) <= CFG.slide_distance + 1e-3


@pytest.mark.parametrize("name", ["loss_short", "loss_reacquire", "loss_sustained", "removal_stress"])
def test_no_actuator_saturation_and_no_unexpected_contact_during_recovery(name, request):
    res = request.getfixturevalue(name)
    tq = mx.torque_stats(res.log, (11.0, float(res.log.phys["t"][-1])))
    assert tq["saturated_substeps"] == 0 and tq["peak_applied_utilisation"] < 0.5
    assert int(res.log.phys["n_unexpected"].sum()) == 0


def test_integrator_states_follow_the_declared_freeze_reset_taper_policy(loss_reacquire):
    c = loss_reacquire.log.ctl
    ph = c["phase"].astype(int)
    k_in = int(np.argmax(ph == 3))                           # first SEARCH update
    k_out = k_in + int(np.argmax(ph[k_in:] != 3))            # first update after it
    # frozen: no integrator moves while the previous output could not be trusted (SEARCH, before the taper starts)
    t_lost = loss_reacquire.sim.controller.t_lost
    k_lost = int(np.searchsorted(c["t"], t_lost))
    assert np.array_equal(c["f_int"][k_in:k_lost], np.full(k_lost - k_in, c["f_int"][k_in]))
    assert np.array_equal(c["tan_int"][k_in:k_lost], np.repeat(c["tan_int"][k_in:k_in + 1], k_lost - k_in, axis=0))
    # reset: f_int is zero from the LOST declaration, and again at the new CONTACT
    assert (c["f_int"][k_lost:k_out + 1] == 0.0).all()
    # taper: the applied integral term is scaled down by a monotonically decreasing smootherstep weight (no command step)
    # over integral_taper_time, after which the frozen states are reset to exactly zero
    m = c["int_scale"][k_lost:k_lost + int(CFG.integral_taper_time / DT_C) - 1]      # up to the update where the taper completes
    assert m[0] > 0.99 and (np.diff(m) <= 1e-12).all() and m[-1] < 0.01
    k_done = k_lost + 10 + int(np.argmax(c["int_scale"][k_lost + 10:] == 1.0))        # the update at which the taper completes
    assert abs(k_done - (k_lost + int(CFG.integral_taper_time / DT_C))) <= 2
    assert (c["tan_int"][k_done] == 0.0).all() and (c["rot_int"][k_done] == 0.0).all()   # reset to exactly zero
    assert (c["f_int"][k_lost:k_out + 1] == 0.0).all()
    # integrators never run between the LOST declaration and the end of the taper; if contact came back earlier than
    # that, the frozen state is simply held until the taper completes, otherwise they stay at zero until CONTACT
    if k_out < k_done:
        assert np.array_equal(c["tan_int"][k_out:k_done - 1], np.repeat(c["tan_int"][k_out:k_out + 1], k_done - 1 - k_out, axis=0))
    else:
        assert (c["tan_int"][k_done:k_out] == 0.0).all() and (c["rot_int"][k_done:k_out] == 0.0).all()
    # after reacquisition the loop re-accumulates from zero
    assert np.linalg.norm(c["tan_int"][max(k_done, k_out) + 400]) > 0.0


def test_conditional_integration_freezes_integrators_when_the_final_torque_saturates():
    """Integral clamps alone do not establish anti-windup: with joint 6 limited far below what sliding needs,
    no integrator may change on an update that follows a saturated one."""
    res_sim = sc.make_force_sim(duration=16.0)
    ctl: ForceApproachController = res_sim.controller
    ctl.torque_limit = ctl.torque_limit.copy()
    ctl.torque_limit[5] = 0.8                                # sliding needs up to ~1.4 N m on joint 6
    log = sc.run_for(res_sim, 16.0)
    c = log.ctl
    sat = c["sat_any"] > 0.5
    assert sat.sum() > 500                                   # saturation actually occurred
    prev_sat = np.concatenate(([False], sat[:-1]))           # an update that follows a saturated one
    j = np.flatnonzero(prev_sat)[1:]
    assert (c["f_int"][j] == c["f_int"][j - 1]).all()
    assert np.array_equal(c["tan_int"][j], c["tan_int"][j - 1])
    assert np.array_equal(c["rot_int"][j], c["rot_int"][j - 1])


def test_abrupt_removal_stress_is_bounded(removal_stress):
    """STRESS TEST (the table is teleported away), not a reacquisition benchmark: the search is bounded and ends in a hold."""
    log, ctl = removal_stress.log, removal_stress.sim.controller
    assert [b for _, _, b in mx.transitions(log)][-2:] == ["search", "hold"]
    z = log.phys["tcp"][:, 2]
    assert z.min() >= ctl.z_floor - 1.5e-3                                        # never falls below the search floor
    assert z[-1] == pytest.approx(ctl.z_surface + CFG.creep_distance, abs=1e-3)   # standoff above the expected surface
    assert z[log.phys["t"] > 12.0].max() < ctl.z_surface + CFG.creep_distance + 2e-3


def test_contact_is_reacquired_only_when_it_physically_exists(loss_sustained):
    """Every CONTACT entry after the loss must coincide with physical tip/table contact: no latching onto inertial
    force while the tool hovers, retreats or is released (the old failure chain)."""
    res = loss_sustained
    p = res.log.phys
    entries = [t for t, _, b in mx.transitions(res.log) if b == "contact" and t > res.info["pull"]["t_start"]]
    assert entries
    for t in entries:
        k = int(np.argmax(p["t"] >= t))
        assert p["present"][k - 20:k + 1].any()


@pytest.mark.parametrize("name", ["loss_short", "loss_reacquire", "loss_sustained"])
def test_reacquisition_is_gentle(name, request):
    """Force peaks: after a CONTACT re-entry the true contact force must not exceed the target (the earlier
    design drove the tool into the table at ~0.1 m/s and peaked at about 31 N)."""
    res = request.getfixturevalue(name)
    peaks = mx.force_peaks(res.log, (11.0, float(res.log.phys["t"][-1])))["peak_after_contact_entry"]
    assert peaks, "no reacquisition happened"
    for _, peak in peaks:
        assert peak < CFG.force_desired * 1.1
    assert mx.force_peaks(res.log, (11.0, float(res.log.phys["t"][-1])))["peak_N"] < CFG.force_desired * 1.1


def test_search_reference_cannot_wind_up_while_a_disturbance_holds_the_tool_away(loss_reacquire):
    ctl = loss_reacquire.sim.controller
    c = loss_reacquire.log.ctl
    ph = c["phase"].astype(int)
    k = np.flatnonzero(ph == 3)
    z_ref, z_meas = c["x_d"][k, 2], c["x"][k, 2]
    lag = z_meas - z_ref
    assert lag.max() > 0.002                                                    # the pull really held the tool away
    assert c["search_rate"][k].min() < 0.2                                      # the governor slowed the descent clock ...
    assert z_ref.min() > ctl.z_good - 4e-3                                      # ... so the reference never sank below the surface
    # the governed reference stays acceleration-continuous and within the declared limits
    zd = c["v_d"][k, 2]
    assert np.abs(np.diff(zd)).max() / DT_C <= CFG.recovery_accel_max * 1.05
    assert np.abs(zd).max() <= CFG.recovery_speed_max
