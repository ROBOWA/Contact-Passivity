"""Controller and human-interaction unit tests."""

import numpy as np

from mujoco_sliding.config import ControllerConfig, HumanForceConfig
from mujoco_sliding.contact_extraction import ContactResult, extract_task_contact
from mujoco_sliding.controller import Phase, SlidingForceController
from mujoco_sliding.human_interaction import HumanForce

from .conftest import press_pad_into_surface


def test_force_error_sign_convention(model, data, handles):
    """If measured F_n is too small, the commanded push must grow."""
    press_pad_into_surface(model, data, handles, depth=0.002)
    cfg = ControllerConfig()
    outs = {}
    for f_n in (0.0, 10.0):  # measured force below / above F_d
        ctl = SlidingForceController(model, handles, cfg)
        ctl.phase = Phase.RAMP
        ctl._t_ramp_start = data.time - cfg.ramp_duration  # ramp finished
        contact = extract_task_contact(model, data, handles)
        contact.f_n = f_n
        outs[f_n] = ctl.update(data, contact)
    assert outs[0.0].e_f > 0 > outs[10.0].e_f
    assert outs[0.0].f_push > outs[10.0].f_push
    # Commanded normal force is downward (negative z) when pushing.
    assert outs[0.0].f_cmd[2] < 0


def test_torque_limits_and_rate_limit(model, data, handles):
    cfg = ControllerConfig(tau_limit=1.0, tau_rate_limit=100.0)
    ctl = SlidingForceController(model, handles, cfg)
    contact = ContactResult()
    out1 = ctl.update(data, contact)
    assert np.all(np.abs(out1.tau) <= cfg.tau_limit + 1e-12)
    out2 = ctl.update(data, contact)
    dmax = cfg.tau_rate_limit * model.opt.timestep
    assert np.all(np.abs(out2.tau - out1.tau) <= dmax + 1e-12)


def test_integral_anti_windup(model, data, handles):
    press_pad_into_surface(model, data, handles, depth=0.002)
    cfg = ControllerConfig()
    ctl = SlidingForceController(model, handles, cfg)
    ctl.phase = Phase.RAMP
    ctl._t_ramp_start = data.time - cfg.ramp_duration
    contact = extract_task_contact(model, data, handles)
    contact.f_n = -100.0  # huge persistent error
    for _ in range(5000):
        ctl.update(data, contact)
    assert cfg.k_i * abs(ctl.integral) <= cfg.integral_limit + 1e-9


def test_torque_decomposition_consistency(model, data, handles):
    """tau_C + tau_U == tau_task (C + U is the identity on the x-z plane)."""
    press_pad_into_surface(model, data, handles, depth=0.002)
    ctl = SlidingForceController(model, handles, ControllerConfig())
    contact = extract_task_contact(model, data, handles)
    out = ctl.update(data, contact)
    np.testing.assert_allclose(out.tau_c + out.tau_u, out.tau_task, atol=1e-12)


def test_human_force_profile():
    cfg = HumanForceConfig(enabled=True, magnitude=3.0, direction=(1, 0, 0),
                           t_start=6.0, t_rise=0.5, t_hold=2.0, t_fall=0.5)
    h = HumanForce(cfg)
    assert np.allclose(h.force(0.0), 0.0)
    assert np.allclose(h.force(5.99), 0.0)
    np.testing.assert_allclose(h.force(6.25), [1.5, 0, 0])   # mid-rise
    np.testing.assert_allclose(h.force(7.5), [3.0, 0, 0])    # hold
    np.testing.assert_allclose(h.force(8.75), [1.5, 0, 0])   # mid-fall
    assert np.allclose(h.force(9.01), 0.0)
    # Disabled profile is identically zero.
    assert np.allclose(HumanForce(HumanForceConfig(enabled=False)).force(7.5), 0.0)


def test_human_power_sign_convention():
    """P_{h->r} > 0 when the human pushes along the EE velocity."""
    f_h = np.array([3.0, 0.0, 0.0])
    v_ee = np.array([0.05, 0.0, 0.0])
    p_h_to_r = f_h @ v_ee
    assert p_h_to_r > 0
    assert -p_h_to_r == -f_h @ v_ee  # P_{r->h} is its negative
    # Opposing push extracts power from the robot.
    assert (-f_h) @ v_ee < 0
