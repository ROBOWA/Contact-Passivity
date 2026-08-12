"""Contact-extraction tests: filtering, signs, Jacobians, projectors."""

import mujoco
import numpy as np

from mujoco_sliding.config import C_PROJ, MODEL_XML_PATH, NORMAL, TANGENT, U_PROJ
from mujoco_sliding.contact_extraction import (
    ModelHandles,
    extract_task_contact,
    site_jacobian,
)
from mujoco_sliding.simulation import reset_to_keyframe

from .conftest import press_pad_into_surface


def test_no_contact_above_surface(model, data, handles):
    res = extract_task_contact(model, data, handles)
    assert not res.active
    assert res.ncontacts == 0
    assert np.allclose(res.f_task, 0.0)
    assert res.f_n == 0.0
    assert res.dist > 0.05  # pad-bottom gap reported even out of contact


def test_contact_active_when_pad_reaches_surface(model, data, handles):
    press_pad_into_surface(model, data, handles, depth=0.002)
    res = extract_task_contact(model, data, handles)
    assert res.active
    assert res.ncontacts >= 1
    assert res.dist < 0.0
    assert abs(res.penetration - 0.002) < 5e-4
    # Contact position is at the surface plane, under the pad.
    assert abs(res.positions[0][2]) < 0.005


def test_normal_force_upward_and_nonnegative(model, data, handles):
    press_pad_into_surface(model, data, handles, depth=0.002)
    res = extract_task_contact(model, data, handles)
    assert res.f_n > 0.0, "force on the EE must push it up, away from the surface"
    assert res.f_task[2] > 0.0
    # The world normal stored for the pad points up (into the pad).
    assert res.normals[0][2] > 0.9


def test_friction_opposes_sliding_velocity(model, data, handles):
    """Kinetic friction opposes the sliding velocity in both directions.

    This is checked dynamically (a short stepped episode with gravity
    compensation and a constant downward push): a single static mj_forward
    with an imposed qvel is not a fair probe, because the soft-friction
    constraint reference can transiently point along the motion.
    """
    for direction in (+1.0, -1.0):
        reset_to_keyframe(model, data)
        press_pad_into_surface(model, data, handles, depth=0.001)
        jp = site_jacobian(model, data, handles.ee_site)
        data.qvel[:] = np.linalg.lstsq(jp, direction * TANGENT * 0.1, rcond=None)[0]
        mujoco.mj_forward(model, data)
        fts, vxs = [], []
        for _ in range(200):
            jp = site_jacobian(model, data, handles.ee_site)
            data.qfrc_applied[:] = data.qfrc_bias + jp.T @ np.array([0.0, 0.0, -8.0])
            mujoco.mj_step(model, data)
            res = extract_task_contact(model, data, handles)
            if res.active and abs(res.v_contact[0]) > 0.02:
                fts.append(res.f_t)
                vxs.append(res.v_contact[0])
        assert len(fts) > 50, "pad must keep sliding in contact"
        assert np.mean(vxs) * direction > 0.0
        assert np.mean(fts) * np.mean(vxs) < 0.0, \
            "friction force must oppose sliding velocity"


def test_extraction_ignores_unrelated_contacts(model, handles):
    # Load a variant with a free-falling debris box that contacts the surface,
    # far away from the robot.
    xml = open(MODEL_XML_PATH).read().replace(
        "</worldbody>",
        '<body name="debris" pos="2.0 0 0.03"><freejoint/>'
        '<geom name="debris_geom" type="box" size="0.03 0.03 0.03" '
        'contype="1" conaffinity="1" mass="0.1"/></body></worldbody>',
    ).replace(
        # extend the keyframe with the debris freejoint (3 pos + 4 quat)
        'qpos="0.0076 1.9930"',
        'qpos="0.0076 1.9930 2.0 0 0.03 1 0 0 0"',
    )
    m2 = mujoco.MjModel.from_xml_string(xml)
    d2 = mujoco.MjData(m2)
    key = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_KEY, "init")
    mujoco.mj_resetDataKeyframe(m2, d2, key)
    for _ in range(200):  # let the debris land; pin the arm above the surface
        d2.qpos[:2] = (0.0076, 1.9930)
        d2.qvel[:2] = 0.0
        mujoco.mj_step(m2, d2)
    assert d2.ncon > 0, "debris-surface contacts exist"
    h2 = ModelHandles.from_model(m2)
    res = extract_task_contact(m2, d2, h2)
    assert not res.active, "pad is above the surface; debris contacts are ignored"
    assert res.ncontacts == 0
    assert np.allclose(res.f_task, 0.0)


def test_jacobian_dimensions_and_velocity_consistency(model, data, handles):
    jp = site_jacobian(model, data, handles.ee_site)
    assert jp.shape == (3, model.nv) == (3, 2)
    data.qvel[:] = [0.3, -0.2]
    mujoco.mj_forward(model, data)
    jp = site_jacobian(model, data, handles.ee_site)
    # Compare J qdot with MuJoCo's own site velocity.
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE,
                             handles.ee_site, vel6, 0)
    np.testing.assert_allclose(jp @ data.qvel, vel6[3:], atol=1e-10)
    # Planarity: no y velocity is reachable.
    assert np.allclose(jp[1], 0.0, atol=1e-12)


def test_projectors():
    # C + U spans exactly the task plane (x and z); CU = 0.
    np.testing.assert_allclose(C_PROJ + U_PROJ, np.diag([1.0, 0.0, 1.0]))
    np.testing.assert_allclose(C_PROJ @ U_PROJ, np.zeros((3, 3)))
    np.testing.assert_allclose(U_PROJ @ U_PROJ, U_PROJ)
    np.testing.assert_allclose(C_PROJ @ C_PROJ, C_PROJ)
    assert TANGENT @ NORMAL == 0.0


def test_task_power_definition(model, data, handles):
    press_pad_into_surface(model, data, handles, depth=0.002,
                           qvel=np.array([0.1, -0.05]))
    res = extract_task_contact(model, data, handles)
    np.testing.assert_allclose(res.p_task, res.f_task @ res.v_contact)
