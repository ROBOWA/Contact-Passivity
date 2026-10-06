import mujoco
import numpy as np

from mujoco_panda.simulation import load_model
from mujoco_panda.ports import ARM_ACTUATORS, ARM_JOINTS


def test_seven_explicit_arm_dofs_and_direct_torque_mapping():
    model, data, h = load_model()
    assert model.nq == model.nv == model.nu == 7
    assert tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in h.joint_ids) == ARM_JOINTS
    assert tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in h.actuator_ids) == ARM_ACTUATORS
    assert np.array_equal(model.actuator_trnid[h.actuator_ids, 0], h.joint_ids)
    assert np.allclose(model.actuator_gainprm[h.actuator_ids, 0], 1)
    assert np.allclose(model.actuator_biasprm[h.actuator_ids, :3], 0)
    command = np.array([1., -2., 3., -4., 5., -6., 7.])
    data.ctrl[h.actuator_ids] = command; mujoco.mj_forward(model, data)
    assert np.allclose(data.actuator_force[h.actuator_ids], command)
    assert np.allclose(data.qfrc_actuator[h.dof_ids], command)
    assert np.allclose(model.actuator_ctrlrange[h.actuator_ids, 1], [87]*4 + [12]*3)


def test_only_tool_table_collision_is_enabled():
    model, _, h = load_model()
    enabled = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    assert set(enabled) == {*h.multipoint_pad_geoms, h.table_geom}
    assert len(h.multipoint_pad_geoms) == 4
    assert len(h.pad_geoms) == 5
    assert len(set(model.geom_bodyid[h.multipoint_pad_geoms])) == 1
