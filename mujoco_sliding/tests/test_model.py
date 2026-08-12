"""Model-structure tests: loading, DOFs, planarity, collision filtering."""

import mujoco
import numpy as np

from mujoco_sliding.config import MODEL_XML_PATH, PAD_GEOM, SURFACE_GEOM


def test_model_loads(model):
    assert model.nq == 2
    assert isinstance(model, mujoco.MjModel)


def test_two_actuated_hinge_dofs_about_y(model):
    assert model.njnt == 2
    assert model.nu == 2
    for j in range(model.njnt):
        assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        # Hinge axis is the local/world y axis => motion stays in x-z.
        np.testing.assert_allclose(model.jnt_axis[j], [0.0, 1.0, 0.0])
    # Torque motors on both joints.
    for a in range(model.nu):
        assert model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT


def test_only_pad_can_collide_with_surface(model):
    surf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SURFACE_GEOM)
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PAD_GEOM)
    for g in range(model.ngeom):
        if g == surf:
            continue
        compatible = (
            model.geom_contype[g] & model.geom_conaffinity[surf]
            or model.geom_contype[surf] & model.geom_conaffinity[g]
        )
        if g == pad:
            assert compatible, "pad must collide with the surface"
        else:
            assert not compatible, f"geom {g} must not collide with the surface"


def test_contact_dimensionality_and_timestep(model):
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PAD_GEOM)
    surf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SURFACE_GEOM)
    assert model.geom_condim[pad] == 3
    assert model.geom_condim[surf] == 3
    assert model.opt.timestep == 0.001
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_initial_pose_above_surface_nonsingular(model, data):
    ee = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")]
    pad_radius = 0.05
    assert ee[2] > pad_radius + 0.05, "pad starts well above the surface"
    assert abs(ee[1]) < 1e-9, "motion is confined to the x-z plane"
    # Elbow well away from the straight-arm singularity.
    assert 0.3 < data.qpos[1] < 2.8


def test_xml_path_packaged():
    assert MODEL_XML_PATH.endswith("planar_two_link.xml")
