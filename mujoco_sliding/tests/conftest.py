"""Shared fixtures. The two full simulation runs are session-scoped so the
whole suite pays for them once."""

import mujoco
import numpy as np
import pytest

from mujoco_sliding.config import SimulationConfig
from mujoco_sliding.contact_extraction import ModelHandles
from mujoco_sliding.simulation import load_model, reset_to_keyframe, run_simulation


@pytest.fixture(scope="session")
def model():
    return load_model()


@pytest.fixture()
def data(model):
    d = mujoco.MjData(model)
    reset_to_keyframe(model, d)
    return d


@pytest.fixture(scope="session")
def handles(model):
    return ModelHandles.from_model(model)


@pytest.fixture(scope="session")
def default_log():
    return run_simulation(SimulationConfig())


@pytest.fixture(scope="session")
def human_log():
    return run_simulation(SimulationConfig().with_human_force())


def press_pad_into_surface(model, data, handles, depth=0.002, qvel=None):
    """Move the pad to a configuration penetrating the surface and run
    mj_forward so contact forces are solved for that state."""
    # Newton iterations on z only, using the site Jacobian.
    from mujoco_sliding.contact_extraction import site_jacobian

    target_z = model.geom_size[handles.pad_geom][0] - depth
    for _ in range(50):
        mujoco.mj_forward(model, data)
        z = data.site_xpos[handles.ee_site][2]
        err = target_z - z
        if abs(err) < 1e-10:
            break
        jz = site_jacobian(model, data, handles.ee_site)[2]
        dq = jz * err / max(jz @ jz, 1e-12)
        data.qpos += dq
    if qvel is not None:
        data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
