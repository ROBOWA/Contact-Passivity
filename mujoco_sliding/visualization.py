"""Shared MuJoCo visualization helpers.

The scripted human force is applied through ``qfrc_applied`` and therefore
has no physical geometry of its own. This module adds a bright arrow to an
``MjvScene`` so the interaction is visible in both the interactive viewer and
the presentation videos.
"""

from __future__ import annotations

import mujoco
import numpy as np


HUMAN_FORCE_RGBA = np.array([0.91, 0.20, 0.55, 1.0], dtype=np.float32)
REFERENCE_RGBA = np.array([1.0, 0.72, 0.10, 1.0], dtype=np.float32)


def add_human_force_arrow(
    scene: mujoco.MjvScene,
    ee_position: np.ndarray,
    force: np.ndarray,
    *,
    nominal_force: float = 3.0,
) -> bool:
    """Add a magenta human-force arrow ending above the end effector.

    The arrow length follows the force magnitude while retaining a small
    minimum length so the start and end of a smooth force ramp remain easy to
    see. ``False`` is returned when the force is zero or the scene has no
    remaining user-geometry slot.
    """
    f_h = np.asarray(force, dtype=float)
    magnitude = float(np.linalg.norm(f_h))
    if magnitude <= 1e-9 or scene.ngeom >= scene.maxgeom:
        return False

    direction = f_h / magnitude
    scale = min(magnitude / max(float(nominal_force), 1e-9), 1.0)
    length = 0.07 + 0.20 * scale

    # End the arrow just above the contact pad. For a blocking force in -x,
    # this places the tail to the right and makes the force visibly push into
    # the robot.
    arrow_end = np.asarray(ee_position, dtype=float) + np.array([0.0, 0.0, 0.13])
    arrow_start = arrow_end - length * direction

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        HUMAN_FORCE_RGBA,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.018,
        arrow_start,
        arrow_end,
    )
    scene.ngeom += 1
    return True


def add_reference_setpoint_marker(
    scene: mujoco.MjvScene,
    x_reference: float,
) -> bool:
    """Add a gold downward arrow at the tangential reference position."""
    if not np.isfinite(x_reference) or scene.ngeom >= scene.maxgeom:
        return False

    # Keep the marker above the surface and slightly off the robot's motion
    # plane. It points to the tangential location without implying a normal
    # position command (normal behavior is force controlled).
    marker_start = np.array([x_reference, -0.07, 0.36], dtype=float)
    marker_end = np.array([x_reference, -0.07, 0.14], dtype=float)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        REFERENCE_RGBA,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.014,
        marker_start,
        marker_end,
    )
    scene.ngeom += 1
    return True
