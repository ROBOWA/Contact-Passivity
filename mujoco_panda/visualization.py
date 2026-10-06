"""Viewer-only force and reference markers (never part of dynamics)."""

import mujoco
import numpy as np


def add_force_arrow(scene, point, force, nominal=7.0):
    magnitude = float(np.linalg.norm(force))
    if magnitude < 1e-8 or scene.ngeom >= scene.maxgeom:
        return
    direction = np.asarray(force) / magnitude
    length = .08 + .18 * min(1., magnitude / nominal)
    end = np.asarray(point) + np.array([0., 0., .10])
    start = end - direction * length
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                        np.zeros(3), np.eye(3).ravel(),
                        np.array([.95, .1, .35, 1.], dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .016, start, end)
    scene.ngeom += 1


def add_reference_marker(scene, position):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    target = np.asarray(position) + np.array([0., 0., .045])
    start = np.asarray(position) + np.array([0., -.13, .17])
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.array([.1, .95, .35, .75], dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                         .012, start, target)
    scene.ngeom += 1
