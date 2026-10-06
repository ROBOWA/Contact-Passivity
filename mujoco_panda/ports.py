"""Spatial contact and human ports with exact generalized-power accounting.

Positive power means the table or human injects energy into the robot.
Every task-contact contribution is evaluated at its own contact point; no
summed-force/average-velocity approximation is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
MULTIPOINT_PAD_GEOMS = tuple(f"tool_contact_{corner}" for corner in
                             ("front_left", "front_right", "rear_left", "rear_right"))


def _id(model, obj, name: str) -> int:
    value = mujoco.mj_name2id(model, obj, name)
    if value < 0:
        raise ValueError(f"model has no {name!r}")
    return value


@dataclass(frozen=True)
class PandaHandles:
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    actuator_ids: np.ndarray
    table_geom: int
    pad_geoms: np.ndarray
    multipoint_pad_geoms: np.ndarray
    single_pad_geom: int
    tool_site: int
    forearm_site: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "PandaHandles":
        joint_ids = np.array([_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS])
        actuator_ids = np.array([_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS])
        qpos_ids = model.jnt_qposadr[joint_ids].copy()
        dof_ids = model.jnt_dofadr[joint_ids].copy()
        multipoint_pads = np.array([_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                                    for name in MULTIPOINT_PAD_GEOMS])
        single_pad = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "tool_contact_single")
        pads = np.r_[multipoint_pads, single_pad]
        return cls(
            joint_ids=joint_ids, qpos_ids=qpos_ids, dof_ids=dof_ids,
            actuator_ids=actuator_ids,
            table_geom=_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table"),
            pad_geoms=pads, multipoint_pad_geoms=multipoint_pads,
            single_pad_geom=single_pad,
            tool_site=_id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site"),
            forearm_site=_id(model, mujoco.mjtObj.mjOBJ_SITE, "forearm_site"),
        )


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int):
    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, site_id)
    return jp, jr


def point_jacobian(model: mujoco.MjModel, data: mujoco.MjData,
                   point: np.ndarray, body_id: int):
    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, np.asarray(point), body_id)
    return jp, jr


@dataclass
class ContactPoint:
    position: np.ndarray
    force: np.ndarray
    moment: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    generalized_force: np.ndarray
    power: float
    distance: float


@dataclass
class ContactPort:
    points: list[ContactPoint] = field(default_factory=list)
    generalized_force: np.ndarray = field(default_factory=lambda: np.zeros(0))
    net_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    net_moment_at_tool: np.ndarray = field(default_factory=lambda: np.zeros(3))
    power: float = 0.0
    normal_force: float = 0.0
    active: bool = False
    gap: float = np.inf


def extract_task_contact(model: mujoco.MjModel, data: mujoco.MjData,
                         h: PandaHandles) -> ContactPort:
    """Return the table wrench on the tool, summed in generalized space."""
    out = ContactPort(generalized_force=np.zeros(model.nv))
    wrench = np.zeros(6)
    tool_origin = data.site_xpos[h.tool_site].copy()
    distances = []
    for i in range(data.ncon):
        c = data.contact[i]
        pad_g2 = c.geom1 == h.table_geom and c.geom2 in h.pad_geoms
        pad_g1 = c.geom1 in h.pad_geoms and c.geom2 == h.table_geom
        if not (pad_g1 or pad_g2):
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        frame = np.asarray(c.frame).reshape(3, 3)
        sign = 1.0 if pad_g2 else -1.0
        force = sign * (frame.T @ wrench[:3])
        moment = sign * (frame.T @ wrench[3:])
        position = np.asarray(c.pos).copy()
        pad_geom = int(c.geom2 if pad_g2 else c.geom1)
        jp, jr = point_jacobian(model, data, position,
                                int(model.geom_bodyid[pad_geom]))
        v, omega = jp @ data.qvel, jr @ data.qvel
        tau = jp.T @ force + jr.T @ moment
        power = float(force @ v + moment @ omega)
        out.points.append(ContactPoint(position, force, moment, v, omega, tau, power, float(c.dist)))
        out.generalized_force += tau
        out.net_force += force
        out.net_moment_at_tool += moment + np.cross(position - tool_origin, force)
        out.power += power
        distances.append(float(c.dist))
    out.active = bool(out.points)
    out.normal_force = float(out.net_force[2])
    if distances:
        out.gap = min(distances)
    else:
        # Every collision element is a sphere with a common rigid pad body.
        pad_bottom = min(data.geom_xpos[geom, 2] - model.geom_size[geom, 0]
                         for geom in h.pad_geoms)
        table_top = data.geom_xpos[h.table_geom, 2] + model.geom_size[h.table_geom, 2]
        out.gap = float(pad_bottom - table_top)
    return out


@dataclass
class SpatialPort:
    site_id: int
    body_id: int
    point: np.ndarray
    wrench: np.ndarray
    jacobian: np.ndarray
    twist: np.ndarray
    generalized_force: np.ndarray
    power: float


def spatial_site_port(model: mujoco.MjModel, data: mujoco.MjData,
                      site_id: int, wrench: np.ndarray) -> SpatialPort:
    """Map a world-frame [force, moment] at a named site to joint space."""
    wrench = np.asarray(wrench, dtype=float)
    jp, jr = site_jacobian(model, data, site_id)
    jac = np.vstack((jp, jr))
    twist = jac @ data.qvel
    tau = jac.T @ wrench
    return SpatialPort(site_id, int(model.site_bodyid[site_id]),
                       data.site_xpos[site_id].copy(), wrench.copy(), jac,
                       twist, tau, float(wrench @ twist))


def apply_spatial_port(model: mujoco.MjModel, data: mujoco.MjData,
                       port: SpatialPort) -> None:
    mujoco.mj_applyFT(model, data, port.wrench[:3], port.wrench[3:],
                      port.point, port.body_id, data.qfrc_applied)


def assert_power_identity(port: SpatialPort, qvel: np.ndarray,
                          atol: float = 1e-10) -> None:
    mapped = float(port.generalized_force @ qvel)
    if not np.isclose(port.power, mapped, atol=atol, rtol=1e-10):
        raise AssertionError(f"spatial-port power mismatch: {port.power} != {mapped}")
