"""Ground-truth task-contact extraction from MuJoCo.

MuJoCo generates the physical pad-surface contact; this module only *reads*
it. No separate plant-side contact model is created.

Contact-frame convention (MuJoCo):
  * ``data.contact[i].frame`` stores the contact frame as a flat 9-vector
    whose ROWS are the frame axes expressed in world coordinates:
    row 0 = contact normal (points from geom1 INTO geom2), rows 1-2 = the
    two tangential friction axes.
  * ``mujoco.mj_contactForce`` returns the 6-vector [f_normal, f_tan1,
    f_tan2, torques] expressed in that contact frame, with the normal
    component nonnegative. This is the force acting on geom2 (the geom the
    normal points into); geom1 receives the opposite force.

Transformation used here: with R = frame.reshape(3, 3) (rows = axes),
``f_world_on_geom2 = R.T @ f_contact[:3]``. If the end-effector pad is
geom1 of the contact the sign is flipped, so ``f_task`` below is ALWAYS the
force exerted on the robot end-effector, in the world frame. The sign
convention is validated experimentally in the tests: during contact the
normal component of ``f_task`` is upward (+z), and the friction component
opposes the tangential sliding velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import EE_SITE, NORMAL, PAD_GEOM, SURFACE_GEOM, TANGENT


@dataclass
class ModelHandles:
    """Cached MuJoCo ids used by extraction and control."""

    surface_geom: int
    pad_geom: int
    ee_site: int
    ee_body: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "ModelHandles":
        pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PAD_GEOM)
        surf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SURFACE_GEOM)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        if min(pad, surf, site) < 0:
            raise ValueError("model is missing a required geom/site name")
        return cls(
            surface_geom=surf,
            pad_geom=pad,
            ee_site=site,
            ee_body=model.geom_bodyid[pad],
        )


@dataclass
class ContactResult:
    """Structured task-contact information for one simulation step."""

    active: bool = False
    ncontacts: int = 0
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    dists: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Penetration is defined from contact distance: dist < 0 means the
    # surfaces overlap, so penetration = max(0, -dist) per contact.
    penetrations: np.ndarray = field(default_factory=lambda: np.zeros(0))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tangents1: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tangents2: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # Per-contact 6D wrench [force(3), torque(3)] in the CONTACT frame,
    # exactly as returned by mj_contactForce.
    wrenches_contact: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    # Per-contact world-frame force ON THE END-EFFECTOR PAD.
    forces_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # Net world-frame task-contact force on the end-effector.
    f_task: np.ndarray = field(default_factory=lambda: np.zeros(3))
    f_n: float = 0.0        # F_n = n^T f_task
    f_t: float = 0.0        # F_t = t^T f_task
    # Velocity of the material contact point on the pad (world frame); falls
    # back to the EE-site velocity when no contact is active.
    v_contact: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Task-contact power P_task = f_task^T v_contact (surface is static).
    p_task: float = 0.0
    # Representative scalar gap/penetration (min dist over matching contacts,
    # or the pad-bottom height above the surface when out of contact).
    dist: float = np.inf
    penetration: float = 0.0


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    """Translational Jacobian J_p (3 x nv) of a site, via mj_jacSite."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp


def point_jacobian(
    model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray, body_id: int
) -> np.ndarray:
    """Translational Jacobian (3 x nv) of a world point on a body, via mj_jac."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, np.asarray(point, dtype=float), body_id)
    return jacp


def extract_task_contact(
    model: mujoco.MjModel, data: mujoco.MjData, handles: ModelHandles
) -> ContactResult:
    """Extract the net pad-surface contact from the current mjData.

    Only contacts whose geom pair is exactly (surface, end_effector_pad) are
    considered; all other contacts in ``data.contact`` are ignored. The
    returned forces reflect the constraint solve of the most recent dynamics
    evaluation (``mj_forward`` or ``mj_step``) — see the simulation loop for
    the one-step-delay bookkeeping.
    """
    res = ContactResult()

    positions, dists, normals, t1s, t2s, wrenches, forces_w = [], [], [], [], [], [], []
    wrench6 = np.zeros(6)

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        pad_is_g2 = g1 == handles.surface_geom and g2 == handles.pad_geom
        pad_is_g1 = g1 == handles.pad_geom and g2 == handles.surface_geom
        if not (pad_is_g1 or pad_is_g2):
            continue  # unrelated contact

        mujoco.mj_contactForce(model, data, i, wrench6)
        frame = np.array(con.frame).reshape(3, 3)  # rows = axes in world
        # Force on geom2 in world coordinates; flip if the pad is geom1.
        f_world_on_g2 = frame.T @ wrench6[:3]
        f_on_pad = f_world_on_g2 if pad_is_g2 else -f_world_on_g2
        normal_world = frame[0] if pad_is_g2 else -frame[0]  # points into the pad

        positions.append(np.array(con.pos))
        dists.append(con.dist)
        normals.append(normal_world)
        t1s.append(frame[1].copy())
        t2s.append(frame[2].copy())
        wrenches.append(wrench6.copy())
        forces_w.append(f_on_pad)

    n = len(positions)
    if n > 0:
        res.active = True
        res.ncontacts = n
        res.positions = np.array(positions)
        res.dists = np.array(dists)
        res.penetrations = np.maximum(0.0, -res.dists)
        res.normals = np.array(normals)
        res.tangents1 = np.array(t1s)
        res.tangents2 = np.array(t2s)
        res.wrenches_contact = np.array(wrenches)
        res.forces_world = np.array(forces_w)
        res.f_task = res.forces_world.sum(axis=0)
        res.dist = float(res.dists.min())
        res.penetration = float(res.penetrations.max())

        # Velocity of the (force-weighted mean) contact point on the pad,
        # using mj_jac at the actual contact position: v_c = J_c qdot.
        mean_pos = res.positions.mean(axis=0)
        jc = point_jacobian(model, data, mean_pos, handles.ee_body)
        res.v_contact = jc @ data.qvel
    else:
        # Out of contact: report the pad-bottom gap above the surface and use
        # the EE-site velocity as the contact-point velocity surrogate.
        pad_center = data.geom_xpos[handles.pad_geom]
        pad_radius = model.geom_size[handles.pad_geom][0]
        res.dist = float(pad_center[2] - pad_radius)  # surface top is z = 0
        jp = site_jacobian(model, data, handles.ee_site)
        res.v_contact = jp @ data.qvel

    res.f_n = float(NORMAL @ res.f_task)
    res.f_t = float(TANGENT @ res.f_task)
    res.p_task = float(res.f_task @ res.v_contact)
    return res
