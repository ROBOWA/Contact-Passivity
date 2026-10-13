"""Numerical inverse kinematics for a site pose, used to set the initial state.

solve_site_ik() finds arm joint angles that place a named site at a target
position and (optionally) orientation. It is a damped least-squares iteration
on the site Jacobian with joint-limit clamping, a null-space pull toward a
preferred configuration (the redundant 7th degree of freedom), a step limit,
and a few deterministic random restarts. Only kinematics are evaluated
(mj_kinematics + mj_comPos on a scratch MjData); the simulation state is not
touched.

Conventions: positions in m in the world frame; quaternions are MuJoCo order
(w, x, y, z); quat_from_rpy(roll, pitch, yaw) builds R = Rz(yaw) Ry(pitch) Rx(roll),
i.e. fixed-axis x-y-z rotations. The orientation error is the rotation vector
(world frame) that takes the current site frame to the desired one.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import ArmIndex, HOME_KEYFRAME


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray            # arm joint angles, rad
    converged: bool
    pos_err: float           # m, residual position error
    ori_err: float           # rad, residual orientation error (0 when orientation is free)
    iterations: int
    limit_margin: float      # rad, smallest distance of q to a joint limit
    seed_index: int          # 0 = provided/home seed, >= 1 random restart

    def summary(self) -> str:
        state = "converged" if self.converged else "NOT converged"
        return (f"IK {state} in {self.iterations} it (seed {self.seed_index}): "
                f"pos err {self.pos_err:.2e} m, ori err {self.ori_err:.2e} rad, "
                f"limit margin {self.limit_margin:.3f} rad, q = {np.array2string(self.q, precision=4)}")


# --------------------------------------------------------------------------- rotations

def quat_from_rpy(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """Quaternion (w, x, y, z) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    qx, qy, qz = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(qx, np.array([1.0, 0, 0]), roll)
    mujoco.mju_axisAngle2Quat(qy, np.array([0, 1.0, 0]), pitch)
    mujoco.mju_axisAngle2Quat(qz, np.array([0, 0, 1.0]), yaw)
    tmp, out = np.zeros(4), np.zeros(4)
    mujoco.mju_mulQuat(tmp, qy, qx)
    mujoco.mju_mulQuat(out, qz, tmp)
    return out


def quat_from_rotmat(rot: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=float).reshape(9))
    return q


def rotmat_from_quat(quat: np.ndarray) -> np.ndarray:
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, dtype=float))
    return rot.reshape(3, 3)


def orientation_error(quat_desired: np.ndarray, rot_current: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) taking the current orientation to the desired one."""
    quat = quat_from_rotmat(rot_current)
    quat_conj, quat_err, err = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(quat_conj, quat)
    mujoco.mju_mulQuat(quat_err, np.asarray(quat_desired, dtype=float), quat_conj)
    mujoco.mju_quat2Vel(err, quat_err, 1.0)
    return err


# --------------------------------------------------------------------------- solver

def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site: str) -> tuple[np.ndarray, np.ndarray]:
    """(position, quaternion wxyz) of a site for the kinematics currently in `data`."""
    sid = model.site(site).id
    return data.site_xpos[sid].copy(), quat_from_rotmat(data.site_xmat[sid])


def solve_site_ik(model: mujoco.MjModel, site: str, pos, quat=None, *, arm: ArmIndex | None = None,
                  q_seed=None, q_pref=None, pos_tol: float = 1e-5, ori_tol: float = 1e-4,
                  max_iter: int = 300, damping: float = 0.02, step_max: float = 0.3,
                  null_gain: float = 0.2, null_tol: float = 1e-4, n_restarts: int = 8,
                  rng_seed: int = 0) -> IKResult:
    """Joint angles placing `site` at `pos` (m) with orientation `quat` (wxyz, or None for free).

    q_seed is the first iterate (default: q_pref, else the home keyframe); q_pref is the
    configuration the redundant degree of freedom is pulled toward (default: the seed).
    Deterministic random restarts inside the joint limits are tried if the seed fails. After the
    task tolerances are met the iteration continues on the redundant degree(s) of freedom until the
    projected pull toward q_pref is below null_tol, so solutions are as close to q_pref as possible.
    """
    arm = arm or ArmIndex.from_model(model)
    sid = model.site(site).id
    pos = np.asarray(pos, dtype=float).reshape(3)
    quat = None if quat is None else np.asarray(quat, dtype=float).reshape(4) / np.linalg.norm(quat)
    lo, hi = model.jnt_range[arm.joint_ids, 0], model.jnt_range[arm.joint_ids, 1]
    home = model.key_qpos[model.key(HOME_KEYFRAME).id, arm.qpos_adr]
    q_pref = np.asarray(q_pref if q_pref is not None else (q_seed if q_seed is not None else home), float).reshape(7)
    q_seed = np.asarray(q_seed if q_seed is not None else q_pref, float).reshape(7)
    rng = np.random.default_rng(rng_seed)
    seeds = [q_seed] + [lo + rng.random(7) * (hi - lo) for _ in range(n_restarts)]

    data = mujoco.MjData(model)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    n_task = 6 if quat is not None else 3
    best: IKResult | None = None

    for seed_index, seed in enumerate(seeds):
        q = np.clip(seed, lo, hi)
        it = 0
        for it in range(1, max_iter + 1):
            data.qpos[arm.qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            e_p = pos - data.site_xpos[sid]
            e_o = orientation_error(quat, data.site_xmat[sid]) if quat is not None else np.zeros(0)
            pos_err, ori_err = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_o)) if quat is not None else 0.0
            task_done = pos_err < pos_tol and ori_err < ori_tol
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            jac = (np.vstack((jacp, jacr)) if quat is not None else jacp)[:, arm.dof_adr]
            err = np.concatenate((e_p, e_o))
            jjt = jac @ jac.T + damping ** 2 * np.eye(n_task)
            dq = jac.T @ np.linalg.solve(jjt, err)              # damped least-squares task step
            pull = np.zeros(7)
            if null_gain > 0:
                # Exact kernel of J (SVD) so the posture pull cannot leak into the task error.
                _, sing, vt = np.linalg.svd(jac)
                rank = int(np.sum(sing > 1e-6 * sing[0]))
                kernel = vt[rank:].T                             # 7 x (7 - rank)
                pull = kernel @ (kernel.T @ (null_gain * (q_pref - q)))
                dq += pull
            # Once the task is met, keep polishing the redundant motion toward q_pref until the
            # projected pull is negligible (or the joint limits block it).
            if task_done and (np.linalg.norm(pull) < null_tol or np.allclose(np.clip(q + pull, lo, hi), q)):
                break
            scale = np.max(np.abs(dq))
            if scale > step_max:
                dq *= step_max / scale
            q = np.clip(q + dq, lo, hi)
        margin = float(np.min(np.minimum(q - lo, hi - q)))
        result = IKResult(q=q.copy(), converged=pos_err < pos_tol and ori_err < ori_tol, pos_err=pos_err,
                          ori_err=ori_err, iterations=it, limit_margin=margin, seed_index=seed_index)
        if best is None or (result.converged and not best.converged) or \
                (result.converged == best.converged and result.pos_err + result.ori_err < best.pos_err + best.ori_err):
            best = result
        if result.converged:
            break
    return best
