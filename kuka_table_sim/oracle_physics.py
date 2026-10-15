"""Known human port and isolated one-interval MuJoCo oracle.

SI units; forces act ON the robot. No controller, callbacks, RNG or recorder is
called by prediction. The immutable model is shared; integration state is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import mujoco
import numpy as np

from .scene import ArmIndex, TimingConfig


def integration_state(model, data):
    sig = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, sig), dtype=np.float64)
    mujoco.mj_getState(model, data, state, sig)
    return state


def restore_state(model, data, state):
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)


@dataclass(frozen=True)
class HumanPulse:
    start: float = 12.0
    ramp: float = 0.2
    hold: float = 1.0
    magnitude: float = 5.0
    direction: tuple = (-1.0, 0.0, 0.0)
    varying: bool = False  # plant samples each substep; predictor ALWAYS holds current force

    def __post_init__(self):
        if self.ramp <= 0 or self.hold < 0 or self.start < 0 or not np.all(np.isfinite(
                [self.start, self.ramp, self.hold, self.magnitude, *self.direction])):
            raise ValueError("invalid pulse")
        if np.shape(self.direction) != (3,) or np.linalg.norm(self.direction) == 0:
            raise ValueError("pulse direction must be a nonzero 3-vector")

    @property
    def end(self):
        return self.start + 2 * self.ramp + self.hold

    def force(self, t):
        u = t - self.start
        if u <= 0 or u >= 2 * self.ramp + self.hold:
            a = 0.0
        elif u < self.ramp:
            a = 0.5 - 0.5 * np.cos(np.pi * u / self.ramp)
        elif u < self.ramp + self.hold:
            a = 1.0
        else:
            a = 0.5 + 0.5 * np.cos(np.pi * (u - self.ramp - self.hold) / self.ramp)
        return self.magnitude * a * np.asarray(self.direction) / np.linalg.norm(self.direction)


class HumanPort:
    def __init__(self, model, body="link5", local=(0.0, 0.04, 0.12)):
        self.model = model
        self.body = model.body(body).id
        self.body_name = body
        self.local = np.asarray(local, dtype=float).copy()
        if self.local.shape != (3,) or not np.all(np.isfinite(self.local)):
            raise ValueError("local point must be a finite 3-vector")
        self.jac = np.empty((3, model.nv))

    def sample(self, data, force):
        """Read after step1, before integration: force, moving point and J at x_k."""
        point = data.xpos[self.body] + data.xmat[self.body].reshape(3, 3) @ self.local
        mujoco.mj_jac(self.model, data, self.jac, None, point, self.body)
        velocity = self.jac @ data.qvel
        torque = self.jac.T @ force
        power = float(force @ velocity)
        defect = power - float(torque @ data.qvel)
        return point, velocity, torque, power, defect

    def apply(self, data, force, base_qfrc):
        data.qfrc_applied[:] = base_qfrc
        sample = self.sample(data, force)
        # mj_applyFT ADDS to the supplied target. Restore its baseline every substep.
        mujoco.mj_applyFT(self.model, data, force, np.zeros(3), sample[0],
                         self.body, data.qfrc_applied)
        return sample


class TaskMeter:
    """The baseline meter, shared by Recorder and the oracle (same arithmetic)."""
    def __init__(self, model):
        self.model = model
        self.tip, self.table = model.geom("tool_tip").id, model.geom("tabletop").id
        self.v6, self.w6 = np.zeros(6), np.zeros(6)

    def sample(self, data):
        m, tip, table = self.model, self.tip, self.table
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_GEOM, tip, self.v6, 0)
        omega, vc = self.v6[:3].copy(), self.v6[3:].copy()
        xt = data.geom_xpos[tip]
        F, M, pc = np.zeros(3), np.zeros(3), np.zeros(3)
        power = wsum = fn_unexp = 0.0
        nt = nu = 0
        pairs = []
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = c.geom1, c.geom2
            mujoco.mj_contactForce(m, data, k, self.w6)
            if (g1 == tip and g2 == table) or (g1 == table and g2 == tip):
                frame = c.frame.reshape(3, 3)
                sign = -1.0 if g1 == tip else 1.0
                f = sign * (frame.T @ self.w6[:3])
                mo = sign * (frame.T @ self.w6[3:])
                vp = vc + np.cross(omega, c.pos - xt)
                power += float(f @ vp + mo @ omega)
                F += f
                M += mo
                mag = float(np.linalg.norm(f)) + 1e-300
                pc += c.pos * mag
                wsum += mag
                nt += 1
            else:
                nu += 1
                fn_unexp += abs(float(self.w6[0]))
                pairs.append((m.geom(g1).name or f"geom{g1}", m.geom(g2).name or f"geom{g2}"))
        pc = pc / wsum if wsum > 1e-200 else xt.copy()
        return dict(F=F, M=M, pc=pc, vc=vc, omega=omega,
                    vp=vc + np.cross(omega, pc - xt), power=power,
                    nt=nt, nu=nu, fn_unexp=fn_unexp, pairs=pairs)


def validate_oracle_model(model, arm):
    """Fail closed on unsupported actuator/stateful callback paths."""
    if model.nu != 7 or model.nv != 7 or model.na or model.nplugin:
        raise ValueError("oracle supports this seven-motor, no-plugin scene only")
    for name in ("control", "passive", "sensor", "act_dyn", "act_gain", "act_bias"):
        get = getattr(mujoco, "get_mjcb_" + name, None)
        if get is not None and get() is not None:
            raise ValueError(f"oracle cannot isolate global mjcb_{name}")
    if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("split-step oracle does not support RK4")
    if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION):
        raise ValueError("actuation disabled")
    a = arm.actuator_ids
    if np.any(model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE):
        raise ValueError("dynamic actuators unsupported")
    if np.any(model.actuator_forcelimited[a]) or np.any(model.jnt_actfrclimited[arm.joint_ids]):
        raise ValueError("additional actuator/joint force clipping unsupported")


@dataclass
class Prediction:
    work: np.ndarray                 # [H, R], deliberately identical in the oracle
    task_work: float
    y: np.ndarray                    # mean Fz, terminal tool v and omega
    final_state: np.ndarray
    min_prefix_work: float
    seconds: float


@dataclass
class Linearization:
    center: np.ndarray
    prediction: Prediction
    gradients: np.ndarray            # 2 x 7
    task_sensitivity: np.ndarray     # 7 x 7
    seconds: float


class OraclePredictor:
    def __init__(self, model, timing: TimingConfig, port: HumanPort):
        self.model, self.timing, self.port = model, timing, port
        self.arm = ArmIndex.from_model(model)
        validate_oracle_model(model, self.arm)
        self.data = mujoco.MjData(model)
        self.meter = TaskMeter(model)
        self.jp, self.jr = np.empty((3, model.nv)), np.empty((3, model.nv))
        self.tcp = model.site("tcp").id
        self.calls = 0

    def predict(self, state, torque, force, base_qfrc=None):
        t0 = perf_counter()
        m, d = self.model, self.data
        torque = np.asarray(torque, dtype=np.float64)
        if torque.shape != (7,) or not np.all(np.isfinite(torque)):
            raise ValueError("candidate must be seven finite joint torques")
        if np.any(np.abs(torque) > self.arm.torque_limit + 1e-10):
            raise ValueError("candidate exceeds torque limits; never differentiate clipped commands")
        restore_state(m, d, state)
        base = d.qfrc_applied.copy() if base_qfrc is None else base_qfrc
        d.ctrl[self.arm.actuator_ids] = torque
        wh = wt = fz = 0.0
        min_prefix = 0.0
        for _ in range(self.timing.control_interval):
            mujoco.mj_step1(m, d)
            sample = self.port.apply(d, force, base)
            mujoco.mj_step2(m, d)
            task = self.meter.sample(d)
            wh += sample[3] * self.timing.timestep
            wt += task["power"] * self.timing.timestep
            fz += task["F"][2] / self.timing.control_interval
            min_prefix = min(min_prefix, wh)
            if not np.allclose(d.qfrc_actuator[self.arm.dof_adr], torque, rtol=0, atol=1e-10):
                raise RuntimeError("actual actuator joint torque differs from candidate")
        final = integration_state(m, d)
        # Terminal kinematics only; no forward dynamics or controller evaluation.
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, self.jp, self.jr, self.tcp)
        y = np.r_[fz, self.jp @ d.qvel, self.jr @ d.qvel]
        self.calls += 1
        return Prediction(np.array([wh, wh]), wt, y, final, min_prefix, perf_counter() - t0)

    def linearize(self, state, center, force, eta=0.02, base_qfrc=None):
        t0 = perf_counter()
        center = np.asarray(center, dtype=float)
        eta = np.broadcast_to(eta, (7,)).astype(float)
        if np.any(eta <= 0) or not np.all(np.isfinite(eta)):
            raise ValueError("finite-difference steps must be positive N m")
        if np.any(np.abs(center) + eta > self.arm.torque_limit):
            raise ValueError("central stencil crosses actuator bound")
        p = self.predict(state, center, force, base_qfrc)
        g, sy = np.empty((2, 7)), np.empty((7, 7))
        for j in range(7):
            step = np.zeros(7)
            step[j] = eta[j]
            plus = self.predict(state, center + step, force, base_qfrc)
            minus = self.predict(state, center - step, force, base_qfrc)
            g[:, j] = (plus.work - minus.work) / (2 * eta[j])
            sy[:, j] = (plus.y - minus.y) / (2 * eta[j])
        return Linearization(center.copy(), p, g, sy, perf_counter() - t0)
