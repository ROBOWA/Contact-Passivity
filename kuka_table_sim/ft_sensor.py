"""Force/torque sensor measurement chain: raw sample, gravity-only and inertia-corrected estimates.

MuJoCo's <force> sensor at `ft_site` reports the force the parent side applies to the sensor body
`ft_sensor` and everything rigidly attached below it (the "supported subtree": the sensor housing and
the tool shaft with its tip, here 0.50 kg), expressed in the site frame. Newton's law for that subtree,
with F_ext the force the environment applies to it, is (world frame, S_w = R S the sensor value)

    m a_com = S_w + F_ext + m g          =>   F_ext = -S_w - m g + m a_com

so the estimates are

    F_gravity_only = -R S - m g                    (exact only when the subtree is not accelerating)
    F_external_est = F_gravity_only + m a_com      (inertia-corrected)

where g = model.opt.gravity and a_com is the translational acceleration of the subtree's centre of mass.
a_com is the mass-weighted mean of the (coordinate) acceleration of each supported body's own COM, so the rotational
terms of an offset COM (alpha x r and omega x (omega x r)) are included; the acceleration of the TCP alone
is not enough. Both identities are checked against the model in tests/test_ft_compensation.py.

Timing. `sample()` must be called right after `mj_step2` (or after `mj_forward`): the force sensor is an
acceleration-stage quantity, so `sensordata`, the site frame, `cacc` and the contacts all describe the
state x_k at the START of the step (time t_k), even though `data.time` and `data.qpos` already moved on.
The sample therefore carries its own timestamp t_k, and the rotation and acceleration are taken from the
same evaluation as the force.

Assumptions of this simulation stage (stated, not validated for hardware): the tool is rigid; its mass and
COM are known exactly; and a_com is an ideal accelerometer, i.e. the exact acceleration the simulator
computed for that evaluation. No noise, bias drift, latency of the acceleration signal, or filtering is
added. No contact information (ncon, mj_contactForce, constraint forces) is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class FTSample:
    """One coherent evaluation of the FT measurement chain."""

    t: float                  # physics evaluation instant of every field below (s)
    raw_site: np.ndarray      # (3,) MuJoCo force sensor output, ft_site frame (N)
    rot: np.ndarray           # (3, 3) world <- site rotation at the same instant
    m_a_com: np.ndarray       # (3,) sum of m_i * a_i over the supported bodies, world frame (N)
    mass: float
    gravity: np.ndarray       # (3,)

    @property
    def raw_world(self) -> np.ndarray:
        return self.rot @ self.raw_site

    @property
    def gravity_only(self) -> np.ndarray:
        """-R S - m g (N): the environment force if the tool were not accelerating."""
        return -self.raw_world - self.mass * self.gravity

    @property
    def corrected(self) -> np.ndarray:
        """-R S - m g + m a_com (N): the estimated external force on the supported subtree."""
        return self.gravity_only + self.m_a_com

    @property
    def a_com(self) -> np.ndarray:
        return self.m_a_com / self.mass


class FTSensor:
    def __init__(self, model: mujoco.MjModel, sensor: str = "ft_force", site: str = "ft_site", body: str = "ft_sensor") -> None:
        self.model = model
        sid = model.sensor(sensor).id
        if model.sensor_type[sid] != mujoco.mjtSensor.mjSENS_FORCE or model.sensor_dim[sid] != 3:
            raise ValueError(f"{sensor} is not a 3-axis force sensor")
        self.adr = int(model.sensor_adr[sid])
        self.site = model.site(site).id
        self.body = model.body(body).id
        if int(model.site_bodyid[self.site]) != self.body:
            raise ValueError(f"site {site} is not attached to body {body}")
        # the supported subtree: the sensor body and every body below it
        parents = model.body_parentid
        self.bodies = [b for b in range(model.nbody) if self._below(b, parents)]
        self.masses = np.array([float(model.body_mass[b]) for b in self.bodies])
        self.mass = float(self.masses.sum())
        if abs(self.mass - float(model.body_subtreemass[self.body])) > 1e-12:
            raise ValueError("supported-subtree mass does not match body_subtreemass")
        self.gravity = model.opt.gravity.copy()
        self._res = np.zeros(6)
        # position of the supported COM in the site frame is recorded for the audit (not used online)
        self.com_site: np.ndarray | None = None

    def _below(self, b: int, parents) -> bool:
        while True:
            if b == self.body:
                return True
            if b == 0:
                return False
            b = int(parents[b])

    def sample(self, data: mujoco.MjData, t_eval: float) -> FTSample:
        """Capture the measurement chain; call right after mj_step2 / mj_forward (see module docstring)."""
        m_a = np.zeros(3)
        for b, mass in zip(self.bodies, self.masses):
            if mass == 0.0:
                continue
            mujoco.mj_objectAcceleration(self.model, data, mujoco.mjtObj.mjOBJ_BODY, b, self._res, 0)
            # [rot; lin] at the body's COM, world axes. MuJoCo returns the PROPER acceleration (it includes
            # gravity, like its accelerometer sensor: a body at rest reads -g), so the gravity vector is
            # added back to get the coordinate acceleration a_com that Newton's law needs. Verified against
            # finite differences of the COM velocity (tests/test_ft_compensation.py).
            m_a += mass * (self._res[3:6] + self.gravity)
        return FTSample(t=float(t_eval), raw_site=data.sensordata[self.adr:self.adr + 3].copy(),
                        rot=data.site_xmat[self.site].reshape(3, 3).copy(), m_a_com=m_a,
                        mass=self.mass, gravity=self.gravity)

    def audit(self, data: mujoco.MjData) -> dict:
        """Facts the compensation relies on, for the documentation and the tests."""
        m = self.model
        com = sum(mass * data.xipos[b] for b, mass in zip(self.bodies, self.masses)) / self.mass
        rot = data.site_xmat[self.site].reshape(3, 3)
        return dict(sensor_body=m.body(self.body).name, supported_bodies=[m.body(b).name for b in self.bodies],
                    masses=self.masses.tolist(), mass=self.mass,
                    com_in_site_frame=(rot.T @ (com - data.site_xpos[self.site])).tolist())
