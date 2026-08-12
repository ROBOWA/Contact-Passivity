"""Perfectly known human disturbance force applied at the end-effector.

No collision object is created for the human. The configured Cartesian force
f_h(t) is applied directly at the end-effector site with ``mujoco.mj_applyFT``
(accumulating into ``data.qfrc_applied``), and the exact applied value is
logged as its own ground-truth channel, separate from the task contact:

    f_measured = f_task + f_h        (synthetic combined channel)

Power sign convention (v_ee = end-effector velocity):
    P_{h->r} = f_h^T v_ee    power the human injects INTO the robot
    P_{r->h} = -f_h^T v_ee   power the robot delivers to the human
"""

from __future__ import annotations

import mujoco
import numpy as np

from .config import HumanForceConfig
from .contact_extraction import ModelHandles


class HumanForce:
    """Smooth trapezoidal (cosine-edged) Cartesian force profile."""

    def __init__(self, cfg: HumanForceConfig):
        self.cfg = cfg
        d = np.asarray(cfg.direction, dtype=float)
        norm = np.linalg.norm(d)
        self._dir = d / norm if norm > 0 else d

    def force(self, t: float) -> np.ndarray:
        """f_h(t) in the world frame; zero outside the configured window."""
        cfg = self.cfg
        if not cfg.enabled:
            return np.zeros(3)
        s = t - cfg.t_start
        if s <= 0.0:
            m = 0.0
        elif s < cfg.t_rise:
            m = 0.5 * (1.0 - np.cos(np.pi * s / cfg.t_rise))
        elif s < cfg.t_rise + cfg.t_hold:
            m = 1.0
        elif s < cfg.t_rise + cfg.t_hold + cfg.t_fall:
            u = (s - cfg.t_rise - cfg.t_hold) / cfg.t_fall
            m = 0.5 * (1.0 + np.cos(np.pi * u))
        else:
            m = 0.0
        return cfg.magnitude * m * self._dir

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        handles: ModelHandles,
        f_h: np.ndarray,
    ) -> None:
        """Accumulate f_h at the EE site into ``data.qfrc_applied``.

        The caller is responsible for clearing ``data.qfrc_applied`` first
        (once per step), since mj_applyFT accumulates.
        """
        if not np.any(f_h):
            return
        point = data.site_xpos[handles.ee_site]
        torque = np.zeros(3)
        mujoco.mj_applyFT(
            model, data, f_h, torque, point, handles.ee_body, data.qfrc_applied
        )
