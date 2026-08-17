"""Task-contact predictors and the residual force decomposition.

The measured end-effector force is the sum of the two ground-truth channels

    F_meas = F_task + F_h            (task contact + scripted human force)

and a task-contact predictor supplies F_task_hat. The residual and the
model-mismatch force follow as

    F_R     = F_meas - F_task_hat    (everything the task model cannot explain)
    F_Delta = F_R - F_h              (pure model mismatch, diagnostics only;
                                      the controller never sees F_h directly
                                      through this channel)

Identities (unit-tested):
  * oracle predictor:                F_R = F_h  and  F_Delta = 0
  * oracle predictor, no human:      F_R = 0
  * imperfect predictor, no human:   F_R = F_Delta

The predictors only use the task-contact channel (the measured *normal* task
force) and the physical end-effector velocity. They never fit the total
measured force, so they cannot explain away the human force.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import NORMAL, TANGENT

PREDICTOR_KINDS = ("oracle", "friction")


@dataclass
class ForceDecomposition:
    """World-frame force channels of one control step (all forces ON the EE)."""

    f_task: np.ndarray        # ground-truth task-contact force
    f_hat: np.ndarray         # predicted task-contact force
    f_h: np.ndarray           # ground-truth human force
    f_meas: np.ndarray = field(init=False)   # F_task + F_h
    f_r: np.ndarray = field(init=False)      # F_meas - F_hat
    f_delta: np.ndarray = field(init=False)  # F_R - F_h

    def __post_init__(self):
        self.f_meas = self.f_task + self.f_h
        self.f_r = self.f_meas - self.f_hat
        self.f_delta = self.f_r - self.f_h


class TaskContactPredictor:
    """Predicts the task-contact force F_task_hat.

    kind="oracle":    F_hat = F_task (perfect model).
    kind="friction":  F_hat_n = F_task_n (true normal component),
                      F_hat_t = -mu_hat * F_task_n * tanh(v_t / v_s),
                      i.e. a regularized Coulomb model whose mu_hat may
                      differ from the plant's true mu = 0.30.
    """

    def __init__(self, kind: str = "oracle", mu_hat: float = 0.30,
                 v_s: float = 0.005):
        if kind not in PREDICTOR_KINDS:
            raise ValueError(f"unknown predictor kind {kind!r}")
        self.kind = kind
        self.mu_hat = mu_hat
        self.v_s = v_s

    def predict(self, f_task: np.ndarray, v_ee: np.ndarray) -> np.ndarray:
        """F_task_hat in the world frame from the task channel and v_ee."""
        if self.kind == "oracle":
            return np.asarray(f_task, dtype=float).copy()
        f_n = float(NORMAL @ f_task)          # true normal task force
        v_t = float(TANGENT @ v_ee)           # physical tangential EE velocity
        f_hat_t = -self.mu_hat * f_n * np.tanh(v_t / self.v_s)
        return TANGENT * f_hat_t + NORMAL * f_n


def decompose(f_task: np.ndarray, f_hat: np.ndarray,
              f_h: np.ndarray) -> ForceDecomposition:
    """Build the residual decomposition from the three force channels."""
    return ForceDecomposition(
        f_task=np.asarray(f_task, dtype=float).copy(),
        f_hat=np.asarray(f_hat, dtype=float).copy(),
        f_h=np.asarray(f_h, dtype=float).copy(),
    )
