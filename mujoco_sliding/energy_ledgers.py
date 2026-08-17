"""Discrete energy ledgers for selective passivation.

Each ledger tracks the energy budget of one external power port of the robot.
Sign convention (p = F^T v_ee): p > 0 means the port injects energy into the
robot (ledger charges), p < 0 means the robot outputs energy through the port
(ledger discharges).

Update rule per control step:

    p >= 0:  E+ = min(E_max, E + dt * p)      (charging saturates at the cap)
    p <  0:  E+ = E + dt * p                  (discharge is NEVER clamped)

A lower-bound violation is deliberately not clamped away: the raw next energy
is stored and returned so violations stay visible in the logs, and the ledger
state itself carries the violation. Keeping the state honest is what makes
the cumulative passivity inequality checkable after the fact.

The residual and human ledgers are NESTED constraints, not additive tanks:
p_R = p_H + p_Delta, so the human power is intentionally observed by both.
There is no energy transfer between them and their sum has no physical
meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import LedgerParams


@dataclass
class LedgerUpdate:
    """Result of one discrete ledger update (all energies in J)."""

    e_prev: float
    p: float                 # port power used for the update [W]
    e_raw: float             # raw next energy BEFORE the charging cap
    e_next: float            # stored next energy (cap applied when charging)
    capped: bool             # charging hit E_max
    below_min: bool          # raw next energy fell below E_min
    margin: float            # e_raw - E_min (negative = violation)


class EnergyLedger:
    """One scalar energy ledger with cap-on-charge and honest discharge."""

    def __init__(self, params: LedgerParams, name: str = "ledger"):
        self.params = params
        self.name = name
        self.e = float(params.e_init)

    def reset(self) -> None:
        self.e = float(self.params.e_init)

    @property
    def headroom(self) -> float:
        """Usable budget above the floor, E - E_min [J]."""
        return self.e - self.params.e_min

    def update(self, p: float, dt: float) -> LedgerUpdate:
        e_prev = self.e
        e_raw = e_prev + dt * p
        if p >= 0.0:
            e_next = min(self.params.e_max, e_raw)
            capped = e_next < e_raw
        else:
            e_next = e_raw          # no lower clamp — violations stay visible
            capped = False
        self.e = e_next
        return LedgerUpdate(
            e_prev=e_prev,
            p=float(p),
            e_raw=e_raw,
            e_next=e_next,
            capped=capped,
            below_min=e_raw < self.params.e_min,
            margin=e_raw - self.params.e_min,
        )
