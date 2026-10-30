"""Controller registry. Planner-based controllers (C1-C4) live in planner.py."""
import numpy as np


class Nominal:
    """C0: nominal hybrid controller, no planner, NO energy filter (explicitly unprotected)."""
    name, filtered, kappa_H = "C0_nominal_unprotected", False, 0.0

    def reset(self, cfg, sc, P):
        pass

    def plan(self, ctx):
        return {"cand": np.array([1.0, 1.0, 1.0])}


class FilterOnly:
    """Nominal policy + servo Ec/BH filter on the booked ledger without margins (diagnostic)."""
    name, filtered, kappa_H = "F_filter_only", True, 0.0

    def reset(self, cfg, sc, P):
        pass

    def plan(self, ctx):
        return {"cand": np.array([1.0, 1.0, 1.0])}


class OracleReference(FilterOnly):
    """ORACLE (truth access): §5.2 reference filter fed with the TRUE BH. Diagnostic only."""
    name, oracle = "ORACLE_BH_ge_S_truth", True


class FilterOnlyInstantaneous(FilterOnly):
    """Ablation: instantaneous-power filter u^T q̇(t_k) ≤ b (no intersample work model)."""
    name, instantaneous_filter = "F_filter_only_instantaneous", True


class Explorer(Nominal):
    """Offline data collection only (unprotected): piecewise-random candidate modifiers every 0.3 s
    give robot motion that varies independently of the hidden intention (helps gain identification)."""
    name = "X_explorer_data_collection"

    def reset(self, cfg, sc, P):
        self.rng = sc["plan_rng"]
        self.cur = np.array([1.0, 1.0, 1.0])
        self.next_t = 0.0

    def plan(self, ctx):
        if ctx.t >= self.next_t:
            self.cur = np.array([self.rng.choice([0.0, 0.5, 1.0, 1.5]), self.rng.choice([1.0, 0.35]),
                                 self.rng.choice([1.0, 0.5])])
            self.next_t = ctx.t + 0.3
        return {"cand": self.cur}
