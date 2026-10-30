"""POST-HOC (designed after observing the sensor-bias failure in the frozen main run).

Adds an optional *robust interval bound* on the past-work error to the planner: with assumed
calibration bounds |b_f,i| <= B_f and |s - ŝ_k| <= B_s,

    e_BH >= (Gaussian part) - B_f ||Δp̂||_1 - B_s Σ_k |G_k| h      (per contact event, summed),

and the planner/servo use BH_book - R as the starting true BH and m_e - R as the margin mean.
The Gaussian margin is kept (double counting -> conservative). Existing controllers are unchanged.
Applied once by this script; the change is recorded under a new freeze label 'posthoc_v1'.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch():
    p = ROOT / "hcu2d" / "servo.py"
    s = p.read_text()
    if "ev_Gabs" not in s:
        s = s.replace("        self.ev_var_n = 0.0\n        self.frozen_mean = 0.0",
                      "        self.ev_var_n = 0.0\n        self.ev_Gabs = 0.0          # Σ|G_k| h (post-hoc robust bound)\n        self.frozen_mean = 0.0")
        s = s.replace("        g = 0.5 * (G0 + G1) * h\n        self.ev_G += g",
                      "        g = 0.5 * (G0 + G1) * h\n        self.ev_G += g\n        self.ev_Gabs += abs(g)")
        s = s.replace("        self.history.append(dict(G=self.ev_G, Gam=self.ev_Gam, dp=self.ev_dp.copy(), var_n=self.ev_var_n,",
                      "        self.history.append(dict(G=self.ev_G, Gabs=self.ev_Gabs, Gam=self.ev_Gam, dp=self.ev_dp.copy(), var_n=self.ev_var_n,")
        s = s.replace("        self.ev_G = self.ev_Gam = self.ev_var_n = 0.0\n",
                      "        self.ev_G = self.ev_Gam = self.ev_var_n = self.ev_Gabs = 0.0\n")
        p.write_text(s)
    p = ROOT / "hcu2d" / "planner.py"
    s = p.read_text()
    if "bias_bound" not in s:
        s = s.replace("                 force_only=False, models_path=None, N_s=None, use_residual=True):",
                      "                 force_only=False, models_path=None, N_s=None, use_residual=True, bias_bound=None):")
        s = s.replace("        self.N_s_override, self.use_residual = N_s, use_residual",
                      "        self.N_s_override, self.use_residual = N_s, use_residual\n        self.bias_bound = bias_bound   # post-hoc robust interval bound (B_f [N], B_s [m])")
        s = s.replace("        Z0 = np.r_[ctx.q, ctx.qd, ctx.xref, ctx.I_f, 0.0, 0.0, led.Ec, led.BH]",
                      "        R = 0.0\n"
                      "        if self.bias_bound is not None:       # post-hoc robust past-work bound\n"
                      "            Bf, Bs = self.bias_bound\n"
                      "            R = Bf * float(np.sum(np.abs(led.ev_dp))) + Bs * led.ev_Gabs\n"
                      "            R += sum(Bf * float(np.sum(np.abs(hh['dp']))) + Bs * hh.get('Gabs', abs(hh['G'])) for hh in led.history)\n"
                      "            m_e = m_e - R\n"
                      "        BH_eff = led.BH - R\n"
                      "        Z0 = np.r_[ctx.q, ctx.qd, ctx.xref, ctx.I_f, 0.0, 0.0, led.Ec, BH_eff]")
        s = s.replace("        mc = rollout_mc(Z0, self.cands, hm, eta, P, cfg, ctx.t, led.BH, u_prev)",
                      "        mc = rollout_mc(Z0, self.cands, hm, eta, P, cfg, ctx.t, BH_eff, u_prev)")
        s = s.replace("            lin = rollout_linear(Z0, self.cands, hm, P, cfg, ctx.t, led.BH, indep_time=self.indep_time)",
                      "            lin = rollout_linear(Z0, self.cands, hm, P, cfg, ctx.t, BH_eff, indep_time=self.indep_time)")
        s = s.replace('        "C4b5": lambda: Planner("C4b5_structured_risk", "structured", beta=5.0),',
                      '        "C4b5": lambda: Planner("C4b5_structured_risk", "structured", beta=5.0),\n'
                      '        "C2_robust": lambda: Planner("C2_robust_bias_bound_POSTHOC", "gm", bias_bound=(1.0, 0.08)),\n'
                      '        "C3_robust": lambda: Planner("C3_robust_bias_bound_POSTHOC", "structured", bias_bound=(1.0, 0.08)),')
        p.write_text(s)
    print("patched")


if __name__ == "__main__":
    patch()
