"""Finite-candidate receding-horizon planners C1-C4 (formulation §10) sharing candidates, horizon,
constraints and servo filter; they differ only in the human predictor and the objective.

  C1 'held'       deterministic held-force predictor, mean-only constraints, no ledger margin
  C2 'gm'         Gauss–Markov (motion-blind) force + Gaussian location, tightened constraints
  C3 'structured' impedance-informed latent-intention GP, tightened constraints, E[J]
  C4 'structured' as C3 with ρ_β(J), β > 0 (risk-averse in this convention)
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

from . import model as M
from .belief import HumanModel, rollout_linear, rollout_mc
from .gp import GaussMarkov, StructuredGP, edge_terms
from .kernels import RBFFeatures
from .servo import kappa

ROOT = Path(__file__).resolve().parent.parent
E_SCALE, H_SCALE = 0.05, 0.02


def entropic(J, beta):
    """Empirical ρ_β with equal non-negative weights (stable log-sum-exp), its delta-method SE and ESS."""
    S = J.shape[-1]
    if beta == 0.0:
        return J.mean(-1), J.std(-1) / np.sqrt(S), np.full(J.shape[:-1], float(S))
    m = (beta * J).max(-1, keepdims=True)
    w = np.exp(beta * J - m)
    rho = (m[..., 0] + np.log(w.mean(-1))) / beta
    se = np.sqrt(w.var(-1) / S) / (abs(beta) * w.mean(-1))
    ess = w.sum(-1) ** 2 / (w ** 2).sum(-1)
    return rho, se, ess


class Planner:
    filtered = True

    def __init__(self, name, predictor, beta=0.0, tighten=True, constraint_moments="linear", indep_time=False,
                 force_only=False, models_path=None, N_s=None, use_residual=True, bias_bound=None):
        self.name, self.predictor, self.beta = name, predictor, float(beta)
        self.tighten, self.cm, self.indep_time, self.force_only = tighten, constraint_moments, indep_time, force_only
        self.models = json.loads(Path(models_path or ROOT / "outputs" / "models" / "gp_frozen.json").read_text())
        self.N_s_override, self.use_residual = N_s, use_residual
        self.bias_bound = bias_bound   # post-hoc robust interval bound (B_f [N], B_s [m])

    # --- lifecycle --------------------------------------------------------------------------------
    def reset(self, cfg, sc, P):
        pl = cfg["planner"]
        self.cfg, self.P, self.N = cfg, P, pl["N"]
        self.N_s = self.N_s_override or pl["N_s"]
        self.kE = kappa(pl["alpha_E"]) if self.tighten else 0.0
        self.kH = kappa(pl["alpha_H"]) if self.tighten else 0.0
        self.kappa_H = self.kH
        self.rng = sc["plan_rng"]
        cands = [(v, k, f) for v in pl["sig_v"] for k in pl["sig_K"] for f in pl["sig_F"]]
        self.cands = np.array(cands + [(-1.0, 1.0, 1.0), (-2.0, 1.0, 1.0)], float)   # + passive backup, + brake
        h = dict(self.models["structured"] if self.use_residual else self.models["structured_no_residual"])
        sens = cfg["sensors"]                    # sensor calibration is configuration, not a learned hyperparameter
        h.update(sig_bs=sens["sigma_bs"], sig_es=sens["sigma_es"], sig_m=sens["sigma_m"])
        self.sgp = StructuredGP(h)
        self.gm = GaussMarkov(self.models["gauss_markov"])
        self.hyp = h
        self.s_post = None
        self.nominal_idx = int(np.where((self.cands == [1.0, 1.0, 1.0]).all(1))[0][0])

    def on_event(self, ev_k0, k):
        self.s_post = None

    def final_s_moments(self, loc):
        return self.s_post if self.s_post is not None else loc.moments()

    # --- human model per planning call ------------------------------------------------------------
    def window(self, ctx):
        k, k0 = ctx.k, ctx.event_k0
        start = max(k0, k - int(round(self.cfg["planner"]["gp_window"] / 1e-3)))
        idx = np.array([i for i in range(start, k) if (i - k0) % 40 == 0], int)
        L = ctx.hist
        t = np.r_[idx * 1e-3, ctx.t]
        q = np.vstack([L["q"][idx], ctx.q[None]])
        qd = np.vstack([L["qd"][idx], ctx.qd[None]])
        y = np.vstack([L["y"][idx], ctx.y[None]])
        q_on = L["q"][k0] if k0 < k else ctx.q
        return t, q, qd, y, k0 * 1e-3, q_on

    def human_model(self, ctx):
        """Returns (HumanModel, s_mean, s_var, bf_mean, bf_cov, cov_s_bf)."""
        led, N, P = ctx.led, self.N, self.P
        t_st = ctx.t + np.arange(N) * self.cfg["planner"]["dt_ctrl"]
        var_ne = led.ev_var_n + led.frozen_var
        common = dict(e_G=led.ev_G, e_Gam=led.ev_Gam, e_dp=led.ev_dp.copy(), e_frozen=led.frozen_mean,
                      e_mean=led.frozen_mean, he=P.he)
        s_mean, s_var = ctx.loc.moments()
        sig_bf = self.hyp["sig_bf"]
        if not ctx.engaged:
            return HumanModel("none", N, mu=np.zeros(0), Lf=np.zeros((0, 0)), Lf_indep=np.zeros((0, 0)), idx={},
                              **common), s_mean, s_var, np.zeros(2), np.eye(2) * sig_bf ** 2, np.zeros(2)
        t, q, qd, y, t_on, q_on = self.window(ctx)
        sbar = ctx.loc.sum / ctx.loc.n if ctx.loc.n > 0 else 0.0
        if self.predictor == "structured":
            post = self.sgp.condition(t, q, qd, y, t_on, q_on, sbar, ctx.loc.n, t_st, ep_times=[ctx.t])
            ix = dict(post["idx"])
            nz = len(post["mu"])
            ix["xi"] = slice(nz, nz + 2 * N)
            ix["ne"] = slice(nz + 2 * N, nz + 2 * N + 1)
            mu = np.r_[post["mu"], np.zeros(2 * N + 1)]
            Sz = post["Sig"] + 1e-12 * np.eye(nz)
            if self.force_only:                                   # ablation: location treated as known
                Sz[ix["s"], :] = 0.0
                Sz[:, ix["s"]] = 0.0
                Sz[ix["s"], ix["s"]] = 1e-14
            Lz = np.linalg.cholesky(Sz + 1e-10 * np.eye(nz))
            Lf = block_diag(Lz, np.eye(2 * N), np.sqrt(max(var_ne, 0.0)) * np.ones((1, 1)))
            Lf_ind = block_diag(np.linalg.cholesky(indep_cov(Sz, ix, N) + 1e-10 * np.eye(nz)), np.eye(2 * N),
                                np.sqrt(max(var_ne, 0.0)) * np.ones((1, 1)))
            c_on, _, u_on, _ = edge_terms(q_on[None], np.zeros((1, 3)), P.he)
            hp = self.hyp
            rho = np.exp(-self.cfg["planner"]["dt_ctrl"] / hp["tau_p"])
            hm = HumanModel("structured", N, mu=mu, Lf=Lf, Lf_indep=Lf_ind, idx=ix, c_on=c_on[0], u_on=u_on[0],
                            K=np.asarray(hp["K"]), D=np.asarray(hp["D"]), feat=RBFFeatures(hp["feat_centers"], hp["feat_width"]),
                            rho=rho, sq=np.sqrt(1 - rho * rho), sig_p=hp["sig_p"], **common)
            si = ix["s"].start
            bfi = ix["bf"]
            s_mean, s_var = float(post["mu"][si]), float(post["Sig"][si, si])
            if self.force_only:
                s_var = 0.0
            self.s_post = (s_mean, s_var)
            return hm, s_mean, s_var, post["mu"][bfi], post["Sig"][bfi, bfi], post["Sig"][si, bfi]
        # exogenous force path: Gauss–Markov (C2) or held (C1)
        if self.predictor == "gm":
            mF, cF = self.gm.predict(t, y, t_st)
            LF = np.linalg.cholesky(cF + 1e-9 * np.eye(N))
            LFi = np.diag(np.sqrt(np.diag(cF)))
        else:
            mF = np.tile(y[-3:].mean(0), (N, 1))
            LF = LFi = np.zeros((N, N))
        ix = {"Fx": slice(0, N), "Fy": slice(N, 2 * N), "s": slice(2 * N, 2 * N + 1), "bf": slice(2 * N + 1, 2 * N + 3),
              "ne": slice(2 * N + 3, 2 * N + 4)}
        mu = np.r_[mF[:, 0], mF[:, 1], s_mean, 0.0, 0.0, 0.0]
        stoch = self.predictor == "gm"
        sd_s = np.sqrt(s_var) if (stoch and not self.force_only) else 0.0
        tail = np.diag([sd_s, sig_bf if stoch else 0.0, sig_bf if stoch else 0.0, np.sqrt(max(var_ne, 0.0)) if stoch else 0.0])
        hm = HumanModel("exogenous", N, mu=mu, Lf=block_diag(LF, LF, tail), Lf_indep=block_diag(LFi, LFi, tail), idx=ix, **common)
        if not stoch:
            return hm, s_mean, 0.0, np.zeros(2), np.zeros((2, 2)), np.zeros(2)
        return hm, s_mean, (0.0 if self.force_only else s_var), np.zeros(2), np.eye(2) * sig_bf ** 2, np.zeros(2)

    # --- planning ----------------------------------------------------------------------------------
    def plan(self, ctx):
        t0 = time.perf_counter()
        P, cfg, N, led = self.P, self.cfg, self.N, ctx.led
        hm, s_m, s_v, bf_m, bf_c, c_sbf = self.human_model(ctx)
        t_model = time.perf_counter() - t0
        if self.predictor == "held":
            m_e, v_e = 0.0, 0.0
        else:
            m_e, v_e = led.error_moments(s_m, s_v, bf_m, bf_c, c_sbf) if ctx.engaged else (led.frozen_mean, led.frozen_var)
        R = 0.0
        if self.bias_bound is not None:       # post-hoc robust past-work bound
            Bf, Bs = self.bias_bound
            R = Bf * float(np.sum(np.abs(led.ev_dp))) + Bs * led.ev_Gabs
            R += sum(Bf * float(np.sum(np.abs(hh['dp']))) + Bs * hh.get('Gabs', abs(hh['G'])) for hh in led.history)
            m_e = m_e - R
        BH_eff = led.BH - R
        Z0 = np.r_[ctx.q, ctx.qd, ctx.xref, ctx.I_f, 0.0, 0.0, led.Ec, BH_eff]
        u_prev = M.policy(ctx.t, ctx.q[None], ctx.qd[None], np.array([ctx.xref]), np.array([ctx.I_f]),
                          ctx.cand[0:1], ctx.cand[1:2], ctx.cand[2:3], P)[0][0]
        stochastic = (self.predictor != "held") and hm.kind != "none"
        S = self.N_s if stochastic else 1
        eta = hm.mu[None] if not stochastic else hm.sample(ctx.rng, S)
        if hm.kind == "none":
            eta = np.zeros((1, 0))
        mc = rollout_mc(Z0, self.cands, hm, eta, P, cfg, ctx.t, BH_eff, u_prev)
        t_mc = time.perf_counter() - t0 - t_model
        C = len(self.cands)
        if stochastic and self.cm == "linear":
            lin = rollout_linear(Z0, self.cands, hm, P, cfg, ctx.t, BH_eff, indep_time=self.indep_time)
            Ec_m, Ec_sd, hH_m, hH_sd = lin["Ec_m"], lin["Ec_sd"], lin["hH_m"], lin["hH_sd"]
        else:
            Ec_m, Ec_sd = mc["Ec"].mean(1), mc["Ec"].std(1)
            hH_m, hH_sd = mc["hH"].mean(1), mc["hH"].std(1)
            lin = None
            if hm.kind == "none" and self.predictor != "held":
                hH_sd = hH_sd + np.sqrt(max(led.frozen_var, 0.0))
        t_lin = time.perf_counter() - t0 - t_model - t_mc
        mE = (Ec_m[:, 1:] - self.kE * Ec_sd[:, 1:] - led.Emin) / E_SCALE
        mH = (hH_m[:, 1:] - self.kH * hH_sd[:, 1:]) / H_SCALE
        margin = np.minimum(mE.min(1), mH.min(1))
        obj, se, ess = entropic(mc["J"], self.beta)
        feas = margin >= 0.0
        if feas.any():
            c = int(np.argmin(np.where(feas, obj, np.inf)))
            status = "ok"
        else:
            c = int(np.argmax(margin))
            status = "planner_infeasible_fallback"
        info = dict(c=c, cand=self.cands[c].tolist(), status=status, n_feas=int(feas.sum()), engaged=bool(ctx.engaged),
                    obj=float(obj[c]), obj_se=float(se[c]), ess=float(ess[c]), margin=float(margin[c]),
                    obj_nominal=float(obj[self.nominal_idx]), feas_nominal=bool(feas[self.nominal_idx]),
                    J_mean=float(mc["J"][c].mean()), J_sd=float(mc["J"][c].std()),
                    hH_m=float(hH_m[c, -1]), hH_sd=float(hH_sd[c, -1]), Ec_m=float(Ec_m[c, -1]), Ec_sd=float(Ec_sd[c, -1]),
                    hH_mc_m=float(mc["hH"][c, :, -1].mean()), hH_mc_sd=float(mc["hH"][c, :, -1].std()),
                    WH_mc_m=float(mc["WH"][c, :, -1].mean()), WH_mc_sd=float(mc["WH"][c, :, -1].std()),
                    WH_mc_q99=float(np.quantile(mc["WH"][c, :, -1], 0.99)) if S > 1 else float(mc["WH"][c, 0, -1]),
                    s_m=float(s_m), s_sd=float(np.sqrt(max(s_v, 0.0))), m_e=float(m_e), sd_e=float(np.sqrt(max(v_e, 0.0))),
                    t_model=t_model, t_mc=t_mc, t_lin=t_lin, n_samples=S,
                    s_out_of_patch=float(out_of_patch(s_m, s_v, cfg)))
        if lin is not None:
            info.update(BH_lin_m=float(lin["BH_m"][c, -1]), BH_lin_sd=float(lin["BH_sd"][c, -1]))
        if ctx.engaged:
            info["stage"] = dict(hH_m=hH_m[c].tolist(), hH_sd=hH_sd[c].tolist(), Ec_m=Ec_m[c].tolist(), Ec_sd=Ec_sd[c].tolist(),
                                 BH_m=(lin["BH_m"][c].tolist() if lin is not None else mc["hH"][c].mean(0).tolist()),
                                 BH_sd=(lin["BH_sd"][c].tolist() if lin is not None else None),
                                 hH_mc_m=mc["hH"][c].mean(0).tolist(), hH_mc_sd=mc["hH"][c].std(0).tolist(),
                                 WH_mc_m=mc["WH"][c].mean(0).tolist(), WH_mc_sd=mc["WH"][c].std(0).tolist(),
                                 WH_mc_q=np.quantile(mc["WH"][c], [0.005, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.995], axis=0).tolist())
        return {"cand": self.cands[c], "m_e": float(m_e), "sd_e": float(np.sqrt(max(v_e, 0.0))), "info": info}


def indep_cov(Sz, ix, N):
    """Deliberately wrong ablation: remove cross-time covariance of the intention latents (and their
    coupling to other latents); keep per-stage blocks [ã_x,ã̇_x,ã_y,ã̇_y]."""
    out = Sz.copy()
    lat = np.r_[np.arange(ix["ax"].start, ix["ady"].stop)]
    out[np.ix_(lat, np.arange(len(Sz)))] = 0.0
    out[np.ix_(np.arange(len(Sz)), lat)] = 0.0
    for k in range(N):
        sel = np.array([ix["ax"].start + k, ix["adx"].start + k, ix["ay"].start + k, ix["ady"].start + k])
        out[np.ix_(sel, sel)] = Sz[np.ix_(sel, sel)]
    return out


def out_of_patch(m, v, cfg):
    from scipy.stats import norm
    sd = np.sqrt(max(v, 1e-16))
    b = cfg["body"]
    return norm.cdf((b["s_min"] - m) / sd) + norm.sf((b["s_max"] - m) / sd)


def make(name, cfg=None):
    """Registry of controllers used in the experiments."""
    from .controllers import Nominal, FilterOnly, OracleReference, FilterOnlyInstantaneous
    beta = (cfg or {}).get("planner", {}).get("beta", 2.0) if cfg else 2.0
    reg = {
        "C0": lambda: Nominal(),
        "C1": lambda: Planner("C1_deterministic_held", "held", tighten=False),
        "C2": lambda: Planner("C2_gauss_markov", "gm"),
        "C3": lambda: Planner("C3_structured_neutral", "structured", beta=0.0),
        "C4": lambda: Planner("C4_structured_risk", "structured", beta=beta),
        "C4neg": lambda: Planner("C4neg_structured_risk_seeking", "structured", beta=-1.0),
        "C4b5": lambda: Planner("C4b5_structured_risk", "structured", beta=5.0),
        "C2_robust": lambda: Planner("C2_robust_bias_bound_POSTHOC", "gm", bias_bound=(1.0, 0.08)),
        "C3_robust": lambda: Planner("C3_robust_bias_bound_POSTHOC", "structured", bias_bound=(1.0, 0.08)),
        "C3_mc": lambda: Planner("C3_mc_constraints", "structured", constraint_moments="mc"),
        "C3_indep": lambda: Planner("C3_indep_time", "structured", indep_time=True),
        "C3_forceonly": lambda: Planner("C3_force_only_uncertainty", "structured", force_only=True),
        "C3_nores": lambda: Planner("C3_no_residual", "structured", use_residual=False),
        "F": lambda: FilterOnly(),
        "Finst": lambda: FilterOnlyInstantaneous(),
        "ORACLE": lambda: OracleReference(),
    }
    c = reg[name]()
    c.key = name
    return c
