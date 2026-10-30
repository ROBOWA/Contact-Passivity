"""Vectorised belief rollouts (formulation §9): Monte Carlo with one consistent joint sample per
rollout, and first-order propagation Φ_{k+1} = A_k Φ_k + B_k E_k with finite-difference Jacobians of
the CLOSED-LOOP stage map (feedback derivatives included).

Dynamic state per row (12): q(3) qd(3) xref If eps(2) Ec BH   (BH = predicted TRUE BH)
Human models (built per planning call by planner.py):
  'none'       : no human contact over the horizon
  'structured' : f = K(a_on + ã_k + τ ã̇_k − p_H) + D(ã̇_k − v_H) + φ(χ)ᵀw + ε  (state-dependent)
  'exogenous'  : f = F_k (motion-blind force path; C2 and the deterministic C1)
"""
import numpy as np

from . import model as M
from .gp import edge_terms

NZ = 12
NLOC = {"structured": 9, "exogenous": 3, "none": 0}
FD_STATE = np.array([1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-6, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6])
FD_LOC = {"structured": np.array([1e-5, 1e-6, 1e-6, 1e-5, 1e-5, 1e-4, 1e-4, 1e-3, 1e-3]),
          "exogenous": np.array([1e-5, 1e-4, 1e-4])}


class HumanModel:
    """Per-planning-call random-input description. η ~ N(mu, Sig) (Sig given via a factor Lf)."""

    def __init__(self, kind, N, **kw):
        self.kind, self.N = kind, N
        self.__dict__.update(kw)

    # --- sampling and row construction -----------------------------------------------------------
    def sample(self, rng, n):
        if self.kind == "none":
            return np.zeros((n, 0))
        return self.mu + rng.standard_normal((n, len(self.mu))) @ self.Lf.T

    def rows(self, eta, P):
        """Map η rows -> per-row human inputs used by stage_step."""
        B = eta.shape[0]
        N = self.N
        if self.kind == "none":
            return {"kind": "none"}
        ix = self.idx
        if self.kind == "structured":
            s = eta[:, ix["s"]][:, 0]
            a_on = self.c_on[None] + s[:, None] * self.u_on[None]
            at = np.stack([eta[:, ix["ax"]], eta[:, ix["ay"]]], -1)            # (B,N,2)
            ad = np.stack([eta[:, ix["adx"]], eta[:, ix["ady"]]], -1)
            w = np.stack([eta[:, ix["wx"]], eta[:, ix["wy"]]], 1)              # (B,2,nw)
            eps0 = np.stack([eta[:, ix["epx"]][:, 0], eta[:, ix["epy"]][:, 0]], -1)
            xi = eta[:, ix["xi"]].reshape(B, 2, N).transpose(0, 2, 1)          # (B,N,2)
            return {"kind": "structured", "s": s, "a_on": a_on, "at": at, "ad": ad, "w": w, "eps0": eps0, "xi": xi}
        s = eta[:, ix["s"]][:, 0]
        F = np.stack([eta[:, ix["Fx"]], eta[:, ix["Fy"]]], -1)
        return {"kind": "exogenous", "s": s, "F": F}

    def E(self, k, Zm):
        """Selector/linear map from η to the stage-k local inputs, per candidate mean state (C,nl,neta)."""
        C, ix, N = Zm.shape[0], self.idx, self.N
        neta = len(self.mu)
        if self.kind == "structured":
            E = np.zeros((C, 9, neta))
            E[:, 0, ix["s"].start] = 1.0
            E[:, 1, ix["ax"].start + k] = 1.0
            E[:, 2, ix["ay"].start + k] = 1.0
            E[:, 3, ix["adx"].start + k] = 1.0
            E[:, 4, ix["ady"].start + k] = 1.0
            _, chi, _, _ = edge_terms(Zm[:, 0:3], Zm[:, 3:6], self.he)
            phi = self.feat(chi)                                   # (C,2,nw)
            E[:, 5, ix["wx"]] = phi[:, 0]
            E[:, 6, ix["wy"]] = phi[:, 1]
            E[:, 7, ix["xi"].start + k] = 1.0
            E[:, 8, ix["xi"].start + N + k] = 1.0
            return E
        E = np.zeros((C, 3, neta))
        E[:, 0, ix["s"].start] = 1.0
        E[:, 1, ix["Fx"].start + k] = 1.0
        E[:, 2, ix["Fy"].start + k] = 1.0
        return E

    def bh0(self, eta, BH_book):
        """Predicted TRUE BH at the planning time: BH_book + e_BH(η) (affine in s, b_f, n_e)."""
        if self.kind == "none":
            return np.full(eta.shape[0], BH_book + self.e_mean)
        ix = self.idx
        e = self.e_G * eta[:, ix["s"]][:, 0] - self.e_Gam - eta[:, ix["bf"]] @ self.e_dp - eta[:, ix["ne"]][:, 0] + self.e_frozen
        return BH_book + e


def stage_step(Z, cand, hum, hm, k, t_k, P, cfg, loc_pert=None, cost=None):
    """Advance all rows over one planning stage (n_sub semi-implicit Euler substeps).

    Z: (B,12) state; cand: (B,3) [σ_v, σ_K, σ_F] (σ_v<0: passive backup); hum: per-row human inputs.
    loc_pert: optional (B, nloc) additive perturbations of the stage-local human inputs (FD).
    cost: optional dict of accumulators (J, uprev) updated in place.
    """
    pl = cfg["planner"]
    n_sub = int(round(pl["dt_ctrl"] / pl["dt_sub"]))
    h = pl["dt_sub"]
    q, qd = Z[:, 0:3].copy(), Z[:, 3:6].copy()
    xref, If, eps = Z[:, 6].copy(), Z[:, 7].copy(), Z[:, 8:10].copy()
    Ec, BH = Z[:, 10].copy(), Z[:, 11].copy()
    sv, sK, sF = cand[:, 0], cand[:, 1], cand[:, 2]
    kind = hum["kind"]
    B = Z.shape[0]
    lp = np.zeros((B, NLOC.get(kind, 0))) if loc_pert is None else loc_pert
    if kind == "structured":
        s = hum["s"] + lp[:, 0]
        a_on = hum["a_on"] + lp[:, 0:1] * hm.u_on[None]
        at = hum["at"][:, k] + lp[:, 1:3]
        ad = hum["ad"][:, k] + lp[:, 3:5]
        goff = lp[:, 5:7]
        rH = M.r_human(s, P.he)
    elif kind == "exogenous":
        s = hum["s"] + lp[:, 0]
        F = hum["F"][:, k] + lp[:, 1:3]
        rH = M.r_human(s, P.he)
    u_sum = np.zeros((B, 3))
    for j in range(n_sub):
        t = t_k + j * h
        con = M.contact(q, qd, P)
        u, vref, dI, fset = M.policy(t, q, qd, xref, If, sv, sK, sF, P, fn=con["fn"], rho_T=con["rho"], vt=con["vt"])
        if kind == "none":
            f = np.zeros((B, 2))
            tauH = np.zeros((B, 3))
        else:
            if kind == "structured":
                pH = M.point(q, rH)
                vH = M.point_vel(q, qd, rH)
                _, chi, _, _ = edge_terms(q, qd, P.he)
                phi = hm.feat(chi)                                     # (B,2,nw)
                g = np.einsum("bin,bin->bi", phi, hum["w"]) + goff
                f = hm.K * (a_on + at + (j * h) * ad - pH) + hm.D * (ad - vH) + g + eps
            else:
                f = F
            tauH = M.gen_force(q, rH, f)
        qdd = (u + con["tau"] + tauH - P.B * qd) / P.Mdiag
        qd_n = qd + h * qdd
        dq = h * qd_n
        Ec = Ec + h * con["DT"] - np.sum(u * dq, 1)
        if kind != "none":
            vHn = M.point_vel(q, qd_n, rH)
            BH = BH + h * np.sum(f * vHn, 1)                          # BH -= h p_H, p_H = -f·v_H
        running = t >= P.t_start
        xref = np.where(running, np.minimum(xref + h * vref, P.x_target), xref)
        sat = np.abs(u[:, 1]) >= P.umax[1] - 1e-9
        If = np.where(sat, If, np.clip(If + h * dI, -P.I_max, P.I_max))
        q, qd = q + dq, qd_n
        u_sum += u
    if kind == "structured":
        xi = hum["xi"][:, k] + lp[:, 7:9]
        eps = hm.rho * eps + hm.sq * hm.sig_p * xi
    Zn = np.concatenate([q, qd, xref[:, None], If[:, None], eps, Ec[:, None], BH[:, None]], 1)
    if cost is not None:
        c = cfg["planner"]["cost"]
        t_end = t_k + n_sub * h
        con = M.contact(q, qd, P)
        tip_x = q[:, 0] + con["rho"][:, 0]
        short = np.maximum(0.0, M.schedule(t_end, P) - tip_x) / c["e_prog"]
        uavg = u_sum / n_sub
        us = np.array(c["u_scale"])
        ell = (c["w_prog"] * short ** 2 + c["w_f"] * ((con["fn"] - P.f_d) / c["e_f"]) ** 2
               + c["w_th"] * (q[:, 2] / c["e_th"]) ** 2 + c["w_u"] * np.sum((uavg / us) ** 2, 1)
               + c["w_du"] * np.sum(((uavg - cost["uprev"]) / us) ** 2, 1))
        cost["J"] += ell
        cost["uprev"] = uavg
    return Zn


def storage_grad(Z, P):
    """S(q,qd) and ∂S/∂z (analytic) for h_H = BH − S."""
    q, qd = Z[:, 0:3], Z[:, 3:6]
    S, _, _ = M.storage(q, qd, P)
    con = M.contact(q, qd, P)
    G = np.zeros((Z.shape[0], NZ))
    d = con["d"]
    G[:, 1] = -P.kn * d
    G[:, 2] = -P.kn * d * con["rho"][:, 0]
    G[:, 3:6] = P.Mdiag * qd
    return S, G


def rollout_mc(Z0, cands, hm, eta, P, cfg, t0, BH_book, u_prev):
    """MC: rows = candidates × samples (common random numbers). Returns J (C,S), Ec,hH (C,S,N+1)."""
    C, S = len(cands), eta.shape[0]
    N = hm.N
    Z = np.repeat(Z0[None], C * S, 0)
    cand = np.repeat(cands, S, 0)
    eta_r = np.tile(eta, (C, 1))
    if hm.kind == "structured":
        Z[:, 8:10] = np.tile(hm.rows(eta, P)["eps0"], (C, 1))
    Z[:, 11] = hm.bh0(eta_r, BH_book)
    hum = hm.rows(eta_r, P)
    cost = {"J": np.zeros(C * S), "uprev": np.repeat(np.asarray(u_prev, float)[None], C * S, 0)}
    Ec = np.zeros((C * S, N + 1))
    hH = np.zeros((C * S, N + 1))
    WH = np.zeros((C * S, N + 1))
    Zs = Z[:, :NZ]
    Sv, _ = storage_grad(Zs, P)
    Ec[:, 0], hH[:, 0] = Zs[:, 10], Zs[:, 11] - Sv
    bh0 = Zs[:, 11].copy()
    for k in range(N):
        Zs = stage_step(Zs, cand, hum, hm, k, t0 + k * cfg["planner"]["dt_ctrl"], P, cfg, cost=cost)
        Sv, _ = storage_grad(Zs, P)
        Ec[:, k + 1], hH[:, k + 1] = Zs[:, 10], Zs[:, 11] - Sv
        WH[:, k + 1] = bh0 - Zs[:, 11]
    J = cost["J"] / N
    res = cfg["planner"]["cost"]
    w_res, R_task = res.get("w_res", 0.0), res.get("R_task", 0.03)
    J = J + w_res * np.maximum(0.0, (R_task - hH[:, -1]) / R_task) ** 2
    sh = (C, S)
    return {"J": J.reshape(sh), "Ec": Ec.reshape(sh + (N + 1,)), "hH": hH.reshape(sh + (N + 1,)),
            "WH": WH.reshape(sh + (N + 1,)), "Z_end": Zs.reshape(sh + (NZ,))}


def rollout_linear(Z0, cands, hm, P, cfg, t0, BH_book, indep_time=False):
    """First-order propagation along each candidate's mean trajectory (η = mean).

    Returns per candidate and stage: mean/sd of Ec, hH, BH (first order), and Φ_N.
    indep_time=True is the deliberately wrong ablation (cross-time latent covariance removed).
    """
    C, N = len(cands), hm.N
    nl = NLOC[hm.kind]
    eta_bar = hm.mu[None]
    hum1 = hm.rows(eta_bar, P)
    neta = len(hm.mu)
    Zm = np.repeat(Z0[None, :NZ], C, 0)
    if hm.kind == "structured":
        Zm[:, 8:10] = hum1["eps0"]
    Zm[:, 11] = hm.bh0(eta_bar, BH_book)[0]
    Phi = np.zeros((C, NZ, neta))
    ix = hm.idx
    Phi[:, 11, ix["s"]] = hm.e_G
    Phi[:, 11, ix["bf"]] = -hm.e_dp
    Phi[:, 11, ix["ne"]] = -1.0
    if hm.kind == "structured":
        Phi[:, 8, ix["epx"].start] = 1.0
        Phi[:, 9, ix["epy"].start] = 1.0
    Lf = hm.Lf_indep if indep_time else hm.Lf
    out = {k: np.zeros((C, N + 1)) for k in ("Ec_m", "Ec_sd", "hH_m", "hH_sd", "BH_m", "BH_sd", "WH_sd")}

    def moments(k, Zm, Phi):
        S, G = storage_grad(Zm, P)
        Psi = Phi @ Lf                                            # (C,NZ,neta)
        gE = Psi[:, 10]
        gB = Psi[:, 11]
        gvec = -G
        gvec[:, 11] += 1.0                                        # h_H = BH - S
        gh = np.einsum("cz,czn->cn", gvec, Psi)
        out["Ec_m"][:, k], out["Ec_sd"][:, k] = Zm[:, 10], np.sqrt(np.sum(gE ** 2, 1))
        out["BH_m"][:, k], out["BH_sd"][:, k] = Zm[:, 11], np.sqrt(np.sum(gB ** 2, 1))
        out["hH_m"][:, k], out["hH_sd"][:, k] = Zm[:, 11] - S, np.sqrt(np.sum(gh ** 2, 1))
        return gB

    gB0 = moments(0, Zm, Phi)
    nd = NZ + nl
    for k in range(N):
        # batch: per candidate [nominal, state perturbations (12), local-input perturbations (nl)]
        Zb = np.repeat(Zm, 1 + nd, 0)
        cand = np.repeat(cands, 1 + nd, 0)
        lp = np.zeros((C * (1 + nd), nl))
        steps = np.concatenate([FD_STATE, FD_LOC.get(hm.kind, np.zeros(0))])
        for i in range(nd):
            rows = np.arange(C) * (1 + nd) + 1 + i
            if i < NZ:
                Zb[rows, i] += steps[i]
            else:
                lp[rows, i - NZ] = steps[i]
        hum = hm.rows(np.repeat(eta_bar, C * (1 + nd), 0), P) if hm.kind != "none" else {"kind": "none"}
        Zn = stage_step(Zb, cand, hum, hm, k, t0 + k * cfg["planner"]["dt_ctrl"], P, cfg, loc_pert=lp if nl else None)
        Zn = Zn.reshape(C, 1 + nd, NZ)
        base = Zn[:, 0]
        Jac = (Zn[:, 1:] - base[:, None]) / steps[None, :, None]   # (C, nd, NZ) rows=inputs
        A = Jac[:, :NZ].transpose(0, 2, 1)                          # (C, NZ, NZ)
        Bm = Jac[:, NZ:].transpose(0, 2, 1)                         # (C, NZ, nl)
        Ek = hm.E(k, Zm) if nl else None                            # (C, nl, neta)
        Phi = A @ Phi + (Bm @ Ek if nl else 0.0)
        Zm = base
        moments(k + 1, Zm, Phi)
    out["Z_end"] = Zm
    out["Phi_N"] = Phi
    return out
