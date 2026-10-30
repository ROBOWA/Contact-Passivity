"""Closed-loop episode: truth plant (physics h) / servo 1 kHz (policy, energy filter, ledger) /
planner 50 Hz (controller.plan). The controller only receives the Context (measured data)."""
import math
import time

import numpy as np

from . import model as M
from .servo import Ledger, project_power, project_tick_energy, kappa
from .truth import TruthPlant

HS = 1e-3  # servo period


class LocBelief:
    """Location-sensor-only Gaussian belief s̄ = s + b_s + ε̄ (used for the booked ledger by all)."""

    def __init__(self, cfg):
        b, s = cfg["body"], cfg["sensors"]
        self.mu0, self.v0 = 0.0, 0.1 ** 2
        self.vb, self.ve = s["sigma_bs"] ** 2, s["sigma_es"] ** 2
        self.reset()

    def reset(self):
        self.n, self.sum = 0, 0.0

    def add(self, z):
        self.n += 1
        self.sum += z

    def moments(self):
        if self.n == 0:
            return self.mu0, self.v0
        vo = self.vb + self.ve / self.n
        prec = 1.0 / self.v0 + 1.0 / vo
        return (self.mu0 / self.v0 + (self.sum / self.n) / vo) / prec, 1.0 / prec


class Context:
    """Read-only measured information handed to the planner (no truth fields)."""
    __slots__ = ("t", "k", "q", "qd", "xref", "I_f", "led", "cand", "P", "cfg", "hist", "event_k0",
                 "loc", "engaged", "rng", "y")


def fork_truth(plant, cand, xref, I_f, t, BH_true0, Ec0, P, cfg, nsub, h):
    """Calibration fork: copy the TRUE simulator and execute candidate `cand` open-loop (no filter, no
    re-planning) over the planning horizon. Returns realized stage-end Ec, BH_true, S, h_H, W_H."""
    import copy
    pl = copy.deepcopy(plant)
    N, every = cfg["planner"]["N"], int(round(cfg["planner"]["dt_ctrl"] / HS))
    z = pl.z.copy()
    WH0, Ec = z[7], Ec0
    rec = dict(Ec=[Ec0], BH=[BH_true0], S=[float(M.storage(z[None, :3], z[None, 3:6], P)[0][0])], WH=[0.0])
    for kk in range(N * every):
        tt = t + kk * HS
        q, qd = z[:3], z[3:6]
        con = M.contact(q, qd, P)
        u, v_ref, dI, _ = M.policy(tt, q, qd, xref, I_f, cand[0], cand[1], cand[2], P, fn=con["fn"], rho_T=con["rho"], vt=con["vt"])
        u = np.clip(u, -P.umax, P.umax)
        for j in range(nsub):
            pl.step(u, h)
        pl.begin_tick()
        z1 = pl.z.copy()
        Ec += HS * min(float(con["DT_credit"]), float(M.contact(z1[:3], z1[3:6], P)["DT_credit"])) - float(u @ (z1[:3] - q))
        xref = min(xref + v_ref * HS, P.x_target) if tt >= P.t_start else xref
        if abs(u[1]) < P.umax[1] - 1e-9:
            I_f = float(np.clip(I_f + dI * HS, -P.I_max, P.I_max))
        z = z1
        if (kk + 1) % every == 0:
            S = float(M.storage(z[None, :3], z[None, 3:6], P)[0][0])
            rec["Ec"].append(Ec)
            rec["BH"].append(BH_true0 - (z[7] - WH0))
            rec["S"].append(S)
            rec["WH"].append(z[7] - WH0)
    rec["hH"] = [b - s for b, s in zip(rec["BH"], rec["S"])]
    return rec


def run(cfg, sc, controller, h_phys=1e-3, keep_log=True, sub_log=False, fork_every=0):
    P = M.Known(cfg)
    t_cfg, acc = cfg["task"], sc["accounts"]
    n_ticks = int(round(t_cfg["T"] / HS))
    nsub = int(round(HS / h_phys))
    h = HS / nsub
    plant = TruthPlant(cfg, sc["events"], sc["q0"])
    led = Ledger(acc["Ec0"], acc["BH0"], acc["Emin"])
    lam_E, lam_H = acc["lam_E"], acc["lam_H"]
    Sigma_m = np.eye(2) * cfg["sensors"]["sigma_m"] ** 2
    noise = sc["noise"]
    loc = LocBelief(cfg)
    ctrl_every = int(round(cfg["planner"]["dt_ctrl"] / HS))
    Wu = 1.0 / np.array(cfg["planner"]["cost"]["u_scale"]) ** 2
    controller.reset(cfg, sc, P)

    L = {k: np.zeros((n_ticks + 1,) + shp) for k, shp in [
        ("q", (3,)), ("qd", (3,)), ("u_nom", (3,)), ("u", (3,)), ("f_true", (2,)), ("y", (2,)),
        ("Ec", ()), ("BH_book", ()), ("plant_int", (6,)), ("fn", ()), ("d", ()), ("xref", ()),
        ("I_f", ()), ("cand", (3,)), ("m_e", ()), ("sd_e", ()), ("s_hat", ()), ("s_var", ()),
        ("s_true", ()), ("engaged", ()), ("filt", ()), ("DT", ()), ("S", ()), ("bH", ()), ("bE", ())]}
    L["s_true"][:] = np.nan
    plan_info, plan_time = [], []
    xref, I_f = t_cfg["x_start"], 0.0
    cand, m_e, sd_e = np.array([1.0, 1.0, 1.0]), 0.0, 0.0
    kap = controller.kappa_H
    cur_ev, ev_k0, loc_j = None, None, 0
    sub = [] if sub_log else None

    def measure(k, z, t):
        ev = plant.engaged_event(t)
        if ev is None:
            return np.zeros(2), np.zeros(2), None
        fx, fy = plant.human(t, *z[:6])[:2]
        f = np.array([fx, fy])
        return f, f + noise["b_f"] + noise["eps_m"][k], ev

    plant.begin_tick()
    z = plant.z.copy()
    f0, y0, ev0 = measure(0, z, 0.0)
    for k in range(n_ticks):
        t = k * HS
        q, qd = z[:3].copy(), z[3:6].copy()
        con = M.contact(q, qd, P)
        S_hat = 0.5 * (P.m * (qd[0] ** 2 + qd[1] ** 2) + P.I * qd[2] ** 2) + float(con["ST"])
        # --- contact-event bookkeeping and location sensing (25 Hz) -------------------------------
        if ev0 is not cur_ev:
            if cur_ev is not None:           # event ended: freeze its past-work error belief
                sm, sv = controller.final_s_moments(loc) if hasattr(controller, "final_s_moments") else loc.moments()
                led.freeze_event(sm, sv)
            cur_ev, ev_k0 = ev0, (k if ev0 is not None else None)
            loc.reset()
            if hasattr(controller, "on_event"):
                controller.on_event(ev_k0, k)
        if cur_ev is not None and (k - ev_k0) % 40 == 0:
            loc.add(cur_ev.s + noise["b_s"] + noise["eps_s"][loc_j % len(noise["eps_s"])])
            loc_j += 1
        s_hat, s_var = loc.moments()
        # --- planner (50 Hz) -------------------------------------------------------------------------
        if k % ctrl_every == 0:
            ctx = Context()
            ctx.t, ctx.k, ctx.q, ctx.qd, ctx.xref, ctx.I_f, ctx.led = t, k, q, qd, xref, I_f, led
            ctx.cand, ctx.P, ctx.cfg, ctx.hist, ctx.event_k0, ctx.loc = cand, P, cfg, L, ev_k0, loc
            ctx.engaged, ctx.rng, ctx.y = cur_ev is not None, sc["plan_rng"], y0
            t0 = time.perf_counter()
            out = controller.plan(ctx)
            plan_time.append(time.perf_counter() - t0)
            cand, m_e, sd_e = np.asarray(out["cand"], float), out.get("m_e", 0.0), out.get("sd_e", 0.0)
            if fork_every and out.get("info") is not None and out["info"].get("engaged") and len(plan_time) % fork_every == 0:
                out["info"]["fork"] = fork_truth(plant, cand, xref, I_f, t, acc["BH0"] - z[7], led.Ec, P, cfg, nsub, h)
            if out.get("info") is not None:
                plan_info.append(dict(t=t, **out["info"]))
        # --- servo: nominal policy + energy filter -------------------------------------------------
        u_nom, v_ref, dI, f_set = M.policy(t, q, qd, xref, I_f, cand[0], cand[1], cand[2], P,
                                          fn=con["fn"], rho_T=con["rho"], vt=con["vt"])
        DT_hat = float(con["DT_credit"])
        Ec_k, BH_k = led.Ec, led.BH
        BH_used = (acc["BH0"] - z[7]) if getattr(controller, "oracle", False) else led.BH
        bH = DT_hat + lam_H * (BH_used + m_e - kap * sd_e - S_hat)
        bE = DT_hat + lam_E * (led.Ec - led.Emin)
        if controller.filtered:
            tau_e = con["tau"] - P.B * qd
            if cur_ev is not None:
                tau_e = tau_e + M.gen_force(q, M.r_human(s_hat, P.he), y0)
            if getattr(controller, "instantaneous_filter", False):
                u, status = project_power(u_nom, qd, min(bH, bE), P.umax, W=Wu)
            else:
                u, status, _ = project_tick_energy(u_nom, qd, tau_e, P.Mdiag, HS, min(bH, bE), P.umax, Wu)
        else:
            u, status = np.clip(u_nom, -P.umax, P.umax), "off"
        # --- truth physics over the tick -----------------------------------------------------------
        for j in range(nsub):
            plant.step(u, h)
            if sub_log:
                sub.append(plant.z.copy())
        plant.begin_tick()
        z1 = plant.z.copy()
        t1 = (k + 1) * HS
        f1, y1, ev1 = measure(k + 1, z1, t1)
        q1, qd1 = z1[:3], z1[3:6]
        DT1 = float(M.contact(q1, qd1, P)["DT_credit"])
        led.book_tick(u, q, q1, DT_hat, DT1, HS)
        if cur_ev is not None:
            rH = M.r_human(s_hat, P.he)
            vh0 = M.point_vel(q, qd, rH)
            vh1 = M.point_vel(q1, qd1, rH)
            c0, s0 = math.cos(q[2]), math.sin(q[2])
            c1, s1 = math.cos(q1[2]), math.sin(q1[2])
            G0 = qd[2] * (y0[0] * -s0 + y0[1] * c0)
            G1 = qd1[2] * (y1[0] * -s1 + y1[1] * c1)
            led.book_human(y0, vh0, y1 if ev1 is cur_ev else np.zeros(2), vh1, HS, G0, G1, s_hat, Sigma_m)
        # --- controller states (executed-command anti-windup) --------------------------------------
        xref = min(xref + v_ref * HS, P.x_target) if t >= P.t_start else xref
        if status in ("free", "off") and abs(u[1]) < P.umax[1] - 1e-9:
            I_f = float(np.clip(I_f + dI * HS, -P.I_max, P.I_max))
        # --- log -----------------------------------------------------------------------------------
        L["q"][k], L["qd"][k], L["u_nom"][k], L["u"][k] = q, qd, u_nom, u
        L["f_true"][k], L["y"][k], L["Ec"][k], L["BH_book"][k] = f0, y0, Ec_k, BH_k
        L["plant_int"][k] = z[6:12]
        L["fn"][k], L["d"][k], L["xref"][k], L["I_f"][k] = con["fn"], con["d"], xref, I_f
        L["cand"][k], L["m_e"][k], L["sd_e"][k], L["s_hat"][k], L["s_var"][k] = cand, m_e, sd_e, s_hat, s_var
        L["engaged"][k] = cur_ev is not None
        L["s_true"][k] = cur_ev.s if cur_ev is not None else np.nan
        L["filt"][k] = {"free": 0, "off": 0, "projected": 1, "infeasible_best_effort": 2}[status]
        L["DT"][k], L["S"][k], L["bH"][k], L["bE"][k] = DT_hat, S_hat, bH, bE
        z, f0, y0, ev0 = z1, f1, y1, ev1
    # Ledger values are logged at the start of each tick; re-log the booked series consistently.
    k = n_ticks
    q, qd = z[:3], z[3:6]
    con = M.contact(q, qd, P)
    L["q"][k], L["qd"][k], L["f_true"][k], L["y"][k], L["plant_int"][k] = q, qd, f0, y0, z[6:12]
    L["fn"][k], L["d"][k], L["xref"][k], L["I_f"][k] = con["fn"], con["d"], xref, I_f
    L["engaged"][k] = ev0 is not None
    L["s_true"][k] = ev0.s if ev0 is not None else np.nan
    L["DT"][k] = con["DT"]
    L["S"][k] = float(M.storage(q, qd, P)[0])
    L["Ec"][k], L["BH_book"][k] = led.Ec, led.BH
    out = {"log": L, "plan_info": plan_info, "plan_time": np.array(plan_time), "ledger": led,
           "events": [dict(family=ev.family, t_on=ev.t_on, t_off=ev.t_off, cause=ev.release_cause, s=ev.s,
                           K=ev.K.tolist(), D=ev.D.tolist(), a_on=ev.a_on) for ev in sc["events"]],
           "h_phys": h, "sub": np.array(sub) if sub_log else None}
    return out
