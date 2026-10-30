"""Independent truth auditor (formulation §4.1, §12-C).

Deliberately does NOT import truth.py or model.py: geometry, the contact law, storage and powers are
re-implemented here from the configuration, and recomputed from the logged tick-rate trajectory
(states, applied commands, true human force, true contact coordinate). Two independent work
estimates are compared: the plant's RK4-integrated work channels and this module's quadrature.
"""
import numpy as np


def _contact(q, qd, c, b):
    th = q[:, 2]
    cs, sn = np.cos(th), np.sin(th)
    rx = cs * b["r_T"][0] - sn * b["r_T"][1]
    ry = sn * b["r_T"][0] + cs * b["r_T"][1]
    vx = qd[:, 0] - qd[:, 2] * ry
    vy = qd[:, 1] + qd[:, 2] * rx
    pen = np.clip(-(q[:, 1] + ry), 0.0, None)
    comp = np.where(pen > 0, np.clip(-vy, 0.0, None), 0.0)
    fn = c["k_n"] * pen + c["b_n"] * pen / (pen + c["delta_s"]) * comp
    slip = np.tanh(vx / c["v_eps"])
    mu = c.get("mu_true", c["mu"])
    diss = c["b_n"] * pen / (pen + c["delta_s"]) * comp ** 2 + mu * fn * vx * slip
    return pen, fn, diss


def _corners(q, b):
    """Lowest y of the four rectangle corners (domain check: only the tool tip may touch the wall)."""
    th = q[:, 2]
    cs, sn = np.cos(th), np.sin(th)
    out = np.full(len(q), np.inf)
    for cx in (-b["L"] / 2, b["L"] / 2):
        for cy in (-b["H"] / 2, b["H"] / 2):
            out = np.minimum(out, q[:, 1] + sn * cx + cs * cy)
    return out


def _trapz_cum(f, dt):
    return np.concatenate([[0.0], np.cumsum(0.5 * dt * (f[1:] + f[:-1]))])


def audit(log, cfg, accounts, dt=1e-3):
    b, c, r = cfg["body"], cfg["contact"], cfg["robot"]
    q, qd, u, fH, s = log["q"], log["qd"], log["u"][:-1], log["f_true"], log["s_true"]
    m, I = b["m"], b["I"]
    B = np.array(r["B_R"])
    pen, fn, DT = _contact(q, qd, c, b)
    S = 0.5 * (m * (qd[:, 0] ** 2 + qd[:, 1] ** 2) + I * qd[:, 2] ** 2) + 0.5 * c["k_n"] * pen ** 2
    DR = (B * qd ** 2).sum(1)
    # human material-point velocity at the TRUE coordinate (migration of estimates never enters)
    th = q[:, 2]
    cs, sn = np.cos(th), np.sin(th)
    sz = np.nan_to_num(s)
    rx, ry = cs * sz - sn * b["h_e"], sn * sz + cs * b["h_e"]
    vhx, vhy = qd[:, 0] - qd[:, 2] * ry, qd[:, 1] + qd[:, 2] * rx
    pH = np.where(np.isnan(s), 0.0, -(fH[:, 0] * vhx + fH[:, 1] * vhy))
    Wa = np.concatenate([[0.0], np.cumsum((u * (q[1:] - q[:-1])).sum(1))])   # exact under ZOH
    WH_q, DT_q, DR_q = _trapz_cum(pH, dt), _trapz_cum(DT, dt), _trapz_cum(DR, dt)
    WHpos_q = _trapz_cum(np.clip(pH, 0, None), dt)
    pi = log["plant_int"]                                    # plant RK4 channels: Wa, WH, DT, DR, pT, WH+
    res_q = (S - S[0]) - Wa + WH_q + DR_q + DT_q              # auditor quadrature
    res_p = (S - S[0]) - pi[:, 0] + pi[:, 1] + pi[:, 3] + pi[:, 2]   # plant integrals, auditor storage
    Ec0, BH0, Emin = accounts["Ec0"], accounts["BH0"], accounts["Emin"]
    CT = log["Ec"] - Ec0 + Wa                                 # Ec = Ec0 + C - Wa  =>  booked credit C
    credit_excess_p = float(np.max(CT - pi[:, 2]))            # >0 means booked credit exceeded true D_T
    credit_excess_q = float(np.max(CT - DT_q))
    BH_true = BH0 - pi[:, 1]
    hH_true = BH_true - S
    V = S + log["Ec"] - Emin
    out = {
        "residual_storage_quad_max": float(np.max(np.abs(res_q))),
        "residual_storage_plant_max": float(np.max(np.abs(res_p))),
        "Wa_plant_vs_exact_max": float(np.max(np.abs(pi[:, 0] - Wa))),
        "WH_plant_vs_quad_max": float(np.max(np.abs(pi[:, 1] - WH_q))),
        "DT_plant_vs_quad_max": float(np.max(np.abs(pi[:, 2] - DT_q))),
        "credit_excess_vs_plant": credit_excess_p,
        "credit_excess_vs_quad": credit_excess_q,
        "credit_valid": credit_excess_p <= 1e-9,
        "Ec_min": float(np.min(log["Ec"])),
        "BH_true_min": float(np.min(BH_true)),
        "BH_book_min": float(np.min(log["BH_book"])),
        "hH_true_min": float(np.min(hH_true)),
        "hH0": float(hH_true[0]),
        "V_min": float(np.min(V)),
        "WH_net_final": float(pi[-1, 1]),
        "WH_pos_final": float(pi[-1, 5]),
        "WH_net_max_prefix": float(np.max(pi[:, 1])),
        "BH_err_final": float((BH0 - pi[-1, 1]) - log["BH_book"][-1]),
        "BH_err_maxabs": float(np.max(np.abs((BH0 - pi[:, 1]) - log["BH_book"]))),
        "bound_V0_ok": bool(np.max(pi[:, 1]) <= V[0] + 1e-6),
        "pH_peak": float(np.max(pH)),
        "fH_peak": float(np.max(np.hypot(fH[:, 0], fH[:, 1]))),
        "fn_peak": float(np.max(fn)),
        "corner_min_y": float(np.min(_corners(q, b))),
        "Ec_violation": bool(np.min(log["Ec"]) < Emin - 1e-9),
        "BH_violation": bool(np.min(BH_true) < -1e-9),
        "hH_violation": bool(np.min(hH_true) < -1e-9),
    }
    series = {"S": S, "BH_true": BH_true, "hH_true": hH_true, "pH": pH, "DT": DT, "fn": fn, "Wa": Wa,
              "res_q": res_q, "res_p": res_p, "CT": CT}
    return out, series
