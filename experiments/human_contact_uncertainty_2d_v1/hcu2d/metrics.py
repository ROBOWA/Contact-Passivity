"""Per-episode metrics (task, interaction, accounts, planner, timing). Truth-side quantities come
from the independent auditor; the controller never sees them."""
import numpy as np

TOL = 1e-6   # J, numerical tolerance for reporting a violation (raw minima are also reported)


def episode_metrics(out, audit, ser, cfg, sc, controller_key):
    L = out["log"]
    P, T = cfg["task"], cfg["body"]
    n = len(L["q"])
    t = np.arange(n) * 1e-3
    q = L["q"]
    c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
    tip_x = q[:, 0] + c * T["r_T"][0] - s * T["r_T"][1]
    x_t, x_s, f_d = P["x_target"], P["x_start"], P["f_d"]
    done = np.where(tip_x >= x_t - 0.005)[0]
    t_done = float(t[done[0]]) if len(done) else np.nan
    end = done[0] if len(done) else n - 1
    wipe = (t >= P["t_start"]) & (np.arange(n) <= end)
    fn = L["fn"]
    # wiping coverage: 1 mm bins of the path visited with adequate normal force
    bins = np.zeros(int(round((x_t - x_s) / 1e-3)), bool)
    ok = (fn >= 0.5 * f_d) & (fn <= 1.5 * f_d)
    bi = np.clip(((tip_x - x_s) / 1e-3).astype(int), 0, len(bins) - 1)
    bins[bi[ok & (tip_x >= x_s) & (tip_x <= x_t)]] = True
    x_nom = np.clip(x_s + P["v_nom"] * np.maximum(0, t - P["t_start"]), x_s, x_t)
    lag = np.maximum(0.0, x_nom - tip_x)
    u = L["u"][:-1]
    umax = np.array(cfg["robot"]["u_max"])
    du = np.diff(u, axis=0) / umax
    contact_lost = wipe & (fn < 0.5) & (t >= P["t_ramp"] + 0.2)
    # release recovery and post-release surge
    rec = []
    for ev in out["events"]:
        if ev["t_off"] is None:
            continue
        k0 = int(round(ev["t_off"] / 1e-3))
        win = slice(k0, min(n, k0 + 1500))
        good = (np.abs(fn[win] - f_d) < 2.0) & (np.abs(q[win, 2]) < 0.03)
        run, k_ok = 0, None
        for i, g in enumerate(good):
            run = run + 1 if g else 0
            if run >= 200:
                k_ok = i - 199
                break
        vtip = L["qd"][win, 0] - L["qd"][win, 2] * (s[win] * T["r_T"][0] + c[win] * T["r_T"][1])
        rec.append(dict(recovery_s=(k_ok * 1e-3 if k_ok is not None else np.nan), surge_v=float(np.max(np.abs(vtip))),
                        fn_peak_post=float(np.max(fn[win])), cause=ev["cause"]))
    pi = out["plan_info"]
    pt = out["plan_time"]
    eng = [p for p in pi if p.get("engaged")]
    m = dict(scenario=sc["name"], seed=sc["seed"], controller=controller_key,
             completed=bool(len(done) > 0), t_done=t_done, tip_x_final=float(tip_x[-1]),
             coverage=float(bins.mean()),
             force_rmse=float(np.sqrt(np.mean((fn[wipe] - f_d) ** 2))) if wipe.any() else np.nan,
             theta_rmse=float(np.sqrt(np.mean(q[wipe, 2] ** 2))) if wipe.any() else np.nan,
             theta_maxabs=float(np.max(np.abs(q[:, 2]))),
             lag_rms=float(np.sqrt(np.mean(lag[wipe] ** 2))) if wipe.any() else np.nan, lag_max=float(lag.max()),
             contact_loss_s=float(contact_lost.sum() * 1e-3), ref_drift_max=float(np.max(L["xref"] - tip_x)),
             du_rms=float(np.sqrt(np.mean(np.sum(du ** 2, 1)))), sat_frac=float(np.mean(np.any(np.abs(u) >= umax - 1e-9, 1))),
             filt_frac=float(np.mean(L["filt"][:-1] > 0)), filt_infeasible_ticks=int(np.sum(L["filt"][:-1] == 2)),
             fn_peak=float(fn.max()),
             release=rec, recovery_s=float(np.nanmax([r["recovery_s"] for r in rec])) if rec else np.nan,
             surge_v=float(max([r["surge_v"] for r in rec])) if rec else np.nan,
             n_plan=len(pi), n_fallback=int(sum(p["status"] != "ok" for p in pi)),
             n_fallback_engaged=int(sum(p["status"] != "ok" for p in eng)),
             plan_ms_p50=float(np.median(pt) * 1e3) if len(pt) else 0.0, plan_ms_p95=float(np.quantile(pt, .95) * 1e3) if len(pt) else 0.0,
             plan_ms_p99=float(np.quantile(pt, .99) * 1e3) if len(pt) else 0.0, plan_ms_max=float(pt.max() * 1e3) if len(pt) else 0.0,
             plan_ms_engaged_p95=float(np.quantile([p["t_mc"] + p["t_lin"] + p["t_model"] for p in eng], .95) * 1e3) if eng else 0.0,
             events=out["events"])
    m.update({k: audit[k] for k in audit})
    m["violation_Ec"] = bool(audit["Ec_min"] < cfg["accounts"]["Emin"] - TOL)
    m["violation_BH"] = bool(audit["BH_true_min"] < -TOL)
    m["violation_hH"] = bool(audit["hH_true_min"] < -TOL)
    m["violation_any"] = m["violation_Ec"] or m["violation_BH"] or m["violation_hH"]
    m["infeasible_init"] = bool(audit["hH0"] < -TOL)
    m["domain_violation"] = bool(audit["corner_min_y"] < -1e-4)   # a body corner below the wall: contact model invalid
    if eng:
        ch = np.array([p["cand"] for p in eng])
        m["sel_sig_v_mean"] = float(np.mean(np.maximum(ch[:, 0], 0)))
        m["sel_sig_K_mean"] = float(np.mean(ch[:, 1]))
        m["sel_sig_F_mean"] = float(np.mean(ch[:, 2]))
        m["sel_backup_frac"] = float(np.mean((ch[:, 0] < 0) & (ch[:, 0] > -1.5)))
        m["sel_brake_frac"] = float(np.mean(ch[:, 0] <= -1.5))
        m["obj_se_mean"] = float(np.mean([p["obj_se"] for p in eng]))
        m["ess_min"] = float(np.min([p["ess"] for p in eng]))
        m["s_err_gp"] = float(np.sqrt(np.mean([(p["s_m"] - sc_s(sc, p["t"])) ** 2 for p in eng])))
        m["s_out_of_patch_max"] = float(np.max([p["s_out_of_patch"] for p in eng]))
        m["sd_e_max"] = float(np.max([p["sd_e"] for p in eng]))
    return m


def sc_s(sc, t):
    for ev in sc["events"]:
        if ev.t_on <= t and (ev.t_off is None or t <= ev.t_off + 0.03):
            return ev.s
    return np.nan


def traces(out, ser, every=20):
    """Downsampled (50 Hz) traces for figures/animation."""
    L = out["log"]
    sl = slice(None, None, every)
    keys = ["q", "qd", "u", "f_true", "y", "Ec", "BH_book", "fn", "xref", "cand", "s_hat", "s_var", "s_true", "engaged",
            "filt", "m_e", "sd_e", "S"]
    tr = {k: L[k][sl] for k in keys}
    for k in ["BH_true", "hH_true", "pH", "S"]:
        tr["true_" + k] = ser[k][sl]
    tr["t"] = (np.arange(len(L["q"])) * 1e-3)[sl]
    return tr
