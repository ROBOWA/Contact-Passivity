"""Analysis of a RunLog: windows, contact absence vs low load, tracking, reference continuity, torque, work.

All functions take explicit evaluation windows (t0, t1) so the initial approach and force ramp are never
mislabelled as steady-contact failures. Geometric contact absence (no tool_tip/tabletop contact in the
physics evaluation), low normal load (true task-contact normal force below a threshold) and the controller's
own FSM counters (UNCERTAIN / LOST events) are reported separately: they are different things.
The table normal is world +z (the scene's tabletop).
"""

from __future__ import annotations

import numpy as np

from .recorder import RunLog

Window = tuple[float, float]


def _sel(t: np.ndarray, window: Window | None) -> np.ndarray:
    if window is None:
        return np.ones(len(t), bool)
    return (t >= window[0]) & (t <= window[1])


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs [i0, i1) of True."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    edges = np.flatnonzero(m[1:] != m[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def event_stats(t: np.ndarray, mask: np.ndarray, dt: float, window: Window | None = None) -> dict:
    """Fraction of the window, number of separate events and longest duration of `mask` being True."""
    sel = _sel(t, window)
    tt, mm = t[sel], mask[sel]
    rs = runs(mm)
    dur = [(b - a) * dt for a, b in rs]
    return dict(fraction=float(mm.mean()) if len(mm) else float("nan"), events=len(rs),
                longest_s=float(max(dur)) if dur else 0.0, total_s=float(mm.sum() * dt),
                first_t=float(tt[rs[0][0]]) if rs else None)


def contact_absence(log: RunLog, window: Window | None = None, low_load_n: float = 0.5) -> dict:
    """Geometric absence (no task contact) and low load (true normal force < low_load_n), separately."""
    p = log.phys
    dt = log.meta["timestep"]
    return dict(window=window,
                geometric_absence=event_stats(p["t"], ~p["present"], dt, window),
                low_load=event_stats(p["t"], p["F_task"][:, 2] < low_load_n, dt, window),
                low_load_threshold_n=low_load_n)


def fsm_events(log: RunLog, window: Window | None = None) -> dict:
    """Controller FSM events inside the window: increments of the controller's own counters (its decisions),
    not physical contact loss."""
    c = log.ctl
    sel = np.flatnonzero(_sel(c["t"], window))
    out = {}
    if len(sel) == 0:
        return out
    before = max(sel[0] - 1, 0)
    for key in ("n_uncertain", "n_lost", "n_reacquired", "n_hold"):
        if key in c:
            out[key] = int(c[key][sel[-1]] - (c[key][before] if sel[0] > 0 else 0))
    return out


def phase_times(log: RunLog) -> dict[str, float | None]:
    """First time of each phase transition (phase codes 0..4)."""
    c = log.ctl
    names = {0: "approach", 1: "contact", 2: "slide", 3: "search", 4: "hold"}
    out: dict[str, float | None] = {}
    ph = c["phase"].astype(int)
    for code, name in names.items():
        idx = np.flatnonzero(ph == code)
        out[name] = float(c["t"][idx[0]]) if len(idx) else None
    return out


def transitions(log: RunLog) -> list[tuple[float, str, str]]:
    c = log.ctl
    names = {0: "approach", 1: "contact", 2: "slide", 3: "search", 4: "hold"}
    ph = c["phase"].astype(int)
    return [(float(c["t"][i]), names[ph[i - 1]], names[ph[i]]) for i in range(1, len(ph)) if ph[i] != ph[i - 1]]


def recovery_intervals(log: RunLog) -> list[tuple[float, float]]:
    """Intervals [t_enter, t_exit) during which the controller was in SEARCH or HOLD (contact recovery)."""
    c = log.ctl
    ph = c["phase"].astype(int)
    out = []
    for a, b in runs((ph == 3) | (ph == 4)):
        out.append((float(c["t"][a]), float(c["t"][min(b, len(c["t"]) - 1)])))
    return out


def force_stats(log: RunLog, window: Window) -> dict:
    """Force errors in the window. `controller` = f_d - f_n (corrected sensor estimate the controller used);
    `truth` = f_d - f_true at physics resolution (f_d of the active control interval)."""
    c, p = log.ctl, log.phys
    sel = _sel(c["t"], window)
    e_ctl = (c["f_d"] - c["f_n"])[sel]
    ci = log.meta["control_interval"]
    k = np.minimum(np.arange(len(p["t"])) // ci, len(c["t"]) - 1)
    f_d_sub = c["f_d"][k]
    psel = _sel(p["t"], window)
    e_true = (f_d_sub - p["F_task"][:, 2])[psel]
    sensor_vs_truth = (p["ft_corr"][:, 2] - p["F_task"][:, 2])[psel]
    return dict(window=window,
                controller_mean_abs=float(np.mean(np.abs(e_ctl))), controller_max_abs=float(np.max(np.abs(e_ctl))),
                truth_mean_abs=float(np.mean(np.abs(e_true))), truth_max_abs=float(np.max(np.abs(e_true))),
                f_true_mean=float(np.mean(p["F_task"][psel, 2])), f_true_std=float(np.std(p["F_task"][psel, 2])),
                sensor_vs_truth_max=float(np.max(np.abs(sensor_vs_truth))))


def tracking_stats(log: RunLog, window: Window, slide_dir=(1.0, 0.0, 0.0)) -> dict:
    """Along-path / cross-path position error and orientation error at control updates (reference and measured
    tcp from the same instant t_k)."""
    c = log.ctl
    sel = _sel(c["t"], window)
    d = np.asarray(slide_dir, float)
    d = d / np.linalg.norm(d)
    e = (c["x_d"][sel] - c["x"][sel])[:, :2]
    along = e @ d[:2]
    cross = e - np.outer(along, d[:2])
    ori = np.degrees(np.linalg.norm(c["e_o"][sel], axis=1))
    return dict(window=window, along_max_mm=float(np.max(np.abs(along)) * 1e3), along_rms_mm=float(np.sqrt(np.mean(along ** 2)) * 1e3),
                cross_max_mm=float(np.max(np.linalg.norm(cross, axis=1)) * 1e3), ori_max_deg=float(np.max(ori)))


def reference_continuity(log: RunLog, window: Window | None = None) -> dict:
    """Continuity of the ACTUAL active reference sequence (x_d, v_d, a_d logged at every control update).

    * pos_step: max |x_d[k+1] - x_d[k]| per axis group (a jump shows up here regardless of the reported velocity)
    * pos_residual: max |x_d[k+1] - x_d[k] - (v_d[k] + v_d[k+1]) dt / 2|  (position change vs reported velocity)
    * vel_residual: max |v_d[k+1] - v_d[k] - (a_d[k] + a_d[k+1]) dt / 2|  (velocity change vs reported acceleration)
    * peak finite-difference speed and acceleration of the reference
    * activation: when the normal reference becomes active (NaN -> finite) the initial position/velocity error
      against the measured state, i.e. the command step a newly activated position loop would see.
    Normal component statistics only use consecutive updates where the normal reference is active.
    """
    c = log.ctl
    dtc = log.meta["control_period"]
    sel = np.flatnonzero(_sel(c["t"], window))
    x, v, a = c["x_d"][sel], c["v_d"][sel], c["a_d"][sel]
    out: dict = {}
    for name, cols in (("tangential", [0, 1]), ("normal", [2])):
        xs, vs, as_ = x[:, cols], v[:, cols], a[:, cols]
        ok = np.isfinite(xs).all(axis=1)
        pair = ok[:-1] & ok[1:]
        dx = np.linalg.norm(xs[1:] - xs[:-1], axis=1)[pair]
        res_p = np.linalg.norm(xs[1:] - xs[:-1] - 0.5 * (vs[1:] + vs[:-1]) * dtc, axis=1)[pair]
        res_v = np.linalg.norm(vs[1:] - vs[:-1] - 0.5 * (as_[1:] + as_[:-1]) * dtc, axis=1)[pair]
        spd = np.linalg.norm(xs[1:] - xs[:-1], axis=1)[pair] / dtc
        acc = np.linalg.norm((xs[2:] - 2 * xs[1:-1] + xs[:-2]), axis=1) / dtc ** 2
        acc = acc[ok[2:] & ok[1:-1] & ok[:-2]]
        out[name] = dict(pos_step_max_mm=float(dx.max() * 1e3) if len(dx) else 0.0,
                         pos_residual_max_mm=float(res_p.max() * 1e3) if len(res_p) else 0.0,
                         vel_residual_max_mm_s=float(res_v.max() * 1e3) if len(res_v) else 0.0,
                         peak_speed_fd=float(spd.max()) if len(spd) else 0.0,
                         peak_accel_fd=float(acc.max()) if len(acc) else 0.0,
                         accel_step_max=float(np.abs(np.diff(np.linalg.norm(as_[ok], axis=1))).max()) if ok.sum() > 1 else 0.0)
    # activation of the normal position loop
    zn = x[:, 2]
    act = np.flatnonzero(np.isfinite(zn[1:]) & ~np.isfinite(zn[:-1])) + 1
    ex = c["x"][sel][:, 2]
    ev = c["v"][sel][:, 2]
    out["normal_activation"] = [dict(t=float(c["t"][sel][i]), pos_err_mm=float((zn[i] - ex[i]) * 1e3),
                                     vel_err_mm_s=float((v[i, 2] - ev[i]) * 1e3)) for i in act]
    return out


def torque_stats(log: RunLog, window: Window | None = None) -> dict:
    """Peak applied torque (actual MuJoCo actuator force) and saturation, plus the commanded value."""
    p, c = log.phys, log.ctl
    lim = np.asarray(log.meta["torque_limit"])
    psel = _sel(p["t"], window)
    util = np.abs(p["tau_applied"][psel]) / lim
    sat_sub = (util >= 1.0 - 1e-9).any(axis=1)
    csel = _sel(c["t"], window)
    cmd = np.abs(c["tau_cmd"][csel]) / lim
    cmd_sat = (cmd >= 1.0 - 1e-9).any(axis=1)
    rs = runs(sat_sub)
    return dict(window=window, peak_applied_utilisation=float(util.max()), peak_commanded_utilisation=float(cmd.max()),
                saturated_substeps=int(sat_sub.sum()), saturated_updates=int(cmd_sat.sum()),
                longest_saturation_s=float(max((b - a) for a, b in rs) * log.meta["timestep"]) if rs else 0.0,
                peak_applied_nm=np.max(np.abs(p["tau_applied"][psel]), axis=0).tolist())


def work_stats(log: RunLog, window: Window | None = None) -> dict:
    """Signed contact work in the window by three quadratures of the same coherent samples (environment on robot).

    rect   sum P_k dt                                 left endpoint (the recorded dW): force and velocity of evaluation k
    trap   sum (P_k + P_k+1) dt / 2                   trapezoid over consecutive evaluations
    next   sum (F_k . v_p,k+1 + M_k . w_k+1) dt       force of evaluation k with the velocity of the NEXT evaluation,
                                                      at the same contact point p_c,k: MuJoCo's semi-implicit integrator
                                                      advances positions with v_k+1, so this is the pairing that matches
                                                      the actual displacement of the step
    The spread of the three at a fixed step is the quadrature part of the discretisation error; it is first order
    in the step and, unlike an exact energy balance, is not zero (the contact is a soft constraint integrated with a
    semi-implicit scheme, and the controller is sampled).
    """
    p = log.phys
    dt = log.meta["timestep"]
    sel = np.flatnonzero(_sel(p["t"], window))
    sel = sel[sel < len(p["t"]) - 1]
    P = p["P_task"]
    rect = float(np.sum(P[sel]) * dt)
    trap = float(np.sum(0.5 * (P[sel] + P[sel + 1])) * dt)
    v_next = p["tip_v"][sel + 1] + np.cross(p["w_tool"][sel + 1], p["p_c"][sel] - p["tip"][sel + 1])
    nxt = float(np.sum(np.einsum("ij,ij->i", p["F_task"][sel], v_next) + np.einsum("ij,ij->i", p["M_task"][sel], p["w_tool"][sel + 1])) * dt)
    has = p["present"][sel] | p["present"][sel + 1]
    nxt = float(np.sum((np.einsum("ij,ij->i", p["F_task"][sel], v_next) + np.einsum("ij,ij->i", p["M_task"][sel], p["w_tool"][sel + 1]))[has]) * dt)
    return dict(window=window, rect_J=rect, trap_J=trap, next_J=nxt,
                positive_J=float(np.sum(np.clip(P[sel], 0, None)) * dt), negative_J=float(np.sum(np.clip(P[sel], None, 0)) * dt),
                peak_power_W=float(np.max(np.abs(P[sel]))) if len(sel) else 0.0)


def force_peaks(log: RunLog, window: Window | None = None) -> dict:
    """Peak true task-contact normal force (physics rate) and, per CONTACT entry, the peak in the first 0.5 s."""
    p = log.phys
    sel = _sel(p["t"], window)
    fn = p["F_task"][:, 2]
    out = dict(window=window, peak_N=float(fn[sel].max()) if sel.any() else 0.0)
    entries = []
    for t, _, b in transitions(log):
        if b == "contact" and (window is None or window[0] <= t <= window[1]):
            m = (p["t"] >= t) & (p["t"] <= t + 0.5)
            entries.append((round(t, 3), float(fn[m].max())))
    out["peak_after_contact_entry"] = entries
    return out
