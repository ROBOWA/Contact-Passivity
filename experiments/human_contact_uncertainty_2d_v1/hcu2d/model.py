"""Controller-side (known) model, vectorised over a leading batch axis.

Geometry/Jacobians (formulation §1), the known task-contact law (§2), storage, and the nominal hybrid
force/impedance policy with candidate modifiers (§10.1). Used by the servo filter, ledger and the
belief rollouts. The truth simulator (truth.py) and the auditor (auditor.py) re-implement the
physics independently.
"""
import numpy as np


class Known:
    """Unpacked known parameters (body, contact, robot, task, controller)."""

    def __init__(self, cfg):
        b, c, r, t, k = cfg["body"], cfg["contact"], cfg["robot"], cfg["task"], cfg["ctrl"]
        self.m, self.I = b["m"], b["I"]
        self.Mdiag = np.array([b["m"], b["m"], b["I"]])
        self.rT = np.array(b["r_T"], float)
        self.he = b["h_e"]
        self.kn, self.bn, self.ds, self.mu, self.ve = c["k_n"], c["b_n"], c["delta_s"], c["mu"], c["v_eps"]
        self.mu_credit = c.get("mu_credit", c["mu"])   # credit uses the lower end of a known μ interval
        self.B = np.array(r["B_R"], float)
        self.umax = np.array(r["u_max"], float)
        self.f_d, self.v_nom = t["f_d"], t["v_nom"]
        self.x_start, self.x_target, self.t_start, self.t_ramp = t["x_start"], t["x_target"], t["t_start"], t["t_ramp"]
        self.K_t, self.D_t, self.k_f, self.k_i, self.I_max = k["K_t"], k["D_t"], k["k_f"], k["k_i"], k["I_max"]
        self.D_n, self.K_th, self.D_th, self.mu_hat = k["D_n"], k["K_th"], k["D_th"], k["mu_hat"]
        self.v_app, self.D_brake, self.I_leak = k["v_app"], k["D_brake"], k["I_leak"]
        self.D_brk = np.array(k["D_brk"], float)


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return c, s


def point(q, r):
    """World position of body point r; q (...,3), r (...,2) or (2,). Returns (...,2)."""
    c, s = rot(q[..., 2])
    r = np.broadcast_to(r, q.shape[:-1] + (2,))
    return np.stack([q[..., 0] + c * r[..., 0] - s * r[..., 1], q[..., 1] + s * r[..., 0] + c * r[..., 1]], -1)


def point_vel(q, qd, r):
    c, s = rot(q[..., 2])
    r = np.broadcast_to(r, q.shape[:-1] + (2,))
    rx, ry = c * r[..., 0] - s * r[..., 1], s * r[..., 0] + c * r[..., 1]
    return np.stack([qd[..., 0] - qd[..., 2] * ry, qd[..., 1] + qd[..., 2] * rx], -1)


def jac(q, r):
    """J(q,r) = [I2, R Ω r] with shape (...,2,3)."""
    c, s = rot(q[..., 2])
    r = np.broadcast_to(r, q.shape[:-1] + (2,))
    rx, ry = c * r[..., 0] - s * r[..., 1], s * r[..., 0] + c * r[..., 1]
    J = np.zeros(q.shape[:-1] + (2, 3))
    J[..., 0, 0] = 1.0
    J[..., 1, 1] = 1.0
    J[..., 0, 2] = -ry
    J[..., 1, 2] = rx
    return J


def r_human(s, he):
    s = np.asarray(s, float)
    return np.stack([s, np.full_like(s, he)], -1)


def gen_force(q, r, F):
    """τ = J(q,r)^T F  -> (...,3)."""
    c, s = rot(q[..., 2])
    r = np.broadcast_to(r, q.shape[:-1] + (2,))
    rx, ry = c * r[..., 0] - s * r[..., 1], s * r[..., 0] + c * r[..., 1]
    return np.stack([F[..., 0], F[..., 1], rx * F[..., 1] - ry * F[..., 0]], -1)


def contact(q, qd, P):
    """Known passive contact law (§2). Returns dict of arrays."""
    c, s = rot(q[..., 2])
    rx, ry = c * P.rT[0] - s * P.rT[1], s * P.rT[0] + c * P.rT[1]
    vtx, vty = qd[..., 0] - qd[..., 2] * ry, qd[..., 1] + qd[..., 2] * rx
    d = np.maximum(0.0, -(q[..., 1] + ry))
    ddp = np.where(d > 0.0, np.maximum(-vty, 0.0), 0.0)
    chi = d / (d + P.ds)
    fn = P.kn * d + P.bn * chi * ddp
    tt = np.tanh(vtx / P.ve)
    ft = -P.mu * fn * tt
    DT = P.bn * chi * ddp ** 2 + P.mu * fn * vtx * tt
    tau = np.stack([ft, fn, rx * fn - ry * ft], -1)
    DT_credit = DT - (P.mu - P.mu_credit) * fn * vtx * tt
    return {"DT_credit": DT_credit, "d": d, "fn": fn, "ft": ft, "DT": DT, "ST": 0.5 * P.kn * d ** 2, "tau": tau,
            "vt": np.stack([vtx, vty], -1), "rho": np.stack([rx, ry], -1)}


def storage(q, qd, P):
    con = contact(q, qd, P)
    SR = 0.5 * (P.m * (qd[..., 0] ** 2 + qd[..., 1] ** 2) + P.I * qd[..., 2] ** 2)
    return SR + con["ST"], SR, con["ST"]


def schedule(t, P):
    """Nominal task schedule x_nom(t) for the tool tip (progress reference for the cost)."""
    return np.clip(P.x_start + P.v_nom * np.maximum(0.0, t - P.t_start), P.x_start, P.x_target)


def policy(t, q, qd, xref, I_f, sig_v, sig_K, sig_F, P, fn=None, rho_T=None, vt=None):
    """Nominal hybrid controller with candidate modifiers (vectorised).

    Returns u_nom (clipped to the actuator box), v_ref (reference rate) and integrator rate.
    sig_v = -1 marks the passive backup (u = 0); sig_v = -2 the dissipative brake u = -D_brk q̇.
    """
    if fn is None or rho_T is None or vt is None:
        con = contact(q, qd, P)
        fn, rho_T, vt = con["fn"], con["rho"], con["vt"]
    ramp = np.clip(t / P.t_ramp, 0.0, 1.0)
    ramp = ramp * ramp * (3 - 2 * ramp)
    f_set = np.abs(sig_F) * P.f_d * ramp
    running = (t >= P.t_start) & (xref < P.x_target)
    v_ref = np.where(running, np.maximum(sig_v, 0.0) * P.v_nom, 0.0)
    pTx = q[..., 0] + rho_T[..., 0]
    Ftx = sig_K * P.K_t * (xref - pTx) + P.D_t * (v_ref - vt[..., 0]) + P.mu_hat * fn * np.tanh(v_ref / P.ve)
    # approach-speed limiter: extra normal damping only when moving toward the wall faster than v_app
    Fty = (-(f_set + P.k_f * (f_set - fn) + I_f) - P.D_n * vt[..., 1]
           + P.D_brake * np.maximum(0.0, -P.v_app - vt[..., 1]))
    tau = -P.K_th * q[..., 2] - P.D_th * qd[..., 2]
    u = np.stack([Ftx, Fty, rho_T[..., 0] * Fty - rho_T[..., 1] * Ftx + tau], -1)
    u = np.clip(u, -P.umax, P.umax)
    backup = (sig_v < 0) & (sig_v > -1.5)            # σ_v = -1: passive backup u = 0
    brake = sig_v <= -1.5                             # σ_v = -2: strictly dissipative brake u = -D q̇ (P_a ≤ 0)
    u = np.where(backup[..., None], 0.0, u)
    u = np.where(brake[..., None], np.clip(-P.D_brk * qd, -P.umax, P.umax), u)
    dI = np.where(fn > 0.0, P.k_i * (f_set - fn), -P.I_leak * I_f)     # integrator leaks out of contact
    return u, v_ref, dI, f_set
