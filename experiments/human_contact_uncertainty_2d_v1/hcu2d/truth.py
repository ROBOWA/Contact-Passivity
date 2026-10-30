"""Ground-truth simulator: planar rigid tool, passive wall contact, impedance human (formulation §1-3, §6).

Scalar pure-Python implementation (fast for one state, and independent of the vectorised predictor in
model.py). Nothing in this module is visible to the controllers except through hcu2d.sensors.
"""
import math
import numpy as np


def smoothstep_int(x):
    """Integral of the smoothstep S(x)=3x^2-2x^3 (0 for x<0, 1 for x>1), with value 0 at x<=0."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return x - 0.5
    return x ** 3 - 0.5 * x ** 4


def smoothstep(x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


class Profile:
    """One axis of a scripted intention displacement Δa(τ), τ = t - t_on (C^2-smooth pieces).

    vel: list of (τ_k, Δv_k, T_k)  velocity changes Δv_k blended by smoothstep over T_k,
    steps: list of (τ_k, d_k, T_k)  minimum-jerk position steps of size d_k over T_k,
    sin: (A, f, T_in) or None      sinusoid A sin(2π f τ) with smoothstep amplitude ramp over T_in.
    """

    def __init__(self, vel=(), steps=(), sin=None):
        self.vel, self.steps, self.sin = list(vel), list(steps), sin

    def __call__(self, tau):
        p = v = 0.0
        for tk, dv, T in self.vel:
            x = (tau - tk) / T
            p += dv * T * smoothstep_int(x)
            v += dv * smoothstep(x)
        for tk, d, T in self.steps:
            x = min(max((tau - tk) / T, 0.0), 1.0)
            p += d * (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5)
            v += d * (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / T if 0.0 < x < 1.0 else 0.0
        if self.sin is not None:
            A, f, T_in = self.sin
            w = 2 * math.pi * f
            r, dr = smoothstep(tau / T_in), (6 * (tau / T_in) * (1 - tau / T_in) / T_in if 0 < tau < T_in else 0.0)
            p += A * r * math.sin(w * tau)
            v += A * (dr * math.sin(w * tau) + r * w * math.cos(w * tau))
        return p, v

    def to_dict(self):
        return {"vel": self.vel, "steps": self.steps, "sin": self.sin}


class HumanEvent:
    """One grasp→release contact event with hidden truth parameters."""

    def __init__(self, t_on, t_off, s, K, D, prof_x, prof_y, residual, rng, cfg_h, grid=2.5e-4, family=""):
        self.t_on, self.t_off_script, self.s = float(t_on), float(t_off), float(s)
        self.K, self.D = np.array(K, float), np.array(D, float)
        self.px, self.py, self.residual, self.family = prof_x, prof_y, bool(residual), family
        self.c_g, self.v_g = cfg_h["c_g"], cfg_h["v_g"]
        self.F_rel, self.T_rel = cfg_h["F_rel"], cfg_h["t_release_ramp"]
        self.t_off = None            # actual release start (script or threshold)
        self.release_cause = None
        self.a_on = None             # grasp point p_Hc(t_on), set by the plant at t_on
        # Physical force variability: exact OU on a fixed fine grid (independent of physics step).
        n = int(math.ceil((t_off - t_on + 0.5) / grid)) + 2
        rho, sp = math.exp(-grid / cfg_h["tau_p"]), cfg_h["sigma_p"]
        eps = np.empty((n, 2))
        eps[0] = rng.normal(0.0, sp, 2)
        xi = rng.normal(0.0, 1.0, (n, 2)) * sp * math.sqrt(1 - rho * rho)
        for k in range(1, n):
            eps[k] = rho * eps[k - 1] + xi[k]
        self.eps, self.grid = eps, grid

    def weight(self, t):
        if t < self.t_on:
            return 0.0
        if self.t_off is None:
            return 1.0
        return max(0.0, 1.0 - (t - self.t_off) / self.T_rel)

    def active(self, t):
        return self.weight(t) > 0.0

    def intention(self, t):
        tau = t - self.t_on
        dx, vx = self.px(tau)
        dy, vy = self.py(tau)
        return self.a_on[0] + dx, self.a_on[1] + dy, vx, vy

    def noise(self, t):
        u = (t - self.t_on) / self.grid
        i = min(max(int(u), 0), len(self.eps) - 2)
        f = min(max(u - i, 0.0), 1.0)
        e = self.eps
        return e[i, 0] + f * (e[i + 1, 0] - e[i, 0]), e[i, 1] + f * (e[i + 1, 1] - e[i, 1])


class TruthPlant:
    """State z = [x, y, th, vx, vy, w, Wa, WH, DTint, DRint, pTint, WHpos] integrated with RK4."""

    NZ = 12

    def __init__(self, cfg, events, q0, qd0=(0.0, 0.0, 0.0)):
        b, r, c = cfg["body"], cfg["robot"], cfg["contact"]
        self.m, self.I = b["m"], b["I"]
        self.rTx, self.rTy, self.he = b["r_T"][0], b["r_T"][1], b["h_e"]
        self.Bx, self.By, self.Bth = r["B_R"]
        self.kn, self.bn, self.ds, self.ve = c["k_n"], c["b_n"], c["delta_s"], c["v_eps"]
        self.mu = c.get("mu_true", c["mu"])          # stress test: true friction may differ from the model
        self.events = list(events)
        self.z = np.zeros(self.NZ)
        self.z[0:3], self.z[3:6] = q0, qd0
        self.t = 0.0

    # --- force laws (scalar) -------------------------------------------------------------------
    def contact(self, x, y, th, vx, vy, w):
        c, s = math.cos(th), math.sin(th)
        rx, ry = c * self.rTx - s * self.rTy, s * self.rTx + c * self.rTy
        vtx, vty = vx - w * ry, vy + w * rx
        gap = y + ry
        if gap >= 0.0:
            return 0.0, 0.0, rx, ry, vtx, vty, 0.0, 0.0
        d = -gap
        dd = -vty
        chi = d / (d + self.ds)
        ddp = dd if dd > 0.0 else 0.0
        fn = self.kn * d + self.bn * chi * ddp
        th_ = math.tanh(vtx / self.ve)
        ft = -self.mu * fn * th_
        DT = self.bn * chi * ddp * ddp + self.mu * fn * vtx * th_
        return ft, fn, rx, ry, vtx, vty, DT, d

    def human(self, t, x, y, th, vx, vy, w):
        """True f_H (world) at time t and the material-point quantities used for power."""
        for ev in self.events:
            wt = ev.weight(t)
            if wt <= 0.0 or ev.a_on is None:
                continue
            c, s = math.cos(th), math.sin(th)
            rx, ry = c * ev.s - s * self.he, s * ev.s + c * self.he
            px, py = x + rx, y + ry
            vhx, vhy = vx - w * ry, vy + w * rx
            ax, ay, adx, ady = ev.intention(t)
            ex, ey = ev.noise(t)
            fx = ev.K[0] * (ax - px) + ev.D[0] * (adx - vhx) + ex
            fy = ev.K[1] * (ay - py) + ev.D[1] * (ady - vhy) + ey
            if ev.residual:
                # patch-reference point r=(0,h_e): rho_c = (-s h_e, c h_e), velocity = v + w*Omega*rho_c
                cx = vx - w * (c * self.he)
                cy = vy - w * (s * self.he)
                fx += -ev.c_g * math.tanh(cx / ev.v_g)
                fy += -ev.c_g * math.tanh(cy / ev.v_g)
            return wt * fx, wt * fy, rx, ry, vhx, vhy
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def rhs(self, t, z, u):
        x, y, th, vx, vy, w = z[0], z[1], z[2], z[3], z[4], z[5]
        ft, fn, rx, ry, vtx, vty, DT, _ = self.contact(x, y, th, vx, vy, w)
        fhx, fhy, hrx, hry, vhx, vhy = self.human(t, x, y, th, vx, vy, w)
        ux, uy, ut = u
        ax = (ux + ft + fhx - self.Bx * vx) / self.m
        ay = (uy + fn + fhy - self.By * vy) / self.m
        al = (ut + rx * fn - ry * ft + hrx * fhy - hry * fhx - self.Bth * w) / self.I
        Pa = ux * vx + uy * vy + ut * w
        pH = -(fhx * vhx + fhy * vhy)
        DR = self.Bx * vx * vx + self.By * vy * vy + self.Bth * w * w
        pT = -(ft * vtx + fn * vty)
        return np.array([vx, vy, w, ax, ay, al, Pa, pH, DT, DR, pT, pH if pH > 0.0 else 0.0])

    def begin_tick(self):
        """Grasp (intention starts at the current material point) and scripted release."""
        t, z = self.t, self.z
        for ev in self.events:
            if ev.a_on is None and t >= ev.t_on - 1e-12:
                c, s = math.cos(z[2]), math.sin(z[2])
                ev.a_on = (z[0] + c * ev.s - s * self.he, z[1] + s * ev.s + c * self.he)
                ev.t_on = t
            if ev.a_on is not None and ev.t_off is None and t >= ev.t_off_script - 1e-12:
                ev.t_off, ev.release_cause = t, "script"

    def engaged_event(self, t=None):
        t = self.t if t is None else t
        for ev in self.events:
            if ev.a_on is not None and ev.weight(t) > 0.0:
                return ev
        return None

    def step(self, u, h):
        """One RK4 step of length h with zero-order-hold u (threshold release checked at the end)."""
        t, z = self.t, self.z
        k1 = self.rhs(t, z, u)
        k2 = self.rhs(t + h / 2, z + h / 2 * k1, u)
        k3 = self.rhs(t + h / 2, z + h / 2 * k2, u)
        k4 = self.rhs(t + h, z + h * k3, u)
        self.z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t = round(t + h, 12)
        for ev in self.events:          # human lets go above the force threshold
            if ev.a_on is not None and ev.t_off is None:
                fx, fy = self.human(self.t, *self.z[:6])[:2]
                if math.hypot(fx, fy) > ev.F_rel:
                    ev.t_off, ev.release_cause = self.t, "force_threshold"
        return self.z

    # --- convenience ---------------------------------------------------------------------------
    def storage(self, z=None):
        z = self.z if z is None else z
        _, _, _, _, _, _, _, d = self.contact(*z[:6])
        SR = 0.5 * (self.m * (z[3] ** 2 + z[4] ** 2) + self.I * z[5] ** 2)
        return SR + 0.5 * self.kn * d * d, SR, 0.5 * self.kn * d * d

    def human_force(self, t=None, z=None):
        t = self.t if t is None else t
        z = self.z if z is None else z
        return self.human(t, *z[:6])

    def contact_now(self, z=None):
        z = self.z if z is None else z
        return self.contact(*z[:6])
