"""Servo-rate (1 kHz) energy filter (formulation §11) and the software ledgers (§4.1-4.2)."""
import math
import numpy as np


def project_power(u_nom, qd, bound, umax, W=None, iters=60):
    """argmin_{|u_i|<=umax_i} ||u-u_nom||_W^2  s.t.  u^T qd <= bound   (exact, monotone bisection).

    Returns (u, status). status: 'free' (unmodified), 'projected', 'infeasible_best_effort'.
    Feasibility: the minimum of u^T qd over the box is -sum(umax|qd|); if bound is below it, the
    most energy-extracting admissible input is returned and flagged.
    """
    u_nom = np.clip(np.asarray(u_nom, float), -umax, umax)
    if float(u_nom @ qd) <= bound:
        return u_nom, "free"
    Winv = np.ones(3) if W is None else 1.0 / np.asarray(W, float)
    lo_val = -float(np.sum(umax * np.abs(qd)))
    if bound < lo_val:
        return -umax * np.sign(qd), "infeasible_best_effort"
    g = Winv * qd
    nu_lo, nu_hi = 0.0, 1.0
    while float(np.clip(u_nom - nu_hi * g, -umax, umax) @ qd) > bound:
        nu_hi *= 2.0
        if nu_hi > 1e12:
            return -umax * np.sign(qd), "infeasible_best_effort"
    for _ in range(iters):
        mid = 0.5 * (nu_lo + nu_hi)
        if float(np.clip(u_nom - mid * g, -umax, umax) @ qd) > bound:
            nu_lo = mid
        else:
            nu_hi = mid
    return np.clip(u_nom - nu_hi * g, -umax, umax), "projected"


def project_tick_energy(u_nom, qd, tau_e, Mdiag, h, bound, umax, W, iters=60):
    """Sampled-data version (§11): constrain the predicted actuator WORK over the ZOH tick.

    Δq(u) ≈ q̇ h + ½ h² M⁻¹(u + τ_e)  (second-order Taylor, external generalised force τ_e held),
    g(u) = uᵀΔq(u)/h = uᵀq̇ + (h/2) uᵀM⁻¹u + (h/2) uᵀM⁻¹τ_e  ≤  bound.
    Convex (ellipsoid ∩ box), contains u=0 when bound ≥ 0. Solved exactly by monotone bisection on
    the multiplier ν of the separable KKT system. Returns (u, status, g(u)).
    """
    u0 = np.clip(np.asarray(u_nom, float), -umax, umax)
    a = qd + 0.5 * h * tau_e / Mdiag          # linear coefficient
    c = h / Mdiag                              # quadratic coefficient (×½ in g)

    def g(u):
        return float(u @ a + 0.5 * np.sum(c * u * u))

    if g(u0) <= bound:
        return u0, "free", g(u0)
    u_min = np.clip(-a / c, -umax, umax)       # minimiser of g over the box
    if g(u_min) > bound:
        return u_min, "infeasible_best_effort", g(u_min)

    def u_of(nu):
        return np.clip((2 * W * u0 - nu * a) / (2 * W + nu * c), -umax, umax)

    lo, hi = 0.0, 1.0
    while g(u_of(hi)) > bound:
        hi *= 4.0
        if hi > 1e14:
            return u_min, "projected", g(u_min)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if g(u_of(mid)) > bound:
            lo = mid
        else:
            hi = mid
    u = u_of(hi)
    return u, "projected", g(u)


class Ledger:
    """Known software ledger: Ec (exact actuator work from encoder increments, conservative credit)
    and booked BH (from measured force and location estimate); plus the BH-error belief bookkeeping."""

    def __init__(self, Ec0, BH0, Emin):
        self.Ec, self.BH, self.Emin = float(Ec0), float(BH0), float(Emin)
        self.Wa = 0.0
        self.CT = 0.0
        self.WH_hat = 0.0
        # e_BH = Gtot*s - Gamma - b_f^T dp - n_e (current event) ; frozen past events add (mean, var)
        self.ev_G = 0.0
        self.ev_Gam = 0.0
        self.ev_dp = np.zeros(2)
        self.ev_var_n = 0.0
        self.ev_Gabs = 0.0          # Σ|G_k| h (post-hoc robust bound)
        self.frozen_mean = 0.0
        self.frozen_var = 0.0
        self.history = []          # per completed event: G, Γ, Δp̂, var(n_e) and the s/b_f moments used

    def book_tick(self, u, q0, q1, DT0, DT1, h):
        dW = float(u @ (q1 - q0))
        C = h * min(DT0, DT1)
        self.Wa += dW
        self.CT += C
        self.Ec += C - dW
        return dW, C

    def book_human(self, y0, vh0, y1, vh1, h, G0=0.0, G1=0.0, s_hat=0.0, Sigma_m=None):
        p_hat = -0.5 * (float(y0 @ vh0) + float(y1 @ vh1))
        self.WH_hat += h * p_hat
        self.BH -= h * p_hat
        g = 0.5 * (G0 + G1) * h
        self.ev_G += g
        self.ev_Gabs += abs(g)
        self.ev_Gam += g * s_hat
        self.ev_dp += 0.5 * h * (vh0 + vh1)
        if Sigma_m is not None:
            v = 0.5 * (vh0 + vh1)
            self.ev_var_n += h * h * float(v @ Sigma_m @ v)
        return p_hat

    def error_moments(self, s_mean, s_var, bf_mean=None, bf_cov=None, cov_s_bf=None):
        """Mean/var of e_BH given the current posterior of (s, b_f) for the active event + frozen."""
        m = self.ev_G * s_mean - self.ev_Gam
        v = self.ev_G ** 2 * s_var + self.ev_var_n
        if bf_mean is not None:
            m -= float(self.ev_dp @ bf_mean)
            v += float(self.ev_dp @ bf_cov @ self.ev_dp)
            if cov_s_bf is not None:
                v -= 2.0 * self.ev_G * float(cov_s_bf @ self.ev_dp)
        return m + self.frozen_mean, v + self.frozen_var

    def freeze_event(self, s_mean, s_var, bf_mean=None, bf_cov=None, cov_s_bf=None):
        self.history.append(dict(G=self.ev_G, Gabs=self.ev_Gabs, Gam=self.ev_Gam, dp=self.ev_dp.copy(), var_n=self.ev_var_n,
                                 s_mean=s_mean, s_var=s_var))
        m, v = self.error_moments(s_mean, s_var, bf_mean, bf_cov, cov_s_bf)
        self.frozen_mean, self.frozen_var = m, v
        self.ev_G = self.ev_Gam = self.ev_var_n = self.ev_Gabs = 0.0
        self.ev_dp = np.zeros(2)


def kappa(alpha):
    """Φ^{-1}(1-α) via the inverse error function (no scipy dependency at servo rate)."""
    from scipy.special import ndtri
    return float(ndtri(1.0 - alpha)) if alpha > 0 else math.inf
