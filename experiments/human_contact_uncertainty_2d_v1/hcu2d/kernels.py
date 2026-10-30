"""Kernels with derivative blocks (formulation §7.2).

Intention kernel k_a(t,t') = SE + constant + linear trend. Notation: ∂1 = d/dt (first argument),
∂2 = d/dt' (second argument). Units: k [m²], ∂k [m²/s], ∂1∂2 k [m²/s²].
P1 Eq. (23) writes the SE kernel as σ² exp(-(z-z')ᵀΛ⁻¹(z-z')) (no factor ½); we use the
equivalent reparameterisation σ² exp(-(t-t')²/(2ℓ²)).
"""
import numpy as np


class IntentKernel:
    def __init__(self, sig_a, ell_a, sig_c, sig_l, T_l, t0=0.0):
        self.sa2, self.ell, self.sc2, self.sl2, self.Tl, self.t0 = sig_a ** 2, ell_a, sig_c ** 2, sig_l ** 2, T_l, t0

    def blocks(self, t1, t2):
        """Return k, ∂1k, ∂2k, ∂1∂2k for all pairs (len(t1) × len(t2))."""
        t1, t2 = np.asarray(t1, float)[:, None], np.asarray(t2, float)[None, :]
        r = t1 - t2
        l2 = self.ell ** 2
        se = self.sa2 * np.exp(-0.5 * r * r / l2)
        a, b = (t1 - self.t0) / self.Tl, (t2 - self.t0) / self.Tl
        k = se + self.sc2 + self.sl2 * a * b
        d1 = -r / l2 * se + self.sl2 * b / self.Tl
        d2 = r / l2 * se + self.sl2 * a / self.Tl
        d12 = (1.0 / l2 - r * r / (l2 * l2)) * se + self.sl2 / self.Tl ** 2
        return k, d1, d2, d12


def ou(t1, t2, sig, tau):
    t1, t2 = np.asarray(t1, float)[:, None], np.asarray(t2, float)[None, :]
    return sig ** 2 * np.exp(-np.abs(t1 - t2) / tau)


class RBFFeatures:
    """Finite residual features φ(χ) (explicitly labelled finite-feature approximation)."""

    def __init__(self, centers, width):
        self.c, self.w = np.asarray(centers, float), float(width)
        self.n = len(self.c)

    def __call__(self, chi):
        chi = np.asarray(chi, float)
        return np.exp(-0.5 * ((chi[..., None] - self.c) / self.w) ** 2)


def se_ard(X1, X2, sig, ells):
    d = (X1[:, None, :] - X2[None, :, :]) / np.asarray(ells)
    return sig ** 2 * np.exp(-0.5 * np.sum(d * d, -1))
