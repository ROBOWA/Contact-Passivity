"""Human-force predictors: structured latent-intention GP (formulation §7), naive GP and
Gauss–Markov baselines (§7.5). All conditioning is on measured data only.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .kernels import IntentKernel, RBFFeatures, ou, se_ard

JITTER = 1e-9


def edge_terms(q, qd, he):
    """c(q): patch-reference point r=(0,h_e); u(θ)=R e1; and their time derivatives (§1)."""
    c_, s_ = np.cos(q[..., 2]), np.sin(q[..., 2])
    c = np.stack([q[..., 0] - he * s_, q[..., 1] + he * c_], -1)
    cd = np.stack([qd[..., 0] - qd[..., 2] * he * c_, qd[..., 1] - qd[..., 2] * he * s_], -1)
    u = np.stack([c_, s_], -1)
    ud = np.stack([-qd[..., 2] * s_, qd[..., 2] * c_], -1)
    return c, cd, u, ud


class StructuredGP:
    """Joint Gaussian over ζ = [ã_x(T),ã̇_x(T),ã_y(T),ã̇_y(T), s, w_x, w_y, b_f, ε_p(T_ep)].

    hyp keys: K(2), D(2), sig_a, ell_a, sig_c, sig_l, T_l, sig_p, tau_p, sig_m, sig_bf,
              mu_s0, sig_s0, sig_bs, sig_es, feat_centers, feat_width, mu_w (2,nw), Sig_w (2,nw,nw), he
    """

    def __init__(self, hyp):
        self.h = hyp
        self.K, self.D = np.asarray(hyp["K"], float), np.asarray(hyp["D"], float)
        self.feat = RBFFeatures(hyp["feat_centers"], hyp["feat_width"])
        self.nw = self.feat.n
        self.mu_w = np.asarray(hyp["mu_w"], float).reshape(2, self.nw)
        self.Sig_w = np.asarray(hyp["Sig_w"], float).reshape(2, self.nw, self.nw)

    def layout(self, N, n_ep):
        nw = self.nw
        idx, o = {}, 0
        for name, n in [("ax", N), ("adx", N), ("ay", N), ("ady", N), ("s", 1), ("wx", nw), ("wy", nw),
                        ("bf", 2), ("epx", n_ep), ("epy", n_ep)]:
            idx[name] = slice(o, o + n)
            o += n
        return idx, o

    def obs_mean_terms(self, t, q, qd, t_on, q_on, chi=None):
        he = self.h["he"]
        c, cd, u, ud = edge_terms(q, qd, he)
        c_on, _, u_on, _ = edge_terms(q_on[None], np.zeros((1, 3)), he)
        base = self.K * (c_on - c) - self.D * cd                      # known offset (n,2)
        hcoef = self.K * (u_on - u) - self.D * ud                      # coefficient of s (n,2)
        chi = cd if chi is None else chi                               # residual input: patch velocity
        return base, hcoef, chi

    def condition(self, t_obs, q_obs, qd_obs, y_obs, t_on, q_on, sbar, n_loc, t_tgt, ep_times=None,
                  return_lml=False, s_known=None):
        """Exact joint Gaussian conditioning (P1 Eq. (24)-(26) structure, extended).

        s_known: if given (offline training with calibrated location), s enters as a known constant.
        """
        h = self.h
        t_obs = np.asarray(t_obs, float)
        n = len(t_obs)
        t_tgt = np.atleast_1d(np.asarray(t_tgt, float))
        N = len(t_tgt)
        ep_times = np.atleast_1d(t_tgt[:1] if ep_times is None else np.asarray(ep_times, float))
        n_ep = len(ep_times)
        idx, nz = self.layout(N, n_ep)
        kern = IntentKernel(h["sig_a"], h["ell_a"], h["sig_c"], h["sig_l"], h["T_l"], t0=t_on)
        base, hco, chi = self.obs_mean_terms(t_obs, q_obs, qd_obs, t_on, q_on)
        use_loc = (n_loc > 0) and (s_known is None)
        ny = 2 * n + (1 if use_loc else 0)
        Kyy = np.zeros((ny, ny))
        C = np.zeros((ny, nz))
        mY = np.zeros(ny)
        Y = np.zeros(ny)
        k, d1, d2, d12 = kern.blocks(t_obs, t_obs)
        kv, d1v, d2v, d12v = kern.blocks(t_obs, t_tgt)
        kT, d1T, d2T, d12T = kern.blocks(t_tgt, t_tgt)
        Kou = ou(t_obs, t_obs, h["sig_p"], h["tau_p"])
        Kou_t = ou(t_obs, ep_times, h["sig_p"], h["tau_p"])
        mu_s0, vs0 = h["mu_s0"], h["sig_s0"] ** 2
        Sz = np.zeros((nz, nz))
        mz = np.zeros(nz)
        for i, (a_, ad_, w_, ep_) in enumerate([("ax", "adx", "wx", "epx"), ("ay", "ady", "wy", "epy")]):
            sl = slice(i * n, (i + 1) * n)
            Ki, Di = self.K[i], self.D[i]
            Phi = self.feat(chi[:, i])
            Kyy[sl, sl] = (Ki * Ki * k + Ki * Di * (d1 + d2) + Di * Di * d12 + Kou
                           + Phi @ self.Sig_w[i] @ Phi.T + h["sig_bf"] ** 2 + h["sig_m"] ** 2 * np.eye(n))
            C[sl, idx[a_]] = Ki * kv + Di * d1v
            C[sl, idx[ad_]] = Ki * d2v + Di * d12v
            C[sl, idx[w_]] = Phi @ self.Sig_w[i]
            C[sl, idx["bf"].start + i] = h["sig_bf"] ** 2
            C[sl, idx[ep_]] = Kou_t
            s_mean = mu_s0 if s_known is None else s_known
            mY[sl] = base[:, i] + s_mean * hco[:, i] + Phi @ self.mu_w[i]
            Y[sl] = y_obs[:, i]
            Sz[idx[a_], idx[a_]] = kT
            Sz[idx[a_], idx[ad_]] = d2T
            Sz[idx[ad_], idx[a_]] = d1T
            Sz[idx[ad_], idx[ad_]] = d12T
            Sz[idx[w_], idx[w_]] = self.Sig_w[i]
            Sz[idx[ep_], idx[ep_]] = ou(ep_times, ep_times, h["sig_p"], h["tau_p"])
            mz[idx[w_]] = self.mu_w[i]
        Sz[idx["bf"], idx["bf"]] = h["sig_bf"] ** 2 * np.eye(2)
        if s_known is None:
            hv = np.concatenate([hco[:, 0], hco[:, 1]])
            Kyy[:2 * n, :2 * n] += vs0 * np.outer(hv, hv)
            C[:2 * n, idx["s"]] = vs0 * hv[:, None]
            Sz[idx["s"], idx["s"]] = vs0
            mz[idx["s"]] = mu_s0
            if use_loc:
                Kyy[-1, :2 * n] = Kyy[:2 * n, -1] = vs0 * hv
                Kyy[-1, -1] = vs0 + h["sig_bs"] ** 2 + h["sig_es"] ** 2 / n_loc
                C[-1, idx["s"]] = vs0
                mY[-1] = mu_s0
                Y[-1] = sbar
        else:
            mz[idx["s"]] = s_known
        Kyy[np.diag_indices(ny)] += JITTER
        cf = cho_factor(Kyy, lower=True)
        alpha = cho_solve(cf, Y - mY)
        mu = mz + C.T @ alpha
        V = solve_triangular(cf[0], C, lower=True)
        Sig = Sz - V.T @ V
        Sig = 0.5 * (Sig + Sig.T)
        out = {"mu": mu, "Sig": Sig, "idx": idx, "t_tgt": t_tgt, "ep_times": ep_times, "prior_mu": mz,
               "prior_Sig": Sz, "Kyy": Kyy, "C": C, "resid": Y - mY}
        if return_lml:
            out["lml"] = float(-0.5 * (Y - mY) @ alpha - np.sum(np.log(np.diag(cf[0]))) - 0.5 * ny * np.log(2 * np.pi))
        return out

    def force_map(self, post, t_star, q_star, qd_star, t_on, q_on):
        """Linear map f(t*) = A ζ + b for a GIVEN (conditioned) robot trajectory at the target times.

        Requires post['t_tgt'] == ep_times == t_star. Returns mean (M,2), cov (2M,2M) ordered [x..., y...].
        """
        idx = post["idx"]
        M = len(t_star)
        base, hco, chi = self.obs_mean_terms(t_star, q_star, qd_star, t_on, q_on)
        nz = len(post["mu"])
        A = np.zeros((2 * M, nz))
        b = np.zeros(2 * M)
        for i, (a_, ad_, w_, ep_) in enumerate([("ax", "adx", "wx", "epx"), ("ay", "ady", "wy", "epy")]):
            rows = slice(i * M, (i + 1) * M)
            A[rows, idx[a_]] = self.K[i] * np.eye(M)
            A[rows, idx[ad_]] = self.D[i] * np.eye(M)
            A[rows, idx["s"]] = hco[:, i:i + 1]
            A[rows, idx[w_]] = self.feat(chi[:, i])
            A[rows, idx[ep_]] = np.eye(M)
            b[rows] = base[:, i]
        mean = A @ post["mu"] + b
        cov = A @ post["Sig"] @ A.T
        return mean.reshape(2, M).T, cov, A, b


class NaiveGP:
    """P1 Alg. 1 step-1 analogue: f_i ~ GP(0, SE_ARD([τ, Δp_i, v_i])) + white (per axis)."""

    def __init__(self, hyp):
        self.h = hyp

    @staticmethod
    def inputs(t, q, qd, t_on, q_on, s_hat, he):
        c, cd, u, ud = edge_terms(q, qd, he)
        c0, _, u0, _ = edge_terms(q_on[None], np.zeros((1, 3)), he)
        p = c + s_hat * u
        v = cd + s_hat * ud
        p0 = c0 + s_hat * u0
        return [np.stack([t - t_on, p[:, i] - p0[0, i], v[:, i]], -1) for i in range(2)]

    def predict(self, X_obs, y_obs, X_star, return_lml=False):
        h = self.h
        out = []
        lml = 0.0
        for i in range(2):
            Kxx = se_ard(X_obs[i], X_obs[i], h["sig_f"], h["ells"]) + (h["sig_n"] ** 2 + JITTER) * np.eye(len(y_obs))
            Ks = se_ard(X_star[i], X_obs[i], h["sig_f"], h["ells"])
            Kss = se_ard(X_star[i], X_star[i], h["sig_f"], h["ells"])
            cf = cho_factor(Kxx, lower=True)
            a = cho_solve(cf, y_obs[:, i])
            V = solve_triangular(cf[0], Ks.T, lower=True)
            out.append((Ks @ a, Kss - V.T @ V))
            lml += -0.5 * y_obs[:, i] @ a - np.sum(np.log(np.diag(cf[0]))) - 0.5 * len(a) * np.log(2 * np.pi)
        return (out, lml) if return_lml else out


class GaussMarkov:
    """C2 predictor: per axis f = μ + η, η OU(σ_f, τ_f), μ ~ N(·, σ_μ²); exogenous (motion-blind)."""

    def __init__(self, hyp):
        self.h = hyp

    def kern(self, t1, t2):
        h = self.h
        return h["sig_mu"] ** 2 + ou(t1, t2, h["sig_f"], h["tau_f"])

    def predict(self, t_obs, y_obs, t_star, return_lml=False):
        h = self.h
        Kxx = self.kern(t_obs, t_obs) + (h["sig_m"] ** 2 + JITTER) * np.eye(len(t_obs))
        Ks = self.kern(t_star, t_obs)
        Kss = self.kern(t_star, t_star)
        cf = cho_factor(Kxx, lower=True)
        V = solve_triangular(cf[0], Ks.T, lower=True)
        cov = Kss - V.T @ V
        means, lml = [], 0.0
        for i in range(2):
            a = cho_solve(cf, y_obs[:, i])
            means.append(Ks @ a)
            lml += -0.5 * y_obs[:, i] @ a - np.sum(np.log(np.diag(cf[0]))) - 0.5 * len(a) * np.log(2 * np.pi)
        out = (np.stack(means, -1), cov)
        return (out, lml) if return_lml else out
