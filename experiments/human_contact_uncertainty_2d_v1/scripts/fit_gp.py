"""Deterministic offline fitting of all human-force predictors on the TRAIN splits only.

Stage 1 (g=0 data): structured-GP intention/noise hyperparameters and population gains by type-II ML
(location known in calibration data).  Stage 2 (residual data): exact Bayesian posterior of the
shared residual-feature weights given stage-1 hyperparameters (intention marginalised per event).
Baselines: naive GP (P1 Alg.1 step 1 analogue) and Gauss–Markov, fitted on the same train split.
Writes outputs/models/gp_frozen.json (frozen before any held-out evaluation).
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params
from hcu2d.gp import StructuredGP, NaiveGP, GaussMarkov, edge_terms
from hcu2d.kernels import IntentKernel, RBFFeatures, ou

cfg = params.default()
HE = cfg["body"]["h_e"]
SIG_M = cfg["sensors"]["sigma_m"]
SUB = 40  # 25 Hz


def load(split):
    return list(np.load(ROOT / "outputs" / "gp_data" / f"{split}.npy", allow_pickle=True))


def sub(ev, t_max=None):
    k = np.arange(0, len(ev["t"]), SUB)
    if t_max is not None:
        k = k[ev["t"][k] <= t_max]
    return ev["t"][k], ev["q"][k], ev["qd"][k], ev["y"][k], ev["f"][k]


BASE_HYP = dict(K=[150.0, 150.0], D=[15.0, 15.0], sig_a=0.03, ell_a=0.5, sig_c=0.02, sig_l=0.03, T_l=1.0,
                sig_p=0.4, tau_p=0.15, sig_m=SIG_M, sig_bf=0.1, mu_s0=0.0, sig_s0=0.1,
                sig_bs=cfg["sensors"]["sigma_bs"], sig_es=cfg["sensors"]["sigma_es"],
                feat_centers=list(np.linspace(-0.24, 0.24, 7)), feat_width=0.06, he=HE)


def hyp_from(theta, base, res_off=True):
    h = dict(base)
    K, D, sa, la, sc, sl, sp, tp = np.exp(theta)
    h.update(K=[K, K], D=[D, D], sig_a=sa, ell_a=la, sig_c=sc, sig_l=sl, sig_p=sp, tau_p=tp)
    nw = len(h["feat_centers"])
    if res_off:
        h["mu_w"] = np.zeros((2, nw)).tolist()
        h["Sig_w"] = np.stack([np.eye(nw) * 0.02 ** 2] * 2).tolist()
    return h


def total_lml(h, data):
    gp = StructuredGP(h)
    tot = 0.0
    for ev in data:
        t, q, qd, y, _ = sub(ev)
        o = gp.condition(t, q, qd, y, ev["t_on"], ev["q_on"], 0.0, 0, t[-1:], return_lml=True, s_known=ev["s"])
        tot += o["lml"]
    return tot


def stage1(data):
    th0 = np.log([150, 15, 0.03, 0.5, 0.02, 0.03, 0.4, 0.15])
    bounds = [(np.log(20), np.log(2000)), (np.log(1), np.log(200)), (np.log(1e-3), np.log(0.5)),
              (np.log(0.05), np.log(5)), (np.log(1e-3), np.log(0.5)), (np.log(1e-3), np.log(0.5)),
              (np.log(0.01), np.log(5)), (np.log(0.01), np.log(2))]
    f = lambda th: -total_lml(hyp_from(th, BASE_HYP), data)
    res = minimize(f, th0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=200))
    # identifiability diagnostic: profile likelihood over the gains on a grid (other params at optimum)
    grid = []
    for K in (75, 150, 300):
        for D in (7.5, 15, 30):
            th = res.x.copy(); th[0], th[1] = np.log(K), np.log(D)
            grid.append(dict(K=K, D=D, lml=-f(th)))
    return res, grid


def stage2(h, data, sig_w0=1.0):
    gp = StructuredGP(h)
    feat = RBFFeatures(h["feat_centers"], h["feat_width"])
    nw = feat.n
    out_mu, out_S, ev_log = [], [], []
    for i in range(2):
        Lam = np.eye(nw) / sig_w0 ** 2
        eta = np.zeros(nw)
        for ev in data:
            t, q, qd, y, _ = sub(ev)
            kern = IntentKernel(h["sig_a"], h["ell_a"], h["sig_c"], h["sig_l"], h["T_l"], t0=ev["t_on"])
            k, d1, d2, d12 = kern.blocks(t, t)
            Ki, Di = h["K"][i], h["D"][i]
            C = (Ki * Ki * k + Ki * Di * (d1 + d2) + Di * Di * d12 + ou(t, t, h["sig_p"], h["tau_p"])
                 + h["sig_bf"] ** 2 + h["sig_m"] ** 2 * np.eye(len(t)))
            base, hco, chi = gp.obs_mean_terms(t, q, qd, ev["t_on"], ev["q_on"])
            m = base[:, i] + ev["s"] * hco[:, i]
            Phi = feat(chi[:, i])
            cf = cho_factor(C, lower=True)
            Lam += Phi.T @ cho_solve(cf, Phi)
            eta += Phi.T @ cho_solve(cf, y[:, i] - m)
        S = np.linalg.inv(Lam)
        out_mu.append(S @ eta)
        out_S.append(0.5 * (S + S.T))
    return np.array(out_mu), np.array(out_S)


def fit_naive(data, s_known=True):
    def lml(th):
        hh = dict(sig_f=np.exp(th[0]), ells=np.exp(th[1:4]), sig_n=np.exp(th[4]))
        gp = NaiveGP(hh)
        tot = 0.0
        for ev in data:
            t, q, qd, y, _ = sub(ev)
            X = NaiveGP.inputs(t, q, qd, ev["t_on"], ev["q_on"], ev["s"], HE)
            _, l = gp.predict(X, y, [x[:1] for x in X], return_lml=True)
            tot += l
        return -tot
    th0 = np.log([5.0, 0.5, 0.03, 0.05, 0.5])
    res = minimize(lml, th0, method="L-BFGS-B", bounds=[(-3, 4), (-4, 3), (-7, 1), (-7, 1), (np.log(0.05), 2)])
    return dict(sig_f=float(np.exp(res.x[0])), ells=np.exp(res.x[1:4]).tolist(), sig_n=float(np.exp(res.x[4])),
                lml=float(-res.fun), success=bool(res.success))


def fit_gm(data):
    def lml(th):
        hh = dict(sig_mu=np.exp(th[0]), sig_f=np.exp(th[1]), tau_f=np.exp(th[2]), sig_m=SIG_M)
        gm = GaussMarkov(hh)
        tot = 0.0
        for ev in data:
            t, q, qd, y, _ = sub(ev)
            _, l = gm.predict(t, y, t[-1:], return_lml=True)
            tot += l
        return -tot
    res = minimize(lml, np.log([3.0, 3.0, 0.5]), method="L-BFGS-B", bounds=[(-4, 3), (-4, 3), (-4, 2)])
    return dict(sig_mu=float(np.exp(res.x[0])), sig_f=float(np.exp(res.x[1])), tau_f=float(np.exp(res.x[2])),
                sig_m=SIG_M, lml=float(-res.fun), success=bool(res.success))


def main():
    t0 = time.time()
    g0, tr = load("train_g0"), load("train_res")
    r1, grid = stage1(g0)
    h = hyp_from(r1.x, BASE_HYP)
    lml_g0 = total_lml(h, g0)
    mu_w, S_w = stage2(h, tr)
    h_res = dict(h, mu_w=mu_w.tolist(), Sig_w=S_w.tolist())
    # evidence for the residual on the training residual split: off vs learned (diagnostic)
    lml_tr_off = total_lml(h, tr)
    naive = fit_naive(tr)
    gm = fit_gm(tr)
    out = dict(structured=h_res, structured_no_residual=h, stage1=dict(theta=r1.x.tolist(), success=bool(r1.success),
               nfev=int(r1.nfev), lml_train_g0=lml_g0, gain_profile=grid), lml_train_res_without_residual=lml_tr_off,
               naive=naive, gauss_markov=gm, train_splits=["train_g0", "train_res"], seconds=time.time() - t0,
               note="Frozen before held-out evaluation. Gains are population values fitted on train_g0; truth gains vary per episode.")
    d = ROOT / "outputs" / "models"; d.mkdir(parents=True, exist_ok=True)
    (d / "gp_frozen.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("structured", "structured_no_residual")}, indent=1))
    print("K,D,sig_a,ell_a,sig_c,sig_l,sig_p,tau_p =", np.round(np.exp(r1.x), 4).tolist())
    print("mu_w x:", np.round(mu_w[0], 3).tolist(), " sd:", np.round(np.sqrt(np.diag(S_w[0])), 3).tolist())


if __name__ == "__main__":
    main()
