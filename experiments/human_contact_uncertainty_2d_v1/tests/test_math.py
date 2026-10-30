"""Mathematical checks (formulation §2-§10). Run: python -m pytest tests -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params, model as M, truth, servo, belief, planner, scenarios, episode, auditor, controllers
from hcu2d.gp import StructuredGP, edge_terms
from hcu2d.kernels import IntentKernel, ou

CFG = params.default()
P = M.Known(CFG)


# ---------------------------------------------------------------------------------------------------
# §2 contact: pointwise identity p_T = Ṡ_T + D_T (compression, unloading, separation)
def test_contact_power_identity_pointwise():
    rng = np.random.default_rng(0)
    n = 4000
    q = np.c_[rng.uniform(-.3, .3, n), rng.uniform(0.095, 0.103, n), rng.uniform(-.1, .1, n)]
    qd = rng.normal(0, 0.1, (n, 3))
    con = M.contact(q, qd, P)
    vt = con["vt"]
    pT = -(con["ft"] * vt[:, 0] + con["fn"] * vt[:, 1])
    d = con["d"]
    STdot = P.kn * d * np.where(d > 0, -vt[:, 1], 0.0)        # dS_T/dt = k δ δ̇ with δ̇ = −v_n
    assert np.all(con["DT"] >= -1e-12)
    assert np.all(con["fn"] >= 0.0)
    np.testing.assert_allclose(pT, STdot + con["DT"], atol=1e-10)
    sep = d == 0
    assert sep.any() and (~sep).any()
    assert np.all(con["fn"][sep] == 0) and np.all(con["DT"][sep] == 0) and np.all(con["ST"][sep] == 0)
    unl = (d > 0) & (vt[:, 1] > 0)                             # unloading: no damping force
    np.testing.assert_allclose(con["fn"][unl], P.kn * d[unl], atol=1e-12)


def _scripted_contact_run(h, T=1.2):
    """Truth plant, no human, scripted input causing compression, unloading, lift-off and re-contact."""
    pl = truth.TruthPlant(CFG, [], np.array([-0.37, 0.101, 0.0]))
    n = int(round(T / h))
    for k in range(n):
        t = k * h
        u = np.array([3.0 * math.sin(4 * t), -12.0 * (1 + math.sin(9 * t)) + 4.0, 0.4 * math.sin(7 * t)])
        pl.begin_tick()
        pl.step(u, h)
    z = pl.z
    S, SR, ST = pl.storage(z)
    S0 = pl.storage(np.r_[np.array([-0.37, 0.101, 0.0, 0, 0, 0]), np.zeros(6)])[0]
    return (S - S0) - z[6] + z[7] + z[8] + z[9], z


def test_composite_energy_balance_converges():
    """ΔS − W_a + W_H + ∫D_R + ∫D_T → 0 under step refinement (non-monotone near contact kinks)."""
    runs = [_scripted_contact_run(h) for h in (2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4)]
    errs = [abs(r) for r, _ in runs]
    Wa = runs[-1][1][6]
    assert errs[-1] < 1e-6
    assert errs[-1] < 1e-3 * errs[0]
    assert max(errs[1:]) < 2e-5 * Wa


# ---------------------------------------------------------------------------------------------------
# §1 location Jacobian and the effect of s on generalised human torque
def test_location_jacobian_finite_difference():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(0, 0.3, 3)
        s = rng.uniform(-0.18, 0.18)
        r = M.r_human(s, P.he)
        J = M.jac(q[None], r)[0]
        Jfd = np.zeros((2, 3))
        for i in range(3):
            dq = np.zeros(3)
            dq[i] = 1e-7
            Jfd[:, i] = (M.point(q[None] + dq, r)[0] - M.point(q[None] - dq, r)[0]) / 2e-7
        np.testing.assert_allclose(J, Jfd, atol=1e-7)
        f = rng.normal(0, 5, 2)
        tau = lambda ss: M.gen_force(q[None], M.r_human(ss, P.he), f[None])[0]
        dtau = (tau(s + 1e-6) - tau(s - 1e-6)) / 2e-6
        c, sn = math.cos(q[2]), math.sin(q[2])
        np.testing.assert_allclose(dtau, [0, 0, -sn * f[0] + c * f[1]], atol=1e-6)   # (R e2)ᵀ f


def test_location_changes_torque_and_future_work():
    """Same nominal force at different moment arms: tangential (along edge) vs normal vs oblique."""
    q = np.array([0.0, 0.1, 0.0])
    s_vals = np.array([-0.15, 0.0, 0.15])
    for f, expect_dep in [((5.0, 0.0), False), ((0.0, -5.0), True), ((-3.5, -3.5), True)]:
        tau = M.gen_force(np.repeat(q[None], 3, 0), M.r_human(s_vals, P.he), np.tile(f, (3, 1)))
        spread = tau[:, 2].max() - tau[:, 2].min()
        assert (spread > 0.5) == expect_dep
    # future work differs with s once the body rotates: same force, θ̇ ≠ 0
    qd = np.array([0.05, 0.0, 0.3])
    v = M.point_vel(np.repeat(q[None], 3, 0), np.repeat(qd[None], 3, 0), M.r_human(s_vals, P.he))
    pH = -(v @ np.array([0.0, -5.0]))
    assert pH.max() - pH.min() > 0.2


# ---------------------------------------------------------------------------------------------------
# §7 kernels: derivative blocks, units, PSD, conditioning signs
def test_kernel_derivatives_and_units():
    k = IntentKernel(0.03, 0.6, 0.02, 0.03, 1.0, t0=0.2)
    t1, t2 = np.linspace(0, 2, 7), np.linspace(0.1, 2.3, 5)
    K, d1, d2, d12 = k.blocks(t1, t2)
    e = 1e-6
    np.testing.assert_allclose(d1, (k.blocks(t1 + e, t2)[0] - k.blocks(t1 - e, t2)[0]) / (2 * e), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(d2, (k.blocks(t1, t2 + e)[0] - k.blocks(t1, t2 - e)[0]) / (2 * e), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(d12, (k.blocks(t1, t2 + e)[1] - k.blocks(t1, t2 - e)[1]) / (2 * e), rtol=1e-5, atol=1e-8)
    # unit change s -> ms: derivative blocks scale by 1e-3 and 1e-6 (per-time units)
    km = IntentKernel(0.03, 600.0, 0.02, 0.03, 1000.0, t0=200.0)
    Km, d1m, d2m, d12m = km.blocks(t1 * 1e3, t2 * 1e3)
    np.testing.assert_allclose(Km, K, rtol=1e-12)
    np.testing.assert_allclose(d1m, d1 * 1e-3, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(d12m, d12 * 1e-6, rtol=1e-10, atol=1e-15)


def _gp_case(seed=0, n=25, N=6):
    import json
    hyp = json.loads((ROOT / "outputs" / "models" / "gp_frozen.json").read_text())["structured"] \
        if (ROOT / "outputs" / "models" / "gp_frozen.json").exists() else None
    if hyp is None:
        pytest.skip("frozen GP model not yet fitted")
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1.5, n))
    q = np.c_[0.05 * t + 0.01 * np.sin(3 * t), 0.098 + 0.001 * np.sin(5 * t), 0.05 * np.sin(2 * t)]
    qd = np.c_[0.05 + 0.03 * np.cos(3 * t), 0.005 * np.cos(5 * t), 0.1 * np.cos(2 * t)]
    y = rng.normal(0, 2, (n, 2))
    gp = StructuredGP(hyp)
    t_st = 1.5 + 0.02 * np.arange(N)
    post = gp.condition(t, q, qd, y, 0.0, q[0], 0.05, 4, t_st, ep_times=[1.5])
    return gp, post, (t, q, qd, y, t_st)


def test_joint_covariance_psd_and_posterior_reduction():
    gp, post, _ = _gp_case()
    J = np.block([[post["Kyy"], post["C"]], [post["C"].T, post["prior_Sig"]]])
    np.testing.assert_allclose(J, J.T, atol=1e-12)
    ev = np.linalg.eigvalsh(J)
    assert ev.min() > -1e-8 * ev.max()
    red = np.linalg.eigvalsh(post["prior_Sig"] - post["Sig"])
    assert red.min() > -1e-9
    assert np.all(np.diag(post["Sig"]) <= np.diag(post["prior_Sig"]) + 1e-12)


def test_conditioning_matches_independent_bruteforce():
    """Independent construction: ã on a fine grid, L_t = K + D d/dt by finite differences, all other
    latents explicit; Gaussian conditioning by brute force. Checks signs/transposes (P1 Eq. 25-26)."""
    gp, post, (t, q, qd, y, t_st) = _gp_case(n=12, N=4)
    h = gp.h
    grid = np.linspace(-0.2, 2.0, 2201)
    dg = grid[1] - grid[0]
    kern = IntentKernel(h["sig_a"], h["ell_a"], h["sig_c"], h["sig_l"], h["T_l"], t0=0.0)
    Kg = kern.blocks(grid, grid)[0]

    def interp_rows(times, deriv):
        R = np.zeros((len(times), len(grid)))
        for r, tt in enumerate(times):
            i = int(np.floor((tt - grid[0]) / dg))
            a = (tt - grid[i]) / dg
            if not deriv:
                R[r, i], R[r, i + 1] = 1 - a, a
            else:                                   # central-difference derivative, interpolated
                R[r, i + 1] += (1 - a) / (2 * dg); R[r, i - 1] -= (1 - a) / (2 * dg)
                R[r, i + 2] += a / (2 * dg);      R[r, i] -= a / (2 * dg)
        return R
    nw = gp.nw
    # latent vector: [grid_x, grid_y, s, b_s, w_x, w_y, bf(2), ε_p values at (t ∪ t_now)]
    tep = np.r_[t, 1.5]
    Kou = ou(tep, tep, h["sig_p"], h["tau_p"])
    Sw = gp.Sig_w
    blocks = [Kg, Kg, np.array([[h["sig_s0"] ** 2]]), np.array([[h["sig_bs"] ** 2]]), Sw[0], Sw[1],
              np.eye(2) * h["sig_bf"] ** 2, Kou, Kou]
    from scipy.linalg import block_diag
    Sx = block_diag(*blocks)
    G = len(grid)
    off = np.cumsum([0] + [b.shape[0] for b in blocks])
    base, hco, chi = gp.obs_mean_terms(t, q, qd, 0.0, q[0])
    n = len(t)
    A = np.zeros((2 * n + 1, Sx.shape[0]))
    mean_y = np.zeros(2 * n + 1)
    for i in range(2):
        rows = slice(i * n, (i + 1) * n)
        A[rows, off[i]:off[i] + G] = gp.K[i] * interp_rows(t, False) + gp.D[i] * interp_rows(t, True)
        A[rows, off[2]] = hco[:, i]
        A[rows, off[4 + i]:off[5 + i]] = gp.feat(chi[:, i])
        A[rows, off[6] + i] = 1.0
        A[rows, off[7 + i]:off[7 + i] + n] = np.eye(n)
        mean_y[rows] = base[:, i] + h["mu_s0"] * hco[:, i] + gp.feat(chi[:, i]) @ gp.mu_w[i]
    A[-1, off[2]] = 1.0
    A[-1, off[3]] = 1.0
    mean_y[-1] = h["mu_s0"]
    Ky = A @ Sx @ A.T + np.diag(np.r_[np.full(2 * n, h["sig_m"] ** 2), h["sig_es"] ** 2 / 4])
    Y = np.r_[y[:, 0], y[:, 1], 0.05]
    # targets: ã_x(t_st), ã̇_x(t_st), s
    T = np.zeros((2 * len(t_st) + 1, Sx.shape[0]))
    T[:len(t_st), :G] = interp_rows(t_st, False)
    T[len(t_st):2 * len(t_st), :G] = interp_rows(t_st, True)
    T[-1, off[2]] = 1.0
    mT = np.r_[np.zeros(2 * len(t_st)), h["mu_s0"]]
    Ct = A @ Sx @ T.T
    mu_bf = mT + Ct.T @ np.linalg.solve(Ky, Y - mean_y)
    Sg_bf = T @ Sx @ T.T - Ct.T @ np.linalg.solve(Ky, Ct)
    ix = post["idx"]
    sel = np.r_[np.arange(ix["ax"].start, ix["ax"].stop), np.arange(ix["adx"].start, ix["adx"].stop), ix["s"].start]
    np.testing.assert_allclose(post["mu"][sel], mu_bf, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(post["Sig"][np.ix_(sel, sel)], Sg_bf, rtol=5e-3, atol=2e-6)


def test_deterministic_zero_covariance_limit():
    """All prior variances -> 0: posterior force = deterministic impedance model, MC spread -> 0."""
    import json
    hyp = json.loads((ROOT / "outputs" / "models" / "gp_frozen.json").read_text())["structured"]
    nw = len(hyp["feat_centers"])
    hyp = dict(hyp, sig_a=1e-7, sig_c=1e-7, sig_l=1e-7, sig_p=1e-7, sig_s0=1e-7, sig_bf=1e-7,
               mu_w=np.zeros((2, nw)).tolist(), Sig_w=np.stack([np.eye(nw) * 1e-14] * 2).tolist(), mu_s0=0.1)
    gp = StructuredGP(hyp)
    t = np.linspace(0, 0.5, 10)
    q = np.c_[0.02 * t, 0.098 + 0 * t, 0.04 * t]
    qd = np.c_[0.02 + 0 * t, 0 * t, 0.04 + 0 * t]
    y = np.zeros((10, 2))
    t_st = 0.5 + 0.02 * np.arange(5)
    q_f = np.c_[0.02 * t_st, 0.098 + 0 * t_st, 0.04 * t_st]
    qd_f = np.c_[0.02 + 0 * t_st, 0 * t_st, 0.04 + 0 * t_st]
    post = gp.condition(t, q, qd, y, 0.0, q[0], 0.1, 5, t_st, ep_times=t_st)
    mean, cov, _, _ = gp.force_map(post, t_st, q_f, qd_f, 0.0, q[0])
    rH = M.r_human(0.1, P.he)
    p_on = M.point(q[0:1], rH)[0]
    det = np.array(hyp["K"]) * (p_on - M.point(q_f, rH)) - np.array(hyp["D"]) * M.point_vel(q_f, qd_f, rH)
    np.testing.assert_allclose(mean, det, atol=1e-5)
    assert np.sqrt(np.max(np.diag(cov))) < 1e-4


def test_temporal_work_covariance():
    """Var(Σ W_i) = Σ_ij Cov(W_i,W_j) ≠ Σ_i Var(W_i); MC samples of the joint posterior agree."""
    gp, post, (t, q, qd, y, t_st) = _gp_case(N=15)
    q_f = np.c_[q[-1, 0] + 0.05 * (t_st - t[-1]), np.full(len(t_st), q[-1, 1]), np.full(len(t_st), q[-1, 2])]
    qd_f = np.tile([0.05, 0.0, 0.0], (len(t_st), 1))
    post = gp.condition(t, q, qd, y, 0.0, q[0], 0.05, 4, t_st, ep_times=t_st)
    mean, cov, A, b = gp.force_map(post, t_st, q_f, qd_f, 0.0, q[0])
    H = len(t_st)
    v = M.point_vel(q_f, qd_f, M.r_human(post["mu"][post["idx"]["s"].start], P.he))
    w = -0.02 * np.r_[v[:, 0], v[:, 1]]                  # W = wᵀ f (location fixed at mean)
    var_full = w @ cov @ w
    var_ind = np.sum(w ** 2 * np.diag(cov))
    rng = np.random.default_rng(3)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(2 * H))
    W = (rng.standard_normal((40000, 2 * H)) @ L.T) @ w
    assert abs(W.var() / var_full - 1) < 0.03
    assert var_full > 1.5 * var_ind                       # positive temporal correlation matters


# ---------------------------------------------------------------------------------------------------
# §10 risk objective sign conventions
def test_entropic_risk_sign_and_limits():
    rng = np.random.default_rng(4)
    J = rng.gamma(2.0, 0.3, (3, 20000))
    m = J.mean(-1)
    r0 = planner.entropic(J, 0.0)[0]
    np.testing.assert_allclose(r0, m)
    rs = planner.entropic(J, 1e-4)[0]
    np.testing.assert_allclose(rs, m + 0.5e-4 * J.var(-1), rtol=1e-5)
    assert np.all(planner.entropic(J, 1.0)[0] > m) and np.all(planner.entropic(J, -1.0)[0] < m)
    theta = -2.0                                           # P3/P4: θ<0 risk-averse
    gamma = -2.0 / theta * np.log(np.mean(np.exp(-theta * J / 2), -1))
    np.testing.assert_allclose(planner.entropic(J, -theta / 2)[0], gamma, rtol=1e-10)
    rho, se, ess = planner.entropic(J, 3.0)
    assert np.all(ess > 1) and np.all(ess <= J.shape[1]) and np.all(se > 0)


# ---------------------------------------------------------------------------------------------------
# §5 backup lemma and reference filter (numerical validation in the stated simplified setting)
def _adversarial_human(rng, t_on=0.0):
    prof = truth.Profile(vel=[(0.0, rng.uniform(-0.3, 0.3), 0.1), (0.4, rng.uniform(-0.6, 0.6), 0.1)],
                         sin=(rng.uniform(0, 0.05), rng.uniform(0.5, 4), 0.05))
    prof_y = truth.Profile(steps=[(0.05, rng.uniform(-0.05, 0.05), 0.2)], sin=(rng.uniform(0, 0.03), rng.uniform(0.5, 4), 0.05))
    return truth.HumanEvent(t_on, 10.0, rng.uniform(-0.18, 0.18), rng.uniform(50, 600, 2), rng.uniform(5, 60, 2),
                            prof, prof_y, True, rng, CFG["human"])


def test_zero_input_backup_lemma():
    rng = np.random.default_rng(5)
    cfg = params.merge(CFG, {"human": {"F_rel": 1e9}})
    for trial in range(6):
        ev = _adversarial_human(rng)
        q0 = np.array([0.0, 0.0985 + rng.uniform(-0.002, 0.001), rng.uniform(-0.05, 0.05)])
        pl = truth.TruthPlant(cfg, [ev], q0, qd0=rng.normal(0, 0.1, 3))
        S0 = pl.storage()[0]
        worst = -np.inf
        for k in range(1500):
            pl.begin_tick()
            pl.step(np.zeros(3), 1e-3)
            worst = max(worst, pl.z[7] - S0)                 # ∫p_H − S(t0) for every prefix
        assert worst <= 1e-7, (trial, worst)


def test_reference_filter_with_exact_accounts():
    """ORACLE filter (true BH, exact Ec) keeps h_H, BH, Ec non-negative (sampled-data numerical check)."""
    for seed in range(3):
        sc = scenarios.make("block_release", seed, CFG)
        out = episode.run(CFG, sc, controllers.OracleReference())
        a, _ = auditor.audit(out["log"], CFG, sc["accounts"])
        assert a["hH_true_min"] > -1e-5 and a["BH_true_min"] > -1e-5 and a["Ec_min"] > -1e-5
        assert a["credit_excess_vs_plant"] <= 1e-9


def test_tick_energy_projection_is_feasible_and_optimal():
    from scipy.optimize import minimize
    rng = np.random.default_rng(6)
    umax = P.umax
    W = 1 / np.array([20.0, 20.0, 2.0]) ** 2
    for _ in range(30):
        u0, qd, te = rng.normal(0, 25, 3), rng.normal(0, 0.2, 3), rng.normal(0, 10, 3)
        b = rng.uniform(-0.5, 1.0)
        u, st, g = servo.project_tick_energy(u0, qd, te, P.Mdiag, 1e-3, b, umax, W)
        if st == "infeasible_best_effort":
            continue
        assert g <= b + 1e-9 and np.all(np.abs(u) <= umax + 1e-12)
        con = {"type": "ineq", "fun": lambda x: b - (x @ (qd + 0.5e-3 * te / P.Mdiag) + 0.5 * np.sum(1e-3 / P.Mdiag * x * x))}
        res = minimize(lambda x: np.sum(W * (x - np.clip(u0, -umax, umax)) ** 2), np.zeros(3), constraints=[con],
                       bounds=list(zip(-umax, umax)), method="SLSQP", options=dict(ftol=1e-14, maxiter=500))
        assert np.sum(W * (u - np.clip(u0, -umax, umax)) ** 2) <= res.fun + 1e-6


# ---------------------------------------------------------------------------------------------------
# §4 ledgers: exact actuator work and the past-work error model
def test_booked_actuator_work_is_exact_and_bh_error_model():
    sc = scenarios.make("moment_arm", 1, CFG)
    out = episode.run(CFG, sc, planner.make("C1", CFG))
    L = out["log"]
    a, ser = auditor.audit(L, CFG, sc["accounts"])
    assert a["Wa_plant_vs_exact_max"] < 1e-9               # ZOH: ∫u·q̇ = uᵀΔq
    assert a["credit_valid"]
    led = out["ledger"]
    ev = sc["events"][0]
    assert ev.t_off is not None and len(led.history) == 1
    hst = led.history[0]
    e_true = a["BH_err_final"]                           # BH_true − BH_book (independent auditor)
    e_det = hst["G"] * ev.s - hst["Gam"]                  # model with the TRUE s (b_f = 0 here)
    sd_n = math.sqrt(hst["var_n"])
    assert abs(e_true - e_det) < 5 * sd_n + 0.1 * abs(e_det) + 2e-4, (e_true, e_det, sd_n)
