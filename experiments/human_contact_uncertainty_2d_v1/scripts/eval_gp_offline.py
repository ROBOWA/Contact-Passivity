"""Offline (conditional) evaluation of human-force predictors on held-out splits.

At decision times inside each held-out contact event, every predictor conditions on the measured
window and predicts the latent (noise-free) human force over the next 0.3 s GIVEN the robot
trajectory (true future, or a counterfactual candidate). Metrics: RMSE, NLPD, interval coverage,
integrated-work calibration (MC over the joint (f, s) posterior; full vs independent-step
covariance), intention inference, and the response of the predicted force to candidate motion
compared with the true counterfactual response. Writes outputs/offline/*.json|npz.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params
from hcu2d.gp import StructuredGP, NaiveGP, GaussMarkov, edge_terms
from hcu2d.model import point_vel, r_human

cfg = params.default()
HE = cfg["body"]["h_e"]
H = 15
DT = 0.02
LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
WIN = 1.6
M = json.loads((ROOT / "outputs" / "models" / "gp_frozen.json").read_text())


def loc_moments(sbar, n, sb=cfg["sensors"]["sigma_bs"], se=cfg["sensors"]["sigma_es"]):
    v0 = 0.1 ** 2
    if n == 0:
        return 0.0, v0
    vo = sb ** 2 + se ** 2 / n
    p = 1 / v0 + 1 / vo
    return (sbar / vo) / p, 1 / p


def true_response(ev, kk, dq, dqd, s):
    """True counterfactual force change for robot-state perturbation (dq,dqd) at ticks kk."""
    q, qd = ev["q"][kk], ev["qd"][kk]
    rH = r_human(np.full(len(kk), s), HE)
    p0 = np.stack([q[:, 0] + np.cos(q[:, 2]) * s - np.sin(q[:, 2]) * HE, q[:, 1] + np.sin(q[:, 2]) * s + np.cos(q[:, 2]) * HE], -1)
    q1, qd1 = q + dq, qd + dqd
    p1 = np.stack([q1[:, 0] + np.cos(q1[:, 2]) * s - np.sin(q1[:, 2]) * HE, q1[:, 1] + np.sin(q1[:, 2]) * s + np.cos(q1[:, 2]) * HE], -1)
    v0, v1 = point_vel(q, qd, rH), point_vel(q1, qd1, rH)
    df = -ev["K"] * (p1 - p0) - ev["D"] * (v1 - v0)
    if ev["residual"]:
        _, c0, _, _ = edge_terms(q, qd, HE)
        _, c1, _, _ = edge_terms(q1, qd1, HE)
        df += -cfg["human"]["c_g"] * (np.tanh(c1 / cfg["human"]["v_g"]) - np.tanh(c0 / cfg["human"]["v_g"]))
    return df


def evaluate(split, rng):
    data = list(np.load(ROOT / "outputs" / "gp_data" / f"{split}.npy", allow_pickle=True))
    S = StructuredGP(M["structured"])
    S0 = StructuredGP(M["structured_no_residual"])
    NG, GM = NaiveGP(M["naive"]), GaussMarkov(M["gauss_markov"])
    rec = []
    for ev in data:
        t_all = ev["t"]
        dur = t_all[-1] - t_all[0]
        for td in np.arange(0.4, dur - 0.32, 0.2):
            kd = int(round(td / 1e-3))                     # tick index within event
            ko = np.arange(max(0, kd - int(WIN / 1e-3)), kd + 1)
            ko = ko[(ko % 40) == 0]                        # 25 Hz grid aligned to grasp
            if len(ko) < 3:
                continue
            kf = kd + np.arange(1, H + 1) * 20              # future ticks (20 ms stages)
            if kf[-1] >= len(t_all):
                continue
            n_loc = kd // 40 + 1
            sbar = float(np.mean(ev["loc"][:n_loc]))
            t_o, q_o, qd_o, y_o = t_all[ko], ev["q"][ko], ev["qd"][ko], ev["y"][ko]
            t_f, q_f, qd_f, f_f = t_all[kf], ev["q"][kf], ev["qd"][kf], ev["f"][kf]
            r = dict(split=split, seed=int(ev["seed"]), family=ev["family"], td=float(td), s=float(ev["s"]))
            s_hat, s_var = loc_moments(sbar, n_loc)
            r["s_hat_sensor"], r["s_sd_sensor"] = s_hat, float(np.sqrt(s_var))
            # perturbations for the counterfactual test
            ramp = (np.arange(1, H + 1) / H)[:, None]
            pert = {"trans_x": (ramp * np.array([0.02, 0, 0]), np.tile([0.02 / 0.3, 0, 0], (H, 1))),
                    "rot": (ramp * np.array([0, 0, 0.05]), np.tile([0, 0, 0.05 / 0.3], (H, 1)))}
            for name, gp in [("S", S), ("S0", S0)]:
                post = gp.condition(t_o, q_o, qd_o, y_o, t_all[0], ev["q_on"], sbar, n_loc, t_f, ep_times=t_f)
                mean, cov, A, b = gp.force_map(post, t_f, q_f, qd_f, t_all[0], ev["q_on"])
                r[name] = score(mean, cov, f_f)
                # joint (f, s) posterior -> integrated work over the horizon (true future robot motion)
                idx = post["idx"]
                is_ = idx["s"].start
                Jfull = np.vstack([A, np.eye(len(post["mu"]))[is_]])
                mu_fs = np.concatenate([mean.T.ravel(), [post["mu"][is_]]])
                cov_fs = Jfull @ post["Sig"] @ Jfull.T
                r[name]["W"] = work_stats(mu_fs, cov_fs, q_f, qd_f, ev, rng, H, f_f)
                cov_ind = cov.copy()
                for i in range(2):                          # independent-step approximation
                    for j in range(2):
                        blk = cov_ind[i * H:(i + 1) * H, j * H:(j + 1) * H]
                        cov_ind[i * H:(i + 1) * H, j * H:(j + 1) * H] = np.diag(np.diag(blk))
                cov_fs_ind = cov_fs.copy()
                cov_fs_ind[:2 * H, :2 * H] = cov_ind
                r[name]["W_indep"] = work_stats(mu_fs, cov_fs_ind, q_f, qd_f, ev, rng, H, f_f)
                # intention inference at the decision time (world-frame a = c_on + s u_on + ã)
                if name == "S":
                    post_now = gp.condition(t_o, q_o, qd_o, y_o, t_all[0], ev["q_on"], sbar, n_loc, [t_all[kd]],
                                            ep_times=[t_all[kd]])
                    c_on, _, u_on, _ = edge_terms(ev["q_on"][None], np.zeros((1, 3)), HE)
                    ia = post_now["idx"]
                    a_hat, a_sd = [], []
                    for i, a_ in enumerate(("ax", "ay")):
                        g = np.zeros(len(post_now["mu"]))
                        g[ia[a_].start] = 1.0
                        g[ia["s"].start] = u_on[0, i]
                        a_hat.append(c_on[0, i] + g @ post_now["mu"])
                        a_sd.append(np.sqrt(g @ post_now["Sig"] @ g))
                    a_true = ev["a"][kd, :2]
                    r["intent"] = dict(err=(np.array(a_hat) - a_true).tolist(), sd=a_sd,
                                       prior_sd=float(np.sqrt(M["structured"]["sig_a"] ** 2 + 0.1 ** 2)),
                                       s_post=float(post_now["mu"][ia["s"].start]),
                                       s_post_sd=float(np.sqrt(post_now["Sig"][ia["s"].start, ia["s"].start])),
                                       corr_w_a=corr_block(post_now, "wx", "ax"))
                # response to candidate motion
                resp = {}
                for pn, (dq, dqd) in pert.items():
                    m1, _, _, _ = gp.force_map(post, t_f, q_f + dq, qd_f + dqd, t_all[0], ev["q_on"])
                    resp[pn] = (m1 - mean)[-1].tolist()
                r[name]["resp"] = resp
            # naive GP (ŝ from the location sensor only)
            Xo = NaiveGP.inputs(t_o, q_o, qd_o, t_all[0], ev["q_on"], s_hat, HE)
            Xf = NaiveGP.inputs(t_f, q_f, qd_f, t_all[0], ev["q_on"], s_hat, HE)
            outN = NG.predict(Xo, y_o, Xf)
            meanN = np.stack([outN[0][0], outN[1][0]], -1)
            covN = np.zeros((2 * H, 2 * H))
            covN[:H, :H], covN[H:, H:] = outN[0][1], outN[1][1]
            r["N"] = score(meanN, covN, f_f)
            r["N"]["W"] = work_stats(np.concatenate([meanN.T.ravel(), [s_hat]]), blockdiag(covN, s_var), q_f, qd_f, ev, rng, H, f_f)
            resp = {}
            for pn, (dq, dqd) in pert.items():
                Xc = NaiveGP.inputs(t_f, q_f + dq, qd_f + dqd, t_all[0], ev["q_on"], s_hat, HE)
                o2 = NG.predict(Xo, y_o, Xc)
                resp[pn] = (np.stack([o2[0][0], o2[1][0]], -1) - meanN)[-1].tolist()
            r["N"]["resp"] = resp
            # Gauss–Markov (motion-blind)
            meanG, covG1 = GM.predict(t_o, y_o, t_f)
            covG = np.zeros((2 * H, 2 * H))
            covG[:H, :H], covG[H:, H:] = covG1, covG1
            r["GM"] = score(meanG, covG, f_f)
            r["GM"]["W"] = work_stats(np.concatenate([meanG.T.ravel(), [s_hat]]), blockdiag(covG, s_var), q_f, qd_f, ev, rng, H, f_f)
            r["GM"]["resp"] = {pn: [0.0, 0.0] for pn in pert}
            # hold-last (deterministic, C1)
            hold = np.tile(y_o[-3:].mean(0), (H, 1))
            r["HOLD"] = dict(rmse=float(np.sqrt(np.mean((hold - f_f) ** 2))),
                             rmse_h=np.sqrt(np.mean((hold - f_f) ** 2, 1)).tolist())
            r["truth_resp"] = {pn: true_response(ev, kf, dq, dqd, ev["s"])[-1].tolist() for pn, (dq, dqd) in pert.items()}
            rec.append(r)
    return rec


def blockdiag(A, b):
    n = A.shape[0]
    out = np.zeros((n + 1, n + 1))
    out[:n, :n] = A
    out[n, n] = b
    return out


def corr_block(post, a, b):
    i, j = post["idx"][a], post["idx"][b]
    S = post["Sig"]
    sd_i, sd_j = np.sqrt(np.diag(S)[i]), np.sqrt(np.diag(S)[j])
    c = S[i, j] / np.outer(sd_i, sd_j)
    return float(np.max(np.abs(c)))


def score(mean, cov, truth):
    Hh = mean.shape[0]
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12)).reshape(2, Hh).T
    z = (truth - mean) / sd
    nlpd = float(np.mean(0.5 * z ** 2 + np.log(sd) + 0.5 * np.log(2 * np.pi)))
    cov_ = {str(l): float(np.mean(np.abs(z) <= norm.ppf(0.5 + l / 2))) for l in LEVELS}
    return dict(rmse=float(np.sqrt(np.mean((mean - truth) ** 2))), rmse_h=np.sqrt(np.mean((mean - truth) ** 2, 1)).tolist(),
                nlpd=nlpd, cover=cov_, mean_sd=float(np.mean(sd)))


def work_stats(mu_fs, cov_fs, q_f, qd_f, ev, rng, Hh, f_true, n=3000):
    """Robot-to-human work over the horizon, W = Σ Δt (−f·v_H(s)); MC over the joint (f, s)."""
    L = np.linalg.cholesky(cov_fs + 1e-10 * np.eye(len(mu_fs)))
    smp = mu_fs + rng.standard_normal((n, len(mu_fs))) @ L.T
    f = np.stack([smp[:, :Hh], smp[:, Hh:2 * Hh]], -1)
    s = smp[:, -1]
    c, sn = np.cos(q_f[:, 2]), np.sin(q_f[:, 2])
    rx = c[None] * s[:, None] - sn[None] * HE
    ry = sn[None] * s[:, None] + c[None] * HE
    vx = qd_f[None, :, 0] - qd_f[None, :, 2] * ry
    vy = qd_f[None, :, 1] + qd_f[None, :, 2] * rx
    W = -DT * np.sum(f[..., 0] * vx + f[..., 1] * vy, 1)
    rH = r_human(np.full(Hh, ev["s"]), HE)
    vt = point_vel(q_f, qd_f, rH)
    Wt = -DT * np.sum(f_true * vt)
    m, sd = float(W.mean()), float(W.std())
    qs = {str(l): [float(np.quantile(W, 0.5 - l / 2)), float(np.quantile(W, 0.5 + l / 2))] for l in LEVELS}
    return dict(mean=m, sd=sd, truth=float(Wt), z=(float(Wt) - m) / max(sd, 1e-12),
                cover={str(l): bool(qs[str(l)][0] <= Wt <= qs[str(l)][1]) for l in LEVELS},
                q99=float(np.quantile(W, 0.99)), q01=float(np.quantile(W, 0.01)))


def main():
    out = ROOT / "outputs" / "offline"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    allrec = {}
    t0 = time.time()
    for split in ["val", "test_g0", "test_id", "test_heldout", "test_mismatch"]:
        allrec[split] = evaluate(split, rng)
        print(split, len(allrec[split]), f"{time.time() - t0:.1f}s", flush=True)
    (out / "offline_records.json").write_text(json.dumps(allrec))


if __name__ == "__main__":
    main()
