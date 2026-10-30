"""Build report tables (outputs/report_tables.md), PDFs and manifest.json.

  python scripts/build_docs.py [--tables] [--pdf] [--manifest]   (default: all)
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from hcu2d.mdpdf import md_to_pdf
import analyze

RUNS = ROOT / "outputs" / "runs"
ORDER = ["C0", "C1", "C2", "C3", "C4"]


def ci(d):
    """'k (Wilson 95% lo–hi %)' for a rate record (n is reported in its own column)."""
    lo, hi = d["wilson95"]
    return f"{d['k']} ({100*lo:.0f}–{100*hi:.0f}%)"


def fmt(x, f="{:.3f}"):
    return "–" if x is None else f.format(x)


def summary(tag):
    p = RUNS / tag / "summary.json"
    if not (RUNS / tag / "metrics.jsonl").exists():
        return None
    rows, err = analyze.load(tag)
    S = analyze.summarize(rows)
    pairs = {}
    ctrls = sorted({r["controller"] for r in rows})
    if "C3" in ctrls:
        for o in ctrls:
            if o != "C3":
                pairs[f"{o}-C3"] = {k: analyze.paired(rows, o, "C3", k) for k in
                                   ["WH_pos_final", "WH_net_final", "coverage", "force_rmse", "tip_x_final", "BH_true_min", "hH_true_min", "du_rms"]}
    res = dict(tag=tag, n_rows=len(rows), n_errors=len(err), summary=S, paired=pairs)
    p.write_text(json.dumps(res, indent=1))
    return res


def main_table(S, scen_order, ctrls=ORDER):
    L = ["| scenario | ctrl | n | completed | true h_H viol. (Wilson 95%) | true BH viol. | Ec viol. | domain exit | W_H⁺ [J] mean (q95) | W_H net [J] | coverage | f_n RMSE [N] | peak f_H [N] | release surge [m/s] | fallbacks/ep | filter active |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc in scen_order:
        for ck in ctrls:
            d = S.get(f"{sc}|{ck}")
            if d is None:
                continue
            L.append(f"| {sc} | {ck} | {d['n']} | {d['completed']['k']}/{d['n']} | {ci(d['violation_hH'])} | {ci(d['violation_BH'])} | "
                     f"{d['violation_Ec']['k']} | {d['domain_violation']['k']} | {fmt(d['WH_pos_final']['mean'])} ({fmt(d['WH_pos_final']['q95'])}) | "
                     f"{fmt(d['WH_net_final']['mean'])} | {fmt(d['coverage']['mean'], '{:.2f}')} | {fmt(d['force_rmse']['mean'], '{:.2f}')} | "
                     f"{fmt(d['fH_peak']['mean'], '{:.1f}')} | {fmt(d['surge_v']['mean'], '{:.3f}')} | {fmt(d['n_fallback']['mean'], '{:.2f}')} | "
                     f"{fmt(d['filt_frac']['mean'], '{:.3f}')} |")
    return "\n".join(L)


def pooled(S, scen, ck, key):
    k = n = 0
    for sc in scen:
        d = S.get(f"{sc}|{ck}")
        if d:
            k += d[key]["k"]
            n += d["n"]
    lo, hi = analyze.wilson(k, n)
    return k, n, lo, hi, analyze.cp_upper(k, n)


def tables():
    out = []
    main = summary("main")
    if main:
        S = main["summary"]
        scen = ["no_human", "block_release", "assist_reverse", "normal_oblique", "moment_arm", "ood_intent", "sensor_bias", "low_BH"]
        out.append("## T1. Main comparison (held-out seeds 100-199, paired across controllers)\n")
        out.append(main_table(S, scen))
        idist = ["no_human", "block_release", "assist_reverse", "normal_oblique", "moment_arm"]
        out.append("\n## T2. Pooled episode-level violation rates (any of true Ec/BH/h_H)\n")
        out.append("| controller | in-distribution scenarios (5) | OOD intent | sensor bias | low BH |\n|---|---|---|---|---|")
        for ck in ORDER:
            cells = []
            for grp in [idist, ["ood_intent"], ["sensor_bias"], ["low_BH"]]:
                k, n, lo, hi, up = pooled(S, grp, ck, "violation_any")
                cells.append(f"{k}/{n} (Wilson {100*lo:.1f}–{100*hi:.1f}%; one-sided 95% upper {100*up:.2f}%)")
            out.append(f"| {ck} | " + " | ".join(cells) + " |")
        out.append("\n## T3. Paired differences vs C3 (mean ± s.e. over common seeds)\n")
        out.append("| scenario | comparison | Δ W_H⁺ [J] | Δ coverage | Δ f_n RMSE [N] | Δ min h_H (true) [J] | Δ du_rms |\n|---|---|---|---|---|---|---|")
        for sc in scen:
            for o in ["C0-C3", "C1-C3", "C2-C3", "C4-C3"]:
                p = main["paired"].get(o, {})
                g = lambda k: p.get(k, {}).get(sc)
                if g("WH_pos_final") is None:
                    continue
                f = lambda d: f"{d['mean']:+.4f} ± {d['se']:.4f}" if d and d.get("se") is not None else "–"
                out.append(f"| {sc} | {o} | {f(g('WH_pos_final'))} | {f(g('coverage'))} | {f(g('force_rmse'))} | {f(g('hH_true_min'))} | {f(g('du_rms'))} |")
        out.append("\n## T2b. Violation magnitudes (true minima): numerical deficits vs material failures\n")
        out.append("| scenario | ctrl | h_H < −1 µJ | < −0.1 mJ | < −1 mJ | worst h_H [mJ] | BH < −1 µJ | < −0.1 mJ | worst BH [mJ] |\n|---|---|---|---|---|---|---|---|---|")
        rows_m, _ = analyze.load("main")
        for sc in scen:
            for ck in ["C1", "C2", "C3", "C4"]:
                R = [r for r in rows_m if r["scenario"] == sc and r["controller"] == ck]
                hh = np.array([r["hH_true_min"] for r in R]); bh = np.array([r["BH_true_min"] for r in R])
                if np.sum(hh < -1e-6) or np.sum(bh < -1e-6):
                    out.append(f"| {sc} | {ck} | {np.sum(hh < -1e-6)} | {np.sum(hh < -1e-4)} | {np.sum(hh < -1e-3)} | {1e3*hh.min():+.3f} | "
                               f"{np.sum(bh < -1e-6)} | {np.sum(bh < -1e-4)} | {1e3*bh.min():+.3f} |")
        out.append("\n## T4. Planner behaviour and timing (main)\n")
        out.append("| scenario | ctrl | mean σ_v sel. | mean σ_K sel. | mean σ_F sel. | backup frac | brake frac | p50 [ms] | p95 engaged [ms] | max [ms] | obj s.e. | min ESS |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for sc in scen:
            for ck in ["C1", "C2", "C3", "C4"]:
                d = S.get(f"{sc}|{ck}")
                if d is None:
                    continue
                out.append(f"| {sc} | {ck} | {fmt(d['sel_sig_v_mean']['mean'], '{:.2f}')} | {fmt(d['sel_sig_K_mean']['mean'], '{:.2f}')} | "
                           f"{fmt(d['sel_sig_F_mean']['mean'], '{:.2f}')} | {fmt(d['sel_backup_frac']['mean'], '{:.3f}')} | {fmt(d['sel_brake_frac']['mean'], '{:.3f}')} | "
                           f"{fmt(d['plan_ms_p50']['mean'], '{:.1f}')} | {fmt(d['plan_ms_engaged_p95']['mean'], '{:.1f}')} | {fmt(d['plan_ms_max']['max'], '{:.0f}')} | "
                           f"{fmt(d['obj_se_mean']['mean'], '{:.4f}')} | {fmt(d['ess_min']['min'], '{:.1f}')} |")
        out.append("\n## T5. Energy audit (main): max residuals over all episodes\n")
        out.append("| ctrl | max |ΔS−W_a+W_H+∫D_R+∫D_T| RK4 channels [J] | auditor trapezoid [J] | max credit excess [J] | wall time per episode [s] |\n|---|---|---|---|---|")
        for ck in ORDER:
            vals = [S[k] for k in S if k.endswith("|" + ck)]
            out.append(f"| {ck} | {max(v['residual_storage_plant_max']['max'] for v in vals):.2e} | {max(v['residual_storage_quad_max']['max'] for v in vals):.2e} | "
                       f"{max(v['credit_excess_vs_plant']['max'] for v in vals):.2e} | {np.mean([v['wall_s']['mean'] for v in vals]):.1f} |")
    inf = summary("infeasible")
    if inf:
        out.append("\n## T6. Infeasible initialisation (h_H(0) < 0; outside the backup set; theorem NOT applicable)\n")
        out.append(main_table(inf["summary"], ["infeasible_init"]))
    abl = [("abl_time", ["block_release", "assist_reverse", "ood_intent"], ["C3", "C3_indep", "C3_mc"], "T7. Temporal covariance and constraint-moment ablations"),
           ("abl_beta", ["block_release", "assist_reverse", "normal_oblique"], ["C4neg", "C3", "C4", "C4b5"], "T8. Risk-sensitivity sweep (β = −1, 0, 2, 5)"),
           ("abl_filter", ["block_release"], ["C0", "F", "Finst", "ORACLE"], "T9. Servo filter: tick-work vs instantaneous vs ORACLE (truth access)"),
           ("abl_residual", ["block_release", "assist_reverse"], ["C3", "C3_nores"], "T10. Residual on/off")]
    for tag, sc, cks, title in abl:
        s = summary(tag)
        if not s:
            continue
        out.append(f"\n## {title} (seeds 100-149)\n")
        out.append(main_table(s["summary"], sc, cks))
        if tag == "abl_beta":
            out.append("\n| scenario | ctrl | mean σ_K selected | mean σ_v selected | mean σ_F selected | coverage | W_H⁺ [J] | du_rms |\n|---|---|---|---|---|---|---|---|")
            for x in sc:
                for ck in cks:
                    d = s["summary"].get(f"{x}|{ck}")
                    if d:
                        out.append(f"| {x} | {ck} | {fmt(d['sel_sig_K_mean']['mean'], '{:.3f}')} | {fmt(d['sel_sig_v_mean']['mean'], '{:.3f}')} | "
                                   f"{fmt(d['sel_sig_F_mean']['mean'], '{:.3f}')} | {fmt(d['coverage']['mean'], '{:.3f}')} | {fmt(d['WH_pos_final']['mean'])} | {fmt(d['du_rms']['mean'], '{:.4f}')} |")
    cr = []
    for c in ["exact", "conservative", "invalid"]:
        s = summary(f"abl_credit_{c}")
        if s:
            for k, d in s["summary"].items():
                sc, ck = k.split("|")
                cr.append(f"| {c} | {sc} | {ck} | {d['n']} | {d['completed']['k']} | {fmt(d['Ec_min']['mean'])} | {fmt(d['credit_excess_vs_plant']['max'], '{:.2e}')} | "
                          f"{d['violation_Ec']['k']} | {d['violation_hH']['k']} | {fmt(d['coverage']['mean'], '{:.2f}')} |")
    if cr:
        out.append("\n## T11. Task-dissipation credit: exact (μ_true = μ_credit = 0.3), conservative (μ_true = 0.35, credit 0.25), INVALID over-credit diagnostic (μ_true = 0.25, credit 0.35)\n")
        out.append("| credit | scenario | ctrl | n | completed | mean min Ec [J] | max credit excess over true ∫D_T [J] | Ec viol. | h_H viol. | coverage |\n|---|---|---|---|---|---|---|---|---|---|")
        out += cr
    sw = RUNS / "sweep_summary.json"
    if sw.exists():
        out.append("\n## T12. Moment-arm × location-uncertainty sweep (moment_arm scenario, 30 seeds per cell)\n")
        out.append("| cell | n | completed | h_H viol. | BH viol. | mean max |BH err| [mJ] | mean max belief σ_e [mJ] | mean max |θ| [rad] | W_H⁺ [J] |\n|---|---|---|---|---|---|---|---|---|")
        for k, d in json.loads(sw.read_text()).items():
            out.append(f"| {k} | {d['n']} | {d['completed']} | {d['viol_hH']} | {d['viol_BH']} | {d['BH_err_maxabs_mJ']:.2f} | {d['sd_e_max_mJ']:.2f} | {d['theta_max']:.3f} | {d['WH_pos']:.3f} |")
    for h in ["0.001", "0.0005", "0.00025"]:
        summary(f"refine_h{h}")
    rf = RUNS / "refinement_summary.json"
    if rf.exists():
        out.append("\n## T13. Physics-step refinement (servo 1 kHz; 10 seeds × 2 scenarios)\n")
        out.append("| ctrl | max residual RK4 channels h=1, 0.5, 0.25 ms [J] | max residual auditor trapezoid [J] |\n|---|---|---|")
        for ck, d in json.loads(rf.read_text()).items():
            out.append(f"| {ck} | {', '.join(f'{x:.1e}' for x in d['residual_plant'])} | {', '.join(f'{x:.1e}' for x in d['residual_quad'])} |")
    ph = summary("posthoc_robust")
    if ph:
        out.append("\n## T16. POST-HOC robust bias bound (fresh seeds 300-349; designed after the frozen run)\n")
        out.append(main_table(ph["summary"], ["sensor_bias", "block_release", "assist_reverse", "moment_arm"],
                              ["C2", "C2_robust", "C3", "C3_robust"]))
    cal = ROOT / "outputs" / "calibration" / "calib_by_scenario.json"
    if cal.exists():
        CB = json.loads(cal.read_text())
        out.append("\n## T14. Forked-truth calibration of predicted margins at the horizon end (0.3 s), by scenario group\n")
        out.append("| group | predictor | quantity | n forks | z mean | z s.d. | coverage 50/80/90/95/99% | lower-tail exceed. (nominal 1%) |\n|---|---|---|---|---|---|---|---|")
        for grp in ["in_distribution_pooled", "block_release", "assist_reverse", "normal_oblique", "moment_arm", "bias_ood"]:
            for ck in ["C3", "C2"]:
                C = CB[grp][ck]
                for q in ["hH_lin", "hH_mc", "BH_lin", "Ec_lin"]:
                    d = C.get(q)
                    if d:
                        out.append(f"| {grp} | {ck} | {q} | {C['n_forks']} | {d['z_mean']:+.2f} | {d['z_sd']:.2f} | "
                                   f"{'/'.join(f'{100*d['cover'][str(l)]:.0f}' for l in [0.5, 0.8, 0.9, 0.95, 0.99])} | {100*d['lower_tail_exceed_01']:.1f}% |")
                w = C["WH_mc"]
                out.append(f"| {grp} | {ck} | W_H (MC quantiles) | {C['n_forks']} | – | – | {'/'.join(f'{100*w['cover'][str(l)]:.0f}' for l in [0.5, 0.8, 0.9, 0.95, 0.99])} | "
                           f"upper > q99.5: {100*w['exceed_q995']:.1f}%, lower < q0.5: {100*w['exceed_q005']:.1f}% |")
                mp = C["mean_power_correction"]
                out.append(f"| {grp} | {ck} | first-order − MC mean work [mJ] | – | {1e3*mp['mean']:+.2f} | – | median {1e3*mp['median']:+.2f}, q05 {1e3*mp['q05']:+.2f}, q95 {1e3*mp['q95']:+.2f} | – |")
    off = ROOT / "outputs" / "offline" / "offline_summary.json"
    if off.exists():
        O = json.loads(off.read_text())
        out.append("\n## T15. Offline predictor evaluation (conditional on the robot trajectory; 0.3 s horizon)\n")
        out.append("| split | predictor | n | force RMSE [N] (h1 / h15) | NLPD | coverage 90/99% | work RMSE [mJ] | work z s.d. | work cov. 90% | indep-step work z s.d. | response slope trans / rot |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for sp in ["val", "test_g0", "test_id", "test_heldout", "test_mismatch"]:
            for m in ["S", "S0", "N", "GM"]:
                d = O[sp][m]
                out.append(f"| {sp} | {m} | {d['n']} | {d['rmse']:.2f} ({d['rmse_h1']:.2f} / {d['rmse_h15']:.2f}) | {d['nlpd']:.2f} | "
                           f"{d['cover']['0.9']:.2f} / {d['cover']['0.99']:.2f} | {1e3*d['W_rmse']:.1f} | {d['W_z_sd']:.2f} | {d['W_cover']['0.9']:.2f} | "
                           f"{fmt(d.get('W_indep_z_sd'), '{:.2f}')} | {d['resp_trans_x_slope']:.2f} / {d['resp_rot_slope']:.2f} |")
            out.append(f"| {sp} | HOLD | – | {O[sp]['HOLD']['rmse']:.2f} | – | – | – | – | – | – | 0 / 0 |")
        out.append("\n| split | intention RMSE [mm] | mean posterior s.d. [mm] (prior 102) | z s.d. | 95% cov. | s posterior RMSE [mm] | sensor-only s RMSE [mm] | median max |corr(w, ã)| |\n|---|---|---|---|---|---|---|---|")
        for sp in ["val", "test_g0", "test_id", "test_heldout", "test_mismatch"]:
            d = O[sp]["intent"]
            out.append(f"| {sp} | {1e3*d['rmse']:.1f} | {1e3*d['mean_post_sd']:.1f} | {d['z_sd']:.2f} | {d['cover95']:.2f} | {1e3*d['s_err_rmse']:.1f} | "
                       f"{1e3*d['s_sensor_err_rmse']:.1f} | {d['max_abs_corr_w_a_median']:.2f} |")
    (ROOT / "outputs" / "report_tables.md").write_text("\n".join(out) + "\n")
    print("tables written:", len(out), "blocks")


def manifest():
    def g(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    files = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            st = p.stat()
            rec = dict(path=str(p.relative_to(ROOT)), bytes=st.st_size)
            if st.st_size < 50_000_000:
                rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            files.append(rec)
    base = (ROOT / "checkpoints" / "base_commit.txt").read_text().strip()
    man = dict(task="isolated 2D PoC: impedance-informed human uncertainty, risk-sensitive selection, Ec + BH",
               folder=str(ROOT), repo_root=g("rev-parse", "--show-toplevel"), base_commit=base,
               head_at_build=g("rev-parse", "HEAD"), branch=g("rev-parse", "--abbrev-ref", "HEAD"),
               repo_status_at_build=g("status", "--porcelain=v1").splitlines(),
               initial_repo_status=(ROOT / "checkpoints" / "initial_git_status.txt").read_text().splitlines(),
               environment=dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
                                cpus=os.cpu_count(), numpy=np.__version__,
                                scipy=__import__("scipy").__version__, matplotlib=__import__("matplotlib").__version__),
               ffmpeg="/opt/anaconda3/envs/517_drake/bin/ffmpeg (6.1.1, invoked by absolute path)",
               literature_dir="/Users/carl/Documents/Contact passivity/Nadia_Papers_2026-10-26",
               built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n_files=len(files), files=files)
    (ROOT / "manifest.json").write_text(json.dumps(man, indent=1))
    print("manifest:", len(files), "files")


def pdfs():
    for name in ["formulation.md", "report.md"]:
        p = ROOT / name
        if p.exists():
            print(name, "->", md_to_pdf(p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    a = ap.parse_args()
    allx = not (a.tables or a.pdf or a.manifest)
    if a.tables or allx:
        tables()
    if a.pdf or allx:
        pdfs()
    if a.manifest or allx:
        manifest()
