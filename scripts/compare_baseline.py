"""Before/after comparison of the contact-baseline repairs.

Extracts the pre-repair controller from git (default 132c14dbc01b0d2b6cfcc03d62fe3755f4ab1a79) into a renamed
package, runs it under the CURRENT stepping loop, recorder, fixtures and metrics, and prints one table. Only
the controller differs between the two columns; measurement chain, scenarios and every metric definition are
shared, so the numbers are directly comparable. The pre-repair controller has no resume request, so the
sustained fixture is run without one for it.

    python scripts/compare_baseline.py [--baseline <sha>] [--out docs/baseline_repair/before_after.md]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kuka_table_sim import metrics as mx                                    # noqa: E402
from kuka_table_sim import scenarios as sc                                  # noqa: E402
from kuka_table_sim.holding import Simulation                               # noqa: E402
from kuka_table_sim.recorder import Recorder                                # noqa: E402
from kuka_table_sim.scene import ArmIndex, TimingConfig, load_model         # noqa: E402

DEFAULT_BASELINE = "132c14dbc01b0d2b6cfcc03d62fe3755f4ab1a79"


def load_baseline_package(sha: str):
    tmp = Path(tempfile.mkdtemp(prefix="kuka_base_"))
    archive = subprocess.run(["git", "-C", str(REPO), "archive", sha, "kuka_table_sim"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
    (tmp / "kuka_table_sim").rename(tmp / "kuka_base")
    for f in (tmp / "kuka_base").glob("*.py"):
        f.write_text(re.sub(r"\bkuka_table_sim\b", "kuka_base", f.read_text()))
    (tmp / "assets").symlink_to(REPO / "assets")
    sys.path.insert(0, str(tmp))
    import importlib
    return importlib.import_module("kuka_base.force_control")


def make_adapter(bfc):
    code = {bfc.APPROACH: 0, bfc.CONTACT: 1, bfc.SLIDE: 2}

    class Adapter(bfc.ForceApproachController):
        """Pre-repair controller + diagnostics() that rebuilds the active reference exactly as its compute() used it."""

        def compute(self, data):
            tau = super().compute(data)
            self.tau_cmd_clipped = tau.copy()
            return tau

        def diagnostics(self):
            n = self.normal
            x_d = self.x_ref - (self.x_ref @ n) * n + self.s_d * self.slide_dir
            v_d = self.sd_dot * self.slide_dir
            a_d = self.sd_ddot * self.slide_dir
            nan = np.where(np.abs(n) > 0.5, np.nan, 0.0)
            if self.phase == bfc.APPROACH:
                x_d, v_d, a_d = x_d + self.z_d * n, v_d + self.zd_dot * n, a_d + self.zd_ddot * n
            else:
                x_d, v_d, a_d = x_d + nan, v_d + nan, a_d + nan
            return dict(phase=code[self.phase], lost=0.0, f_n=self.f_n, f_d=self.f_d, f_cmd=self.f_cmd, f_int=self.f_int,
                        tan_int=self.tan_int.copy(), rot_int=self.rot_int.copy(), int_scale=1.0, x_d=x_d, v_d=v_d, a_d=a_d,
                        e_o=np.zeros(3), tau_cmd=self.tau_cmd_clipped, sat_any=float(np.any(self.saturated)),
                        n_uncertain=0, n_lost=self.n_releases, n_reacquired=0, n_hold=0)

        def request_resume(self):               # the pre-repair controller has no HOLD
            pass

    return Adapter


def make_sim(factory, duration):
    timing = TimingConfig()
    model = load_model(timing)
    arm = ArmIndex.from_model(model)
    sim = Simulation(model, timing, factory(model, arm, timing.control_period))
    sim.recorder = Recorder(sim, duration)
    sim.reset()
    return sim


def run_all(factory, cfg_cls):
    out = {}
    # nominal
    sim = make_sim(lambda m, a, dt: factory(m, a, dt, cfg_cls()), 25.0)
    out["nominal"] = sc.ScenarioResult("nominal", sim, sc.run_for(sim, 25.0))
    # fast free-air descent
    sim = make_sim(lambda m, a, dt: factory(m, a, dt, cfg_cls(approach_speed=1.0, slide_distance=0.0)), 12.0)
    out["fast"] = sc.ScenarioResult("fast-approach", sim, sc.run_for(sim, 12.0))
    # loss fixtures (same pulls as scenarios.LOSS_FIXTURES)
    for kind, fx in sc.LOSS_FIXTURES.items():
        sim = make_sim(lambda m, a, dt: factory(m, a, dt, cfg_cls()), fx["duration"])
        pull = sc.WristPull(t_start=12.0, hold=fx["hold"], force=fx["force"], ramp=fx["ramp"])
        sim.disturbance = pull.bind(sim.model)
        out[f"loss-{kind}"] = sc.ScenarioResult(f"loss-{kind}", sim, sc.run_for(sim, fx["duration"]), dict(pull=pull.__dict__, t_pull_end=pull.t_end))
    # abrupt removal stress
    sim = make_sim(lambda m, a, dt: factory(m, a, dt, cfg_cls()), 24.0)
    tb = sim.model.body("table").id
    done = {"x": False}

    def hook(s):
        if not done["x"] and s.data.time >= 12.0:
            s.model.body_pos[tb][2] -= 1.0
            done["x"] = True

    out["removal"] = sc.ScenarioResult("removal-stress", sim, sc.run_for(sim, 24.0, hook), dict(t_remove=12.0))
    return out


def summarize(res: dict) -> dict:
    s = {}
    n = res["nominal"]
    log = n.log
    t0 = mx.phase_times(log)["slide"]
    win = (t0, float(log.phys["t"][-1]))
    f = mx.force_stats(log, win)
    tr = mx.tracking_stats(log, win)
    tq = mx.torque_stats(log)
    s["nominal"] = dict(f_mean=f["truth_mean_abs"], f_max=f["truth_max_abs"], along=tr["along_max_mm"], cross=tr["cross_max_mm"],
                        sat=tq["saturated_substeps"], events=sum(mx.fsm_events(log).values()), slide_start=t0)
    fa = res["fast"].log
    p = fa.phys
    first_true = float(p["t"][np.argmax(p["present"])])
    t_contact = next((a for a, _, b in mx.transitions(fa) if b == "contact"), None)
    s["fast"] = dict(first_true=first_true, t_contact=t_contact, false=t_contact is not None and t_contact < first_true,
                     peak_f_used=float(np.nanmax(fa.ctl["f_n"][fa.ctl["t"] < first_true])) if (fa.ctl["t"] < first_true).any() else 0.0)
    for key in ("loss-short", "loss-reacquire", "loss-sustained", "removal"):
        r = res[key]
        lg = r.log
        t_ev = r.info.get("pull", {}).get("t_start", r.info.get("t_remove", 12.0))
        w = (t_ev - 0.5, float(lg.phys["t"][-1]))
        rc = mx.reference_continuity(lg, w)
        tq = mx.torque_stats(lg, w)
        ab = mx.contact_absence(lg, w)
        tt = mx.transitions(lg)
        tt_after = [x for x in tt if x[0] > t_ev]
        # CONTACT entries while no physical tip/table contact exists (false reacquisition)
        ci = lg.meta["control_interval"]
        c = lg.ctl
        false_entries = 0
        for tx, _, b in tt_after:
            if b == "contact":
                i = int(np.searchsorted(c["t"], tx))
                lo = max(i * ci - 40, 0)
                false_entries += int(not lg.phys["present"][lo:i * ci + 1].any())
        norm_act = rc["normal_activation"]
        z = lg.phys["tcp"][lg.phys["t"] >= t_ev, 2]
        fp = mx.force_peaks(lg, (t_ev, float(lg.phys["t"][-1])))
        s[key] = dict(force_peak=fp["peak_N"], trans=len(tt_after), false_entries=false_entries, abs_events=ab["geometric_absence"]["events"],
                      abs_total=ab["geometric_absence"]["total_s"], tan_step=rc["tangential"]["pos_step_max_mm"],
                      act_err=max((abs(a["pos_err_mm"]) for a in norm_act), default=0.0), peak_util=tq["peak_applied_utilisation"],
                      sat_sub=tq["saturated_substeps"], sat_long=tq["longest_saturation_s"], z_max=float(z.max()), z_min=float(z.min()),
                      final=mx.transitions(lg)[-1][2] if tt else "?", surface=r.sim.controller.z_surface)
    return s


def table(before: dict, after: dict, sha: str) -> str:
    L = []
    add = L.append
    add(f"Before = pre-repair controller `{sha[:10]}`; after = current checkout. Same stepping loop, recorder, fixtures and metrics for both.\n")
    add("| Mechanism / scenario | Metric | Before | After |")
    add("| --- | --- | --- | --- |")
    b, a = before["nominal"], after["nominal"]
    for lab, k, fmt in (("mean absolute force error", "f_mean", "{:.4f} N"), ("max absolute force error", "f_max", "{:.4f} N"), ("along-path error max", "along", "{:.2f} mm"),
                        ("cross-path error max", "cross", "{:.3f} mm"), ("saturated substeps", "sat", "{:d}"), ("recovery/loss events", "events", "{:d}")):
        add(f"| A nominal sliding (25 s, 10 N) | {lab} | {fmt.format(b[k])} | {fmt.format(a[k])} |")
    b, a = before["fast"], after["fast"]
    add(f"| B free-air, 1 m/s descent | first physical contact | {b['first_true']:.3f} s | {a['first_true']:.3f} s |")
    add(f"| B free-air, 1 m/s descent | CONTACT entered at | {b['t_contact']:.3f} s ({'FALSE, before any contact' if b['false'] else 'genuine'}) | {a['t_contact']:.3f} s ({'FALSE' if a['false'] else 'genuine'}) |")
    add(f"| B free-air, 1 m/s descent | peak force used by control before contact | {b['peak_f_used']:.2f} N | {a['peak_f_used']:.2e} N |")
    for key, name in (("loss-short", "C short interruption"), ("loss-reacquire", "C loss + reacquisition"), ("loss-sustained", "C sustained loss"), ("removal", "C removal STRESS TEST")):
        b, a = before[key], after[key]
        add(f"| {name} | FSM transitions after the event | {b['trans']} | {a['trans']} |")
        add(f"| {name} | CONTACT entries without physical contact | {b['false_entries']} | {a['false_entries']} |")
        add(f"| {name} | physical absence: events / total | {b['abs_events']} / {b['abs_total']:.2f} s | {a['abs_events']} / {a['abs_total']:.2f} s |")
        add(f"| {name} | max in-plane reference step per 1 ms | {b['tan_step']:.2f} mm | {a['tan_step']:.3f} mm |")
        add(f"| {name} | normal position-loop activation error | {b['act_err']:.1f} mm | {a['act_err']:.2f} mm |")
        add(f"| {name} | peak true contact force after the event | {b['force_peak']:.1f} N | {a['force_peak']:.1f} N |")
        add(f"| {name} | peak applied torque / limit | {b['peak_util'] * 100:.0f} % | {a['peak_util'] * 100:.0f} % |")
        add(f"| {name} | saturated substeps (longest) | {b['sat_sub']} ({b['sat_long']:.3f} s) | {a['sat_sub']} ({a['sat_long']:.3f} s) |")
        add(f"| {name} | tip height after the event: max / min | {b['z_max']:.3f} / {b['z_min']:.3f} m | {a['z_max']:.3f} / {a['z_min']:.3f} m |")
        add(f"| {name} | final phase | {b['final']} | {a['final']} |")
    add("")
    add("Notes. (1) Row B: the physical first-contact time differs because the pre-repair controller's false CONTACT changed the motion. "
        "(2) C short interruption: the pre-repair controller keeps pushing and rides through this 11 N wrist pull without losing contact; the new "
        "policy treats a 5 N force deficit lasting 10 ms as a contact problem and backs off, so it loses contact for 0.25 s and spends about 1.3 s "
        "recovering. That is a deliberate sensitivity trade-off, not an improvement. (3) C sustained: neither run is given a resume request here "
        "(the pre-repair controller has none), so the new controller ends in HOLD; the full SEARCH -> HOLD -> resume -> reacquire sequence is in "
        "`reports/loss-sustained.txt`. (4) C removal is a STRESS TEST in which the table is teleported away, not a reacquisition benchmark.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    bfc = load_baseline_package(args.baseline)
    Adapter = make_adapter(bfc)
    before = summarize(run_all(lambda m, a, dt, cfg: Adapter(m, a, dt, cfg), bfc.ForceControlConfig))
    from kuka_table_sim.force_control import ForceApproachController, ForceControlConfig
    after = summarize(run_all(lambda m, a, dt, cfg: ForceApproachController(m, a, dt, cfg), ForceControlConfig))
    md = table(before, after, args.baseline)
    print(md)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
