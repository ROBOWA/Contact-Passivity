"""Regenerate auditable figures, compact evidence and the acceptance manifest.

python -m kuka_table_sim.oracle_report --root results/oracle/final \
  --validation results/oracle/validation_final --out docs/oracle_joint_qp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import hashlib
import subprocess

import numpy as np

from .oracle_experiments import TOLERANCES, save_json, source_hash
from .recorder import RunLog


def plot_runs(root, out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["tight-off", "tight-active", "lower-budget"]
    fig, axs = plt.subplots(4,2,figsize=(13,12),constrained_layout=True)
    for label in labels:
        z=np.load(root/label/"oracle.npz")
        b=RunLog.load(root/label/"baseline_log.npz")
        c={k[5:]:z[k] for k in z.files if k.startswith("ctl__")}
        p={k[6:]:z[k] for k in z.files if k.startswith("phys__")}
        budget=json.loads((root/label/"metadata.json").read_text())["oracle"]["energy_init"][0]
        axs[0,0].plot(c["t"]+.001,c["energy_after"][:,0],label=label)
        # Full physics samples preserve brief force/power spikes. A fixed
        # stride can skip the very transient this experiment must expose.
        axs[0,1].plot(p["t"],p["normal_force"],label=label)
        axs[1,0].plot(p["t"],p["power"],label=label)
        axs[1,1].plot(p["t"],p["velocity"][:,0],label=label)
        axs[2,0].plot(c["t"],c["intervention"],label=label)
        axs[2,1].plot(b.ctl["t"],1000*(b.ctl["x_d"][:,0]-b.ctl["x"][:,0]),label=label)
        axs[3,0].plot(c["t"],(c["predicted"][:,0]-c["actual"][:,0])*1e6,label=label)
        axs[3,1].plot(c["t"],(c["affine"][:,0]-c["predicted"][:,0])*1e6,label=label)
    ylabels=["account H = R (J)","actual normal force (N)","human power (W)","human point vx (m/s)",
             "max |torque correction| (N m)","along-path error (mm)","predicted − actual work (µJ)",
             "affine − nonlinear work (µJ)"]
    for ax,yl in zip(axs.flat,ylabels):
        ax.set(xlim=(9.,14.),xlabel="simulation time (s)",ylabel=yl)
        ax.legend(fontsize=8)
    axs[0,0].axhline(0,color="k",ls="--",lw=.8)
    axs[0,1].axhline(10,color="k",ls="--",lw=.8)
    fig.savefig(out/"active_comparison.png",dpi=140)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    observed_floor=0.
    for label in ("tight-active","task-cost","stress","held-out-active","elbow-tight"):
        z=np.load(root/label/"oracle.npz")
        axs[0,0].plot(z["ctl__t"]+.001,z["ctl__energy_after"][:,0],label=label)
        axs[0,1].plot(z["phys__t"],z["phys__prefix_energy"][:,0]*1e6,label=label)
        observed_floor=min(observed_floor,float(np.min(z["phys__prefix_energy"][:,0]))*1e6)
        b=RunLog.load(root/label/"baseline_log.npz")
        axs[1,0].plot(b.phys["t"],b.phys["F_task"][:,2],label=label)
        axs[1,1].plot(b.ctl["t"],b.ctl["f_int"],label=label)
    for ax,yl in zip(axs.flat,["boundary energy (J)","substep energy near floor (µJ)","normal force (N)","force integrator (N s)"]):
        ax.set(xlim=(9.3,14.),xlabel="simulation time (s)",ylabel=yl)
        ax.legend(fontsize=7)
    axs[0,0].axhline(0,color="k",ls="--")
    axs[0,1].axhline(0,color="k",ls="--")
    axs[0,1].set_ylim(min(-5,1.1*observed_floor),50)
    fig.savefig(out/"failures_and_recovery.png",dpi=140)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    for label in ("tight-shadow","varying"):
        z=np.load(root/label/"oracle.npz")
        axs[0].plot(z["ctl__t"],(z["ctl__predicted"][:,0]-z["ctl__actual"][:,0])*1e6,label=label)
    axs[0].set(xlim=(9.4,11.),xlabel="time (s)",ylabel="hold prediction − actual work (µJ)")
    axs[0].legend()
    z=np.load(root/"tight-active"/"oracle.npz")
    axs[1].hist(z["tick_seconds"]*1000,bins=70)
    axs[1].set(xlabel="wall time per 1 ms simulation tick (ms)",ylabel="count",yscale="log")
    fig.savefig(out/"forecast_and_runtime.png",dpi=140)
    plt.close(fig)


def compact_case(folder, dest):
    z=np.load(folder/"oracle.npz")
    out={}
    for k in ("t","energy_before","energy_after","actual","predicted","affine","intervention","attempts","min_prefix","gross"):
        out["ctl__"+k]=z["ctl__"+k]
    csel=((z["ctl__t"]>=9.25)&(z["ctl__t"]<=11.6)&
          ((z["ctl__intervention"]>1e-7)|np.any(z["ctl__gradient"]!=0,axis=(1,2))))
    for k in ("t","tau_nom","tau_exec","tau_previous","gradient"):
        out["audit__"+k]=z["ctl__"+k][csel]
    psel=((z["phys__t"]>=9.25)&(z["phys__t"]<=11.6)&np.any(z["phys__force"]!=0,axis=1))
    for k in ("t","force","velocity","power","task_power","prefix_energy","power_identity"):
        out["phys__"+k]=z["phys__"+k][psel]
    # Broad task/recovery trace is strided; account/work above remain full precision/full rate.
    b=RunLog.load(folder/"baseline_log.npz")
    fn=b.phys["F_task"][:,2]
    transitions=np.flatnonzero(np.diff(b.phys["present"].astype(int))!=0)
    tidx=np.unique(np.r_[np.arange(0,len(fn),40),np.argmin(fn),np.argmax(fn),transitions,transitions+1])
    for k in ("t","tcp","F_task","present"):
        out["task__"+k]=b.phys[k][tidx]
    for k in ("t","x_d","x","f_int","tan_int","rot_int","phase"):
        out["taskctl__"+k]=b.ctl[k][::20]
    out["compaction_json"]=np.array(json.dumps(dict(
        account_work="all control boundaries, float64",
        audit="only intervention or nonzero diagnostic-gradient ticks in 9.25–11.6 s",
        physics="all nonzero-human-force samples in 9.25–11.6 s; omitted human power is zero",
        task="every 40 physics steps plus global force extrema/contact transitions; control every 20 ticks")))
    out["metadata_json"]=np.array((folder/"metadata.json").read_text())
    np.savez_compressed(dest,**out)


def generate(root, validation, out):
    root,validation,out=map(Path,(root,validation,out))
    out.mkdir(parents=True,exist_ok=True)
    raw=out/"raw"
    raw.mkdir(exist_ok=True)
    summary=json.loads((root/"summary.json").read_text())
    val=json.loads((validation/"validation.json").read_text())
    metas={k:json.loads((root/k/"metadata.json").read_text()) for k in summary}
    save_json(out/"metrics.json",summary)
    save_json(out/"validation.json",val)
    retained={"no-human-off","tight-off","tight-shadow","tight-active","stress",
              "varying","elbow","held-out-active","elbow-tight"}
    for k in summary:
        if k in retained:
            compact_case(root/k,raw/(k+".npz"))
        elif (raw/(k+".npz")).exists():
            # Remove only this generator's known obsolete compact case file.
            (raw/(k+".npz")).unlink()
        if (root/k/"failure.json").exists():
            shutil.copy2(root/k/"failure.json",raw/(k+"_failure.json"))
            shutil.copy2(root/k/"failure_snapshot.npz",raw/(k+"_failure_snapshot.npz"))
    for name in ("scans.npz","gradient_response.png","one_sided_error.png"):
        shutil.copy2(validation/name,out/name)
    # Keep accepted/rejected QP history for the main case and all failing trials.
    for k in summary:
        if k=="tight-active" or summary[k]["status"]=="FAIL":
            shutil.copy2(root/k/"events.json",raw/(k+"_events.json"))
    plot_runs(root,out,summary)
    gates=[]
    def gate(name,ok,command,evidence,value=None,tolerance=None,reason=None):
        gates.append(dict(gate=name,status="NOT RUN" if ok is None else "PASS" if ok else "FAIL",
                          command=command,evidence=evidence,value=value,tolerance=tolerance,
                          failure_reason=reason if ok is not True else None))
    current_sha=source_hash()
    sha=next(iter(metas.values()))["source_sha256"]
    revision=next(iter(metas.values()))["commit"]
    # Postprocessing repairs do not require re-running identical physics. Verify
    # the recorded full source hash against its Git tree, and permit ONLY this
    # reporting module to differ. Every numerical/control source must match.
    repo=Path(__file__).resolve().parent.parent
    def git(*args):
        return subprocess.run(["git","-C",str(repo),*args],capture_output=True,check=True).stdout
    paths=git("ls-tree","--name-only",revision,"kuka_table_sim/").decode().splitlines()
    h=hashlib.sha256()
    for name in sorted(p for p in paths if p.endswith(".py")):
        h.update(Path(name).name.encode())
        h.update(git("show",f"{revision}:{name}"))
    changed=git("diff","--name-only",revision,"--","kuka_table_sim").decode().splitlines()
    untracked=git("ls-files","--others","--exclude-standard","--","kuka_table_sim").decode().splitlines()
    consistent=(all(m["source_sha256"]==sha and m["commit"]==revision for m in metas.values()) and
                val["source_sha256"]==sha and h.hexdigest()==sha and not untracked and
                all(p=="kuka_table_sim/oracle_report.py" for p in changed))
    gate("final_source_provenance",consistent,"python -m kuka_table_sim.oracle_report", "raw/* metadata_json; validation.json",
         dict(numerical_source_sha256=sha,postprocessing_source_sha256=current_sha,changed_files=changed),
         "recorded Git source hash; only reporting module may differ","stale or modified numerical/control source")
    testfile=root.parent/"final_tests.txt"
    testtext=testfile.read_text() if testfile.exists() else ""
    gate("physical_QP_and_regression_suite",True if "61 passed" in testtext else False if testtext else None,
         ".venv/bin/python -m pytest -q",str(testfile),testtext.strip().splitlines()[-1:] if testtext else None,
         "61 passed","see exact pytest output")
    if testfile.exists(): shutil.copy2(testfile,out/"final_tests.txt")
    before=root.parent/"baseline.npz"
    after=root.parent/"baseline_final.npz"
    baseline_ok=None
    if before.exists() and after.exists():
        a,b=RunLog.load(before),RunLog.load(after)
        baseline_ok=all(np.array_equal(a.phys[k],b.phys[k],equal_nan=True) for k in a.phys) and all(
            np.array_equal(a.ctl[k],b.ctl[k],equal_nan=True) for k in a.ctl)
    gate("unchanged_25s_baseline",baseline_ok,".venv/bin/python -m kuka_table_sim.scenarios nominal --log results/oracle/baseline_final.npz",
         "results/oracle/baseline{,_final}.npz",baseline_ok,"bitwise physics and control arrays","baseline changed or not reproduced")
    for fname in ("baseline.txt","baseline_final.txt"):
        if (root.parent/fname).exists(): shutil.copy2(root.parent/fname,out/fname)
    def parity(labels):
        logs=[RunLog.load(root/l/"baseline_log.npz") for l in labels]
        return all(all(np.array_equal(log.phys[k],logs[0].phys[k],equal_nan=True) for k in logs[0].phys)
                   and all(np.array_equal(log.ctl[k],logs[0].ctl[k],equal_nan=True) for k in logs[0].ctl) for log in logs[1:])
    gate("no_human_transparency",parity(["no-human-off","no-human-shadow","no-human-active"]),
         ".venv/bin/python -m kuka_table_sim.oracle_experiments matrix","raw/no-human-*.npz",True,"bitwise","mode perturbs nominal trajectory")
    gate("ample_budget_transparency",parity(["ample-off","ample-active"]),"matrix","raw/ample-*.npz",True,"bitwise","ample active perturbs nominal")
    gate("human_power_and_oracle_identities",all(m["power_identity_max_W"]<=1e-12 and m["account_equality_max_J"]==0 for m in summary.values()),
         "pytest + matrix","metrics.json",max(m["power_identity_max_W"] for m in summary.values()),1e-12,"power or H=R mismatch")
    held=[m["prediction_max_abs_J"] for k,m in summary.items() if k!="varying" and m["prediction_max_abs_J"] is not None]
    gate("held_force_execution_prediction",max(held)<1e-11,"matrix","metrics.json",max(held),1e-11,"execution/prediction mismatch")
    gate("gradients_and_empirical_envelope",val["status"]=="PASS",".venv/bin/python -m kuka_table_sim.oracle_validation --out results/oracle/validation_final",
         "validation.json; scans.npz",val["held_out_max_over_J"],val["epsilon_linearization_J"],"declared empirical domain exceeded")
    primary=summary["tight-active"]
    energy_ok=primary["status"]=="PASS" and primary["account_min_margin_J"]>=-1e-10 and primary["activity"]["intervention_ticks"]>0 and primary["net_outgoing_J"]<summary["tight-off"]["net_outgoing_J"]
    gate("main_active_energy",energy_ok,".venv/bin/python -m kuka_table_sim.oracle_experiments case --name resistance --mode active --budget .12 --task-weight 1",
         "raw/tight-active.npz; metrics.json",primary["account_min_margin_J"],-1e-10,"active run incomplete, no useful intervention, or account violation")
    # Independently recompute physical bounds from logged candidate/previous/center.
    max_box=max_energy=0.
    for label,m in metas.items():
        z=np.load(root/label/"oracle.npz")
        ct=z["ctl__t"]
        for e in json.loads((root/label/"events.json").read_text()):
            for a in e.get("attempts",[]):
                if "torque" not in a: continue
                tau=np.array(a["torque"])
                idx=int(np.argmin(np.abs(ct-e["t"])))
                previous=z["ctl__tau_previous"][idx]
                if abs(ct[idx]-e["t"]) > 1e-6 and (root/label/"failure_snapshot.npz").exists():
                    previous=np.load(root/label/"failure_snapshot.npz")["previous"]
                cfg=m["oracle"]["config"]
                limits=np.array([320,320,176,176,110,40,40])
                residual=max(float(np.max(np.abs(tau)-limits)),
                             float(np.max(np.abs(tau-previous)-np.array(cfg["rate"])*.001)),
                             float(np.max(np.abs(tau-np.array(a["center"]))-np.array(a["rho"]))))
                max_box=max(max_box,residual)
                affine=np.array(a["center_work"])+np.array(a["gradient"])@(tau-np.array(a["center"]))
                energy=np.array(a["energy"])+affine-a["epsilon_linearization"]-cfg["epsilon_prediction"]-np.array(a["minimum"])
                max_energy=max(max_energy,-float(np.min(energy)))
    gate("logged_QP_physical_residuals",max_box<=1e-7 and max_energy<=1e-10,"matrix + oracle_report",
         "raw/*_events.json; full results/oracle/final/*/events.json",dict(torque_Nm=max_box,energy_J=max_energy),
         dict(torque_Nm=1e-7,energy_J=1e-10),"recomputed original-unit constraint residual")
    task_ok=(primary["force"]["original_10N_mean_abs"]<=.5 and primary["force"]["original_10N_max_abs"]<=3 and
             primary["contact"]["geometric_absence"]["total_s"]==0 and primary["tracking"]["cross_max_mm"]<=3 and
             primary["tracking"]["along_max_mm"]<=15 and primary["tracking"]["ori_max_deg"]<=2)
    gate("main_active_task_quality",task_ok,"matrix","metrics.json; active_comparison.png",
         dict(force_max_N=primary["force"]["original_10N_max_abs"],contact_absence_s=primary["contact"]["geometric_absence"]["total_s"]),
         dict(force_mean_N=.5,force_max_N=3,contact_absence_s=0),"hard energy constraint causes force spikes/contact interruption")
    rec=primary["recovery"]
    gate("release_recovery",None if rec is None else rec["force"]["truth_max_abs"]<=.5 and rec["tracking"]["along_max_mm"]<=5,
         "matrix","metrics.json; failures_and_recovery.png",rec,"last-second max force error .5 N, along error 5 mm","recovery tracking gate exceeded")
    stress=summary["stress"]
    stress_ok=(stress["status"]=="PASS" and stress["account_min_margin_J"]>=-1e-10) or (
        stress["status"]=="FAIL" and (root/"stress"/"failure_snapshot.npz").exists())
    gate("stress_outcome_is_explicit",stress_ok,"matrix","raw/stress.npz; metrics.json",
         dict(trial_status=stress["status"],failure=stress["failure"],min_endpoint_margin_J=stress["account_min_margin_J"]),
         "completed with valid endpoints OR explicit failure + snapshot","unreported stress outcome")
    gate("held_out_active",summary["held-out-active"]["status"]=="PASS","matrix","raw/held-out-active*; metrics.json",
         summary["held-out-active"]["failure"],"completed trial","held-out affine QP infeasible within configured bounds")
    gate("varying_force_error_measured",summary["varying"]["prediction_max_abs_J"]>1e-11,"matrix","forecast_and_runtime.png; metrics.json",
         summary["varying"]["prediction_max_abs_J"],"measured; shadow only","forecast mismatch was not exposed")
    manifest=dict(overall="PARTIAL" if any(g["status"]!="PASS" for g in gates) else "PASS",
                  source_sha256=sha,postprocessing_source_sha256=current_sha,gates=gates,
                  limits="Oracle simulation; empirical margins; endpoint energy only; no hardware/unknown-contact/real-time certificate")
    save_json(out/"acceptance.json",manifest)
    write_report(out,summary,val,metas,manifest)
    return manifest


def write_report(out,s,v,metas,manifest):
    m=s["tight-active"]
    off=s["tight-off"]
    meta=metas["tight-active"]
    rows=[]
    for name,a in s.items():
        rows.append(f"| {name} | {a['status']} | {a['completed_time']:.3f} | {a['net_outgoing_J']:.9f} | {a['account_min_margin_J']:.3g} | {a['force']['original_10N_max_abs']:.3f} | {a['contact']['geometric_absence']['total_s']:.4f} | {a['tracking']['along_max_mm']:.3f} | {a['activity']['intervention_ticks']} |")
    gates="\n".join(f"| {g['gate']} | {g['status']} |" for g in manifest["gates"])
    text=f'''# KUKA oracle joint-torque QP — {manifest['overall']}

The unchanged nominal baseline and shadow prediction are validated. The main active
fixture completes and limits cumulative NET human output, but strict task preservation
and held-out active feasibility fail. This is a partial research result, not a completed
task-preserving passivity controller. No robot hardware was accessed.

## Reproduction and provenance

Starting checkout `dbff1c1` / `main`, clean. Work branch
`feature/kuka-oracle-joint-qp`. Source revision for these runs: `{meta['commit']}`;
source SHA256 `{meta['source_sha256']}`. Artifact/reporting commits may be later;
the provenance gate verifies that all numerical/control files still match this
recorded Git tree. Seed 0 (deterministic fixtures); validation scan seed 7341.
Hardware `{meta['hardware']}`; software `{meta['versions']}`.
XML hashes and unchanged nominal gains/trajectory are in `baseline_final.txt`
and full `results/oracle/final/*/baseline_log.npz` metadata.

```
.venv/bin/python -m pytest -q
.venv/bin/python -m kuka_table_sim.scenarios nominal --log results/oracle/baseline_final.npz
.venv/bin/python -m kuka_table_sim.oracle_validation --out results/oracle/validation_final
.venv/bin/python -m kuka_table_sim.oracle_experiments matrix --out results/oracle/final
.venv/bin/python -m kuka_table_sim.oracle_report --root results/oracle/final --validation results/oracle/validation_final --out docs/oracle_joint_qp
```

The matrix deliberately returns exit 2 when held-out/optional active trials fail;
all logs are still saved. A single failed case also exits 2. Unsupported inputs
fail closed. `results/` holds full logs and is ignored; `raw/` contains compact
full-precision account/work data for nine representative cases, force/power physics samples around interaction,
strided task recovery traces preserving global force extrema and contact transitions,
rejected candidates and failure snapshots. `compaction_json` declares retained
samples; omitted human powers are zero. No
float32 conversion is applied to work/accounts. Regenerate figures with the report
command after reproducing logs; do not treat a missing test as passing.

## Timing, ports and accounts

Control dt=0.001 s, physics h=0.00025 s, four split physics steps per held COMPLETE
torque command. Nominal bias/gravity is computed once by the existing controller.
`ArmIndex` checks gear=1, fixed gain=1, no bias, joint transmission; the oracle
additionally excludes actuator dynamics, extra force clamps, plugins, RK4, disabled
actuation and global dynamics callbacks. Every rollout and executed physics sample
checks actual `qfrc_actuator` against the candidate. No downstream command clipping
is hidden. Rate/trust bounds are simulation choices, not hardware specifications:
rho=(80,80,60,60,40,20,20) N m; rate=rho/dt N m/s. Torque limits are unchanged.

Known human point: link5 local (0,.04,.12) m; second point link3 local
(.02,.04,.16) m. The main force is −x, 5 N, cosine ramp .2 s / hold 1 s /
ramp .2 s, starting 9.5 s. Unfiltered point velocity is +x in this interval;
the measured negative power confirms resistance. Assisting case reverses force.
Cartesian force is sampled at the control boundary, then held WORLD-fixed while
point/J are recomputed at every physics sample. The predictor receives only this
current force. The varying-force check samples the plant at each physics step
while the predictor continues holding the current value; it runs in shadow.

Human power uses actual link point J qdot; applied input uses mj_applyFT after
restoring base qfrc_applied every substep. Force is applied exactly once. Task
work restricts contact to tool_tip/tabletop, with world-frame force ON the tool,
material-point velocity including rotation, and any contact moment. The baseline
meter was extracted without changing its arithmetic and is shared with rollouts.
Independent tests use constraint power, a force+moment at body COM, and a condim=6
fixture to check the sign, frame and moment term.

Sampling: step1 evaluates kinematics at x_k; human point/J/velocity are saved
before integration. step2 solves contact at x_k and advances q/qdot. The task
meter uses still-current x_k cvel/contact forces; it never pairs solved forces
with the new qdot. Work is signed left rectangle P_k*h. Tests independently
check these stage semantics, generalized constraint power and timestep halving.

Full `mjSTATE_INTEGRATION` snapshots include time, q/qdot, act/history, warmstart,
ctrl, external forces, equality flags, mocap, user and plugin fields exposed by
installed MuJoCo. Each candidate restores the SAME snapshot in a separate MjData,
overrides the torque and human force, and advances exactly one interval. The
shared model is immutable during prediction. No integrator/reference/FSM/RNG/
recorder callbacks are called. Replay and candidate-order invariance are tested.

W_T_nominal=W_T_actual; W_R=W_H EXACTLY. Both accounts update only from actual
executed signed work, with explicit E_init and E_min=0. No recharge except
positive work, no account clipping or per-cycle resets. Off/shadow violations
are retained. Gross outgoing energy is separately logged and is NOT bounded by
the net-work constraint. Control-boundary minima and observed substep minima
are separate; negative substep margins can occur even when endpoints pass.
This is not a continuous-time passivity certificate.

## Gradients, margin and QP

Seven-variable delta=tau−ORIGINAL tau_nom, full joint space. Central double
precision stencil uses 15 rollouts with eta=.02 N m; .01/.02/.04 checks identify
noise. Stencils crossing actuator bounds are rejected explicitly. Validation
uses 15 snapshots and {v['samples']} single/mixed-direction commands over the
declared trust box. Calibration and held-out trajectories are separate.
Normalized-to-5-N max one-sided affine overprediction: calibration
{v['calibration_max_over_J']:.6g} J, held-out {v['held_out_max_over_J']:.6g} J.
Margin epsilon_linearization=2.5e-5 J * |f|/5 N; deterministic held-force
epsilon_prediction=1e-11 J. These are empirical resolutions/envelopes on the
logged validation set, not uniform certified bounds. At accepted candidates a
fresh nonlinear rollout checks E+W_hat−epsilon_prediction >= E_min, so the
actual accepted point does not rely on its affine remainder bound. No-human
work/margin is exactly zero only under the guaranteed held-zero-input fixture.

Cost H=diag(1/torque_scale²)+S_nom^T W_y S_nom. Scale=(20,20,10,10,5,2,2)
N m; task outputs are average normal force, terminal linear/angular velocity,
normalized by (1 N,.01 m/s x3,.1 rad/s x3). Main weight=1; weight=0 and 100
are reported as explicit ablations. Nominal task sensitivity and H freeze
during retries. Objective whitening preserves the optimum; exactly equal
energy rows consolidate to the tighter budget, then both rows are checked in
original units. OSQP eps_abs/eps_rel=1e-9, max_iter=20000, polishing on. A small
bound-active KKT recovery handles solver nonconvergence without changing bounds
or using slack; an independent dense-H analytic optimum test exercises it.

Torque/rate/trust and affine energy residuals are checked in N m/J, regardless
of solver status. Fresh nonlinear check follows every solved candidate. Up to
three relinearizations shrink the trust region and shift the center from the
original snapshot; the objective always remains anchored to tau_nom. Nominal
zero correction is skipped only if it already satisfies bounds and the same
energy reserve. Rejecting a candidate or an empty box never executes it.
An impossible affine halfspace/box intersection stops the trial with state and
diagnostics; it is NOT a proof of global nonlinear infeasibility or a hardware
fallback. Integrators use their existing clamps and freeze after filter
intervention, extending the existing previous-saturation condition. No inverse
Cartesian interpretation of general joint correction is invented.

## Quantitative comparison

Every row has the same nominal initial state, plant, trajectory, and controller.
Matched cases differ only by explicitly logged mode/budget/task cost/force fixture.
Force max is relative to the ORIGINAL 10 N target; lowering f_d during SEARCH
does not hide contact loss. Whole steady/contact/recovery window starts 8.71 s.

| case | trial | completed s | net outgoing J | min endpoint margin J | max 10 N error N | contact absent s | along max mm | intervention ticks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Primary net-output reduction: {(1-m['net_outgoing_J']/off['net_outgoing_J'])*100:.3f}%.
Primary gross outgoing={m['gross_outgoing_J']:.9f} J (separate from net), minimum
observed substep margin={m['observed_substep_min_margin_J']:.6g} J. Endpoint
largest violation={m['largest_boundary_violation_J']:.3g} J. Both accounts have
identical work/gradients by design. Full per-scenario force mean/max, along/cross/
orientation, power/speed, signed task/human/residual work, failure/retry categories,
constraint activity, integrator bounds and final-second recovery metrics are
in `metrics.json`; event histories preserve accepted and rejected candidates.

The active energy demonstration has a material task cost: force transients and
temporary contact interruption. It must not be described as constant-force
sliding during intervention. Recovery is exercised through the existing FSM.
Held-out and elbow tight-budget failures expose a local one-tick feasibility
limit; budget feasibility is not monotonic and no universal boundary is claimed.
The report intentionally remains PARTIAL when these/task-quality gates fail.

## Runtime

Main full tick wall seconds (median/p95/p99/max): `{m['tick_seconds']}`.
Gradient-only active ticks: `{m['active_gradient_seconds']}`;
QP solve active ticks: `{m['active_qp_seconds']}`.
All snapshot/rollout/gradient/QP/nonlinear/retry quantiles are in `metrics.json`.
Total {m['wall_seconds']:.3f} wall seconds for {m['completed_time']:.3f} simulation
seconds (factor {m['realtime_factor']:.3f}). Benchmarks have no visualization or
synchronous per-step disk output; arrays/events are buffered, plots/files written
after timing. Memory allocation and Python logging remain in the measured path.
Median speed or a 1 ms simulated interval does not establish 1 kHz wall-clock
performance; tail latency and intervening-gradient ticks exceed the deadline.

## Audit, final gates and research decision

Three evidence-backed rounds and repairs are in `reviews.md`; exact source/test
and run provenance are in `acceptance.json`. Final figures are regenerated from
the matching final logs, including the failed trials. Numerical/control sources
must match the recorded Git revision and SHA; reporting-only repairs are tracked
separately and do not change the physics logs. Final manifest:

| gate | status |
|---|---|
{gates}

Next decision: whether the research scope should add a viability/braking-reserve
policy and stronger task constraints, rather than expecting this greedy one-tick
hard-energy QP to preserve contact under depleted budgets. Those would be explicit
extensions; this implementation does not silently add them or enlarge budgets.

References: [MuJoCo simulation/state](https://mujoco.readthedocs.io/en/stable/programming/simulation.html),
[MuJoCo API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html),
[OSQP solver](https://osqp.org/docs/solver/).
'''
    (out/"report.md").write_text(text)
    morning=f'''# 早晨摘要：{manifest['overall']}

已实现已知前臂作用点、完整状态 MuJoCo oracle、off/shadow/active、七变量关节力矩
QP、双账户、非线性候选复核、有限重试和显式失败快照。基线保持不变。

主案例 active 完整执行，净对人输出由 {off['net_outgoing_J']:.9f} J 降至
{m['net_outgoing_J']:.9f} J（减少 {(1-m['net_outgoing_J']/off['net_outgoing_J'])*100:.2f}%），
账户在控制边界的最小裕量 {m['account_min_margin_J']:.3g} J。
但法向力最大偏差 {m['force']['original_10N_max_abs']:.3f} N，离开桌面
{m['contact']['geometric_absence']['total_s']:.4f} s；保接触/任务质量不合格。
held-out 紧预算与部分其他 active 案例显式失败，不能宣称整体完成或普遍可行。
子步最小裕量 {m['observed_substep_min_margin_J']:.3g} J；端点约束不是连续时间保证。

建议先读：`docs/oracle_joint_qp/report.md`、`acceptance.json`、`reviews.md`，
再看 `active_comparison.png`；完整日志位于 `results/oracle/final/`。

```
.venv/bin/python -m pytest -q
.venv/bin/python -m kuka_table_sim.oracle_experiments case --name resistance --mode active --budget .12 --task-weight 1 --out results/oracle/reproduce
.venv/bin/python -m kuka_table_sim.oracle_experiments matrix --out results/oracle/final
```

失败试验退出码为 2，日志仍保留；不要把这个退出码当成未保存结果。
下一步需要明确决定：是否把制动能量储备/可行性维护与更强的任务约束纳入新阶段。
没有改动摩擦、接触参数、步长、名义增益或 10 N 目标来掩盖问题。
'''
    (out/"morning_summary_zh.md").write_text(morning)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path("results/oracle/final"))
    ap.add_argument("--validation",type=Path,default=Path("results/oracle/validation_final"))
    ap.add_argument("--out",type=Path,default=Path("docs/oracle_joint_qp"))
    args=ap.parse_args()
    result=generate(args.root,args.validation,args.out)
    print(result["overall"],[(g["gate"],g["status"]) for g in result["gates"]])
    return 0 if result["overall"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
