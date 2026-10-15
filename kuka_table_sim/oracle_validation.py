"""Finite-difference and one-sided affine error validation, separate calibration
and held-out trajectories; independent execution uses mj_step and J^T f.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from .joint_qp import QPConfig
from .oracle import OracleFilter
from .oracle_experiments import save_json, source_hash, fixture
from .oracle_physics import HumanPort, HumanPulse, OraclePredictor, integration_state, restore_state


def execution_copy(model, timing, state, torque, force, body, local):
    d = mujoco.MjData(model)
    restore_state(model, d, state)
    base = d.qfrc_applied.copy()
    bid = model.body(body).id
    jp = np.empty((3, model.nv))
    d.ctrl[:] = torque
    wh = wt = 0.
    for _ in range(timing.control_interval):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        point = d.xpos[bid] + d.xmat[bid].reshape(3,3) @ local
        mujoco.mj_jac(model, d, jp, None, point, bid)
        qv = d.qvel.copy()
        wh += float(force @ (jp @ qv)) * timing.timestep
        d.qfrc_applied[:] = base + jp.T @ force
        mujoco.mj_step(model, d)
        wt += float(qv @ d.qfrc_constraint) * timing.timestep
    return np.array([wh,wh]), wt, d.qpos.copy(), d.qvel.copy()


def run_validation(out):
    from .scenarios import make_force_sim
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = QPConfig()
    rng = np.random.default_rng(7341)
    records, snapshots, summary = [], [], []
    # Force direction change and alternate point are separate measured fixtures.
    cases = [
        ("calibration", "resistance", [9., 9.6, 10.1, 10.75, 11.1], None),
        ("calibration", "direction-change", [12.2, 12.7, 13.1],
         HumanPulse(start=12., ramp=.2,hold=1.,magnitude=5.,direction=(1.,0.,0.))),
        ("held-out", "held-out", [10.2,10.6,11.05,11.3],None),
        ("held-out", "elbow", [9.6,10.1,10.75],None),
    ]
    for split, name, times, override in cases:
        pulse, body, local = fixture("resistance" if override else name)
        pulse = override or pulse
        sim = make_force_sim(record=False)
        sim.oracle = OracleFilter(sim,"off",pulse,body,local)
        predictor = OraclePredictor(sim.model,sim.timing,HumanPort(sim.model,body,local))
        for target in times:
            sim.advance(int(round((target - sim.data.time)/sim.timing.timestep)))
            mujoco.mj_step1(sim.model,sim.data)
            sim.data.qfrc_applied[:] = sim.oracle.base_qfrc
            state = integration_state(sim.model,sim.data)
            tau = sim.tau.copy()  # current held command, center declared explicitly
            force = pulse.force(target)
            lins = [predictor.linearize(state,tau,force,e,sim.oracle.base_qfrc) for e in (.01,.02,.04)]
            lin = lins[1]
            gscale_abs = max(float(np.max(np.abs(l.gradients-lin.gradients))) for l in lins)
            gmag = float(np.max(np.abs(lin.gradients)))
            sid = len(snapshots)
            snapshots.append(dict(state=state, tau=tau,force=force))
            directions = list(np.eye(7))
            # Mixed directions include all joints; finite samples are empirical.
            directions += [rng.uniform(-1.,1.,7) for _ in range(4)]
            replay_error = qerr = verr = maxover = maxabs = 0.
            worst = None
            for di, direction in enumerate(directions):
                delta_dir = np.asarray(cfg.rho) * direction
                for frac in (-1.,-.5,-.1,0.,.1,.5,1.):
                    delta = frac * delta_dir
                    command = tau + delta
                    if np.any(np.abs(command) > sim.arm.torque_limit):
                        continue
                    pred = predictor.predict(state, command, force, sim.oracle.base_qfrc)
                    actual, task, q, v = execution_copy(sim.model,sim.timing,state,command,force,body,local)
                    final = mujoco.MjData(sim.model)
                    restore_state(sim.model,final,pred.final_state)
                    qerr = max(qerr,float(np.max(np.abs(q-final.qpos))))
                    verr = max(verr,float(np.max(np.abs(v-final.qvel))))
                    affine = lin.prediction.work + lin.gradients @ delta
                    err = float(affine[0] - actual[0])
                    if err > maxover:
                        maxover, worst = err, dict(direction=di, fraction=frac, delta=delta.tolist())
                    maxabs = max(maxabs,abs(err))
                    replay_error = max(replay_error,float(np.max(np.abs(actual-pred.work))))
                    records.append(dict(snapshot=sid,direction=di,fraction=frac,delta=delta,
                                        prediction=pred.work[0], execution=actual[0],affine=affine[0],
                                        task_prediction=pred.task_work,task_execution=task))
            # A-B-A checks full state; no data, controller or recorder mutations allowed.
            p0 = predictor.predict(state,tau,force,sim.oracle.base_qfrc)
            predictor.predict(state,tau+np.ones(7)*.5,force,sim.oracle.base_qfrc)
            p1 = predictor.predict(state,tau,force,sim.oracle.base_qfrc)
            isolated = np.array_equal(integration_state(sim.model,sim.data),state)
            repeat = np.array_equal(p0.final_state,p1.final_state)
            summary.append(dict(id=sid,split=split,case=name,t=target,body=body,force=force.tolist(),
                                gradient_max_J_Nm=gmag,eta_max_difference_J_Nm=gscale_abs,
                                max_one_sided_affine_over_J=maxover,max_abs_affine_error_J=maxabs,
                                replay_max_work_J=replay_error,replay_max_q_rad=qerr,replay_max_qvel_rad_s=verr,
                                isolated=bool(isolated),candidate_order_invariant=bool(repeat),worst=worst))
    np.savez_compressed(out/"scans.npz",**{k:np.asarray([r[k] for r in records]) for k in records[0]},
                        **{"snap__"+k:np.asarray([s[k] for s in snapshots]) for k in snapshots[0]})
    def normalized(r):
        return r["max_one_sided_affine_over_J"] * cfg.margin_force_reference / max(np.linalg.norm(r["force"]),1e-30)
    calibration = max(normalized(r) for r in summary if r["split"]=="calibration")
    heldout = max(normalized(r) for r in summary if r["split"]=="held-out")
    report = dict(source_sha256=source_hash(),seed=7341,eta_Nm=[.01,.02,.04],rho_Nm=cfg.rho,
                  samples=len(records),snapshots=summary,calibration_max_over_J=calibration,
                  held_out_max_over_J=heldout,margin_note="max errors normalized to 5 N; zero force exact",
                  epsilon_prediction_J=cfg.epsilon_prediction,
                  epsilon_linearization_J=cfg.epsilon_linearization,
                  status="PASS" if max(calibration,heldout)<=cfg.epsilon_linearization and
                  all(r["isolated"] and r["candidate_order_invariant"] and r["replay_max_work_J"]<1e-11 and
                      r["replay_max_q_rad"]<1e-11 and r["replay_max_qvel_rad_s"]<1e-10 for r in summary) else "FAIL")
    save_json(out/"validation.json",report)
    plot_scans(out)
    return report


def plot_scans(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    z = np.load(out/"scans.npz")
    # Resistance onset, steady resistance, direction change, held-out.
    ids = [1,2,6,9]
    fig,axs = plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax, sid in zip(axs.flat,ids):
        # Show a mixed-joint direction alongside all-direction one-sided errors.
        sel=(z["snapshot"]==sid)&(z["direction"]==7)
        ax.plot(z["fraction"][sel],z["execution"][sel]*1e6,"o",label="execution copy")
        ax.plot(z["fraction"][sel],z["prediction"][sel]*1e6,"-",label="nonlinear oracle")
        ax.plot(z["fraction"][sel],z["affine"][sel]*1e6,"--",label="affine")
        ax.set(xlabel="fraction of mixed trust-region direction",ylabel="human work (µJ)",title=f"Snapshot {sid}")
        ax.legend(fontsize=8)
    fig.savefig(out/"gradient_response.png",dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5),constrained_layout=True)
    ax.scatter(z["snapshot"],(z["affine"]-z["execution"])*1e6,s=5,alpha=.4)
    ax.axhline(QPConfig().epsilon_linearization*1e6,color="r",ls="--",label="declared empirical margin")
    ax.set(xlabel="snapshot id (0–7 calibration; 8–14 held out)",ylabel="affine − actual work (µJ)")
    ax.legend()
    fig.savefig(out/"one_sided_error.png",dpi=150)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=Path("results/oracle/validation"))
    args=ap.parse_args()
    r=run_validation(args.out)
    print({k:v for k,v in r.items() if k!="snapshots"})
    return 0 if r["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
