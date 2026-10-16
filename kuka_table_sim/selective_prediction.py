"""Causal joint-torque oracle for genuinely different H/R/whole-port ledgers.

Plant dynamics are an explicit simulation oracle; the nominal TASK contact
model is independent and may be mismatched. This tests selective accounting,
not robustness to an unknown plant. No future scripted force is accessible.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import mujoco
from .energy_state import HybridSnapshot,controller_bytes
from .oracle_physics import HumanPort,TaskMeter,integration_state,validate_oracle_model
from .selective_ports import TaskModel,capped_update,account_powers


@dataclass
class Proposal:
    next: HybridSnapshot
    computed_controller: bytes
    nominal: np.ndarray
    torque: np.ndarray
    trace: dict
    terminal_human_velocity: np.ndarray
    y: np.ndarray
    seconds: float


class SelectivePredictor:
    def __init__(self,sim,body,local,nominal_friction,cfg):
        self.template=sim;self.model=sim.model;self.arm=sim.arm;self.cfg=cfg
        self.timing=sim.timing
        if sim.timing.control_interval!=4 or abs(sim.timing.timestep-.00025)>1e-15:raise ValueError('fixed split stepping required')
        validate_oracle_model(sim.model,sim.arm)
        self.port=HumanPort(sim.model,body,local);self.meter=TaskMeter(sim.model)
        self.task_model=TaskModel(sim.model,body,local,nominal_friction)
        self.tcp=sim.model.site('tcp').id
        self.jp=np.empty((3,7));self.jr=np.empty((3,7))

    def interval(self,snapshot,force,target=1.,torque=None):
        started=perf_counter();m=self.model;branch=snapshot.fork(self.template);d=branch.data;ctl=branch.controller
        mujoco.mj_step1(m,d)
        ctl.stroke.clock.advance(float(d.time));ctl.stroke.clock.set_target(target)
        nominal=ctl.compute(d)
        torque=nominal.copy() if torque is None else np.asarray(torque,dtype=float).copy()
        if np.any(np.abs(torque)>self.arm.torque_limit+1e-10):raise ValueError('candidate exceeds actuator limits')
        ctl.filter_intervened=bool(np.max(np.abs(torque-nominal))>self.cfg.torque_tolerance_Nm)
        computed=controller_bytes(ctl);d.ctrl[self.arm.actuator_ids]=torque
        E=snapshot.energy.copy();rows=[]
        for j in range(4):
            t=float(d.time);mujoco.mj_step1(m,d)
            state=integration_state(m,d)
            hat=self.task_model.sample(state,torque,force,snapshot.base_qfrc)
            point,vh,human_tau,ph,identity=self.port.apply(d,force,snapshot.base_qfrc)
            q,qd=d.qpos.copy(),d.qvel.copy()
            mujoco.mj_step2(m,d);ctl.observe(m,d,t)
            task=self.meter.sample(d)
            p,delta=account_powers(ph,task['power'],hat['power'])
            before=E.copy();E,discarded=capped_update(E,p*self.timing.timestep,self.cfg.caps)
            if not np.allclose(d.qfrc_actuator[self.arm.dof_adr],torque,atol=1e-10,rtol=0):raise RuntimeError('actuator mismatch')
            rows.append(dict(t=t,power=p,delta=delta,task_power=task['power'],task_hat_power=hat['power'],
                 normal_force=task['F'][2],present=task['nt']>0,unexpected=task['nu'],
                 energy_before=before,energy=E.copy(),discarded=discarded,point=point.copy(),
                 human_velocity=vh.copy(),human_tau=human_tau.copy(),power_identity=identity,
                 force=force.copy(),q=q,qd=qd,stage=j,torque=torque.copy(),
                 task_force=task['F'].copy(),task_hat_force=hat['F'].copy()))
        trace={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        final=integration_state(m,d)
        next_state=HybridSnapshot(final,controller_bytes(ctl),E,torque.copy(),snapshot.base_qfrc.copy(),snapshot.n_steps+4,snapshot.n_updates+1)
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        vh_terminal=self.port.sample(d,force)[1].copy()
        mujoco.mj_jacSite(m,d,self.jp,self.jr,self.tcp)
        y=np.r_[np.mean(trace['normal_force']),self.jp@d.qvel,self.jr@d.qvel]
        return Proposal(next_state,computed,nominal.copy(),torque,trace,vh_terminal,y,perf_counter()-started)
