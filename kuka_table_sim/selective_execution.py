"""Independent execution and buffered evidence for selective comparisons."""
from time import perf_counter
import numpy as np
import mujoco
from .holding import Simulation
from .energy_state import HybridSnapshot,GovernedController,controller_from_bytes
from .oracle_physics import HumanPort,TaskMeter,integration_state
from .selective_ports import TaskModel,capped_update,account_powers
from .selective_prediction import SelectivePredictor
from .selective_qp import filtered,constraints
from .joint_qp import FilterFailure


class SelectiveSimulation(Simulation):
    def step(self):
        m,d,g=self.model,self.data,self.guard;t=float(d.time)
        control=self.n_steps%4==0
        mujoco.mj_step1(m,d)
        if self.recorder is not None:self.recorder.pre(self,t)
        if control:
            p=g.select(self)
            self.controller=controller_from_bytes(p.computed_controller,m)
            d.ctrl[self.arm.actuator_ids]=p.torque
            self.n_control_updates+=1
        g.pre(self,t)
        mujoco.mj_step2(m,d)
        self.controller.observe(m,d,t)
        if self.recorder is not None:self.recorder.post(self,t,control)
        g.post(self,t)
        self.n_steps+=1


class SelectiveGuard:
    def __init__(self,sim,mode,profile,body,local,friction,cfg):
        self.mode,self.profile,self.cfg=mode,profile,cfg
        sim.controller=GovernedController.from_controller(sim.controller,cfg.governor_transition_s)
        sim.__class__=SelectiveSimulation;sim.guard=self
        self.predictor=SelectivePredictor(sim,body,local,friction,cfg)
        # Executed work never comes from the selected rollout's trace.
        self.task_model=TaskModel(sim.model,body,local,friction)
        self.port=HumanPort(sim.model,body,local);self.meter=TaskMeter(sim.model)
        self.energy=np.asarray(cfg.initial).copy();self.previous=sim.data.ctrl[sim.arm.actuator_ids].copy()
        self.base=sim.data.qfrc_applied.copy();self.force=np.zeros(3);self.last_force=np.zeros(3)
        self.paused=False;self.rows=[];self.phys=[];self.events=[];self.snapshot=None

    def select(self,sim):
        started=perf_counter();t=float(sim.data.time)
        self.force=np.asarray(self.profile.force(t),float)
        if self.force.shape!=(3,) or not np.all(np.isfinite(self.force)):raise ValueError('measured force')
        increment=float(np.linalg.norm(self.force-self.last_force));self.last_force=self.force.copy()
        if increment>self.cfg.force_rate_N_s*.001+1e-12:raise FilterFailure('force_change_class_exceeded',dict(increment=increment))
        if self.mode=='K4' and np.linalg.norm(self.force)<=self.cfg.release_force_N:self.paused=False
        target=0. if self.mode=='K4' and self.paused else 1.
        self.snapshot=HybridSnapshot.capture(sim,self.energy,self.previous,self.base)
        try:p,attempts,timing=filtered(self.predictor,self.snapshot,self.force,self.mode,target)
        except FilterFailure as exc:
            self.events.append(dict(t=t,failed=True,reason=exc.reason,**exc.diagnostics));raise
        self.selected=p
        intervened=bool(np.max(np.abs(p.torque-p.nominal))>self.cfg.torque_tolerance_Nm)
        if self.mode=='K4' and intervened and timing['human_trigger']:self.paused=True
        ctl=controller_from_bytes(p.computed_controller,sim.model);clock=ctl.stroke.clock
        names,v,tol=constraints(p,self.snapshot,self.force,self.mode,self.cfg)
        self.row=dict(t=t,energy_before=self.energy.copy(),tau_nom=p.nominal.copy(),tau_exec=p.torque.copy(),
                      tau_previous=self.previous.copy(),alpha=clock.alpha,sigma=clock.sigma,alpha_target=clock.target,
                      intervention=intervened,qp_attempts=len(attempts),human_trigger=timing['human_trigger'],
                      force_increment=increment,min_constraint=float(np.min(v)) if len(v) else 0.,
                      seconds_select=perf_counter()-started,seconds_gradient=timing['gradient'],seconds_qp=timing['qp'],
                      predicted_work=np.sum(p.trace['power'],axis=0)*sim.timing.timestep,
                      predicted_energy=p.next.energy.copy())
        if attempts:self.events.append(dict(t=t,failed=False,attempts=attempts))
        return p

    def pre(self,sim,t):
        self.hat=self.task_model.sample(integration_state(sim.model,sim.data),self.selected.torque,self.force,self.base)
        self.sample=self.port.apply(sim.data,self.force,self.base)
        self.q,self.qd=sim.data.qpos.copy(),sim.data.qvel.copy()

    def post(self,sim,t):
        task=self.meter.sample(sim.data);point,vh,ht,ph,identity=self.sample
        p,delta=account_powers(ph,task['power'],self.hat['power']);before=self.energy.copy()
        self.energy,discard=capped_update(before,p*sim.timing.timestep,self.cfg.caps)
        j=sim.n_steps%4;prediction=self.selected.trace
        errors=np.array([np.max(np.abs(p-prediction['power'][j])),np.max(np.abs(self.energy-prediction['energy'][j]))])
        actual_torque=sim.data.qfrc_actuator[sim.arm.dof_adr].copy()
        self.phys.append(dict(t=t,power=p,delta=delta,task_power=task['power'],task_hat_power=self.hat['power'],
            energy_before=before,energy=self.energy.copy(),discarded=discard,present=task['nt']>0,unexpected=task['nu'],
            normal_force=task['F'][2],force=self.force.copy(),human_velocity=vh.copy(),point=point.copy(),
            human_tau=ht.copy(),power_identity=identity,q=self.q,qd=self.qd,torque=actual_torque,
            task_force=task['F'].copy(),task_hat_force=self.hat['F'].copy(),prediction_error=errors))
        self.phys[-1].update(task_moment=task['M'].copy(),task_point_velocity=task['vp'].copy(),
            task_omega=task['omega'].copy(),task_hat_moment=self.hat['M'].copy(),
            task_hat_point_velocity=self.hat['vp'].copy(),task_hat_omega=self.hat['omega'].copy())
        if errors[0]>1e-7 or errors[1]>1e-10:raise FilterFailure('executed_prediction_mismatch',dict(errors=errors))
        if np.max(np.abs(actual_torque-self.selected.torque))>1e-10:raise FilterFailure('actuator_mismatch')
        if j==3:
            self.row['energy']=self.energy.copy();self.rows.append(self.row);self.previous=self.selected.torque.copy()
            if np.max(np.abs(integration_state(sim.model,sim.data)-self.selected.next.plant))>1e-10:raise FilterFailure('state_replay_mismatch')

    def arrays(self):
        def pack(rows):return {k:np.asarray([r[k] for r in rows]) for k in rows[0]} if rows else {}
        return pack(self.rows),pack(self.phys)
