"""Opt-in execution. No changes to the frozen Simulation/controller modules.

Predictions branch BEFORE compute. Only an accepted post-compute controller is
installed, once; the live plant then integrates and observes independently.
Cached plans require exact complete-state AND measured-force equality. A plan
contains at most 20 ms of continued progress followed by the declared backup;
its reserve includes that reaction delay and every physics prefix.
"""
from dataclasses import asdict
from time import perf_counter
import json
import numpy as np
import mujoco

from .holding import Simulation
from .energy_state import HybridSnapshot,GovernedController,controller_bytes,controller_from_bytes,clone_controller
from .energy_prediction import BranchPredictor,backup_rollout,GuardConfig,Backup
from .energy_qp import filtered_interval,nonlinear_ok
from .joint_qp import FilterFailure,QPConfig


class EnergySimulation(Simulation):
    guard=None

    def reset(self):
        super().reset()
        if self.guard is not None:
            g=self.guard
            g.energy=g.initial.copy();g.previous=self.tau.copy();g.base=self.data.qfrc_applied.copy()
            g.counterfactual.reset(self.model,self.data,self.q_home)
            g.counterfactual.observe(self.model,self.data,float(self.data.time))
            g.rows=[];g.phys=[];g.events=[];g.cache=[];g.cache_steps=0
            g.cache_force=g.cache_signature=g.selected=g.snapshot=None

    def step(self):
        if self.guard is None:return super().step()
        m,d,g=self.model,self.data,self.guard;t=float(d.time)
        control=self.n_steps%self.timing.control_interval==0
        mujoco.mj_step1(m,d)
        if self.recorder is not None:self.recorder.pre(self,t)
        if control:
            proposal=g.select(self)
            self.controller=controller_from_bytes(proposal.computed_controller,m)
            d.ctrl[self.arm.actuator_ids]=proposal.torque
            self.n_control_updates+=1
        g.pre(self,t)
        mujoco.mj_step2(m,d)
        self.controller.observe(m,d,t)
        g.counterfactual.observe(m,d,t)
        if self.recorder is not None:self.recorder.post(self,t,control)
        g.post(self,t)
        self.n_steps+=1


class EnergyGuard:
    def __init__(self,sim,variant,pulse,body,local,energy,minimum=(0.,0.),config=None):
        if variant not in ('B','C','D'):raise ValueError(variant)
        self.variant=variant;self.pulse=pulse;self.cfg=config or GuardConfig()
        self.energy=np.asarray(energy,dtype=float).copy();self.minimum=np.asarray(minimum,dtype=float).copy()
        if self.energy.shape!=(2,) or np.any(self.energy<self.minimum):raise ValueError('invalid accounts')
        self.initial=self.energy.copy();self.base=sim.data.qfrc_applied.copy()
        self.previous=sim.data.ctrl[sim.arm.actuator_ids].copy()
        self.counterfactual=clone_controller(sim.controller)
        if variant in ('C','D'):sim.controller=GovernedController.from_controller(sim.controller,self.cfg.transition)
        sim.__class__=EnergySimulation;sim.guard=self
        self.predictor=BranchPredictor(sim,body,local,self.cfg)
        self.rows=[];self.phys=[];self.events=[];self.cache=[];self.cache_force=None;self.cache_signature=None
        self.cache_steps=0
        self.selected=None;self.force=np.zeros(3);self.snapshot=None

    def plan(self,snapshot,force,target):
        """20 ms reaction-delay backup; finite horizon begins after this delay.
        All states/observers evolve closed-loop. Invalid/unsettled tails reject
        the WHOLE plan, even if the first command alone has enough energy.
        """
        first,attempts,timing=filtered_interval(self.predictor,snapshot,force,target,
                                               self.variant=='D',self.minimum)
        path=[first];current=first.next
        # A delayed backup permits exact plan reuse under a held measured force.
        # A zero target starts braking now; it does not wait for a later trigger.
        if target>0:
            for _ in range(19):
                p=self.predictor.interval(current,force,target)
                rate=np.asarray(QPConfig().rate)*self.cfg.rate_factor*.001
                if (not self.predictor.task_ok(p.trace) or
                    np.any(np.abs(p.torque-current.previous)>rate+self.cfg.torque_tolerance)):
                    raise FilterFailure('reaction_delay_task_or_rate')
                path.append(p);current=p.next
        tail=backup_rollout(self.predictor,current,force)
        if tail.status!='SETTLED':
            raise FilterFailure('backup_'+tail.status.lower(),dict(reason=tail.reason,reserve=tail.reserve,horizons=tail.horizons,
                                 elapsed_ticks=len(tail.intervals),timing=timing))
        return path+tail.intervals,attempts,dict(**timing,backup=tail.seconds),tail

    @staticmethod
    def reserve(path):
        work=0.;minimum=0.
        for p in path:
            minimum=min(minimum,work+float(np.min(p.trace['prefix'])));work+=p.work
        return max(0.,-minimum)

    def select(self,sim):
        start=perf_counter();t=float(sim.data.time)
        self.force=np.asarray(self.pulse.force(t),float) # ONLY real control-boundary measurement
        snap=HybridSnapshot.capture(sim,self.energy,self.previous,self.base);self.snapshot=snap
        attempts=[];reserve=0.;backup_status='not_used';horizons=[];timing={};cache_hit=False
        if self.variant=='B' or not np.any(self.force):
            selected,attempts,timing=filtered_interval(self.predictor,snap,self.force,
                                                      1. if self.variant!='B' else None,
                                                      self.variant=='D',self.minimum)
            # Zero reserve only under the primary guaranteed held-zero fixture.
            backup_status='held_zero' if self.variant!='B' else 'not_used'
            self.cache=[]
        else:
            valid_cache=(bool(self.cache) and self.cache_steps>0 and np.array_equal(self.cache_force,self.force) and
                         self.cache_signature==snap.signature())
            path=None;tail=None;rejections=[]
            if valid_cache:
                reserve=self.reserve(self.cache)
                valid_cache=bool(np.all(self.energy-self.minimum>=reserve+self.cfg.reserve_margin))
            if valid_cache:
                path=self.cache;cache_hit=True;backup_status='SETTLED_cached'
            else:
                for target in self.cfg.targets:
                    try:
                        candidate,qa,tm,bt=self.plan(snap,self.force,target)
                        required=self.reserve(candidate)
                        if np.any(self.energy-self.minimum<required+self.cfg.reserve_margin):
                            raise FilterFailure('reserve_budget',dict(required=required,available=(self.energy-self.minimum).tolist()))
                        path,attempts,timing,tail=candidate,qa,tm,bt
                        self.cache_steps=20 if target>0 else len(candidate)
                        reserve=required;backup_status=bt.status;horizons=bt.horizons
                        break
                    except (FilterFailure,ValueError) as exc:
                        rejections.append(dict(target=target,reason=getattr(exc,'reason',str(exc)),diagnostics=getattr(exc,'diagnostics',{})))
                if path is None:
                    self.events.append(dict(t=t,rejections=rejections))
                    raise FilterFailure('no_verified_reserve_candidate',dict(rejections=rejections))
            proposed=path[0]
            # Re-evaluate even cached actions from CURRENT complete state.
            clock=controller_from_bytes(proposed.computed_controller,sim.model).stroke.clock
            selected=self.predictor.interval(snap,self.force,clock.target,proposed.torque)
            if not np.allclose(selected.next.plant,proposed.next.plant,atol=1e-11,rtol=0):
                raise FilterFailure('cache_replay_mismatch')
            self.cache=path[1:];self.cache_force=self.force.copy()
            self.cache_steps-=1
            self.cache_signature=selected.next.signature()
            if rejections:self.events.append(dict(t=t,rejections=rejections))
        ok,why=nonlinear_ok(selected,self.energy,self.minimum,self.predictor,self.variant=='D')
        if not ok:raise FilterFailure('final_'+why)
        rate=np.asarray(QPConfig().rate)*self.cfg.rate_factor*.001
        if np.any(np.abs(selected.torque-self.previous)>rate+self.cfg.torque_tolerance):raise FilterFailure('final_rate')
        cf=clone_controller(self.counterfactual);ungoverned=cf.compute(sim.data)
        self.counterfactual=cf # committed only after selection
        ctl=controller_from_bytes(selected.computed_controller,sim.model)
        clock=getattr(ctl.stroke,'clock',None)
        self.row=dict(t=t,energy_before=self.energy.copy(),tau_nom=selected.nominal.copy(),tau_exec=selected.torque.copy(),
                      tau_ungoverned=ungoverned.copy(),tau_previous=self.previous.copy(),
                      x_ungoverned=cf.x_ref.copy(),
                      alpha=clock.alpha if clock else 1.,alpha_dot=clock.alpha_dot if clock else 0.,
                      alpha_target=clock.target if clock else 1.,sigma=clock.sigma if clock else t,
                      reserve=reserve,backup_status=backup_status,reserve_margin=self.cfg.reserve_margin,
                      predicted=selected.work,predicted_prefix=selected.trace['prefix'].copy(),
                      cache_hit=cache_hit,seconds_select=perf_counter()-start,**{'seconds_'+k:v for k,v in timing.items()})
        if attempts or horizons:self.events.append(dict(t=t,attempts=attempts,horizons=horizons))
        self.work=0.;self.gross=0.;self.task_work=0.;self.min_energy=self.energy.copy();self.selected=selected
        self.previous=selected.torque.copy();self.stage=0
        return selected

    def pre(self,sim,t):
        point,velocity,tau,power,identity=self.predictor.port.apply(sim.data,self.force,self.base)
        ctl=sim.controller;tcp=self.predictor.tcp
        self.sample=dict(t=t,point=point.copy(),velocity=velocity.copy(),force=self.force.copy(),power=power,power_identity=identity,
                         qvel=sim.data.qvel.copy(),human_tau=tau.copy(),stage=self.stage,tick=sim.n_control_updates-1,
                         orientation=ctl.orientation_error(sim.data.site_xmat[tcp].reshape(3,3)),
                         cross_error=float((sim.data.site_xpos[tcp]-ctl.x_ref)@np.cross(ctl.normal,ctl.slide_dir)))

    def post(self,sim,t):
        s=self.sample;dt=sim.timing.timestep;dw=s['power']*dt
        self.energy+=dw;self.work+=dw;self.gross+=max(0.,-dw)
        self.min_energy=np.minimum(self.min_energy,self.energy)
        task=self.predictor.meter.sample(sim.data);self.task_work+=task['power']*dt
        s.update(work=dw,energy=self.energy.copy(),normal_force=task['F'][2],present=task['nt']>0,
                 unexpected=task['nu'],torque=sim.data.qfrc_actuator[sim.arm.dof_adr].copy())
        self.phys.append(s)
        if not np.allclose(s['torque'],self.previous,rtol=0,atol=1e-10):raise FilterFailure('actual_torque_mismatch')
        if np.min(self.energy-self.minimum)<-self.cfg.energy_tolerance:raise FilterFailure('executed_energy_prefix')
        self.stage+=1
        if self.stage==sim.timing.control_interval:
            self.row.update(actual=self.work,gross=self.gross,task_work=self.task_work,energy_after=self.energy.copy(),min_prefix=self.min_energy.copy())
            self.rows.append(self.row)

    def arrays(self):
        keys=set().union(*(r.keys() for r in self.rows)) if self.rows else set()
        c={k:np.asarray([r.get(k,0.) for r in self.rows]) for k in keys}
        p={k:np.asarray([r[k] for r in self.phys]) for k in self.phys[0]} if self.phys else {}
        return c,p

    def save_failure(self,path,sim):
        now=HybridSnapshot.capture(sim,self.energy,self.previous,self.base)
        extra=dict(force=self.force,minimum=self.minimum,config_json=np.array(json.dumps(asdict(self.cfg))),
                   counterfactual_blob=np.frombuffer(controller_bytes(self.counterfactual),dtype=np.uint8),
                   cache_force=self.cache_force if self.cache_force is not None else np.zeros(3),
                   cache_signature=np.array(self.cache_signature or ''),cache_steps=np.array(self.cache_steps),
                   cache_torques=np.asarray([p.torque for p in self.cache]),
                   cache_targets=np.asarray([controller_from_bytes(p.computed_controller,sim.model).stroke.clock.target for p in self.cache]))
        now.save(path/'failure_complete.npz',**extra)
        if self.snapshot is not None:self.snapshot.save(path/'failure_pre.npz',**extra)
