"""Stage 2 one-interval proposals and causal closed-loop backup rollouts.

Only a measured Cartesian force VECTOR enters this module. There is no pulse
object or future force query. All controller, sensor, FSM and clock state evolves
in an isolated branch with the same 1 ms / four 0.25 ms stepping contract.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from time import perf_counter
import numpy as np
import mujoco

from .energy_state import HybridSnapshot,controller_bytes,controller_from_bytes
from .oracle_physics import HumanPort,TaskMeter,integration_state,validate_oracle_model
from .joint_qp import QPConfig


@dataclass(frozen=True)
class GuardConfig:
    transition: float=.1
    damping: float=40.             # backup human-point damping, N s/m
    horizons: tuple=(.02,.05,.1,.25,.5,1.,2.)
    terminal_persistence: float=.02
    terminal_speed: float=.001     # human point norm, m/s
    terminal_power: float=.002     # abs human power, W
    terminal_tool_speed: float=.001
    terminal_angular_speed: float=.01
    energy_margin: float=1e-8      # deterministic held-input numerical resolution; measured in tests
    reserve_margin: float=1e-5     # empirical planning guard, J; not a future-force guarantee
    force_lower: float=7.
    force_upper: float=13.
    normal_speed: float=.005
    orientation: float=2.*np.pi/180
    cross_error: float=.003
    task_start: float=8.71
    energy_tolerance: float=1e-10
    torque_tolerance: float=1e-7
    task_tolerance: float=1e-7
    eta: float=.02
    retries: int=3
    targets: tuple=(1.,.5,0.)
    task_weight: float=1.
    rate_factor: float=1.

    def __post_init__(self):
        positive=('transition','damping','terminal_persistence','terminal_speed','terminal_power',
                  'terminal_tool_speed','terminal_angular_speed','energy_margin','reserve_margin',
                  'normal_speed','orientation','cross_error','eta','rate_factor')
        if any(not np.isfinite(getattr(self,k)) or getattr(self,k)<=0 for k in positive):raise ValueError('finite positive guard configuration required')
        if self.task_weight!=1.:raise ValueError('matched A/B/C/D task weight is frozen at 1')
        if not self.horizons or tuple(sorted(self.horizons))!=tuple(self.horizons) or min(self.horizons)<=0:raise ValueError('ordered positive horizons required')
        if self.force_lower>=self.force_upper or not self.targets or self.targets[0]!=1 or self.targets[-1]!=0 or any(not 0<=a<=1 for a in self.targets):raise ValueError('invalid task band or progression search')


@dataclass
class Interval:
    next: HybridSnapshot
    computed_controller: bytes
    nominal: np.ndarray
    torque: np.ndarray
    trace: dict
    y: np.ndarray
    seconds: float

    @property
    def work(self): return float(self.trace['prefix'][-1])


class BranchPredictor:
    def __init__(self,sim,body,local,config):
        self.model,self.timing,self.arm=sim.model,sim.timing,sim.arm
        self.template=sim
        self.cfg=config
        if (self.timing.control_interval!=4 or abs(self.timing.timestep-.00025)>1e-15 or
            abs(self.timing.control_period-.001)>1e-15):raise ValueError('Stage 2 requires 1 ms / four 0.25 ms steps')
        if sim.disturbance is not None or sim.oracle is not None:raise ValueError('unsupported external execution hook')
        self.port=HumanPort(self.model,body,local)
        self.meter=TaskMeter(self.model)
        self.tcp=self.model.site('tcp').id
        validate_oracle_model(self.model,self.arm)
        for name in ('contactfilter','time'):
            get=getattr(mujoco,'get_mjcb_'+name,None)
            if get is not None and get() is not None: raise ValueError('stateful global callback '+name)
        self.jp=np.zeros((3,self.model.nv));self.jr=np.zeros_like(self.jp)

    def prepare(self,snapshot,target=None):
        branch=snapshot.fork(self.template)
        mujoco.mj_step1(self.model,branch.data)
        ctl=branch.controller
        if hasattr(ctl.stroke,'clock'):
            ctl.stroke.clock.advance(float(branch.data.time))
            if target is not None: ctl.stroke.clock.set_target(target)
        tau=ctl.compute(branch.data)
        return branch,tau

    def interval(self,snapshot,force,target=None,torque=None,damping=0.):
        start=perf_counter();branch,nominal=self.prepare(snapshot,target)
        if torque is None:
            torque=nominal.copy()
            if damping:
                velocity=self.port.sample(branch.data,force)[1]
                torque-=self.port.jac.T@(damping*velocity)
        return self.advance(branch,snapshot,nominal,np.asarray(torque),force,start)

    def advance(self,branch,snapshot,nominal,torque,force,start=None):
        start=perf_counter() if start is None else start
        m,d,ctl=self.model,branch.data,branch.controller
        if np.any(np.abs(torque)>self.arm.torque_limit+1e-10):
            raise ValueError('candidate torque outside actuator bounds')
        ctl.filter_intervened=bool(np.max(np.abs(torque-nominal))>self.cfg.torque_tolerance)
        computed=controller_bytes(ctl)
        d.ctrl[self.arm.actuator_ids]=torque
        rows=[];work=0.
        for j in range(self.timing.control_interval):
            t=float(d.time);mujoco.mj_step1(m,d)
            point,vh,th,ph,identity=self.port.apply(d,force,snapshot.base_qfrc)
            mujoco.mj_jacSite(m,d,self.jp,self.jr,self.tcp)
            v=self.jp@d.qvel;omega=self.jr@d.qvel
            ori=ctl.orientation_error(d.site_xmat[self.tcp].reshape(3,3))
            x=d.site_xpos[self.tcp].copy()
            cross=float((x-ctl.x_ref)@np.cross(ctl.normal,ctl.slide_dir))
            q=d.qpos.copy();qd=d.qvel.copy()
            mujoco.mj_step2(m,d)
            ctl.observe(m,d,t)
            task=self.meter.sample(d)
            if not np.allclose(d.qfrc_actuator[self.arm.dof_adr],torque,rtol=0,atol=1e-10):
                raise RuntimeError('actual torque mismatch')
            work+=ph*self.timing.timestep
            rows.append(dict(t=t,point=point.copy(),velocity=vh.copy(),power=ph,power_identity=identity,
                             prefix=work,normal_force=task['F'][2],normal_velocity=v[2],orientation=ori,
                             cross_error=cross,present=task['nt']>0,unexpected=task['nu'],
                             tool_velocity=v,angular_velocity=omega,task_power=task['power'],q=q,qd=qd,x=x))
        trace={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        next_state=HybridSnapshot(integration_state(m,d),controller_bytes(ctl),snapshot.energy+work,
                                  torque.copy(),snapshot.base_qfrc.copy(),snapshot.n_steps+4,snapshot.n_updates+1)
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        mujoco.mj_jacSite(m,d,self.jp,self.jr,self.tcp)
        y=np.r_[np.mean(trace['normal_force']),self.jp@d.qvel,self.jr@d.qvel]
        return Interval(next_state,computed,nominal.copy(),torque.copy(),trace,y,perf_counter()-start)

    def task_ok(self,trace):
        active=trace['t']>=self.cfg.task_start-1e-10
        if not np.any(active): return True
        return bool(np.all(trace['present'][active]) and np.max(trace['unexpected'][active])==0 and
                    np.min(trace['normal_force'][active])>=self.cfg.force_lower-self.cfg.task_tolerance and
                    np.max(trace['normal_force'][active])<=self.cfg.force_upper+self.cfg.task_tolerance and
                    np.max(np.abs(trace['normal_velocity'][active]))<=self.cfg.normal_speed+self.cfg.task_tolerance and
                    np.max(np.linalg.norm(trace['orientation'][active],axis=1))<=self.cfg.orientation+self.cfg.task_tolerance and
                    np.max(np.abs(trace['cross_error'][active]))<=self.cfg.cross_error+self.cfg.task_tolerance)


@dataclass
class Backup:
    status: str
    reserve: float
    intervals: list
    horizons: list
    force: np.ndarray
    signature: str
    seconds: float
    reason: str|None=None


def backup_rollout(predictor,snapshot,force):
    """Reserve is worst PHYSICS prefix of THIS backup, never added to energy."""
    start=perf_counter();cfg=predictor.cfg
    current=snapshot;work=0.;minimum=0.;intervals=[];study=[];settled=0;checkpoint=0
    max_ticks=round(cfg.horizons[-1]/predictor.timing.control_period)
    rate=np.asarray(QPConfig().rate)*cfg.rate_factor*predictor.timing.control_period
    branch,nominal=predictor.prepare(snapshot,0.)
    for k in range(max_ticks):
        try:
            if k:
                mujoco.mj_step1(predictor.model,branch.data)
                if hasattr(branch.controller.stroke,'clock'):
                    branch.controller.stroke.clock.advance(float(branch.data.time))
                    branch.controller.stroke.clock.set_target(0.)
                nominal=branch.controller.compute(branch.data)
            velocity=predictor.port.sample(branch.data,force)[1]
            torque=nominal-predictor.port.jac.T@(cfg.damping*velocity)
            p=predictor.advance(branch,current,nominal,torque,force)
        except (ValueError,RuntimeError) as exc:
            return Backup('INVALID',max(0.,-minimum),intervals,study,force.copy(),snapshot.signature(),perf_counter()-start,str(exc))
        prefix=work+p.trace['prefix'];minimum=min(minimum,float(np.min(prefix)));work+=p.work
        intervals.append(p)
        if np.max(np.abs(p.torque-current.previous)-rate)>cfg.torque_tolerance:
            return Backup('INVALID',max(0.,-minimum),intervals,study,force.copy(),snapshot.signature(),perf_counter()-start,'backup_rate')
        if not predictor.task_ok(p.trace):
            return Backup('INVALID',max(0.,-minimum),intervals,study,force.copy(),snapshot.signature(),perf_counter()-start,'backup_task')
        clock=getattr(branch.controller.stroke,'clock',None)
        if clock is not None:
            # Terminal alpha belongs to the terminal state. Advance only a
            # private CLOCK copy, without another controller compute/observe.
            import copy
            clock=copy.deepcopy(clock);clock.advance(float(branch.data.time))
        terminal=(np.max(np.abs(p.trace['power']))<=cfg.terminal_power and
                  np.max(np.linalg.norm(p.trace['velocity'],axis=1))<=cfg.terminal_speed and
                  np.max(np.linalg.norm(p.trace['tool_velocity'],axis=1))<=cfg.terminal_tool_speed and
                  np.max(np.linalg.norm(p.trace['angular_velocity'],axis=1))<=cfg.terminal_angular_speed and
                  (clock is None or (clock.alpha<=.01 and abs(clock.alpha_dot)<=.1)))
        settled=settled+1 if terminal else 0
        elapsed=(k+1)*predictor.timing.control_period
        while checkpoint<len(cfg.horizons) and elapsed>=cfg.horizons[checkpoint]-1e-9:
            study.append(dict(horizon=cfg.horizons[checkpoint],reserve_J=max(0.,-minimum),terminal=bool(terminal),persistence_ticks=settled))
            checkpoint+=1
        if settled>=round(cfg.terminal_persistence/predictor.timing.control_period)+1:
            return Backup('SETTLED',max(0.,-minimum),intervals,study,force.copy(),snapshot.signature(),perf_counter()-start)
        current=p.next
    return Backup('UNRESOLVED',max(0.,-minimum),intervals,study,force.copy(),snapshot.signature(),perf_counter()-start,'terminal_not_persistent')
