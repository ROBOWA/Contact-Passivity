"""Stage 2 opt-in virtual progression and complete hybrid-state snapshots.

Frozen Stage 1 modules are deliberately unmodified. Models are immutable and
shared; all evolving controller/observer/trajectory arrays are private copies.
Snapshot controller blobs are trusted local pickle artifacts, never untrusted inputs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import io
import pickle

import mujoco
import numpy as np

from .force_control import ForceApproachController
from .holding import Simulation
from .oracle_physics import integration_state, restore_state
from .trajectories import quintic_coef, poly_eval, stop_trajectory, StrokeSequencer


class ProgressClock:
    def __init__(self, transition=.1):
        self.transition = float(transition)
        self.reset()

    def reset(self, t=0.):
        self.t = self.sigma = float(t)
        self.alpha, self.alpha_dot, self.alpha_ddot, self.target = 1., 0., 0., 1.
        self.epoch, self.coef = float(t), np.array([1.,0,0,0,0,0])

    def _integral(self, a, b):
        a, b = np.clip([a-self.epoch,b-self.epoch],0,self.transition)
        powers = np.arange(1,7)
        return float(np.sum(self.coef * (b**powers-a**powers)/powers))

    def advance(self, t):
        t=float(t)
        if t < self.t-1e-12:
            raise ValueError("virtual clock cannot run backward")
        if self.alpha==1. and self.target==1. and self.alpha_dot==0. and self.sigma==self.t:
            self.sigma=t  # exactly transparent before any slowdown
        else:
            end=self.epoch+self.transition
            self.sigma += self._integral(self.t,min(t,end)) if self.t<end else 0.
            self.sigma += max(0.,t-max(self.t,end))*self.target
        self.t=t
        if t>=self.epoch+self.transition:
            self.alpha,self.alpha_dot,self.alpha_ddot=self.target,0.,0.
        else:
            self.alpha,self.alpha_dot,self.alpha_ddot=poly_eval(self.coef,max(0.,t-self.epoch))

    def set_target(self, target):
        target=float(target)
        if not 0<=target<=1:
            raise ValueError("alpha target outside [0,1]")
        if target==self.target: return
        c=quintic_coef(self.alpha,self.alpha_dot,self.alpha_ddot,target,0.,0.,self.transition)
        vals=np.array([poly_eval(c,u) for u in np.linspace(0,self.transition,101)])
        if np.min(vals[:,0]) < -1e-10 or np.max(vals[:,0])>1+1e-10:
            raise ValueError("retargeting would overshoot alpha bounds")
        if np.max(np.abs(vals[:,1])) > 40.:
            raise ValueError("alpha rate exceeds declared 40/s limit")
        self.target,self.epoch,self.coef=target,self.t,c


class ProgressStroke:
    """Time adapter around the original sequencer; its spatial endpoints stay fixed.
    Actual contact recovery switches to a physical-time stop initialized from the
    GOVERNED position/velocity/acceleration, never from an unscaled velocity.
    Energy pauses do not call suspend or change the contact FSM.
    """
    def __init__(self, sequence, transition=.1):
        self.sequence=sequence
        self.clock=ProgressClock(transition)
        self.recovery=False

    def __getattr__(self, name):
        seq=self.__dict__.get("sequence")
        if seq is None: raise AttributeError(name)
        return getattr(seq,name)

    def reset(self):
        self.sequence.reset(); self.clock.reset(); self.recovery=False

    def eval(self,t):
        s,v,a=self.sequence.eval(self.clock.sigma)
        return s,v*self.clock.alpha,a*self.clock.alpha**2+v*self.clock.alpha_dot

    def start(self,t): self.sequence.start(self.clock.sigma)

    def suspend(self,t):
        if self.sequence.state==StrokeSequencer.MOVING:
            s,v,a=self.eval(t)
            self.sequence.traj=stop_trajectory(t,s,v,a,self.sequence.stop_time,self.sequence.a_max,
                                               p_bounds=(0.,self.sequence.D))
            self.sequence.state=StrokeSequencer.STOPPING
        elif self.sequence.state==StrokeSequencer.DWELL:
            self.sequence.state=StrokeSequencer.SUSPENDED
        self.clock.reset(t); self.recovery=True

    def resume(self,t):
        self.sequence.resume(self.clock.sigma)
        self.recovery=False


class GovernedController(ForceApproachController):
    @classmethod
    def from_controller(cls, controller, transition=.1):
        out=clone_controller(controller)
        out.__class__=cls
        out.stroke=ProgressStroke(out.stroke,transition)
        return out

    def compute(self,data):
        self.stroke.clock.advance(float(data.time))
        return super().compute(data)


def clone_controller(controller):
    return copy.deepcopy(controller,{id(controller.model):controller.model})


class _Writer(pickle.Pickler):
    def persistent_id(self,obj):
        return "immutable_mujoco_model" if isinstance(obj,mujoco.MjModel) else None


class _Reader(pickle.Unpickler):
    def __init__(self,stream,model): super().__init__(stream);self.model=model
    def persistent_load(self,key):
        if key!="immutable_mujoco_model": raise pickle.UnpicklingError(key)
        return self.model


def controller_bytes(controller):
    stream=io.BytesIO();_Writer(stream,protocol=5).dump(controller);return stream.getvalue()


def controller_from_bytes(blob,model):
    return _Reader(io.BytesIO(blob),model).load()


def _state_digest(obj,h):
    """Hash values, not pickle's incidental array alias/memo representation.
    Persistent models are replaced by None for this read-only state traversal.
    No rounding: a changed bit in any dynamic numeric field invalidates cache.
    """
    h.update((type(obj).__module__+'.'+type(obj).__qualname__).encode())
    if isinstance(obj,np.ndarray):
        h.update(str(obj.dtype).encode());h.update(str(obj.shape).encode());h.update(obj.tobytes())
    elif isinstance(obj,dict):
        for k in sorted(obj,key=str):_state_digest(k,h);_state_digest(obj[k],h)
    elif isinstance(obj,(list,tuple)):
        h.update(str(len(obj)).encode())
        for v in obj:_state_digest(v,h)
    elif hasattr(obj,'__dict__'):_state_digest(vars(obj),h)
    else:h.update(pickle.dumps(obj,protocol=5))


@dataclass
class HybridSnapshot:
    plant: np.ndarray
    controller: bytes
    energy: np.ndarray
    previous: np.ndarray
    base_qfrc: np.ndarray
    n_steps: int
    n_updates: int

    @classmethod
    def capture(cls,sim,energy,previous,base_qfrc):
        return cls(integration_state(sim.model,sim.data),controller_bytes(sim.controller),
                   np.array(energy,copy=True),np.array(previous,copy=True),np.array(base_qfrc,copy=True),
                   sim.n_steps,sim.n_control_updates)

    def signature(self):
        h=hashlib.sha256(self.plant.tobytes()+self.previous.tobytes()+self.base_qfrc.tobytes())
        _state_digest(controller_from_bytes(self.controller,None),h)
        return h.hexdigest()

    def save(self,path,**extra):
        np.savez_compressed(path,plant=self.plant,controller_blob=np.frombuffer(self.controller,dtype=np.uint8),
                            energy=self.energy,previous=self.previous,base_qfrc=self.base_qfrc,
                            n_steps=self.n_steps,n_updates=self.n_updates,**extra)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            return cls(z['plant'].copy(),z['controller_blob'].tobytes(),z['energy'].copy(),
                       z['previous'].copy(),z['base_qfrc'].copy(),int(z['n_steps']),int(z['n_updates']))

    def fork(self,sim):
        out=Simulation.__new__(Simulation)
        out.model,out.timing,out.arm=sim.model,sim.timing,sim.arm
        out.q_home=sim.q_home.copy();out.data=mujoco.MjData(sim.model)
        restore_state(sim.model,out.data,self.plant)
        out.controller=controller_from_bytes(self.controller,sim.model)
        out.n_steps,out.n_control_updates=self.n_steps,self.n_updates
        out.recorder=out.oracle=out.disturbance=None
        return out
