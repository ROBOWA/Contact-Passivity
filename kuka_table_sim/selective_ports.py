"""Independent task-model work and three physically declared interaction accounts.

Positive power enters the robot. H is actual known-point human power; delta is
actual task-contact power minus nominal-model task power at the SAME actual
state/velocity. R=H+delta and W=H+actual task power. Different spatial ports are
combined through their individual wrench/velocity duals, never wrench subtraction.
"""
from dataclasses import dataclass
import copy
import hashlib
import numpy as np
import mujoco
from .oracle_physics import HumanPort, TaskMeter, integration_state, restore_state
from .scene import ArmIndex

HUMAN, RESIDUAL, WHOLE = 0, 1, 2
ACCOUNT_NAMES = ('H', 'R', 'W')


@dataclass(frozen=True)
class SelectiveConfig:
    initial: tuple = (.12, .30, .30)
    caps: tuple = (.15, .30, .30)
    floors: tuple = (0., 0., 0.)
    peak_human_W: float = .15
    barrier_rate: float = 5.
    safe_energy_J: float = 2.5e-5
    force_rate_N_s: float = 60.
    governor_transition_s: float = .1
    release_force_N: float = .02
    eta_Nm: float = .02
    retries: int = 3
    energy_guard_J: float = 1e-10
    power_guard_W: float = 1e-7
    energy_tolerance_J: float = 1e-10
    power_tolerance_W: float = 1e-7
    torque_tolerance_Nm: float = 1e-7
    task_weight: float = 1.

    def __post_init__(self):
        for key in ('initial','caps','floors'):
            value=tuple(float(v) for v in getattr(self,key));object.__setattr__(self,key,value)
            if len(value)!=3 or not np.all(np.isfinite(value)):raise ValueError(key)
        if np.any(np.array(self.initial)<self.floors) or np.any(np.array(self.initial)>self.caps):raise ValueError('account initialization')
        if any(getattr(self,k)<=0 or not np.isfinite(getattr(self,k)) for k in ('peak_human_W','barrier_rate','safe_energy_J','force_rate_N_s','governor_transition_s','eta_Nm')):raise ValueError('positive finite parameters required')


def capped_update(energy, work, caps):
    raw=np.asarray(energy,dtype=float)+np.asarray(work,dtype=float)
    discarded=np.maximum(raw-np.asarray(caps),0.)
    return raw-discarded,discarded


def account_powers(human, task, nominal_task):
    delta=float(task-nominal_task)
    return np.array([human,human+delta,human+task],dtype=float),delta


class TaskModel:
    """A separate MuJoCo nominal contact model; plant parameters never change.

    Only the explicit tip/table pair's tangential friction coefficients differ.
    The actual integration snapshot is restored before EACH evaluation, including
    warmstart/control/external forces. Contact power uses x_k velocities after
    the corresponding step2 solve, prior to refreshing kinematics.
    """
    def __init__(self, plant, body, local, friction):
        self.model=copy.deepcopy(plant)
        pair=self.model.pair('tip_tabletop').id
        self.plant_friction=plant.pair_friction[pair].copy()
        if friction is not None:self.model.pair_friction[pair,:2]=float(friction)
        self.nominal_friction=self.model.pair_friction[pair].copy()
        self.data=mujoco.MjData(self.model)
        self.meter=TaskMeter(self.model);self.port=HumanPort(self.model,body,local)
        self.arm_ids=ArmIndex.from_model(self.model).actuator_ids
        self.parameter_hash=hashlib.sha256(self.nominal_friction.tobytes()+self.model.pair_solref.tobytes()+self.model.pair_solreffriction.tobytes()).hexdigest()

    def sample(self, state, torque, force, base):
        m,d=self.model,self.data
        restore_state(m,d,state)
        mujoco.mj_step1(m,d)
        d.ctrl[self.arm_ids]=torque
        self.port.apply(d,force,base)
        mujoco.mj_step2(m,d)
        return self.meter.sample(d)
