"""MuJoCo-based planar contact-sliding simulation.

Proof of concept for the contact-model-aware selective-passivation project:
a two-link planar robot approaches a surface, regulates a normal contact
force, and slides tangentially, while ground-truth task-contact and
human-disturbance channels are logged separately.
"""

from .config import (
    C_PROJ,
    NORMAL,
    TANGENT,
    U_PROJ,
    ControllerConfig,
    HumanForceConfig,
    SimulationConfig,
)
from .contact_extraction import ContactResult, ModelHandles, extract_task_contact
from .controller import Phase, SlidingForceController
from .human_interaction import HumanForce

_SIMULATION_EXPORTS = {"load_model", "run_simulation", "save_log", "steady_slide_metrics"}


def __getattr__(name):
    # Lazy so that `python -m mujoco_sliding.simulation` does not import the
    # simulation module twice (once here, once as __main__).
    if name in _SIMULATION_EXPORTS:
        from . import simulation

        return getattr(simulation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "C_PROJ",
    "NORMAL",
    "TANGENT",
    "U_PROJ",
    "ContactResult",
    "ControllerConfig",
    "HumanForce",
    "HumanForceConfig",
    "ModelHandles",
    "Phase",
    "SimulationConfig",
    "SlidingForceController",
    "extract_task_contact",
    "load_model",
    "run_simulation",
    "save_log",
    "steady_slide_metrics",
]
