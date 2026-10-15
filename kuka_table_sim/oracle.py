"""Optional off/shadow/active instrumentation attached to the existing Simulation."""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from .joint_qp import QPConfig, FilterFailure, accepted_command, torque_box
from .oracle_physics import HumanPort, HumanPulse, OraclePredictor, TaskMeter, integration_state


class OracleFilter:
    def __init__(self, sim, mode="off", pulse=None, body="link5", local=(0., .04, .12),
                 energy_init=(1., 1.), energy_min=(0., 0.), config=None, shadow_stride=25):
        if mode not in ("off", "shadow", "active"):
            raise ValueError("mode must be off/shadow/active")
        if sim.disturbance is not None:
            raise ValueError("oracle fixture cannot forecast an arbitrary disturbance callback")
        self.mode, self.pulse = mode, pulse or HumanPulse(magnitude=0.)
        self.port = HumanPort(sim.model, body, local)
        self.predictor = OraclePredictor(sim.model, sim.timing, HumanPort(sim.model, body, local))
        self.meter = TaskMeter(sim.model)
        self.cfg = config or QPConfig()
        self.init, self.minimum = np.asarray(energy_init, float), np.asarray(energy_min, float)
        if self.init.shape != (2,) or self.minimum.shape != (2,) or not np.all(np.isfinite(np.r_[self.init, self.minimum])):
            raise ValueError("two finite account initial values and lower bounds required")
        if shadow_stride < 1:
            raise ValueError("shadow_stride must be positive")
        self.shadow_stride = shadow_stride
        self.reset(sim)

    def reset(self, sim):
        self.energy = self.init.copy()  # explicit trial reset only, never per-cycle recharge
        self.base_qfrc = sim.data.qfrc_applied.copy()
        self.previous = sim.tau.copy()
        self.force = np.zeros(3)
        self.rows, self.phys, self.events = [], [], []
        self.snapshot = None
        self.failure = None
        sim.controller.filter_intervened = False

    def command(self, sim, nominal):
        start = perf_counter()
        d = sim.data
        self.force = self.pulse.force(float(d.time))
        d.qfrc_applied[:] = self.base_qfrc
        ts = perf_counter()
        self.snapshot = integration_state(sim.model, d)
        timing = dict(snapshot=perf_counter() - ts, rollout=0., gradient=0., qp=0., nonlinear=0., retry=0.)
        torque = nominal.copy()
        pred = affine = np.full(2, np.nan)
        attempts = []
        self.grad = np.zeros((2, 7))
        try:
            if self.mode != "off":
                p = self.predictor.predict(self.snapshot, nominal, self.force, self.base_qfrc)
                timing["rollout"] = p.seconds
                pred = affine = p.work.copy()
                if self.mode == "active":
                    # If zero correction satisfies every hard row it is the exact
                    # minimizer of either positive definite objective. Reuse the
                    # fresh nonlinear nominal check; no unnecessary QP/stencil.
                    lo, hi = torque_box(nominal, nominal, self.previous, sim.arm.torque_limit,
                                        sim.timing.control_period, self.cfg)
                    ep = 0. if np.all(self.force == 0) else self.cfg.epsilon_prediction
                    el = self.cfg.epsilon_linearization * float(np.linalg.norm(self.force)) / self.cfg.margin_force_reference
                    nominal_ok = (np.min(self.energy + pred - ep - el - self.minimum) >= 0 and
                                  np.all(lo <= 0) and np.all(hi >= 0))
                    if not nominal_ok:
                        a = accepted_command(self.predictor, self.snapshot, nominal, self.force, self.energy,
                                             self.minimum, self.previous, self.cfg, self.base_qfrc)
                        torque, pred, affine, attempts = a.torque, a.prediction.work, a.affine, a.attempts
                        self.grad = a.linear.gradients
                        for k, v in a.timing.items():
                            timing[k] += v
                elif len(self.rows) % self.shadow_stride == 0:
                    lin = self.predictor.linearize(self.snapshot, nominal, self.force, self.cfg.eta, self.base_qfrc)
                    timing["gradient"] = lin.seconds
                    self.grad = lin.gradients
                    # A shadow candidate is validated only on its own execution
                    # branch in the gate suite, never against the nominal plant.
            self.modification = float(np.max(np.abs(torque - nominal)))
            sim.controller.filter_intervened = self.mode == "active" and self.modification > 1e-7
            self.tick = dict(t=float(d.time), energy_before=self.energy.copy(),
                             tau_nom=nominal.copy(), tau_exec=torque.copy(), tau_previous=self.previous.copy(),
                             force=self.force.copy(), predicted=pred.copy(), affine=affine.copy(), gradient=self.grad.copy(),
                             intervention=self.modification, attempts=len(attempts),
                             **{f"seconds_{k}": v for k, v in timing.items()})
            self.tick["seconds_command"] = perf_counter() - start
            if attempts:
                self.events.append(dict(t=float(d.time), attempts=attempts))
            self.previous = torque.copy()  # This command WILL execute; rejection raises before assignment.
            self.wh = self.wt = self.gross = 0.
            self.min_prefix = self.energy.copy()
            self.substeps = 0
            return torque
        except FilterFailure as exc:
            self.events.append(dict(t=float(d.time), failed=True, reason=exc.reason,
                                    attempts=exc.diagnostics.get("attempts", [])))
            self.failure = dict(t=float(d.time), reason=exc.reason, diagnostics=exc.diagnostics,
                                energy=self.energy.tolist(), nominal=nominal.tolist(), force=self.force.tolist(),
                                seconds_failed_command=perf_counter() - start)
            raise

    def pre(self, sim, t):
        f = self.pulse.force(t) if self.pulse.varying else self.force
        self.sample = self.port.apply(sim.data, f, self.base_qfrc)
        self.applied_force = f.copy()
        self.qd_pre = sim.data.qvel.copy()

    def post(self, sim, t):
        d, h = sim.data, sim.timing.timestep
        if not np.allclose(d.qfrc_actuator[sim.arm.dof_adr], self.previous, rtol=0, atol=1e-10):
            raise FilterFailure("execution_torque_mismatch")
        if sim.recorder is not None:
            rp, i = sim.recorder.phys, sim.recorder.row - 1
            power, fn, unexpected = rp["P_task"][i], rp["F_task"][i, 2], rp["n_unexpected"][i]
        else:
            task = self.meter.sample(d)
            power, fn, unexpected = task["power"], task["F"][2], task["nu"]
        point, velocity, human_tau, ph, defect = self.sample
        self.wh += ph * h
        self.wt += power * h
        self.gross += max(-ph, 0.) * h
        self.min_prefix = np.minimum(self.min_prefix, self.energy + self.wh)
        self.phys.append(dict(t=t, point=point.copy(), velocity=velocity.copy(), force=self.applied_force.copy(),
                              power=ph, task_power=power, power_identity=defect,
                              prefix_energy=self.energy + self.wh, normal_force=fn,
                              constraint_power=float(self.qd_pre @ d.qfrc_constraint), unexpected=unexpected))
        self.substeps += 1
        if self.substeps == sim.timing.control_interval:
            self.energy += self.wh  # R=H, actual signed work ONLY. No clipping.
            self.tick.update(actual=np.array([self.wh, self.wh]), task_work=self.wt,
                             nominal_task_work=self.wt, energy_after=self.energy.copy(),
                             min_prefix=self.min_prefix.copy(), gross=self.gross)
            self.rows.append(self.tick)
            if self.mode == "active" and np.min(self.energy - self.minimum) < -self.cfg.energy_tolerance:
                self.failure = dict(t=float(d.time), reason="actual_account_violation", energy=self.energy.tolist())
                raise FilterFailure("actual_account_violation", self.failure)

    def arrays(self):
        def stack(rows):
            return {k: np.asarray([r[k] for r in rows]) for k in rows[0]} if rows else {}
        return stack(self.rows), stack(self.phys)

    def metadata(self):
        return dict(mode=self.mode, pulse=asdict(self.pulse), body=self.port.body_name, local=self.port.local.tolist(),
                    energy_init=self.init.tolist(), energy_min=self.minimum.tolist(), config=asdict(self.cfg),
                    shadow_stride=self.shadow_stride, failure=self.failure,
                    accounts="R=H exactly; task nominal is actual task contact work",
                    quadrature="left rectangle per physics evaluation; endpoint account enforcement")
