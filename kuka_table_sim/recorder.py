"""Buffered, stage-coherent diagnostics for a simulation run.

Read-only: the recorder never calls mj_forward/mj_step and never writes to `MjData`, so enabling it cannot
change the trajectory (checked bitwise in tests/test_timing.py). Files are written and plots are drawn
only after the run, from `RunLog`.

Evaluation instant
    Every row of the PHYSICS log describes ONE evaluation: the physics step that starts at t_k with state
    x_k. Pre-integration quantities (q, qvel, site poses, cvel, tool velocities) are read after mj_step1;
    acceleration-stage quantities (FT sample, contact forces, contact points, qfrc_constraint, applied
    actuator force) are read after mj_step2, where they still describe x_k (see holding.py). `t` is t_k.
    No quantity is read from a different instant.

Contact power and work (task contact = tool_tip <-> tabletop only)
    For each task contact the force F_c and moment M_c (about the contact point p_c) that the environment
    applies to the TOOL come from mj_contactForce (contact frame -> world, sign fixed by which geom is the
    tip). The velocity is that of the tool's material point at p_c, v_p = v_tip_centre + omega x (p_c -
    x_tip_centre), and omega is the tool angular velocity, both from the same evaluation (cvel). Then
        P = sum_c ( F_c . v_p + M_c . omega )          [W, signed, positive = environment does work ON the robot]
    The table is fixed, so there is no work on the environment. Because the contact Jacobian is evaluated at
    p_c, P equals qvel . qfrc_constraint exactly when no other constraint is active; this identity is
    logged (`P_constraint`) as an independent check. Unexpected contacts (anything that is not
    tool_tip/tabletop) are counted and their normal force summed in `fn_unexpected`, never added to P.
    Work uses the left-endpoint rectangle rule per physics step, dW_k = P_k * dt, because force and velocity
    are available from the same evaluation (a trapezoid and a force-times-displacement variant are computed
    in metrics.py for the discretisation study). Nothing is absolute-valued, smoothed or clipped.

Control rows (one per control update) hold the controller's `diagnostics()`: phase, the FT sample consumed
and its timestamp, raw/gravity-only/corrected force, the force used, references, integrators, torque
components, saturation, counters. Measured tcp pose/velocity and applied actuator torque are taken from the
physics row of the same instant (index j * control_interval). The torque command is held for the whole
control interval; `tau_applied` in the physics rows shows it step by step.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import mujoco
import numpy as np

from .ft_sensor import FTSensor
from .scene import ASSETS_DIR, TCP_SITE, TOOL_TIP_GEOM

TABLETOP_GEOM = "tabletop"
STROKE_CODE = {"idle": 0, "moving": 1, "dwell": 2, "stopping": 3, "suspended": 4, "finished": 5}


@dataclass
class RunLog:
    phys: dict[str, np.ndarray]
    ctl: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    def save(self, path: Path, stride: int = 1, window: tuple[float, float] | None = None, float32: bool = False) -> None:
        """Write a compressed .npz. `stride` keeps every n-th physics row (and every n/control_interval-th control
        row), `window` = (t0, t1) restricts both logs, `float32` stores everything except time and work as float32
        (compact artefacts). The saved metadata records stride, window and precision."""
        out: dict[str, np.ndarray] = {}
        pmask = np.ones(len(self.phys["t"]), bool)
        cmask = np.ones(len(self.ctl["t"]), bool)
        if window is not None:
            pmask &= (self.phys["t"] >= window[0]) & (self.phys["t"] <= window[1])
            cmask &= (self.ctl["t"] >= window[0]) & (self.ctl["t"] <= window[1])
        pidx = np.flatnonzero(pmask)[::max(1, stride)]
        cidx = np.flatnonzero(cmask)[::max(1, stride // int(self.meta.get("control_interval", 1)))]
        keep64 = {"t", "W_cum", "dW", "t_ft"}
        for prefix, d, idx in (("phys", self.phys, pidx), ("ctl", self.ctl, cidx)):
            for k, v in d.items():
                a = v[idx]
                if float32 and a.dtype == np.float64 and k not in keep64:
                    a = a.astype(np.float32)
                out[f"{prefix}__{k}"] = a
        meta = dict(self.meta, saved_stride=stride, saved_window=window, saved_float32=float32)
        out["meta_json"] = np.array(json.dumps(meta, default=_json_default))
        np.savez_compressed(path, **out)

    @staticmethod
    def load(path: Path) -> "RunLog":
        z = np.load(path, allow_pickle=False)
        phys = {k[6:]: z[k] for k in z.files if k.startswith("phys__")}
        ctl = {k[5:]: z[k] for k in z.files if k.startswith("ctl__")}
        return RunLog(phys, ctl, json.loads(str(z["meta_json"])))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def git_info(repo: Path | None = None) -> dict:
    repo = repo or Path(__file__).resolve().parent.parent
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return dict(commit=run("rev-parse", "HEAD") or "unknown", branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(run("status", "--porcelain", "--untracked-files=no")))


class Recorder:
    def __init__(self, sim, duration: float, legacy_mixed: bool = False) -> None:
        self.sim = sim
        m = sim.model
        self.m = m
        self.ci = sim.timing.control_interval
        self.dt = float(m.opt.timestep)
        self.legacy_mixed = legacy_mixed
        n = int(round(duration / self.dt)) + 8
        nc = n // self.ci + 8
        self.cap, self.ccap = n, nc
        arm = sim.arm
        self.arm = arm
        self.tip = m.geom(TOOL_TIP_GEOM).id
        self.table = m.geom(TABLETOP_GEOM).id
        self.tcp = m.site(TCP_SITE).id
        self.link7 = m.body("link7").id
        self.ft = FTSensor(m)
        z = lambda *shape: np.zeros(shape)
        self.phys: dict[str, np.ndarray] = dict(
            t=z(n), q=z(n, 7), qd=z(n, 7), tcp=z(n, 3), tcp_v=z(n, 3), tip=z(n, 3),
            present=np.zeros(n, bool), n_task=np.zeros(n, np.int8), n_unexpected=np.zeros(n, np.int16), fn_unexpected=z(n),
            F_task=z(n, 3), M_task=z(n, 3), p_c=z(n, 3), v_c=z(n, 3), w_tool=z(n, 3),
            tip_v=z(n, 3), P_task=z(n), P_constraint=z(n), dW=z(n),
            ft_raw=z(n, 3), ft_grav=z(n, 3), ft_corr=z(n, 3), a_com=z(n, 3),
            tau_applied=z(n, 7), xfrc=z(n, 3), control_update=np.zeros(n, bool))
        if legacy_mixed:
            self.phys["P_mixed"] = z(n)
        self.ctl: dict[str, np.ndarray] | None = None
        self._ctl_rows = 0
        self.row = 0
        self.unexpected_pairs: dict[tuple[str, str], int] = {}
        self._qd_pre = np.zeros(m.nv)
        self._v6 = np.zeros(6)
        self._w6 = np.zeros(6)
        self._jp = np.zeros((3, m.nv))

    # ------------------------------------------------------------------ hooks called by Simulation
    def on_reset(self, sim) -> None:
        self.row = 0
        self._ctl_rows = 0
        self.unexpected_pairs = {}

    def pre(self, sim, t_k: float) -> None:
        """After mj_step1: state x_k and velocity-stage quantities."""
        d = sim.data
        i = self.row
        if i >= self.cap:
            raise RuntimeError("recorder capacity exceeded; construct it with a longer duration")
        P = self.phys
        P["t"][i] = t_k
        P["q"][i] = d.qpos[self.arm.qpos_adr]
        P["qd"][i] = d.qvel[self.arm.dof_adr]
        self._qd_pre[:] = d.qvel
        P["tcp"][i] = d.site_xpos[self.tcp]
        P["tip"][i] = d.geom_xpos[self.tip]
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_SITE, self.tcp, self._v6, 0)
        P["tcp_v"][i] = self._v6[3:6]

    def post(self, sim, t_k: float, is_control: bool) -> None:
        """After mj_step2: acceleration-stage quantities of the SAME evaluation (t_k, x_k)."""
        d, m = sim.data, self.m
        i = self.row
        P = self.phys
        tip, table = self.tip, self.table
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, tip, self._v6, 0)   # cvel still at x_k
        omega, v_centre = self._v6[:3].copy(), self._v6[3:].copy()
        x_tip = d.geom_xpos[tip]
        F = np.zeros(3)
        M = np.zeros(3)
        power = 0.0
        wsum = 0.0
        pc = np.zeros(3)
        n_task = n_unexp = 0
        fn_unexp = 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = c.geom1, c.geom2
            mujoco.mj_contactForce(m, d, k, self._w6)
            if (g1 == tip and g2 == table) or (g1 == table and g2 == tip):
                frame = c.frame.reshape(3, 3)                      # rows: normal, tangent1, tangent2
                sign = -1.0 if g1 == tip else 1.0                  # mj_contactForce: force on geom2 by geom1
                f = sign * (frame.T @ self._w6[:3])
                mo = sign * (frame.T @ self._w6[3:])
                v_p = v_centre + np.cross(omega, c.pos - x_tip)
                power += float(f @ v_p + mo @ omega)
                F += f
                M += mo
                mag = float(np.linalg.norm(f)) + 1e-300
                pc += c.pos * mag
                wsum += mag
                n_task += 1
            else:
                n_unexp += 1
                fn_unexp += abs(float(self._w6[0]))
                key = (m.geom(g1).name or f"geom{g1}", m.geom(g2).name or f"geom{g2}")
                self.unexpected_pairs[key] = self.unexpected_pairs.get(key, 0) + 1
        p_c = pc / wsum if wsum > 1e-200 else x_tip.copy()
        P["present"][i] = n_task > 0
        P["n_task"][i] = n_task
        P["n_unexpected"][i] = n_unexp
        P["fn_unexpected"][i] = fn_unexp
        P["F_task"][i] = F
        P["M_task"][i] = M
        P["p_c"][i] = p_c
        P["v_c"][i] = v_centre + np.cross(omega, p_c - x_tip)
        P["w_tool"][i] = omega
        P["tip_v"][i] = v_centre
        P["P_task"][i] = power
        P["dW"][i] = power * self.dt                               # left-endpoint rectangle, same evaluation
        P["P_constraint"][i] = float(self._qd_pre @ d.qfrc_constraint)
        if self.legacy_mixed:                                      # what a post-step logger would compute
            mujoco.mj_jacSite(m, d, self._jp, None, self.tcp)
            P["P_mixed"][i] = float(F @ (self._jp @ d.qvel))      # J(x_k) with qvel(x_k+1), force from x_k
        # FT chain of this evaluation (the sample the controller consumes if it is the last one before an update)
        ctl = sim.controller
        s = getattr(ctl, "ft_latest", None)
        if s is None or s.t != t_k:
            s = self.ft.sample(d, t_k)
        P["ft_raw"][i] = s.raw_site
        P["ft_grav"][i] = s.gravity_only
        P["ft_corr"][i] = s.corrected
        P["a_com"][i] = s.a_com
        P["tau_applied"][i] = d.actuator_force[self.arm.actuator_ids]
        P["xfrc"][i] = d.xfrc_applied[self.link7, :3]
        P["control_update"][i] = is_control
        if is_control:
            self._record_control(sim, t_k)
        self.row += 1

    def _record_control(self, sim, t_k: float) -> None:
        diag_fn = getattr(sim.controller, "diagnostics", None)
        diag = diag_fn() if diag_fn is not None else {}
        diag = dict(diag)
        if "stroke_state" in diag:
            diag["stroke_state"] = STROKE_CODE.get(diag["stroke_state"], -1)
        diag["t"] = t_k
        diag["tau_ctrl"] = sim.data.ctrl[self.arm.actuator_ids].copy()      # the held (clipped) command
        if self.ctl is None:
            self.ctl = {k: np.full((self.ccap,) + np.shape(v), np.nan) for k, v in diag.items()}
        j = self._ctl_rows
        for k, v in diag.items():
            if k not in self.ctl:
                self.ctl[k] = np.full((self.ccap,) + np.shape(v), np.nan)
            self.ctl[k][j] = v
        self._ctl_rows += 1

    # ------------------------------------------------------------------ result
    def finish(self) -> RunLog:
        n, nc, ci = self.row, self._ctl_rows, self.ci
        phys = {k: v[:n].copy() for k, v in self.phys.items()}
        ctl = {k: v[:nc].copy() for k, v in (self.ctl or {"t": np.zeros((nc,))}).items()}
        phys["W_cum"] = np.cumsum(phys["dW"])                      # work done through the end of each step
        phys["ctl_index"] = np.arange(n) // ci
        idx = np.arange(nc) * ci
        ctl["x"] = phys["tcp"][idx]
        ctl["v"] = phys["tcp_v"][idx]
        ctl["tau_applied"] = phys["tau_applied"][idx]
        pad = np.concatenate((phys["dW"], np.zeros(ci)))
        ctl["W_interval"] = np.array([pad[j * ci:(j + 1) * ci].sum() for j in range(nc)])
        sim_ = self.sim
        cfg = getattr(sim_.controller, "cfg", None)
        files = {}
        for name in ("kuka_table_scene.xml", "iiwa14_torque.xml", "polishing_tool.xml"):
            f = ASSETS_DIR / name
            if f.exists():
                files[name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        meta = dict(
            **git_info(), mujoco=mujoco.__version__, numpy=np.__version__, python=platform.python_version(),
            platform=platform.platform(), timestep=self.dt, control_period=sim_.timing.control_period,
            control_interval=ci, n_physics_steps=n, n_control_updates=nc,
            controller=type(sim_.controller).__name__, config=asdict(cfg) if is_dataclass(cfg) else None,
            model_file_sha256=files, integrator=mujoco.mjtIntegrator(self.m.opt.integrator).name,
            cone=mujoco.mjtCone(self.m.opt.cone).name, gravity=self.m.opt.gravity.tolist(),
            pair_solreffriction=self.m.pair_solreffriction.tolist() if self.m.npair else None,
            tool_mass=self.ft.mass, torque_limit=self.arm.torque_limit.tolist(),
            unexpected_pairs={f"{a}|{b}": c for (a, b), c in self.unexpected_pairs.items()},
            conventions=dict(
                row="one physics evaluation: step starting at t_k with state x_k; every field is from that evaluation",
                work="dW = P*dt, left-endpoint rectangle; P = sum F_c.v_p + M_c.omega, environment on robot, signed",
                force_sign="F_task, ft_corr, f_n: force of the environment ON the tool; +z (table normal) = pressing",
                ft_sample="control update at t_k consumes the sample of the step that started at t_k - dt (t_ft)",
                torque="tau_ctrl is constant during the control interval; tau_applied is MuJoCo actuator_force per substep"))
        return RunLog(phys, ctl, meta)
