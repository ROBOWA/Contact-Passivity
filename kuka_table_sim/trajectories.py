"""Acceleration-continuous 1-D reference trajectories built from polynomial segments.

Everything the force controller commands as a position reference (approach and search
descent, stopping a moving reference, tangential strokes) is a `Trajectory1D`: a list of
polynomial segments (degree <= 5) whose boundaries match in position, velocity and
acceleration. The planners start from an arbitrary (p, v, a) state, so a recovery motion
can begin from the reference that is currently active instead of snapping to a fixed
position.

`ApproachProfile` and `SlideProfile` are the original closed-form nominal profiles; they
are kept as the analytic reference that the generic planners are tested against.
"""

from __future__ import annotations

import math

import numpy as np

INF = math.inf
_GRID = 400   # samples used to find peak speed / acceleration of a segment


# --------------------------------------------------------------------------- quintic algebra

def quintic_coef(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, T: float) -> np.ndarray:
    """Coefficients c0..c5 of p(tau) = sum c_i tau^i with the given boundary state at tau = 0 and T."""
    c3 = (20 * (p1 - p0) - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T ** 2) / (2 * T ** 3)
    c4 = (30 * (p0 - p1) + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T ** 2) / (2 * T ** 4)
    c5 = (12 * (p1 - p0) - 6 * (v1 + v0) * T - (a0 - a1) * T ** 2) / (2 * T ** 5)
    return np.array([p0, v0, a0 / 2.0, c3, c4, c5], dtype=float)


def poly_eval(c: np.ndarray, tau: float) -> tuple[float, float, float]:
    p = c[0] + tau * (c[1] + tau * (c[2] + tau * (c[3] + tau * (c[4] + tau * c[5]))))
    v = c[1] + tau * (2 * c[2] + tau * (3 * c[3] + tau * (4 * c[4] + tau * 5 * c[5])))
    a = 2 * c[2] + tau * (6 * c[3] + tau * (12 * c[4] + tau * 20 * c[5]))
    return float(p), float(v), float(a)


def peak_speed_accel(c: np.ndarray, T: float) -> tuple[float, float]:
    tau = np.linspace(0.0, T, _GRID)
    v = c[1] + tau * (2 * c[2] + tau * (3 * c[3] + tau * (4 * c[4] + tau * 5 * c[5])))
    a = 2 * c[2] + tau * (6 * c[3] + tau * (12 * c[4] + tau * 20 * c[5]))
    return float(np.max(np.abs(v))), float(np.max(np.abs(a)))


def _search_T(make, v_max: float, a_max: float, v_floor: float, a_floor: float, T_min: float):
    """Smallest feasible T >= T_min, or None. Bisection assumes the peaks decrease with T."""
    def ok(T: float) -> bool:
        pv, pa = peak_speed_accel(make(T), T)
        return pv <= max(v_max, v_floor) * (1 + 1e-9) and pa <= max(a_max, a_floor) * (1 + 1e-9)

    lo = max(T_min, 1e-3)
    if ok(lo):
        return lo, make(lo)
    hi = lo
    for _ in range(40):
        hi *= 2.0
        if ok(hi):
            break
    else:
        return None
    lo = hi / 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi, make(hi)


# Peaks of a segment are evaluated on a 400-point grid, so a planned limit can be exceeded by ~1e-5 relative.
def _min_feasible_T(make, v_max: float, a_max: float, v_floor: float, a_floor: float, T_min: float) -> tuple[float, np.ndarray]:
    """Shortest T >= T_min for which `make(T)` respects both limits (boundary states excepted).

    The declared limits apply to references that start inside them, which is what the controller
    produces. A start state that is itself outside them (for example a measured velocity far above the
    declared reference speed) can make the speed limit unreachable; the speed limit is then dropped
    and finally both, and the caller gets the best-effort plan rather than an exception.
    """
    for vm, am in ((v_max, a_max), (INF, a_max), (INF, INF)):
        found = _search_T(make, vm, am, v_floor, a_floor, T_min)
        if found is not None:
            return found
    T = max(T_min, 1.0)
    return T, make(T)


def plan_quintic(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float,
                 v_max: float = INF, a_max: float = INF, T_min: float = 0.0) -> tuple[float, np.ndarray]:
    """Quintic (p0, v0, a0) -> (p1, v1, a1) with the shortest duration >= T_min that keeps the peak
    speed <= v_max and the peak acceleration <= a_max (boundary states excepted)."""
    return _min_feasible_T(lambda T: quintic_coef(p0, v0, a0, p1, v1, a1, T), v_max, a_max,
                           max(abs(v0), abs(v1)), max(abs(a0), abs(a1)), T_min)


def plan_to_velocity(p0: float, v0: float, a0: float, v1: float, v_max: float = INF, a_max: float = INF,
                     T_min: float = 0.0) -> tuple[float, np.ndarray]:
    """Quintic that changes the velocity from v0 to v1 and ends with zero acceleration; the end position
    is the average-velocity position p0 + (v0 + v1) T / 2 so the speed changes monotonically."""
    return _min_feasible_T(lambda T: quintic_coef(p0, v0, a0, p0 + 0.5 * (v0 + v1) * T, v1, 0.0, T), v_max, a_max,
                           max(abs(v0), abs(v1)), abs(a0), T_min)


# --------------------------------------------------------------------------- piecewise trajectory

class Trajectory1D:
    """Piecewise-polynomial reference; `t` arguments are absolute times. The last segment may be
    open-ended (T = inf). Before t0 the initial state is returned."""

    def __init__(self, t0: float, segments: list[tuple[float, np.ndarray]]) -> None:
        self.t0 = float(t0)
        self.segments = [(float(T), np.asarray(c, dtype=float)) for T, c in segments]
        self.starts = np.concatenate(([0.0], np.cumsum([T for T, _ in self.segments[:-1]]))) if self.segments else np.zeros(1)

    @property
    def duration(self) -> float:
        return float(sum(T for T, _ in self.segments))

    @property
    def t_end(self) -> float:
        return self.t0 + self.duration

    def __call__(self, t: float) -> tuple[float, float, float]:
        tau = max(0.0, t - self.t0)
        for (T, c), s in zip(self.segments, self.starts):
            if tau - s < T or T == INF:
                return poly_eval(c, tau - s)
        T, c = self.segments[-1]
        return poly_eval(c, T)

    def start_state(self) -> tuple[float, float, float]:
        return self(self.t0)


def hold_trajectory(t0: float, p: float) -> Trajectory1D:
    return Trajectory1D(t0, [(INF, np.array([p, 0, 0, 0, 0, 0.0]))])


def stop_trajectory(t0: float, p0: float, v0: float, a0: float, stop_time: float, a_max: float = INF,
                    p_bounds: tuple[float, float] | None = None) -> Trajectory1D:
    """Bring a moving reference to rest with a quintic of at least `stop_time` and peak acceleration
    <= a_max, then hold. At rest and with zero acceleration this is a plain hold. With `p_bounds` the rest
    position is clipped into [lo, hi] (the stroke range), so a stop near the end of a stroke cannot run past it."""
    if abs(v0) < 1e-12 and abs(a0) < 1e-12:
        return hold_trajectory(t0, p0)

    def rest_position(T: float) -> float:
        p1 = p0 + 0.5 * v0 * T
        return min(max(p1, p_bounds[0]), p_bounds[1]) if p_bounds is not None else p1

    T, c = _min_feasible_T(lambda T: quintic_coef(p0, v0, a0, rest_position(T), 0.0, 0.0, T), INF, a_max,
                           abs(v0), abs(a0), stop_time)
    p1 = poly_eval(c, T)[0]
    return Trajectory1D(t0, [(T, c), (INF, np.array([p1, 0, 0, 0, 0, 0.0]))])


def move_to_trajectory(t0: float, p0: float, v0: float, a0: float, p1: float, v_max: float = INF,
                       a_max: float = INF, T_min: float = 0.0) -> Trajectory1D:
    """Quintic from (p0, v0, a0) to rest at p1 (peak speed/acceleration limited, duration >= T_min), then hold."""
    T, c = plan_quintic(p0, v0, a0, p1, 0.0, 0.0, v_max=v_max, a_max=a_max, T_min=T_min)
    return Trajectory1D(t0, [(T, c), (INF, np.array([p1, 0, 0, 0, 0, 0.0]))])


def descent_trajectory(t0: float, z0: float, v0: float, a0: float, z_creep: float, z_floor: float,
                       v_peak: float, v_creep: float, a_max: float = INF, ramp_in_time: float = 0.4,
                       stop_time: float = 0.5) -> Trajectory1D:
    """Approach / search descent: quintic from (z0, v0, a0) down to the creep point at the creep speed,
    constant creep speed, then a quintic stop that comes to rest exactly at z_floor.

    * from rest at z0 above the creep point with a_max = inf this is the nominal approach profile
      (quintic duration chosen so that the peak speed equals v_peak);
    * if z0 is already below the creep point the quintic is replaced by a ramp from (v0, a0) to the creep
      speed (`ramp_in_time` or longer if a_max requires);
    * if there is no room left above the floor for the creep and the stop, the reference just stops.
    """
    delta = 0.5 * v_creep * stop_time                 # distance covered by the final stop
    z_stop_start = z_floor + delta
    if z0 <= z_floor + 2.0 * delta:
        return stop_trajectory(t0, z0, v0, a0, stop_time, a_max)
    segments: list[tuple[float, np.ndarray]] = []
    if z0 > z_creep:
        T1, c1 = plan_quintic(z0, v0, a0, z_creep, -v_creep, 0.0, v_max=v_peak, a_max=a_max)
        z_start_creep = z_creep
    else:
        T1, c1 = plan_to_velocity(z0, v0, a0, -v_creep, v_max=v_peak, a_max=a_max, T_min=ramp_in_time)
        z_start_creep = poly_eval(c1, T1)[0]
    segments.append((T1, c1))
    creep_len = z_start_creep - z_stop_start
    if creep_len > 0:
        segments.append((creep_len / v_creep, np.array([z_start_creep, -v_creep, 0, 0, 0, 0.0])))
        z_stop_from = z_stop_start
    else:
        z_stop_from = z_start_creep
    # final stop at the floor
    T_stop = max(stop_time, 2.0 * (z_stop_from - z_floor) / v_creep) if z_stop_from > z_floor else stop_time
    segments.append((T_stop, quintic_coef(z_stop_from, -v_creep, 0.0, z_floor, 0.0, 0.0, T_stop)))
    segments.append((INF, np.array([z_floor, 0, 0, 0, 0, 0.0])))
    return Trajectory1D(t0, segments)


# --------------------------------------------------------------------------- tangential strokes

class StrokeSequencer:
    """Rest-to-rest strokes 0 -> D -> 0 -> ... that can be suspended and resumed without a jump.

    Uninterrupted, it reproduces `SlideProfile` exactly (stroke time 1.875 D / v_peak; the next stroke
    starts at the exact end time of the previous one). `suspend(t)` replaces the running segment by a
    bounded stop from the current (s, v, a); `resume(t)` plans from the current state to the endpoint of
    the interrupted stroke, so the reference never returns to the stroke origin.
    """

    IDLE, MOVING, DWELL, STOPPING, SUSPENDED, FINISHED = "idle", "moving", "dwell", "stopping", "suspended", "finished"

    def __init__(self, distance: float, v_peak: float, cycles: int = 0, dwell: float = 0.0,
                 stop_time: float = 0.6, a_max: float = INF) -> None:
        self.D, self.v_peak, self.cycles, self.dwell = float(distance), float(v_peak), int(cycles), float(dwell)
        self.stop_time, self.a_max = float(stop_time), float(a_max)
        self.reset()

    def reset(self) -> None:
        self.state = self.IDLE
        self.strokes = 0                  # completed strokes
        self.s_rest = 0.0
        self.traj: Trajectory1D | None = None
        self._dwell_until = 0.0

    @property
    def target(self) -> float:
        return self.D if self.strokes % 2 == 0 else 0.0

    @property
    def active(self) -> bool:
        return self.state in (self.MOVING, self.DWELL)

    @property
    def enabled(self) -> bool:
        return self.D > 0 and self.v_peak > 0

    def _plan(self, t: float, s: float, v: float, a: float) -> None:
        target = self.target
        d = abs(target - s)
        if d < 1e-4 and abs(v) < 1e-4:             # interrupted a hair short of the end: count it and go on
            self.strokes += 1
            target = self.target
            d = abs(target - s)
        T0 = 1.875 * d / self.v_peak
        T, c = plan_quintic(s, v, a, target, 0.0, 0.0, v_max=self.v_peak, a_max=self.a_max, T_min=T0)
        self.traj = Trajectory1D(t, [(T, c), (INF, np.array([target, 0, 0, 0, 0, 0.0]))])
        self.state = self.MOVING

    def _advance(self, t: float) -> None:
        while True:
            if self.state == self.MOVING and t >= self.traj.t0 + self.traj.segments[0][0]:
                t_end = self.traj.t0 + self.traj.segments[0][0]
                self.s_rest = self.target
                self.strokes += 1
                if self.cycles > 0 and self.strokes >= 2 * self.cycles:
                    self.state = self.FINISHED
                    self.traj = Trajectory1D(t_end, [(INF, np.array([self.s_rest, 0, 0, 0, 0, 0.0]))])
                    return
                if self.dwell > 0:
                    self.state, self._dwell_until = self.DWELL, t_end + self.dwell
                    self.traj = Trajectory1D(t_end, [(INF, np.array([self.s_rest, 0, 0, 0, 0, 0.0]))])
                else:
                    self._plan(t_end, self.s_rest, 0.0, 0.0)
            elif self.state == self.DWELL and t >= self._dwell_until:
                self._plan(self._dwell_until, self.s_rest, 0.0, 0.0)
            elif self.state == self.STOPPING and t >= self.traj.t0 + self.traj.segments[0][0]:
                self.s_rest = poly_eval(self.traj.segments[0][1], self.traj.segments[0][0])[0]
                self.state = self.SUSPENDED
                self.traj = Trajectory1D(self.traj.t0 + self.traj.segments[0][0], [(INF, np.array([self.s_rest, 0, 0, 0, 0, 0.0]))])
            else:
                return

    def eval(self, t: float) -> tuple[float, float, float]:
        if self.traj is None:
            return self.s_rest, 0.0, 0.0
        self._advance(t)
        return self.traj(t)

    def start(self, t: float) -> None:
        """Begin the first stroke from rest at the current rest position."""
        if self.enabled and self.state in (self.IDLE, self.SUSPENDED):
            self._plan(t, self.s_rest, 0.0, 0.0)

    def suspend(self, t: float) -> None:
        if self.state in (self.MOVING,):
            s, v, a = self.eval(t)
            if self.state != self.MOVING:               # the stroke happened to finish exactly now
                return
            self.traj = stop_trajectory(t, s, v, a, self.stop_time, self.a_max, p_bounds=(0.0, self.D))
            self.state = self.STOPPING
        elif self.state == self.DWELL:
            self.state = self.SUSPENDED

    def resume(self, t: float) -> None:
        if self.state == self.STOPPING:
            s, v, a = self.eval(t)
            if self.state == self.STOPPING:
                self._plan(t, s, v, a)
        elif self.state == self.SUSPENDED:
            self._plan(t, self.s_rest, 0.0, 0.0)

    @property
    def moving(self) -> bool:
        return self.state in (self.MOVING, self.STOPPING)


# --------------------------------------------------------------------------- nominal closed forms

class ApproachProfile:
    """Original closed-form nominal approach: quintic from rest to the creep point at the creep speed
    (duration chosen so that the peak speed equals v_peak), then constant creep speed, clamped at
    z_floor. Reference implementation for `descent_trajectory`."""

    def __init__(self, z0: float, z_creep: float, z_floor: float, v_peak: float, v_creep: float) -> None:
        self.z0, self.z_creep, self.z_floor, self.v_creep = float(z0), float(z_creep), float(z_floor), float(v_creep)
        dz = self.z_creep - self.z0
        if dz >= 0.0 or v_peak <= 0.0:
            self.T, self.coef = 0.0, None
            self.z_creep = self.z0
            return
        lo, hi = abs(dz) / v_peak, 2.0 * abs(dz) / max(v_peak, 1e-9)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if peak_speed_accel(quintic_coef(self.z0, 0.0, 0.0, self.z_creep, -self.v_creep, 0.0, mid), mid)[0] > v_peak:
                lo = mid
            else:
                hi = mid
        self.T = hi
        self.coef = quintic_coef(self.z0, 0.0, 0.0, self.z_creep, -self.v_creep, 0.0, self.T)

    def __call__(self, t: float) -> tuple[float, float, float]:
        if self.coef is not None and t < self.T:
            return poly_eval(self.coef, t)
        z = self.z_creep - self.v_creep * max(0.0, t - self.T)
        if z <= self.z_floor:
            return self.z_floor, 0.0, 0.0
        return float(z), -self.v_creep, 0.0


class SlideProfile:
    """Original closed-form stroke profile s(t) in [0, D]: rest-to-rest quintics 0 -> D -> 0 -> ...,
    stroke time T = 1.875 D / v_peak, optional dwell, cycle limit (0 = unlimited)."""

    def __init__(self, distance: float, v_peak: float, cycles: int = 0, dwell: float = 0.0) -> None:
        self.D = float(distance)
        self.T = 1.875 * self.D / v_peak if v_peak > 0 and self.D > 0 else 0.0
        self.cycles = int(cycles)
        self.dwell = float(dwell)

    def strokes_completed(self, t: float) -> int:
        if self.T <= 0 or t <= 0:
            return 0
        n = int(t // (self.T + self.dwell))
        return min(n, 2 * self.cycles) if self.cycles > 0 else n

    def __call__(self, t: float) -> tuple[float, float, float]:
        if self.T <= 0 or t <= 0:
            return 0.0, 0.0, 0.0
        period = self.T + self.dwell
        n = int(t // period)
        if self.cycles > 0 and n >= 2 * self.cycles:
            return 0.0, 0.0, 0.0
        tau = t - n * period
        forward = n % 2 == 0
        if tau >= self.T:
            return (self.D if forward else 0.0), 0.0, 0.0
        u = tau / self.T
        p = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        v = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / self.T
        a = (60 * u - 180 * u ** 2 + 120 * u ** 3) / self.T ** 2
        if forward:
            return self.D * p, self.D * v, self.D * a
        return self.D * (1 - p), -self.D * v, -self.D * a
