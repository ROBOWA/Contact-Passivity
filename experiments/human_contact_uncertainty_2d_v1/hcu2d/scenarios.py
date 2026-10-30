"""Scenario generation with paired hidden truth (intent, gains, location, noise) per seed.

The same (scenario, seed) yields identical hidden human parameters, OU paths and sensor-noise
sequences for every controller; human forces are regenerated from the same hidden intention and
the actual closed-loop state (no force-trace replay).
"""
import numpy as np

from .truth import HumanEvent, Profile

FAMILIES_TRAIN = ("hold", "assist", "reverse", "normal_in", "oblique_out")
FAMILIES_HELDOUT = ("oscillate", "fast_assist")


def _gains(rng, cfg, mismatch=1.0):
    h = cfg["human"]
    K = np.array(h["K"]) * np.exp(rng.normal(0, h["gain_logstd"], 2)) * mismatch
    D = np.array(h["D"]) * np.exp(rng.normal(0, h["gain_logstd"], 2)) * mismatch
    return K, D


def intent_profiles(family, rng, scale=1.0):
    """Scripted Δa(τ) per axis for an intent family (world frame = wall frame)."""
    u = lambda a, b: rng.uniform(a, b)
    if family == "hold":
        px = Profile(steps=[(u(0.2, 0.8), u(-0.006, 0.006), 0.6)])
        py = Profile(steps=[(u(0.2, 0.8), u(-0.004, 0.004), 0.6)])
    elif family == "assist":
        px = Profile(vel=[(0.0, scale * u(0.05, 0.12), 0.3)])
        py = Profile(steps=[(0.3, u(-0.005, 0.005), 0.5)])
    elif family == "fast_assist":                       # held-out speed range
        px = Profile(vel=[(0.0, u(0.16, 0.22), 0.25)])
        py = Profile()
    elif family == "reverse":
        v1, v2, t1 = u(0.05, 0.12), -u(0.04, 0.09), u(0.8, 1.2)
        px = Profile(vel=[(0.0, v1, 0.3), (t1, v2 - v1, 0.4)])
        py = Profile()
    elif family == "normal_in":
        px = Profile(steps=[(0.2, u(-0.01, 0.01), 0.5)])
        py = Profile(steps=[(0.1, -scale * u(0.02, 0.05), 0.5)])
    elif family == "oblique_out":
        px = Profile(steps=[(0.1, -scale * u(0.02, 0.05), 0.6)])
        py = Profile(steps=[(0.1, scale * u(0.02, 0.05), 0.6)])
    elif family == "oscillate":                          # held-out family
        px = Profile(sin=(u(0.02, 0.04), u(1.0, 1.5), 0.3))
        py = Profile(sin=(u(0.005, 0.015), u(1.0, 1.5), 0.3))
    elif family == "push_normal_fixed":                  # moment-arm scenario: identical nominal force
        px = Profile()
        py = Profile(steps=[(0.05, -0.04, 0.4)])
    else:
        raise ValueError(family)
    return px, py


SCENARIOS = {
    # name: dict(family, t_on, t_off, s range, residual, extra)
    "no_human": dict(events=[]),
    "block_release": dict(events=[dict(family="hold", t_on=1.6, t_off=4.0)]),
    "assist_reverse": dict(events=[dict(family="reverse", t_on=1.6, t_off=4.2)]),
    "normal_oblique": dict(events=[dict(family="normal_in", t_on=1.4, t_off=2.8),
                                   dict(family="oblique_out", t_on=4.0, t_off=5.4)]),
    "moment_arm": dict(events=[dict(family="push_normal_fixed", t_on=1.6, t_off=3.6)]),
    "bias_ood": dict(events=[dict(family="oscillate", t_on=1.6, t_off=4.4)], b_f=[0.8, -0.5], b_s=0.06),
    "ood_intent": dict(events=[dict(family="oscillate", t_on=1.6, t_off=4.4)]),
    "sensor_bias": dict(events=[dict(family="reverse", t_on=1.6, t_off=4.2)], b_f=[0.8, -0.5], b_s=0.06),
    "low_BH": dict(events=[dict(family="hold", t_on=1.6, t_off=4.0)]),
    "gp_data": dict(events=[dict(family="hold", t_on=1.0, t_off=3.6)]),
    "infeasible_init": dict(events=[dict(family="hold", t_on=1.6, t_off=4.0)], BH0=0.004, preload=0.002),
}


def make(name, seed, cfg, overrides=None):
    """Build the hidden truth and sensor-noise sequences for (scenario, seed)."""
    spec = dict(SCENARIOS[name])
    spec.update(overrides or {})
    ss = np.random.SeedSequence([hash_name(name), int(seed)])
    r_h, r_ou, r_sens, r_plan = [np.random.default_rng(s) for s in ss.spawn(4)]
    b, t = cfg["body"], cfg["task"]
    events = []
    for k, e in enumerate(spec["events"]):
        mis = spec.get("gain_mismatch", 1.0)
        K, D = _gains(r_h, cfg, mis)
        fam = e["family"] if spec.get("family") is None else spec["family"]
        px, py = intent_profiles(fam, r_h, spec.get("intent_scale", 1.0))
        if "s" in spec:
            s = float(spec["s"])
        else:
            s = float(r_h.uniform(-0.15, 0.15))
        jitter = r_h.uniform(-0.1, 0.1)
        residual = spec.get("residual", True)
        events.append(HumanEvent(e["t_on"] + round(jitter / 0.02) * 0.02, e["t_off"] + round(jitter / 0.02) * 0.02,
                                 s, K, D, px, py, residual, r_ou, cfg["human"], family=fam))
    n_ticks = int(round(t["T"] / 1e-3))
    sens = cfg["sensors"]
    b_s = spec.get("b_s", sens["b_s"])
    b_s = float(r_sens.normal(0.0, sens["sigma_bs"])) if b_s is None else float(b_s)
    noise = {"eps_m": r_sens.normal(0.0, sens["sigma_m"], (n_ticks + 1, 2)),
             "eps_s": r_sens.normal(0.0, sens["sigma_es"], n_ticks // 40 + 2),
             "b_f": np.array(spec.get("b_f", sens["b_f"]), float), "b_s": b_s}
    gap0 = -spec["preload"] if "preload" in spec else t["gap0"]
    q0 = np.array([t["x_start"] - b["r_T"][0], gap0 - b["r_T"][1], 0.0])
    acc = dict(cfg["accounts"])
    if name == "low_BH":
        acc["BH0"] = spec.get("BH0", 0.02)
    if "BH0" in spec:
        acc["BH0"] = spec["BH0"]
    return {"name": name, "seed": int(seed), "events": events, "noise": noise, "q0": q0,
            "accounts": acc, "plan_rng": r_plan, "spec": {k: v for k, v in spec.items() if k != "events"},
            "event_specs": spec["events"]}


def hash_name(name):
    return int(sum((i + 1) * ord(c) for i, c in enumerate(name)) % 100003)


def describe(sc):
    return [{"family": ev.family, "t_on": ev.t_on, "t_off_script": ev.t_off_script, "s": ev.s,
             "K": ev.K.tolist(), "D": ev.D.tolist(), "residual": ev.residual,
             "profile_x": ev.px.to_dict(), "profile_y": ev.py.to_dict()} for ev in sc["events"]]
