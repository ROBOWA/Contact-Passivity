"""Frozen default configuration (mirrors formulation.md §13); JSON round-trip helpers."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT = {
    "body": {"m": 2.0, "I": 0.03, "L": 0.40, "H": 0.10, "r_T": [0.12, -0.10], "h_e": 0.05,
             "s_min": -0.18, "s_max": 0.18},
    "robot": {"B_R": [1.0, 1.0, 0.005], "u_max": [40.0, 40.0, 4.0]},
    "contact": {"k_n": 5000.0, "b_n": 150.0, "delta_s": 0.5e-3, "mu": 0.3, "v_eps": 5e-3},
    "task": {"f_d": 10.0, "v_nom": 0.08, "x_start": -0.25, "x_target": 0.25, "t_start": 0.6,
             "t_ramp": 0.5, "gap0": 0.002, "T": 9.0},
    "ctrl": {"K_t": 600.0, "D_t": 50.0, "k_f": 0.6, "k_i": 8.0, "I_max": 10.0, "D_n": 40.0,
             "K_th": 40.0, "D_th": 2.0, "mu_hat": 0.3, "v_app": 0.03, "D_brake": 400.0, "I_leak": 5.0,
             "D_brk": [40.0, 40.0, 1.5]},
    "human": {"K": [150.0, 150.0], "D": [15.0, 15.0], "gain_logstd": 0.2, "sigma_p": 0.4,
              "tau_p": 0.15, "c_g": 1.0, "v_g": 0.05, "F_rel": 35.0, "t_release_ramp": 0.02},
    "sensors": {"sigma_m": 0.25, "loc_rate": 25.0, "sigma_es": 0.03, "sigma_bs": 0.02,
                "b_f": [0.0, 0.0], "b_s": None},
    "accounts": {"Ec0": 0.6, "BH0": 0.15, "Emin": 0.0, "lam_E": 5.0, "lam_H": 5.0},
    "planner": {"dt_ctrl": 0.02, "N": 15, "dt_sub": 0.005, "N_s": 48, "alpha_E": 0.01,
                "alpha_H": 0.01, "beta": 2.0, "sig_v": [0.0, 0.5, 1.0, 1.5], "sig_K": [1.0, 0.35],
                "sig_F": [1.0, 0.5], "gp_window": 1.6, "gp_rate": 25.0,
                "cost": {"w_prog": 1.0, "e_prog": 0.02, "w_f": 1.0, "e_f": 3.0, "w_th": 0.5,
                         "e_th": 0.05, "w_u": 0.05, "u_scale": [20.0, 20.0, 2.0], "w_du": 0.05,
                         "w_res": 2.0, "R_task": 0.03}},
    "physics": {"h": 1e-3},
}


def default():
    return copy.deepcopy(DEFAULT)


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def save(cfg, path):
    Path(path).write_text(json.dumps(cfg, indent=1, sort_keys=True))


def load(path):
    return merge(DEFAULT, json.loads(Path(path).read_text()))
