"""Run one configurable Panda scenario headlessly."""

import argparse
import json
from dataclasses import asdict

from .config import MODES
from .experiments import SCENARIOS, scenario
from .simulation import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=SCENARIOS, default="C_ee_oblique_blocking")
    p.add_argument("--mode", choices=MODES, default="C3_dual_ledger_qp")
    p.add_argument("--duration", type=float)
    p.add_argument("--output", default="results_panda/single")
    p.add_argument("--governor", action="store_true")
    args = p.parse_args()
    cfg = scenario(args.scenario); cfg.passivation.mode = args.mode
    cfg.controller.governor.enabled = args.governor
    if args.duration is not None: cfg.duration = args.duration
    result = run(cfg); result.save(args.output)
    print(json.dumps(asdict(result.metrics), indent=2))


if __name__ == "__main__": main()
