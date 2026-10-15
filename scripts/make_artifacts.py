"""Regenerate docs/baseline_repair/: reports, plots and compact raw RunLogs for every scenario.

    python scripts/make_artifacts.py [--only nominal loss-reacquire ...]

Logs are windowed around the transients, optionally strided, and stored as float32 (time and work stay float64);
nothing is smoothed or resampled. Load one with `RunLog.load(path)`; `python -m kuka_table_sim.plot LOG --kind ...`
redraws a figure from it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kuka_table_sim import scenarios as sc                  # noqa: E402
from kuka_table_sim.plot import plot_scenario, plot_work     # noqa: E402

OUT = REPO / "docs" / "baseline_repair"
# name -> (log window, stride)
LOG_WINDOWS = {
    "nominal": ((6.65, 7.35), 1),                 # touchdown transient at full rate (0.25 ms physics, 1 ms control)
    "free-air-static": (None, 16),
    "free-air-dynamic": (None, 4),
    "fast-approach": ((0.0, 2.6), 4),
    "loss-short": ((11.9, 12.6), 1),
    "loss-reacquire": ((11.9, 13.2), 1),
    "loss-sustained": ((11.8, 26.0), 16),
    "removal-stress": ((11.8, 18.0), 16),
}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    for sub in ("reports", "logs", "plots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    names = args.only or [n for n in sc.NAMES]
    for name in names:
        t0 = time.perf_counter()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res, ok = sc.run_by_name(name)
        text = buf.getvalue() + f"(wall {time.perf_counter() - t0:.1f} s)\n"
        (OUT / "reports" / f"{name}.txt").write_text(text)
        if isinstance(res, sc.ScenarioResult):
            window, stride = LOG_WINDOWS[name]
            res.log.save(OUT / "logs" / f"{name}.npz", stride=stride, window=window, float32=True)
            plot_scenario(res, OUT / "plots" / f"{name}.png")
            if name == "nominal":
                plot_work(res.log, OUT / "plots" / "nominal-work.png", title="nominal: contact power and work")
        print(f"{name:18s} {'PASS' if ok else 'FAIL'}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
