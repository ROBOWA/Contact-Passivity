# View the recorded comparisons

On this Mac:

```sh
cd /Users/carl/code/contactSimulation-selective-20261016-263136e
PYTHONDONTWRITEBYTECODE=1 /Users/carl/code/contactSimulation/.venv/bin/mjpython scripts/selective_viewer.py
```

The window starts paused at 9.2 s in the main TCP-blocking K4 trial. Press P
to play. It replays the actual recorded joint poses and displays recorded force,
human power and H/R/W accounts. No controller or experiment is rerun.

| Key | Action |
|---|---|
| 0, 1, 2, 3, 4 | Switch K0–K4 at the same timeline position |
| P or Space | Pause/play |
| Left/right arrows | Seek backward/forward .25 s |
| R | Return to the starting time and pause |
| N / B | Next/previous fixture |
| comma / period | Half/double playback speed |
| Q | Close viewer |
| Mouse drag / scroll | Orbit, pan, zoom |

Magenta arrows show the measured human force at its actual moving application
point. The gold dot shows the tangential reference above the TCP height.
The force and energy readouts come from the logs. Unprotected negative accounts
remain visible. Stopped trials freeze at their last recorded sample and say
STOPPED; there is no invented continuation. The small-budget fixture contains
only K3/K4, so other mode keys display an unavailable notice.

Start another fixture or time:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/carl/code/contactSimulation/.venv/bin/mjpython \
  scripts/selective_viewer.py --case F_help_block --mode K3 --start 9.2 --play
```

Playback requires the full logs under `results/selective/final/`; regenerate
them with `scripts/selective_matrix.py` if checking out this branch elsewhere.
Use `--root OTHER_LOG_DIRECTORY` for another recorded matrix. The compact
Git-tracked plot archive does not include every robot joint pose.

The original live nominal viewer remains available:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/carl/code/contactSimulation/.venv/bin/mjpython \
  -m kuka_table_sim.run --controller force viewer
```
