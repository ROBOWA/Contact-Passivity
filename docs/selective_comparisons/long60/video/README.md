# Recorded K4 experiment

[Play the 60-second MP4](kuka_C_blocking_K4_60s.mp4).

1920 x 1080, 30 fps, H.264/yuv420p, normal speed, no audio. The full recorded
simulation runs from 0 to 60 s. A 5 N TCP-blocking interaction occurs at
9.5–10.9 s; the robot then continues sliding back and forth for about 49 s.
This is the exact-model K4 dual-account controller with its governor enabled.

The movie renders saved joint poses and displays logged force, power, accounts
and reference-clock speed. It does not rerun the controller or integrate the
plant. Human outgoing power is signed: negative values indicate human input.
H/R are protected; W is monitored and may become negative as task work proceeds.
The magenta force illustration is raised .06 m for visibility above the table,
with a thin leader to its actual application point. The gold dot marks the
tangential reference above the TCP height.

`kuka_C_blocking_K4_60s.frames.csv` maps every video frame to its original
recorded sample and measurements. The matching JSON contains source/video
SHA256 hashes, the pose-sequence hash, encoder version, and decode validation.
All 1,800 frames decode successfully; decoded duration is 60.0 s. The poster
shows the interaction at 10.4 s. Six focused video-timing/viewer tests passed.

Reproduce from the worktree root, with FFmpeg on PATH or the optional
`imageio-ffmpeg` package in a video environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/carl/code/contactSimulation/.venv/bin/python \
  scripts/selective_video.py --out results/selective/kuka_K4_60s_reproduction.mp4
```

Alternatively supply `--ffmpeg /path/to/ffmpeg`. The existing numerical Python
environment was left unchanged; this recording used the imageio-ffmpeg 0.6.0
wheel in `/tmp/kuka-video-encoder` for its FFmpeg 7.1 binary. The full input logs
are in `results/selective/long60/C_blocking/K4/`. Use a new output filename.
