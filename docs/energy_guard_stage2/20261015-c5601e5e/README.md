# Stage 2 — primary engineering gates PASS

[Technical report and full comparison](final_02/report.md) ·
[Acceptance gates](final_02/acceptance.json) ·
[中文早间摘要](final_02/morning_summary_zh.md) ·
[Reproduction commands](reproduction.md) ·
[Three review rounds](reviews.md)

The opt-in D controller completes the main/historical regressions and all three
untouched held-outs within the frozen energy/contact/progress/recovery targets.
The 0.001 J stress case explicitly stops at 9.533 s; it is not a completed task.
All 70 tests pass. All 65 matched trial outcomes, including failures, are retained.
This is an offline, current-held-force oracle validation; it does not establish
recursive feasibility under changing human force or 1 kHz execution.

Numerical/control commit: `81dffa64109752a9a027975e7b831daa4ff91e86`.
Later commits contain tests, audit tools and reporting evidence.
New branch: `feature/kuka-energy-reserve-contact-20261015-c5601e5e`.
No remote push or merge was performed. Exact push command is in reproduction.md.

The original worktree `/Users/carl/code/contactSimulation`, frozen Stage 1 branch,
protected branch tips and old outputs remain intact. Final verification is in
[preservation_final.json](preservation_final.json); the verified external backup
holds all 540 pre-existing ignored/untracked evidence files. Full new logs remain
under `results/energy_guard_stage2/20261015-c5601e5e/`; committed audit traces retain
every task-window physics sample and every nonzero human-power sample in float64.
The only omitted powers are exactly zero. Complete failure states and lossless
candidate histories are retained; see [raw manifest](final_02/raw_manifest.json).
