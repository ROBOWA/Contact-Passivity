# Canonical source and controller semantics audit

The canonical source tree is Stage 2 `41359f20430c930eec5afaebffa2cfbc482f07da`;
the numerical package was frozen at `81dffa64109752a9a027975e7b831daa4ff91e86`.
This audit reads code and raw logs independently of figure styling. Local-only
2-DOF inputs live in the protected original worktree; their role is to establish
scientific questions and signal semantics, never to substitute 2-DOF traces for KUKA.

The original mismatch is independently configured: planar plant friction is
`0.3` in `mujoco_sliding/models/planar_two_link.xml:41`, while scenario B/D
sets predictor `mu_hat=0.20` (`experiments.py:121-130`).
`contact_model.py:53-88` uses the predictor coefficient and constructs the
task-error/residual decomposition; the raw powers satisfy its declared
identity. No matching KUKA decomposition is silently inferred.

## Naming and causal comparisons

`paper/README.md:53-58`, `paper/results_section.tex:20-42` and
`mujoco_sliding/experiments.py:466-484` establish that paper 2-DOF C3 and C4
both run implementation `C3_dual_ledger_qp`, differing only in governor enable.
The legacy `C4_dual_ledger_safe_scalar` is a different diagnostic. The module
docstring in `paper/make_figures.py:5-14` mentions that legacy mode as part of
the wider data bundle; the actual figure functions and governor loader are
authoritative about which traces the report uses.

KUKA UF/A/B/C/D are a different naming system. UF is off, A is the frozen
Stage 1 endpoint filter, B checks physics prefixes, C adds finite reserve,
backup selection and virtual progression, and D additionally checks task
constraints in nominal/candidate acceptance and adds explicit QP rows.
See `kuka_table_sim/energy_experiments.py:99-109`,
`energy_guard.py:63-66,123-177`, and `energy_qp.py:65-125`.
C already checks task validity of every reaction-delay and backup segment
(`energy_guard.py:100-103`, `energy_prediction.py:190-192`). Thus C/D equality
does not imply that C has no task checking. B/D is not governor off/on.

## Ports, frames and actual work

The force is ON the robot in world coordinates, applied to a body-fixed
point, link5 local `(0, .04, .12)` in the main run and link3 local
`(.02, .04, .16)` in the elbow fixture (`energy_experiments.py:33-39`).
`oracle_physics.py:71-86` recomputes the moving world point and its Jacobian,
uses `v_H = J_point(q) qdot`, computes `p_H = F_H dot v_H`, verifies
`p_H = (J_point.T F_H) dot qdot`, and applies force exactly once after
restoring base generalized forces. Human moment is explicitly zero in
`mj_applyFT`; task-contact moment power is included by `TaskMeter`
(`oracle_physics.py:97-129`). TCP velocity belongs to task tracking,
not to the forearm force meter. A force sign is not enough to label helping:
the measured power sign determines charging/discharging.

Physics `h=.00025 s`, control `dt=.001 s`, four samples per held complete
joint-torque command (`energy_prediction.py:72-74`). Human point/J/velocity
are read after step1 and before integration; contact is solved at that same
state by step2. Work uses the recorder's signed left rectangle `p_H*h`,
not trapezoids or decimated power (`recorder.py:155-195`,
`energy_guard.py:196-218`). The old 2-DOF runtime instead explicitly pairs
the held force with next-step end-effector velocity
(`mujoco_sliding/passivity_qp.py:780-798`). This sampling distinction must be
preserved; one cannot infer a common quadrature from matching signal names.

## Ledgers, limits and forecasts

Both KUKA energy entries receive identical scalar actual human work, with no
upper/lower clipping: Stage 1 `oracle.py:123-136`; Stage 2
`energy_guard.py:208-211`. The predictor likewise broadcasts its scalar work
to both accounts (`energy_prediction.py:135`). Equal initial budgets imply
exactly coincident H/R. These are not independent model-error accounts.
The task meter is separate and is not subtracted from a measured forearm
wrench. `p_delta` is not an implemented KUKA signal; treating a missing
power as a measured zero would invent the experiment.

The minimum defaults to zero (`energy_guard.py:56-60`); the main initial
budget comes from case metadata, not a plotting constant. No finite E_max,
human peak-power constraint or human force cap is implemented. Cumulative
NET outgoing is `-sum(dW_H)`; gross outgoing is `sum(max(-dW_H,0))`.
Recharge can make gross exceed the net budget. Raw negative prefixes stay
visible, including A endpoint-valid/physics-invalid behavior. Whole-port
offline integration has no controller threshold or stopping semantics.

The predictor has one shared immutable model and receives only the measured
current force vector, not the future scripted pulse
(`energy_prediction.py:1-5,69-84`; `energy_guard.py:117-124`). It conditions
the task-contact forecast on that currently held human force/location.
Its nominal no-governor controller is a diagnostic counterfactual evaluated
on the actual current plant state (`energy_guard.py:178-187`), not an online
input to the selected governed controller. No independently configured
plant/predictor mismatch residual meter exists. The finite backup horizon
and empirical reserve guard do not prove recursive feasibility or a bound
for an arbitrary within-tick force change.

## References, intervention and release

`ProgressStroke` preserves original spatial endpoints and changes progression
through a coherent virtual clock (`energy_state.py:69-105`). It does not
capture the old PoC stop-time spatial anchor. Tangential direction is world
+x by the actual default controller configuration
(`force_control.py:108,133,174-183`), with desired normal contact 10 N.
The internal FSM may lower its commanded force target during recovery;
task quality must still be measured against the original 10 N target.

The decision variable is seven joint torques. Panel F8(d) must show a
declared torque intervention norm [N m]. Difference from governed nominal
and difference from the diagnostic ungoverned nominal answer different
questions and must be labeled separately. There is no unique scalar
tangential force inverse. First task-QP intervention, torque correction,
alpha-target change and actual alpha slowdown are distinct events.

Original release metrics start at ramp-out onset, using a fixed one-second
window (`mujoco_sliding/experiments.py:145-148`; `paper/make_figures.py:619-637,744-758`).
KUKA plots must annotate both ramp-out and pulse end; fixed windows should
be reported from both anchors. A stopped trace has no post-stop samples,
recovery, or assumed continuation. The tiny-budget D stop is a computational
trial stop before executing an unverified command, not physical emergency braking.

## Archived signal completeness

`scripts/energy_stage2_report.py:83-98` preserves every physics sample from
8.71 s and every nonzero-human-power sample. It also preserves all control
samples. Omitted pre-task human powers are exactly zero by the compaction
contract, but pre-task task-contact powers are not recoverable from compact
traces. A whole-port task-work integral beginning before 8.71 s needs the
preserved FULL `baseline_log.npz`; otherwise label its initial time 8.71 s.
Do not zero-fill pre-task task power. Original control-recorded position
and velocity signals are sampled at 1 ms and should not be presented as
full 0.25 ms physics tracking extrema.

## Claim boundaries

No original family is an exact controller/scenario reproduction for KUKA.
F3/F5/F7/F8 are adaptations. F4/F6 mismatch questions are unsupported, while
one combined nonempty diagnostic can show the current account identity.
Archived D has zero recorded solver calls and no selected accepted QP rows in
all 13 cases, and C/D executed trajectories
coincide. This supports the current reserve/governor/backup behavior in those
fixtures, not active task-QP superiority. A separate focused QP test verifies
the implemented rows, without promoting that test to a matrix demonstration.
Some rejected candidate branches discard their QP timing/attempt diagnostics
after reserve-budget or reaction-delay rejection (`energy_guard.py:143-153`),
so the record cannot exclude unlogged solver work inside those rejected plans.
No report may infer global robot passivity, a certified model-error envelope,
general human safety, gross/peak limits, real-time operation or hardware
emergency-stop behavior from this evidence.
