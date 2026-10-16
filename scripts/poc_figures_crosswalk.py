"""Canonical scientific crosswalk; reporting only, with no controller imports.

Run from the repository root with --out NEW_REPORT_ROOT. The figure builder may
call write_crosswalk(out) to reproduce the same CSV and source audit narrative.
Line citations refer to the untouched Stage 2 source tree at 41359f2.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARCHIVE = "docs/energy_guard_stage2/20261015-c5601e5e/final_02"
POC = "docs/kuka_poc_figures/20261016-41359f2"

CROSSWALK = [
    dict(figure="F3", original_filename="fig3_false_depletion.pdf",
         original_question="Does exempting useful modeled task work avoid false whole-port depletion?",
         original_function="paper/make_figures.py:340-395 (fig3)",
         original_scenario="A_exact_no_human",
         original_controllers="2-DOF C0 nominal; C1 whole-port QP; C3 dual-ledger QP, governor off",
         original_panels="a: E_W, E_R, E_H and floors [J]; b: F_n and desired F_n [N]; c: v_t and desired v_t [m/s]",
         original_signals="sp_e_w/sp_e_r/sp_e_h; f_n/f_n_desired; ee_vel[:,0]/vx_desired; sp_active/sp_active_any intervention",
         original_data="results_passivation_iter2/A_exact_no_human/{C0_nominal,C1_whole_port_qp,C3_dual_ledger_qp}/log.npz",
         kuka_sources=f"PLOTTED: {POC}/inputs/nohuman-zero__D/executed.npz, preserving full baseline/guard data including startup; OTHER POLICIES AUDITED: {ARCHIVE}/raw/nohuman-{{zero,ample}}__{{UF,A,B,C,D}}/trace.npz",
         kuka_controllers="KUKA UF/A/B/C/D; no closed-loop whole-port controller",
         kuka_panels="a: actual E_H=E_R and floor [J], optional offline task-work account [J]; b: normal tool/table contact and original target [N]; c: TCP along-path velocity and executed governed reference [m/s]",
         kuka_signal_mapping="energy[:,0:2]; normal_force; reference__v[:,0], reference__v_d[:,0]; task_power*h offline only",
         status="adapted comparison", supported_subset="No-human task-work exemption and matched transparency; optional offline whole-port work diagnostic",
         adaptation="Plot supported no-human transparency and separately labeled offline task work; the missing closed-loop whole-port baseline prevents a false-depletion stopping comparison.",
         unsupported_claim="Offline ledger depletion cannot show that a whole-port controller would stop the robot.",
         required_followup="Implement and separately validate a genuine whole-port controller in a future controller project.",
         source_citations="paper/results_section.tex:168-203; kuka_table_sim/oracle.py:123-136; kuka_table_sim/energy_guard.py:208-218; scripts/energy_stage2_report.py:83-98"),
    dict(figure="F4", original_filename="fig4_mismatch_residual.pdf",
         original_question="Does task-model error deplete the residual ledger while leaving the human ledger unchanged?",
         original_function="paper/make_figures.py:402-454 (fig4)", original_scenario="B_mismatch_no_human",
         original_controllers="2-DOF C3 dual-ledger QP and C1 whole-port QP, governor off; independently configured friction predictor",
         original_panels="a: E_H, E_R and floor/interventions [J,s]; b: signed mismatch p_delta and human p_H [W]",
         original_signals="sp_e_h/sp_e_r; sp_p_delta_act/sp_p_h_act; 50 ms display mean only for p_delta",
         original_data="results_passivation_iter2/B_mismatch_no_human/{C1_whole_port_qp,C3_dual_ledger_qp}/log.npz",
         kuka_sources=f"PLOTTED COMBINED F4/F6: {POC}/inputs/main__D/executed.npz; NO-HUMAN CONTEXT IN F3: {POC}/inputs/nohuman-zero__D/executed.npz; source audit of shared immutable model and same-work accounts",
         kuka_controllers="KUKA D identity diagnostic only",
         kuka_panels="Optional supported diagnostic: coincident no-human E_H/E_R and measured p_H; p_delta is not implemented",
         kuka_signal_mapping="energy[:,0:2]; power; p_delta=not implemented (never substitute zeros)",
         status="unsupported", supported_subset="Existing-account identity diagnostic; combine with F6 rather than duplicate it",
         adaptation="Document that both KUKA accounts receive the same human work and that there is no independently configured mismatch-accounting experiment.",
         unsupported_claim="Constant E_R without human input does not establish detection or accounting of model mismatch.",
         required_followup="A separately configured plant/predictor mismatch and a physically consistent residual-work meter need future controller work.",
         source_citations="mujoco_sliding/experiments.py:121-125,155-161; mujoco_sliding/models/planar_two_link.xml:41; mujoco_sliding/contact_model.py:53-88; mujoco_sliding/passivity_qp.py:688-704,789-798; kuka_table_sim/energy_prediction.py:72-81,135; kuka_table_sim/energy_guard.py:208-211"),
    dict(figure="F5", original_filename="fig5_human_limiting.pdf",
         original_question="How do dedicated human-port constraints alter power and cumulative work during blocking?",
         original_function="paper/make_figures.py:461-504 (fig5)", original_scenario="C_oracle_blocking",
         original_controllers="2-DOF C0 nominal; C2 residual-only QP; C3 dual-ledger QP, governor off",
         original_panels="a: -p_H, positive robot-to-human [W]; b: cumulative NET outgoing human work [J]; c: raw E_H and floor [J]",
         original_signals="-sp_p_h_act; -cumsum(sp_p_h_act)*meta_timestep; sp_e_h; actually enforced power limit and energy floor",
         original_data="results_passivation_iter2/C_oracle_blocking/{C0_nominal,C2_residual_qp,C3_dual_ledger_qp}/log.npz",
         kuka_sources=f"{ARCHIVE}/raw/main__{{UF,A,B,C,D}}/{{trace.npz,metadata.json,events.json.gz,failure.json}}",
         kuka_controllers="KUKA UF nominal matched reference; A endpoint; B prefix; C reserve/governor; D adds task rows",
         kuka_panels="a: -p_H [W], no enforced peak-power line; b: net and separately identified gross outgoing [J]; c: raw E_H and implemented zero floor [J]",
         kuka_signal_mapping="-power; -cumsum(power*h); cumsum(maximum(-power*h,0)); energy[:,0]; actual control intervention and termination events",
         status="adapted comparison", supported_subset="Complete executed blocking traces and current KUKA policy ablations, including stops",
         adaptation="Use KUKA UF/A/B/C/D as their actual policy roles, retaining signed-energy failures and early stops without implying residual-only or peak-power baselines.",
         unsupported_claim="A signed-energy budget is neither a gross-work bound nor a human peak-power or force cap; KUKA has no 2-DOF C2 equivalent.",
         required_followup="Peak-power protection and an independent mismatch/residual policy would require separately designed controller extensions.",
         source_citations="paper/results_section.tex:206-226; kuka_table_sim/energy_experiments.py:33-39,99-109; kuka_table_sim/oracle_physics.py:61-86; kuka_table_sim/energy_guard.py:208-218; kuka_table_sim/energy_qp.py:65-81"),
    dict(figure="F6", original_filename="fig6_combined_decomposition.pdf",
         original_question="Can simultaneous human interaction and model mismatch be separated as p_R=p_H+p_delta?",
         original_function="paper/make_figures.py:511-573 (fig6)", original_scenario="D_mismatch_blocking",
         original_controllers="2-DOF C3 dual-ledger QP, governor off; friction predictor mismatch",
         original_panels="a: E_H and E_R [J]; b: signed human, mismatch and residual powers [W]; original presentation ends at ramp-out onset",
         original_signals="sp_e_h/sp_e_r; sp_p_h_act/sp_p_delta_act/sp_p_r_act; separate signed lines preferable to old stacked display",
         original_data="results_passivation_iter2/D_mismatch_blocking/C3_dual_ledger_qp/log.npz",
         kuka_sources=f"{ARCHIVE}/raw/main__D/trace.npz; actual same-work source audit",
         kuka_controllers="KUKA D, coincident-account diagnostic only",
         kuka_panels="Supported identity subset: E_H=E_R [J] and measured p_H [W]; mismatch/residual decomposition is not implemented",
         kuka_signal_mapping="energy[:,0:2]; power; p_R label permitted only as implemented copy of p_H; p_delta=not implemented",
         status="unsupported", supported_subset="Combined F4/F6 figure may show account identity without inventing mismatch power",
         adaptation="Show the implemented coincident human-work accounts and identify the absent model-mismatch mechanism instead of manufacturing a separating residual trace.",
         unsupported_claim="Coincident H/R curves cannot demonstrate two-account model-error separation or a dual-account advantage.",
         required_followup="Implement and validate common-port or correctly transformed multiport residual-work decomposition in a distinct future controller change.",
         source_citations="paper/results_section.tex:56-93; mujoco_sliding/passivity_qp.py:688-704,789-798; kuka_table_sim/oracle.py:123-136; kuka_table_sim/energy_prediction.py:135; kuka_table_sim/energy_guard.py:208-211"),
    dict(figure="F7", original_filename="fig7_charge_discharge.pdf",
         original_question="Does positive human work recharge a ledger that is later discharged during blocking?",
         original_function="paper/make_figures.py:580-612 (fig7)", original_scenario="F_helping_blocking",
         original_controllers="2-DOF C3 dual-ledger QP, governor off; two continuous force pulses",
         original_panels="a: E_H with implemented floor/cap [J]; b: signed p_H, helping and blocking phases [W]",
         original_signals="sp_e_h; sp_p_h_act; actual capped per-step ledger update",
         original_data="results_passivation_iter2/F_helping_blocking/C3_dual_ledger_qp/log.npz",
         kuka_sources=f"{POC}/inputs/help-then-block__D/executed.npz; full new raw {POC}/inputs/new_run_raw/help_then_block_01/guard.npz; repeated run {POC}/inputs/help-then-block-repeat__D/executed.npz; frozen configuration {POC}/inputs/new_runs_config.json; one complete state/account history per run",
         kuka_controllers="KUKA D, unchanged guard/controller configuration; optional UF matched reference",
         kuka_panels="a: raw uncapped E_H and zero floor [J]; b: measured signed p_H [W] with phase shading based on actual work sign",
         kuka_signal_mapping="physics work/energy; force dot actual human-point velocity; no finite E_max and no discarded charging work",
         status="adapted comparison", supported_subset="Bidirectional work accounting in one continuous KUKA run; no cap experiment",
         adaptation="Use a newly frozen continuous force profile and retain uncapped charging, without importing the original finite cap or power threshold.",
         unsupported_claim="Force direction alone does not prove helping; separate runs cannot be spliced into a charging/discharging experiment; no cap is implemented.",
         required_followup="A finite upper energy cap would require explicit future controller design; it is outside this plotting project.",
         source_citations="mujoco_sliding/experiments.py:86-94,137-139; paper/results_section.tex:243-256; kuka_table_sim/oracle_physics.py:71-86; kuka_table_sim/energy_guard.py:117-120,208-211"),
    dict(figure="F8", original_filename="fig8_governor.pdf",
         original_question="Does an isolated governor prevent accumulated reference debt and the blocking-release catch-up transient?",
         original_function="paper/make_figures.py:619-758 (fig8)", original_scenario="governor_compare/C3_dual_ledger_qp/{governor_off,governor_on}",
         original_controllers="2-DOF paper C3/C4: the SAME implementation C3_dual_ledger_qp with governor false/true; never legacy C4_dual_ledger_safe_scalar",
         original_panels="a: human force/trigger [N]; b: actual/governed/original reference position [m]; c: tangential actual/governed velocity [m/s]; d: scalar tangential command [N]; e: normal-force error [N]; f: E_H/E_R [J]",
         original_signals="norm(f_h); ee_pos[:,0]/x_desired/sp_x_ref_original; ee_vel[:,0]/vx_desired; sp_u[:,0]; f_n-f_n_desired; sp_e_h/sp_e_r",
         original_data="results_passivation_iter2/governor_compare/C3_dual_ledger_qp/{governor_off,governor_on}/log.npz",
         kuka_sources=f"{ARCHIVE}/raw/main__{{UF,A,B,C,D}}/trace.npz; preserved full baseline_log/guard for ungoverned diagnostic reference; extension_main_D_01",
         kuka_controllers="Adapted KUKA A/B versus C/D interaction/recovery; C/D equality is the actual task-row ablation, not governor off/on",
         kuka_panels="Six shared-time panels; d becomes infinity norm of actual joint-torque intervention [N m]; original time reference marked diagnostic; whole interaction and recovery retained",
         kuka_signal_mapping="a: norm(force_H); b: x[:,0], x_des[:,0], paired UF x_des[:,0] as original-time diagnostic (not D online input); c: v[:,0], v_des[:,0]; d: max(abs(tau_exec-tau_ungoverned),axis=1), with A/UF tau_nom already ungoverned; e: normal_force-10; f: energy_H/energy_R; governed-nominal difference max(abs(tau_exec-tau_nom)) is a SEPARATE TABLE metric",
         status="adapted comparison", supported_subset="Current reserve/governor/backup policy interaction and recovery, with fixed windows and stopped traces unpadded",
         adaptation="Compare the implemented coupled reserve/governor policies, replace unavailable scalar command by joint-torque intervention, and state that the virtual clock preserves spatial endpoints rather than capturing the old moving anchor.",
         unsupported_claim="B versus D changes reserve/governor, backup selection and task protections; it is not an isolated governor ablation; joint torque has no unique inverse Cartesian command.",
         required_followup="A separately validated governor-only switch with fixed reserve/backup policy is required for the original causal governor claim.",
         source_citations="paper/README.md:53-58; mujoco_sliding/experiments.py:145-148,466-484; paper/make_figures.py:619-637,744-758; kuka_table_sim/energy_state.py:69-105; kuka_table_sim/energy_guard.py:63-66,123-177,181-187"),
]

ROLES = [
    ("2-DOF", "C0", "C0_nominal", "Nominal force/motion controller", "none", "off", "KUKA UF is the matched nominal reference role, with a different robot/controller"),
    ("2-DOF", "C1", "C1_whole_port_qp", "Whole measured contact-port QP", "E_W", "off", "No KUKA equivalent; offline work is diagnostic only"),
    ("2-DOF", "C2", "C2_residual_qp", "Model-aware residual-only QP", "E_R", "off", "No KUKA independent mismatch-residual equivalent"),
    ("2-DOF", "C3", "C3_dual_ledger_qp", "Independent human/residual QP and human power row", "E_H and E_R", "off", "No exact KUKA controller equivalent"),
    ("2-DOF", "C4", "C3_dual_ledger_qp", "Same QP as 2-DOF C3; only governor enabled", "E_H and E_R", "on", "Legacy C4_dual_ledger_safe_scalar is excluded"),
    ("KUKA", "UF", "OracleFilter(mode=off)", "Matched unfiltered reference; actual signed accounts logged", "E_H=E_R monitored", "off", "Useful nominal comparison; no enforced energy limit"),
    ("KUKA", "A", "OracleFilter(mode=active)", "Frozen Stage 1 endpoint joint-torque QP", "E_H=E_R endpoints", "off", "Can violate a physics prefix while endpoints pass"),
    ("KUKA", "B", "EnergyGuard(variant=B)", "Physics-prefix enforcement and nonlinear replay", "E_H=E_R every physics prefix", "off", "No reserve/governor; may explicitly stop"),
    ("KUKA", "C", "EnergyGuard(variant=C)", "B plus verified finite-horizon backup reserve and virtual-time governor", "E_H=E_R every physics prefix", "on, coupled to reserve/backup", "Backup/reaction-path task checks already exist in C"),
    ("KUKA", "D", "EnergyGuard(variant=D)", "C plus explicit per-stage nonlinear task acceptance and QP task rows", "E_H=E_R every physics prefix", "on, same as C", "Zero recorded selected-path QP solves in archived D matrix; equality with C is retained"),
]

COVERAGE = [
    ("Known human-point signed work accounting", "demonstrated", "Actual application-point J qdot; independent generalized power identity; no human moment applied"),
    ("No-human task-work exemption/transparency", "demonstrated", "Matched UF/A/B/C/D no-human traces; whole-port closed-loop false depletion remains missing"),
    ("Hard raw physics-prefix human-work protection", "demonstrated", "B/C/D executed samples, including explicit stops; finite fixtures only"),
    ("Reserve/governor interaction and recovery", "demonstrated", "Main and historical D completion with recovery; stress stops; current-held-force forecast only"),
    ("Bidirectional human ledger", "demonstrated on continuous frozen fixture", "Two fresh completed help/block runs independently audited with identical executed signals and initial/final states; measured charging retained without a finite upper cap"),
    ("Explicit per-stage task QP rows", "partially tested", "Implemented and exercised by separate focused test; archived D scenarios have zero recorded solver calls or selected accepted QP rows; rejected-plan solver diagnostics are incomplete"),
    ("Whole-port false-depletion causal comparison", "missing", "No closed-loop KUKA whole-port baseline"),
    ("Independent model-mismatch residual ledger", "missing", "Current actual R equals H; immutable shared model; no separating p_delta meter"),
    ("Isolated governor on/off ablation", "missing", "Reserve policy and governor are coupled in available variants"),
    ("Human peak-power/force or gross-energy protection", "missing", "Only signed work is constrained; no imposed human force/power cap"),
    ("Finite upper energy cap", "missing", "KUKA accounts charge without clipping"),
    ("Global/continuous passivity and recursive feasibility", "missing", "Finite sampled fixtures and finite-horizon policy; no invariant-set certificate"),
    ("Real-time 1 kHz execution", "missing", "Offline Python rollouts and measured wall-clock deadlines; simulated 1 ms is not a runtime claim"),
]


SEMANTICS = """# Canonical source and controller semantics audit

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
"""


def _csv(path, rows, names=None):
    with path.open("w", newline="") as handle:
        if names is None:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(handle)
            writer.writerow(names)
            writer.writerows(rows)


def write_crosswalk(out):
    out = Path(out)
    support = out / "support"
    support.mkdir(parents=True, exist_ok=True)
    _csv(out / "figure_crosswalk.csv", CROSSWALK)
    _csv(support / "controller_roles.csv", ROLES,
         ("system", "paper_or_variant_label", "implementation", "meaning", "accounts", "governor", "mapping_limit"))
    _csv(support / "proposal_coverage.csv", COVERAGE, ("capability", "status", "evidence_or_gap"))
    (support / "source_semantics.md").write_text(SEMANTICS)
    (support / "crosswalk.json").write_text(json.dumps(CROSSWALK, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_crosswalk(args.out)


if __name__ == "__main__":
    main()
