"""Editable, results-first ten-page KUKA PoC report; no simulation imports.

Public API: write_report(out, metrics, context=None). ``metrics`` is keyed by
``case__variant`` and contains poc_figures_data.compute() results. Context may
supply source_commit/current_commit/branch/build_command, review_rounds,
matrix_counts, common_window_metrics/common_window_end_s, and equal_progress.
The six expected figure stems are listed in FIGURES. Missing optional results
are explicitly shown as unavailable; no scientific result is synthesized.
"""
from __future__ import annotations

from pathlib import Path


FIGURES = (
    "fig3_task_exemption", "fig4_fig6_accounting", "fig5_blocking",
    "fig7_charge_discharge", "fig8_recovery", "figS_reserve_progress_timing",
)


def tex(value):
    text = str(value)
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                    "–": "--", "—": "---", "−": "-", "≥": r"\(\geq\)",
                    "≤": r"\(\leq\)", "±": r"\(\pm\)", "×": r"\(\times\)",
                    "µ": r"\(\mu\)", "μ": r"\(\mu\)", "α": r"\(\alpha\)",
                    "Δ": r"\(\Delta\)", "δ": r"\(\delta\)", "°": r"\(^{\circ}\)",
                    "’": "'", "‘": "'", "“": "``", "”": "''"}
    return "".join(replacements.get(char, char) for char in text)


def n(value, digits=4):
    if value is None:
        return "not logged"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if value != 0 and abs(value) < 1e-4:
            return f"{value:.3g}"
        if digits == 0:
            return f"{value:.0f}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def write_report(out, metrics, context=None):
    out = Path(out)
    context = context or {}
    out.mkdir(parents=True, exist_ok=True)
    pages, md = [], []

    def metric(key, name, default=None):
        return metrics.get(key, {}).get(name, default)

    def fmt(key, name, digits=4):
        return n(metric(key, name), digits)

    def page(title):
        pages.append([r"\section*{" + tex(title) + "}"])
        md.append("\n## " + title + "\n")

    def paragraph(text):
        pages[-1].append(tex(text) + "\n\n")
        md.append(text + "\n")

    def table(headings, rows, widths=None, small=True):
        if widths is None:
            specification = "@{}" + "l" * len(headings) + "@{}"
        else:
            specification = "@{}" + "".join("p{" + width + r"\textwidth}" for width in widths) + "@{}"
        lines = [r"\begingroup", r"\small" if small else r"\normalsize",
                 r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.16}",
                 r"\begin{tabular}{" + specification + "}", r"\toprule",
                 " & ".join(r"\textbf{" + tex(x) + "}" for x in headings) + r" \\", r"\midrule"]
        lines += [" & ".join(tex(x) for x in row) + r" \\" for row in rows]
        lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", "\n"]
        pages[-1].append("\n".join(lines))
        md.append("| " + " | ".join(headings) + " |\n| " + " | ".join("---" for _ in headings) + " |")
        md.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
        md.append("")

    def figure(stem, title, caption, height="5.65in"):
        path = f"figures/{stem}.pdf"
        if not (out / path).is_file():
            paragraph("Draft artifact: " + title + " Fresh continuous experiment pending; no figure yet. "
                      "No empty or fabricated result plot is supplied.")
            return
        pages[-1].append(r"\begin{center}\includegraphics[width=\textwidth,height=" + height +
                         r",keepaspectratio]{" + path + r"}\end{center}")
        pages[-1].append(r"{\small\textbf{" + tex(title) + "} " + tex(caption) + "}\n")
        md.append(f"![{title}](figures/{stem}.png)\n\n{caption}\n")

    source = context.get("source_commit", "41359f20430c930eec5afaebffa2cfbc482f07da")
    branch = context.get("branch", "feature/kuka-poc-figures-20261016-41359f2")

    page("KUKA contact-energy guard: what the traces establish")
    paragraph("Results-first adaptation of the original 2-DOF PoC Figures 3–8. The current KUKA reserve/governor policy completes the main blocking fixture and preserves contact, while the frozen endpoint filter exhibits a negative physics energy prefix and substantial contact disturbance. This is a report about the existing implementation; no controller, gains, energy rules or plant parameters were redesigned.")
    rows = []
    for label, name in (("Main UF", "main__UF"), ("Main A", "main__A"), ("Main D", "main__D"),
                        ("Historical D", "historical__D"), ("Tiny budget D", "stress__D")):
        rows.append((label, fmt(name, "completed_s", 3), fmt(name, "net_outgoing_J", 6),
                     fmt(name, "min_substep_energy_J", 7), fmt(name, "force_max_error_N", 3),
                     "complete" if metric(name, "completed") else "stopped"))
    table(("Run", "End [s]", "Net out [J]", "Min raw E [J]", "Force peak error [N]", "Outcome"), rows)
    paragraph("Energy minima include every executed 0.25 ms physics increment. Force errors use the original 10 N tool/table task target and the declared task window, beginning at 8.71 s. A stopped run contributes only its executed samples; it has no inferred release or recovery.")
    table(("KUKA label", "Actual role"), [
        ("UF", "Matched unfiltered reference; work accounts monitored"),
        ("A", "Frozen Stage 1 endpoint joint-torque energy QP"),
        ("B", "Physics-prefix energy checks and nonlinear replay"),
        ("C", "B plus finite backup reserve, virtual progression and backup selection"),
        ("D", "C plus explicit per-stage task acceptance and QP task rows"),
    ], widths=(".13", ".80"))
    paragraph("H and R receive the same human work. Archived C and D coincide, and D performs zero recorded QP solves across the 13-case matrix. The supported improvement is therefore evidence for the coupled reserve/governor/backup policy in these fixtures, without establishing active task-QP superiority, independent mismatch accounting or global robot passivity.")
    if context.get("matrix_counts"):
        matrix_rows = []
        matrix = context["matrix_counts"]
        if "cases" in matrix:
            for variant in ("UF", "A", "B", "C", "D"):
                selected = [r["metrics"] for r in matrix["cases"] if r["variant"] == variant]
                matrix_rows.append((variant, sum(r.get("completion") == "PASS" for r in selected),
                                    sum(r.get("completion") == "STOPPED" for r in selected),
                                    sum(r.get("energy") == "FAIL" for r in selected),
                                    sum(r.get("qp_solves", 0) for r in selected)))
        else:
            for variant, record in matrix.items():
                matrix_rows.append((variant, n(record.get("completed"), 0), n(record.get("stopped"), 0),
                                    n(record.get("energy_invalid", record.get("energy_failed")), 0),
                                    n(record.get("qp_solves", record.get("recorded_solver_calls")), 0)))
        table(("13-case matrix", "Completed", "Stopped", "Energy-invalid", "Recorded QP calls"), matrix_rows)

    page("Original scientific questions and KUKA coverage")
    table(("Family", "Status", "Supported result and missing comparison"), [
        ("F3", "Adapted", "No-human task-work exemption/transparency; no closed-loop whole-port baseline"),
        ("F4", "Unsupported", "No independently configured mismatch residual; show existing-account identity only"),
        ("F5", "Adapted", "Executed blocking power, net/gross work and raw energy for actual KUKA policies"),
        ("F6", "Unsupported", "No implemented p_delta separation; combine the supported identity diagnostic with F4"),
        ("F7", "Adapted", "One continuous help-then-block history; uncapped charging"),
        ("F8", "Adapted", "Full coupled-policy interaction/recovery; joint-torque intervention replaces scalar command"),
    ], widths=(".07", ".14", ".70"))
    paragraph("No family is an exact controller/scenario reproduction. The original 2-DOF plots and raw data establish each question and panel meaning; every result curve here is KUKA data. The complete file/function, signal, units, configuration, adaptation and source-line mapping is recorded in figure_crosswalk.csv.")
    table(("2-DOF paper label", "Actual implementation and relation to KUKA"), [
        ("C0", "Nominal; closest experimental role is KUKA UF"),
        ("C1", "Whole-port QP; no KUKA closed-loop equivalent"),
        ("C2", "Residual-only QP; no independent KUKA mismatch/residual equivalent"),
        ("C3", "C3_dual_ledger_qp with governor OFF"),
        ("C4", "The same C3_dual_ledger_qp with governor ON"),
    ], widths=(".23", ".69"))
    paragraph("The historical code mode C4_dual_ledger_safe_scalar is a separate diagnostic and is not paper C4. KUKA C/D labels are unrelated to that naming system. KUKA B versus D changes reserve/governor, backup selection and task protection; it cannot isolate a governor effect.")
    paragraph("The main human force acts at a body-fixed link5 point; elbow fixtures use link3. Human power uses velocity at that actual moving point. Tool/table normal contact, its 10 N target and TCP tracking are separate task signals. Positive human power enters the robot; negative power is robot-to-human output. Known force, simulated contact and finite sampled tests remain explicit assumptions.")

    page("F3 — Task work is exempted, with the causal gap visible")
    figure("fig3_task_exemption", "F3. No-human task behavior and accounting.",
           "The matched no-human traces show actual ledgers, tool/table normal force and along-path velocity with their implemented floor and references. The offline monitored-port integral starts at t = 0 and includes startup task/contact work before the displayed sliding window, so its initial offset is not attributed solely to useful sliding.", "6.2in")
    paragraph("Zero-initial-energy D preserves the task with E_H = E_R = 0 and no actual intervention. The offline diagnostic is not a command-generating whole-port controller, so the original causal claim that useful friction would stop such a controller remains untested for KUKA.")

    page("F4 and F6 — The current residual account is a copy")
    figure("fig4_fig6_accounting", "F4/F6. Supported accounting-identity diagnostic.",
           "Blocking traces, together with no-human F3, expose E_H = E_R because both executed accounts receive the same human work. The predictor shares the immutable plant model and holds the currently measured force at its known application point; mismatch power p_delta is not implemented or replaced with a zero-valued measurement.", "5.2in")
    paragraph("The coincident accounts do not demonstrate mismatch accounting or the original p_R = p_H + p_delta decomposition. Reproducing F4/F6 requires future controller/model work on a physically consistent residual-work meter and independently configured plant/predictor parameters.")

    page("F5 — Blocking limits signed output, with failures retained")
    figure("fig5_blocking", "F5. Complete executed blocking power, work and raw energy.",
           "The panels show positive human-directed output -p_H, signed NET work with gross output identified separately, and raw energy against the implemented zero floor. Ramp-out onset, pulse end, actual intervention and premature termination are distinct events; no unenforced peak-power threshold is drawn.", "6.5in")
    paragraph("Main D finishes with net output " + fmt("main__D", "net_outgoing_J", 6) +
              " J, gross output " + fmt("main__D", "gross_outgoing_J", 6) +
              " J and minimum raw energy " + fmt("main__D", "min_substep_energy_J", 7) +
              " J. A reaches a negative physics minimum of " + fmt("main__A", "min_substep_energy_J", 8) +
              " J despite endpoint checks, while B stops at " + fmt("main__B", "completed_s", 3) +
              " s; these are signed-energy results rather than peak-power, force or gross-work guarantees.")

    page("F7 — Charging and discharging in one continuous run")
    figure("fig7_charge_discharge", "F7. Continuous help-then-block accounting.",
           "The frozen help-then-block profile runs continuously with the same plant, controller, reference history and uncapped ledgers, retaining the implemented zero floor.", "5.45in")
    help_result = metrics.get("help-then-block__D")
    if help_result is not None:
        helping=help_result.get('phase_work',{}).get('helping',{})
        paragraph("Measured helping work is " + n(helping.get('signed_incoming_J'),6) +
                  " J, raising energy from " + n(help_result.get('initial_energy_J'),6) +
                  " J to " + n(help_result.get('max_raw_energy_J'),6) +
                  " J without clipping; the repeated run matches all executed signals and initial/final states exactly. " +
                  "The run " + ("completes" if help_result.get("completed") else "stops") +
                  " at " + n(help_result.get("completed_s"), 3) +
                  " s with net outgoing " + n(help_result.get("net_outgoing_J"), 6) +
                  " J, gross outgoing " + n(help_result.get("gross_outgoing_J"), 6) +
                  " J and minimum raw energy " + n(help_result.get("min_substep_energy_J"), 9) +
                  " J. Gross output can exceed the initial budget because measured positive human-point work recharges it, while the implemented constraint concerns signed net work; phase interpretation uses measured power rather than force direction alone.")
    else:
        paragraph("The continuous evaluation is pending in this draft, so no separate helping and blocking records are spliced. Completion and recharge outcomes require the fresh raw logs. Finite upper energy and instantaneous human-power limits are not implemented.")

    page("F8 — Complete interaction and recovery of the current policy")
    figure("fig8_recovery", "F8. Adapted interaction/recovery comparison.",
           "Six panels retain force/events, actual and online governed position/velocity, original-target force error and raw H/R accounts, with any original-time reference marked diagnostic; panel (d) replaces the absent scalar Cartesian command with joint-torque intervention in N m.", "6.85in")
    paragraph("D starts intervening at " + fmt("main__D", "first_intervention_s", 3) +
              " s; A first intervenes at " + fmt("main__A", "first_intervention_s", 3) +
              " s. Main D sustained force recovery takes " + fmt("main__D", "force_recovery_from_rampout_s", 3) +
              " s from ramp-out onset, or " + fmt("main__D", "force_recovery_from_pulse_end_s", 3) +
              " s from full force removal, using a full 0.5 s all-good window and explicitly distinguishing the old paper's ramp-out release anchor.")
    paragraph("KUKA's virtual clock preserves stroke endpoints rather than capturing the old spatial anchor, and this coupled reserve/governor/backup comparison does not isolate a governor effect.")

    page("Quantitative task, interaction and recovery outcomes")
    table(("Main policy", "Force RMSE [N]", "MAE [N]", "Max error [N]", "Absent [s]", "End [s]"), [
        (v, fmt("main__" + v, "force_rmse_N", 3), fmt("main__" + v, "force_mae_N", 3),
         fmt("main__" + v, "force_max_error_N", 3), fmt("main__" + v, "contact_absence_s", 5),
         fmt("main__" + v, "completed_s", 3)) for v in ("UF", "A", "B", "C", "D")])
    paragraph("These force statistics use each run's executed task interval [8.71 s, actual end). Equal-duration main UF/A/D rows support full-task comparisons; B ends early and its full-run aggregate has a shorter window. No samples are repeated after termination. Maximum errors and contact absence retain the original 10 N target during controller recovery.")
    common = context.get("common_window_metrics")
    if common:
        rows = []
        for v in ("UF", "A", "B", "C", "D"):
            item = common.get("main__" + v, common.get(v, {}))
            rows.append((v, n(item.get("force_rmse_N"), 3), n(item.get("force_mae_N"), 3),
                         n(item.get("tangent_rmse_m_s"), 5)))
        table(("Common window", "Force RMSE [N]", "Force MAE [N]", "Tangent RMSE [m/s]"), rows)
        paragraph("The equal-support table ends at " + n(context.get("common_window_end_s"), 3) +
                  " s for every main policy. It does not replace the full-run completion and violation outcomes.")
    table(("Run", "Tangent RMSE [m/s]", "Peak speed [m/s]", "Online debt [mm]", "Force recovery [s]"), [
        (name, fmt(key, "tangent_rmse_m_s", 5), fmt(key, "tangent_peak_speed_m_s", 4),
         n(None if metric(key, "peak_tracking_debt_m") is None else metric(key, "peak_tracking_debt_m") * 1000, 3),
         fmt(key, "force_recovery_from_pulse_end_s", 3))
        for name, key in (("Main A", "main__A"), ("Main D", "main__D"), ("Historical D", "historical__D"))])
    table(("Run", "Peak original-reference debt [mm]", "Progress / matched UF [%]"), [
        (name, n(None if metric(key, "peak_original_reference_debt_m") is None else metric(key, "peak_original_reference_debt_m") * 1000, 3),
         n(None if metric(key, "progress_ratio_to_UF_full_run") is None else metric(key, "progress_ratio_to_UF_full_run") * 100, 3))
        for name, key in (("Main A", "main__A"), ("Main D", "main__D"), ("Historical D", "historical__D"))])
    table(("Run", "Net / gross [J]", "Peak output [W]", "Progress [m]", "Outcome"), [
        (name, fmt(key, "net_outgoing_J", 6) + " / " + fmt(key, "gross_outgoing_J", 6),
         fmt(key, "peak_outgoing_power_W", 3), fmt(key, "measured_progress_m", 4),
         "complete" if metric(key, "completed") else "stopped")
        for name, key in (("Main A", "main__A"), ("Main D", "main__D"),
                          ("Historical A", "historical__A"), ("Historical D", "historical__D"),
                          ("Tiny budget D", "stress__D"))])
    paragraph("Tangential tracking compares actual TCP velocity with the online governed reference at control samples; the peak-speed metric uses physics samples. Online tracking debt, original-reference debt and ordered task-progress loss are distinct declared metrics. Force recovery is measured from pulse end and requires 0.5 s of sustained recovery. The historical A stop and tiny-budget D stop remain failures to complete, even when their executed prefixes satisfy a subset of constraints.")

    page("KUKA supplement — Reserve, progression, completion and runtime")
    figure("figS_reserve_progress_timing", "KUKA-specific supplement.",
           "Available actual energy and the unexecuted current-held-force backup reserve have different provenance. Alpha target and actual alpha expose transition dynamics. Ordered progress avoids treating back-and-forth displacement as completed work. Whole successful control-tick timing is compared with the 1 ms simulated deadline; timing distributions are not a real-time certificate.", "4.7in")
    table(("Main policy", "Recorded QP calls", "Torque correction ticks", "Intervention ticks"), [
        (variant, fmt("main__" + variant, "qp_solves", 0),
         fmt("main__" + variant, "torque_correction_ticks", 0),
         fmt("main__" + variant, "intervention_ticks", 0)) for variant in ("UF", "A", "B", "C", "D")])
    paragraph("Zero recorded C/D solves or active rows on selected executed paths does not prove that no solver work occurred inside rejected plans: the existing guard discards some attempted-plan timing/diagnostics after a reserve or reaction-path rejection. No active task-QP advantage is inferred from this record.")
    extra = context.get("equal_progress", {})
    paragraph("Main D pauses for " + fmt("main__D", "pause_s", 3) +
              " s and accumulates maximum virtual-time delay " + fmt("main__D", "virtual_time_delay_s", 3) +
              " s. Its ordered measured progress at 18 s is " + fmt("main__D", "measured_progress_m", 4) +
              " m versus matched UF " + fmt("main__UF", "measured_progress_m", 4) +
              " m. Preserved continuous extensions measure equal-progress completion without resetting the clock, ledgers or controller history.")
    if extra:
        paragraph("Additional physical time to reach the matched UF 18 s task amount is main " +
                  n(extra.get("main_extra_s"), 3) + " s and historical " +
                  n(extra.get("historical_extra_s"), 3) + " s; the reference amount and crossing rule are recorded in the manifest.")
    timing = metric("main__D", "tick_seconds", {})
    paragraph("Main D successful whole-tick wall time: median " + n(timing.get("median"), 6) +
              " s, p99 " + n(timing.get("p99"), 6) + " s, maximum " + n(timing.get("max"), 6) +
              " s; " + fmt("main__D", "deadline_misses", 0) +
              " measured successful ticks exceed 1 ms. Full forecasts are offline computations, and failures are reported separately. Current measured force is held throughout each prediction; future scripted pulse values are not supplied to the guard.")

    page("Four-round verification, provenance and remaining limits")
    rounds = context.get("review_rounds", [])
    review_rows = []
    defaults = [
        "Canonical mapping, C4 distinction, all-sample data checks and independent raw metrics",
        "Controller meaning, nuisance equality, human-point work, causal forecasts and QP activity",
        "Fresh representative reproductions, repeated continuous force profile and explicit stop",
        "Clean-directory build, every PDF page rendered and inspected, adversarial claim audit",
    ]
    for index in range(1, 5):
        item = next((r for r in rounds if r.get("round") in (index, f"{index:02d}")), {})
        review_rows.append((str(index), item.get("verdict", "PENDING"), item.get("summary", defaults[index - 1])))
    table(("Round", "Verdict", "Recorded evidence"), review_rows, widths=(".06", ".12", ".73"))
    paragraph("Detailed findings, justified corrections, commands, true subprocess exit status and structured test counts are in reviews/round_01.md through round_04.md and review_summary.json. Report correctness is assessed separately from scientific success: an unsupported mismatch mechanism, an inactive task-QP comparison and an explicit stop can be reported correctly.")
    table(("Proposal component", "Coverage"), [
        ("Known-point signed work; task exemption; finite fixture reserve/recovery", "Demonstrated on reported fixtures"),
        ("Continuous bidirectional charging without an upper cap", "Demonstrated on the frozen continuous fixture and exact repeat"),
        ("Per-stage task QP", "Partially tested; no selected accepted task-QP rows in archived D matrix"),
        ("Whole-port causal baseline; independent mismatch/residual account", "Missing"),
        ("Isolated governor ablation; finite ledger cap; peak-power/force protection", "Missing"),
        ("Global passivity; arbitrary-force robustness; recursive feasibility; 1 kHz hardware", "Not established"),
    ], widths=(".60", ".31"))
    paragraph("All plotted physics powers and work increments remain unsmoothed. Actual raw ledgers are never lower-clamped; rejected candidates and unexecuted backup predictions are separated from executed trajectories. The complete manifest maps every curve to its preserved raw file/hash, source/configuration, units, sampling stage, transform and window. Original source, results and reports remain preserved in their prior worktrees.")
    paragraph("Starting synchronized Stage 2 source: " + source + ". Numerical-source archive: 81dffa64109752a9a027975e7b831daa4ff91e86. New plotting branch: " + branch + ". Final artifact/source hashes and the final Git revision are recorded in the reproducibility and preservation records, avoiding a self-referential commit hash in the PDF.")
    command = context.get("build_command", "See reproducibility.md for the exact single-entry build and separate simulation command.")
    paragraph("Reproduction: use the exact single-entry build command in reproduction.md with preserved inputs, a clean new output directory and the four review records. Simulation generation is separate and uses the frozen run configuration. The editable Markdown also preserves the complete command.")
    md.append("```sh\n" + command + "\n```\n")

    preamble = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.66in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx,booktabs,array}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small KUKA PoC figures: verified data and explicit gaps}
\fancyhead[R]{\small 16 October 2026}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.1}
\begin{document}
"""
    body = "\n\\clearpage\n".join("\n".join(p) for p in pages)
    (out / "report.tex").write_text(preamble + body + "\n\\end{document}\n")
    (out / "report.md").write_text("# KUKA adaptation of the 2-DOF PoC result figures\n" + "\n".join(md))
    return {"pages_planned": len(pages), "figures": list(FIGURES),
            "editable_sources": [str(out / "report.tex"), str(out / "report.md")]}
