# Round 1 — source, semantics and data integrity

Verdict: PASS for the reported archival data. Initial successful report: `build_initial_03`; corrected report: `build_round01_03`, both 10 pages. F7 is explicitly pending in this draft.

Inspect: read all six canonical mappings and the actual KUKA/paper implementations, including the paper C4 naming trap, different work-sampling stages, actual link-point velocity, shared H/R update and absent mismatch/cap/power mechanism. Full input clocks and all physics prefixes were checked. The independent bundle audit recomputed every common headline without importing plotting helpers; exact raw signal bytes, short negative energy prefixes and stopped-run endpoints remain intact.

Findings and correction: boundary timestamp roundoff produced tiny negative elapsed recovery (~3e-12 s); clamp only elapsed duration to zero. Machine manifests now distinguish source-domain/display-domain and executed/offline/predicted curves. F8 fonts/embedding were too small and the PDF reproduction command overflowed; enlarge F8 and preserve the exact command in Markdown/reproduction instructions. No numerical controller change.

Revalidation: corrected build exit 0; independent bundle audit exit 0 (`support/round01_revalidated.json`); 132 shared matrix metrics agree within 6.66e-16 (tolerance1e-8), with additional independent common-window/provenance comparisons in the audit. The PDF remains10 pages with no overfull/missing-glyph warnings. Page7 was rendered/inspected after correction. Main D RMSE remains0.128521019 N and A minimum remains-1.113335274e-6 J. Structured tests are recorded in support/audit_tests_02.json (11 tests, exit 0; no stdout-parsed gate).

Remaining: F4/F6 remain scientifically unsupported; no whole-port closed-loop or isolated governor baseline. Fresh help/block logs will be included and repeated before sealing. Exact commands, statuses, source identifiers and hashes are in round_01.json and the build/audit manifests.
