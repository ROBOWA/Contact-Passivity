# KUKA selective comparisons — implemented and measured

20/32 trials completed; stopped trials remain in the figures. In the primary TCP block, K4 completed 18 s, reduced net human output by 60.9% relative to K0, and reached 0.148221 W peak human output. Its maximum 10 N force error was 1.647 N and contact absence was 0.00000 s. It achieved 87.3% of K0 measured progress in the same 18 s. K3 stopped at 10.754 s; this supports a completion benefit for the governor on this fixture, not a measured K3 post-release recovery peak.

These are new closed-loop KUKA comparisons inspired by the previous 2-DOF reference. K0 nominal; K1 whole monitored port; K2 residual only; K3 H/R plus human peak power, governor OFF; K4 identical K3 QP, governor ON. The actual plant and nominal 10 N controller are unchanged. A separate contact model evaluates predicted task work at each actual state. Only its explicit tip/table tangential friction changes (1 versus .20). Seven joint torques and a virtual reference clock adapt the reference architecture; they are not its original two-variable Cartesian controller or spatial-anchor governor.

| Controller | Protected accounts | Human peak | Governor |
|---|---|---|---|
| K0 | monitored only | no | off |
| K1 | W=H+actual task work | no | off |
| K2 | R=H+task-model mismatch | no | off |
| K3 | H and R | .15 W | off |
| K4 | same as K3 | .15 W | on |

| Case | Mode | End s | Result | H net out J | H gross out J | min H J | min R J | peak H W | max force error N | lost contact s | QP attempts |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_exact_nohuman | K0 | 18.000 | COMPLETE | -0.000000 | 0.000000 | 0.12 | 0.3 | 0.000000 | 0.050 | 0.00000 | 0 |
| A_exact_nohuman | K1 | 12.541 | STOP | -0.000000 | 0.000000 | 0.12 | 0.3 | 0.000000 | 1.854 | 0.00000 | 3229 |
| A_exact_nohuman | K2 | 18.000 | COMPLETE | -0.000000 | 0.000000 | 0.12 | 0.3 | 0.000000 | 0.050 | 0.00000 | 0 |
| A_exact_nohuman | K3 | 18.000 | COMPLETE | -0.000000 | 0.000000 | 0.12 | 0.3 | 0.000000 | 0.050 | 0.00000 | 0 |
| A_exact_nohuman | K4 | 18.000 | COMPLETE | -0.000000 | 0.000000 | 0.12 | 0.3 | 0.000000 | 0.050 | 0.00000 | 0 |
| B_mismatch_nohuman | K0 | 18.000 | COMPLETE | -0.000000 | 0.000000 | 0.12 | 0.0670424 | 0.000000 | 0.050 | 0.00000 | 0 |
| B_mismatch_nohuman | K1 | 12.541 | STOP | -0.000000 | 0.000000 | 0.12 | 0.214862 | 0.000000 | 1.854 | 0.00000 | 3229 |
| B_mismatch_nohuman | K2 | 18.000 | COMPLETE | -0.000000 | 0.000000 | 0.12 | 0.0670424 | 0.000000 | 0.050 | 0.00000 | 0 |
| B_mismatch_nohuman | K3 | 18.000 | COMPLETE | -0.000000 | 0.000000 | 0.12 | 0.0670424 | 0.000000 | 0.050 | 0.00000 | 0 |
| B_mismatch_nohuman | K4 | 18.000 | COMPLETE | -0.000000 | 0.000000 | 0.12 | 0.0670424 | 0.000000 | 0.050 | 0.00000 | 0 |
| C_blocking | K0 | 18.000 | COMPLETE | 0.187435 | 0.187435 | -0.067435 | 0.112565 | 0.295100 | 0.145 | 0.00000 | 0 |
| C_blocking | K1 | 10.624 | STOP | 0.132250 | 0.132250 | -0.01225 | 0.16775 | 0.265072 | 3.069 | 0.00000 | 214 |
| C_blocking | K2 | 18.000 | COMPLETE | 0.187435 | 0.187435 | -0.067435 | 0.112565 | 0.295100 | 0.145 | 0.00000 | 0 |
| C_blocking | K3 | 10.754 | STOP | 0.109175 | 0.109175 | 0.0108255 | 0.190825 | 0.148221 | 14.751 | 0.00000 | 588 |
| C_blocking | K4 | 18.000 | COMPLETE | 0.073209 | 0.073209 | 0.0467912 | 0.226791 | 0.148221 | 1.647 | 0.00000 | 40 |
| D_combined | K0 | 18.000 | COMPLETE | 0.187435 | 0.187435 | -0.067435 | -0.214466 | 0.295100 | 0.145 | 0.00000 | 0 |
| D_combined | K1 | 10.624 | STOP | 0.132250 | 0.132250 | -0.01225 | 0.137941 | 0.265072 | 3.069 | 0.00000 | 214 |
| D_combined | K2 | 10.752 | STOP | 0.171101 | 0.171101 | -0.0511006 | 0.0753639 | 0.295100 | 0.198 | 0.00000 | 48 |
| D_combined | K3 | 10.754 | STOP | 0.109175 | 0.109175 | 0.0108255 | 0.183927 | 0.148221 | 14.751 | 0.00000 | 588 |
| D_combined | K4 | 14.948 | STOP | 0.073209 | 0.073209 | 0.0467912 | 0.0136669 | 0.148221 | 1.647 | 0.00000 | 153 |
| E_forearm_block | K0 | 18.000 | COMPLETE | 0.241623 | 0.241623 | -0.121623 | 0.0583767 | 0.238858 | 0.397 | 0.00000 | 0 |
| E_forearm_block | K1 | 10.335 | STOP | 0.132274 | 0.132274 | -0.0122744 | 0.167726 | 0.244181 | 3.017 | 0.00000 | 242 |
| E_forearm_block | K2 | 18.000 | COMPLETE | 0.241623 | 0.241623 | -0.121623 | 0.0583767 | 0.238858 | 0.397 | 0.00000 | 0 |
| E_forearm_block | K3 | 10.232 | STOP | 0.088968 | 0.088968 | 0.0310318 | 0.211032 | 0.148201 | 5.082 | 0.00000 | 655 |
| E_forearm_block | K4 | 18.000 | COMPLETE | 0.039143 | 0.043966 | 0.076052 | 0.256052 | 0.148201 | 0.930 | 0.00000 | 10 |
| F_help_block | K0 | 18.000 | COMPLETE | 0.025306 | 0.138273 | 0.0117267 | 0.161727 | 0.189865 | 0.247 | 0.00000 | 0 |
| F_help_block | K1 | 13.715 | STOP | 0.022128 | 0.135096 | 0.0149045 | 0.164904 | 0.189865 | 3.874 | 0.00000 | 3721 |
| F_help_block | K2 | 18.000 | COMPLETE | 0.025306 | 0.138273 | 0.0117267 | 0.161727 | 0.189865 | 0.247 | 0.00000 | 0 |
| F_help_block | K3 | 18.000 | COMPLETE | 0.010692 | 0.123659 | 0.0263407 | 0.176341 | 0.148221 | 0.295 | 0.00000 | 383 |
| F_help_block | K4 | 18.000 | COMPLETE | -0.031180 | 0.081788 | 0.068212 | 0.218212 | 0.148221 | 1.417 | 0.00000 | 31 |
| S_starved | K3 | 10.363 | STOP | 0.000939 | 0.001011 | 5.56623e-05 | 0.299056 | 0.004108 | 17.481 | 0.00000 | 1263 |
| S_starved | K4 | 18.000 | COMPLETE | -0.065725 | 0.006546 | 0.000682986 | 0.293771 | 0.029066 | 11.047 | 0.00000 | 60 |

| Reference comparison | New KUKA evidence |
|---|---|
| F3: avoid charging predictable task work | K1 first filters at 10.927 s and stops at 12.541 s (nonlinear_retries_exhausted); K2/K3/K4 match nominal bitwise. This is accounting/filter evidence, not proof of unavoidable physical infeasibility. |
| F4: model error appears separately | No-human mismatch gives maximum |pR-pH|=0.084082 W and minimum R=0.067042 J; H remains .12 J. No residual intervention is required in this finite window. |
| F5: dedicated human protection | Residual-only K2 reaches H=-0.067435 J and 0.295100 W peak output. K4 completes with H=0.046791 J minimum and peak 0.148221 W. K3 only demonstrates a protected prefix. |
| F6: simultaneous human and mismatch | Executed power decomposition is independently checked for every mode. Completion and task quality must be read separately from that accounting identity. |
| F7: bounded bidirectional charging | K4 helping/blocking discards 0.082968 J of H charging at the .15 J cap; final H=0.068212 J. Net H output=-0.031180 J, gross=0.081788 J. |
| F8: governor changes recovery | K3/K4 have identical pre-governor trajectories and constraints. Primary K3 stops before release, so its post-release peak is unavailable; this comparison establishes K4 completion on this fixture, not the reference paper's numerical recovery reduction. |

This is deterministic simulation evidence, not a mathematical proof of general safety or reproduction of the reference paper's numerical results. Prediction uses true-plant dynamics with current measured force held for 1 ms; the task model alone is mismatched. The terminal human guard assumes force increments no greater than 60 N/s times 1 ms. Finite-difference rows are checked by fresh nonlinear rollouts; stopped cases expose solver, nonlinear acceptance, or local feasibility limits. No verified fallback, hardware execution, continuous-time certificate, or universal task-preservation claim is made. Gross outgoing energy is not bounded by a net-work account. K3/K4 have no Stage 2 reserve/backup or hard contact-quality constraints.

Numerical source: `b98d443a9e4b0acf376dd5ba02f0cb2f2e0b5a1fca5ac9f5d1bee90c3ecf319b` at `dfc09dfcb402cdc99dd8138d5d71f0c6e8ea205e`. Audit: **PASS**. Regression plus report-audit suite: 91 tests, 0 failures, 0 errors. All executed H/R/W ledgers, cap discard, physical power duals, actuator bounds, peak/CBF rows and accepted QP residuals were recalculated from raw arrays. Audit PASS denotes consistent implementation evidence; it does not convert stopped trials into successful controllers.

Full logs: `results/selective/final/` (ignored by Git). Full-rate, float64 account/force/progress plot data, complete final/failure states, and lossless QP histories for all 32 cases: `raw/`. Do not compare a stopped prefix with an 18 s total without the completion column. Exact-model K2/K3/K4 trajectories match K0 bitwise. Human and residual powers coincide only when the nominal task model is exact; budgets remain distinct.

![F3 adaptation: predictable task work and whole-port depletion](figures/fig3_exact.png)

Exact model, no human. K2/K3/K4 should be transparent; K1 accounts for ordinary task work. Crosses and dotted lines mark explicit stops, never zero-padded continuations.

![F4 adaptation: task-model mismatch without human input](figures/fig4_mismatch.png)

Actual friction remains 1; nominal pair friction is .20. Nonzero residual power with zero human power establishes independent accounting. It does not establish robustness to unknown plant dynamics.

![F5 adaptation: dedicated human energy and peak-power protection](figures/fig5_human.png)

All modes use the same 5 N TCP block. K3/K4 enforce the .15 W human peak bound and H/R rows. K2 has the larger residual budget and no separate H constraint. Compare completion as well as safe prefixes.

![F6 adaptation: simultaneous human input and task-model mismatch](figures/fig6_combined.png)

At every executed sample, pR=pH+(pT-pT_hat). H and R have independently initialized budgets. The cap is applied separately to each account; the raw reconstruction checks both identities.

![F7 adaptation: helping, upper-cap saturation, then blocking](figures/fig7_charge.png)

One continuous trial: helping starts at 9.5 s, blocking at 10.3 s. H charges only to .15 J; discarded charging is explicitly logged. Net outgoing work and gross outgoing work are distinct.

![F8 adaptation: matched K3/K4, governor OFF/ON](figures/fig8_governor.png)

K3/K4 share the QP, bounds, budgets and model; only the reference-clock governor changes. A stopped K3 trace cannot establish a post-release catch-up peak. Joint correction is not a reconstructed Cartesian command.

![Matched governor comparison through helping and blocking](figures/fig9_help_recovery.png)

This additional full-window comparison exposes the cost in progress and the available recovery evidence after the 11.7 s release. No unexecuted tail is extrapolated.

![Second human port: forearm rather than TCP](figures/fig10_forearm.png)

Same world-fixed blocking pulse at link5 local (0,.04,.12) m. Human power uses that actual moving point Jacobian, not TCP velocity.

![Small-budget challenge: H initialized at .001 J](figures/fig11_starved.png)

Only the initial H budget changes. Explicit failures are retained with complete pre-action and failure snapshots; no safe-fallback or viability guarantee is inferred.
