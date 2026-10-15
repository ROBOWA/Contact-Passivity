Before = pre-repair controller `132c14dbc0`; after = current checkout. Same stepping loop, recorder, fixtures and metrics for both.

| Mechanism / scenario | Metric | Before | After |
| --- | --- | --- | --- |
| A nominal sliding (25 s, 10 N) | mean absolute force error | 0.0186 N | 0.0186 N |
| A nominal sliding (25 s, 10 N) | max absolute force error | 0.0498 N | 0.0498 N |
| A nominal sliding (25 s, 10 N) | along-path error max | 3.95 mm | 3.95 mm |
| A nominal sliding (25 s, 10 N) | cross-path error max | 0.069 mm | 0.069 mm |
| A nominal sliding (25 s, 10 N) | saturated substeps | 0 | 0 |
| A nominal sliding (25 s, 10 N) | recovery/loss events | 0 | 0 |
| B free-air, 1 m/s descent | first physical contact | 0.836 s | 2.257 s |
| B free-air, 1 m/s descent | CONTACT entered at | 0.007 s (FALSE, before any contact) | 2.260 s (genuine) |
| B free-air, 1 m/s descent | peak force used by control before contact | 1.52 N | 3.82e-14 N |
| C short interruption | FSM transitions after the event | 0 | 3 |
| C short interruption | CONTACT entries without physical contact | 0 | 0 |
| C short interruption | physical absence: events / total | 0 / 0.00 s | 1 / 0.25 s |
| C short interruption | max in-plane reference step per 1 ms | 0.05 mm | 0.050 mm |
| C short interruption | normal position-loop activation error | 0.0 mm | 0.00 mm |
| C short interruption | peak true contact force after the event | 11.1 N | 10.0 N |
| C short interruption | peak applied torque / limit | 18 % | 20 % |
| C short interruption | saturated substeps (longest) | 0 (0.000 s) | 0 (0.000 s) |
| C short interruption | tip height after the event: max / min | 0.159 / 0.159 m | 0.160 / 0.159 m |
| C short interruption | final phase | slide | slide |
| C loss + reacquisition | FSM transitions after the event | 5 | 3 |
| C loss + reacquisition | CONTACT entries without physical contact | 1 | 0 |
| C loss + reacquisition | physical absence: events / total | 2 / 4.39 s | 1 / 0.96 s |
| C loss + reacquisition | max in-plane reference step per 1 ms | 99.98 mm | 0.050 mm |
| C loss + reacquisition | normal position-loop activation error | 0.0 mm | 0.00 mm |
| C loss + reacquisition | peak true contact force after the event | 10.0 N | 10.0 N |
| C loss + reacquisition | peak applied torque / limit | 69 % | 21 % |
| C loss + reacquisition | saturated substeps (longest) | 0 (0.000 s) | 0 (0.000 s) |
| C loss + reacquisition | tip height after the event: max / min | 0.469 / 0.159 m | 0.170 / 0.159 m |
| C loss + reacquisition | final phase | slide | slide |
| C sustained loss | FSM transitions after the event | 19 | 2 |
| C sustained loss | CONTACT entries without physical contact | 6 | 0 |
| C sustained loss | physical absence: events / total | 4 / 27.90 s | 1 / 27.95 s |
| C sustained loss | max in-plane reference step per 1 ms | 99.98 mm | 0.029 mm |
| C sustained loss | normal position-loop activation error | 0.0 mm | 0.00 mm |
| C sustained loss | peak true contact force after the event | 10.0 N | 10.0 N |
| C sustained loss | peak applied torque / limit | 100 % | 21 % |
| C sustained loss | saturated substeps (longest) | 472 (0.065 s) | 0 (0.000 s) |
| C sustained loss | tip height after the event: max / min | 0.647 / 0.159 m | 0.180 / 0.159 m |
| C sustained loss | final phase | approach | hold |
| C removal STRESS TEST | FSM transitions after the event | 3 | 2 |
| C removal STRESS TEST | CONTACT entries without physical contact | 1 | 0 |
| C removal STRESS TEST | physical absence: events / total | 1 / 12.00 s | 1 / 12.00 s |
| C removal STRESS TEST | max in-plane reference step per 1 ms | 99.78 mm | 0.029 mm |
| C removal STRESS TEST | normal position-loop activation error | 246.4 mm | 0.02 mm |
| C removal STRESS TEST | peak true contact force after the event | 10.0 N | 10.0 N |
| C removal STRESS TEST | peak applied torque / limit | 100 % | 20 % |
| C removal STRESS TEST | saturated substeps (longest) | 324 (0.081 s) | 0 (0.000 s) |
| C removal STRESS TEST | tip height after the event: max / min | 0.579 / -0.099 m | 0.169 / 0.149 m |
| C removal STRESS TEST | final phase | approach | hold |

Notes. (1) Row B: the physical first-contact time differs because the pre-repair controller's false CONTACT changed the motion. (2) C short interruption: the pre-repair controller keeps pushing and rides through this 11 N wrist pull without losing contact; the new policy treats a 5 N force deficit lasting 10 ms as a contact problem and backs off, so it loses contact for 0.25 s and spends about 1.3 s recovering. That is a deliberate sensitivity trade-off, not an improvement. (3) C sustained: neither run is given a resume request here (the pre-repair controller has none), so the new controller ends in HOLD; the full SEARCH -> HOLD -> resume -> reacquire sequence is in `reports/loss-sustained.txt`. (4) C removal is a STRESS TEST in which the table is teleported away, not a reacquisition benchmark.
