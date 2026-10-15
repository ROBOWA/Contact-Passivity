# Stage 2 reproduction

Use the new isolated worktree. Original Stage 1 files and outputs must remain
untouched. All commands below select fresh, collision-resistant output paths.
The environment symlink references the existing compatible environment read-only;
do not install packages into it. Python bytecode writes are disabled.

```sh
cd /Users/carl/code/contactSimulation-stage2-20261015-c5601e5e
export PYTHONDONTWRITEBYTECODE=1
run_id=$(.venv/bin/python -c 'import uuid; print(uuid.uuid4().hex)')
out="results/energy_guard_stage2/reproduction-$run_id"
docs="docs/energy_guard_stage2/20261015-c5601e5e"

.venv/bin/python scripts/energy_stage2_matrix.py \
  --out "$out/matrix" --jobs 4 \
  --heldout "$docs/heldout_manifest.json" --config "$docs/frozen_config.json"
.venv/bin/python -m pytest -q | tee "$out/tests.txt"
.venv/bin/python -m kuka_table_sim.energy_validation --out "$out/validation"
.venv/bin/python scripts/energy_stage2_preservation.py \
  --manifest "$docs/preservation_manifest.json" --out "$out/preservation.json"

.venv/bin/python -m kuka_table_sim.energy_experiments case \
  --case main --variant D --config "$docs/frozen_config.json" --out "$out/repeat-main-D"
.venv/bin/python -m kuka_table_sim.energy_experiments case \
  --case stress --variant D --config "$docs/frozen_config.json" --out "$out/repeat-stress-D"

.venv/bin/python -m kuka_table_sim.energy_experiments extend \
  --from-run "$out/matrix/main/main/D" --reference-run "$out/matrix/main/main/UF" \
  --until 20 --out "$out/extension-main-D"
.venv/bin/python -m kuka_table_sim.energy_experiments extend \
  --from-run "$out/matrix/historical/historical/D" \
  --reference-run "$out/matrix/historical/historical/UF" \
  --until 20 --out "$out/extension-historical-D"

.venv/bin/python scripts/energy_stage2_report.py \
  --matrix-parent "$out/matrix" --validation "$out/validation" \
  --tests "$out/tests.txt" --preservation "$out/preservation.json" \
  --freeze "$docs/numerical_freeze.json" \
  --repeats "$out/repeat-main-D" "$out/repeat-stress-D" \
  --out "$docs/reproduction-$run_id"
```

Expected research/control stops return exit 2 from an individual matrix/case;
logs and complete failure states are still written. The matrix scheduler accepts
exit 2 as a recorded outcome and returns 1 for a process error. Tests must actually
pass; missing cases/tests are never interpreted as passes. Frozen-tree hashes
allow later report-only commits without relabeling the numerical implementation.
Do not retune after viewing new held-out outcomes and continue calling them held out.

The initial hindsight-only diagnosis is separately reproducible, also into a new path:

```sh
.venv/bin/python -m kuka_table_sim.energy_diagnostic --out "$out/offline-timing"
```

No remote publication was authorized. Push only the new branch if/when desired:

```sh
git push -u origin feature/kuka-energy-reserve-contact-20261015-c5601e5e
```
