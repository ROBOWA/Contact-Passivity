# Repository consolidation — 2026-10-27

The single working repository is `/Users/carl/code/contactPassivity`, on `main`
at the latest inspected development commit, `5e792cf7157449a9806c9c3aeb6e3b5464780b3a`.
The former four worktrees and the external Stage 2 preservation backup were
consolidated here at the user's request. No remote branch was pushed.

| Former folder | Recorded revision | Retained content |
| --- | --- | --- |
| `contactSimulation` | `32c4866` | Oracle source in Git history; original results and environment |
| `contactSimulation-stage2-20261015-c5601e5e` | `41359f2` | Stage 2 source/report in latest tree; full `results/energy_guard_stage2/` |
| `contactSimulation-poc-20261016-41359f2` | `263136e` | PoC reports, full `results/kuka_poc_figures/` and archived report inputs |
| `contactSimulation-selective-20261016-263136e` | `5e792cf` | Latest source/reports; full `results/selective/`, replay recordings and videos |
| `contactSimulation-stage2-preservation-20261015-c5601e5e` | Historical backup | Exact backup under `archives/preservation/` |

All existing development branches, remote-tracking refs and Git history remain.
The previous local `main` tip also has the tag
`archive/pre-consolidation-main-20261027`. The old planar compatibility-fix
remote branch remains separate; its code was not automatically mixed into the
newer KUKA implementation.

## Integrity and historical paths

`source_manifest.json` inventories 6,619 source file records with SHA-256 hashes.
Historical tracked versions are retained as their exact Git blobs and commits.
Untracked/ignored research files are merged into their original relative paths;
different-content collisions are retained under `archives/worktree-extras/`.
The independent preservation backup is retained in full. `verification.json`
records the successful check before old worktrees were removed.

Recheck integrity from any location:

```bash
/Users/carl/code/contactPassivity/.venv/bin/python /Users/carl/code/contactPassivity/scripts/verify_consolidation.py
```

The manifest covers 1,123 unique historical Git blobs and 3,063 additional files.
Rebuildable Python/pytest caches and linked-worktree `.git` pointers were excluded;
the main repository's Git database and Python environment were retained.
Environment launchers and activation scripts now point to `contactPassivity`.

Original reports, failure records and preservation manifests keep their original
contents, including historical absolute paths, branch names and frozen revisions.
Those paths describe where the evidence was produced. Use `source_manifest.json`
and `verify_consolidation.py` for integrity checks after relocation; old
worktree-preservation audits intentionally compare their original live paths and
branch state and are not relocation checks. Numerical controller sources and
simulation assets remain identical to the latest development commit.

## Working here

`main` now includes the successive KUKA baseline, oracle, Stage 2,
figure/report and selective-comparison stages. Stage 2 and K0–K4 remain distinct
experimental controller families. The planar and Panda packages, papers and
their data remain available as historical backends.

Keep new simulation outputs under `results/`, using a new output directory for
each run. `results/` and `archives/` are local data and are ignored by Git; the
consolidation manifest is retained under `docs/`. Existing large research data
was preserved rather than re-added to Git.

## Validation after relocation

- KUKA regression suite: **97 passed in 80.61 s**.
- Historical source/data/backup integrity: **PASS**, all 6,619 source records.
- Integrity verifier accepts intact data and rejects modified data.
- A fresh 25 s nominal contact simulation: **PASS**, no contact-loss events.
- Preserved C_blocking/K4 recording: **72,000 samples**, COMPLETE.
- Git connectivity and installed dependency checks: **PASS**.
- Exactly one Git worktree remains, at `contactPassivity`; old project folders
  are removed and no research-data symlink points at them.

See `validation.json`, `verification_final.json` and `verifier_smoke.json`.
