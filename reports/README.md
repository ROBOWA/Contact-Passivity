# Reports

Self-contained HTML reports (figures embedded as data URIs — no external
assets, no network access needed). GitHub does not render HTML in the repo
view; download a file and open it locally, or use the hosted links below.

| File | Language | What it covers |
| --- | --- | --- |
| [`selective-passivation-evidence.html`](selective-passivation-evidence.html) | English | The proof of concept and what it establishes: eight findings from 33 simulations, scope limits, open questions. **Start here.** |
| [`selective-passivation-evidence.zh.html`](selective-passivation-evidence.zh.html) | 中文 | Same content, Chinese edition. Key figures cross-checked against the English version. |
| [`validation-fixes.zh.html`](validation-fixes.zh.html) | 中文 | Engineering record of the five repairs made in this branch, with before/after measurements. Superseded for most purposes by the code comments and `mujoco_sliding/README.md`. |

Hosted copies (private artifacts, same content):

* English — https://claude.ai/code/artifact/2503edd7-85c6-48c2-b907-010348376d19
* 中文 — https://claude.ai/code/artifact/f2ec2623-be85-458a-b286-e0add84d92b4
* 修复记录 — https://claude.ai/code/artifact/4561062c-abdf-4624-8e9d-c022a2ebbee0

## Provenance

Generated from a full regeneration at 1 kHz on MuJoCo 3.11.0 (3.3.1 agrees to
≈1e-12; 46/46 unit tests pass on both). Every number traces to
`results_passivation/summary.json`, `acceptance.json` and `sweep/sweep.csv`,
which are gitignored build artifacts — reproduce them with:

```bash
python -m mujoco_sliding.experiments --all --sweep --animate
```

The reports are snapshots. They do **not** update when the pipeline is re-run,
so if the numbers change, regenerate the reports too.
