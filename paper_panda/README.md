# Panda companion report

This folder contains a standalone companion to the planar report in `paper/`.
It applies the same selective-passivation analysis to the tuned seven-joint
Panda evidence in `results_panda/`; the original report is not modified.

From the repository root, regenerate the figures with:

```bash
python paper_panda/make_figures.py
```

Compile the report with:

```bash
cd paper_panda
tectonic main.tex
```

The compiled deliverable is `paper_panda/main.pdf`. The report deliberately
shows both raw and filtered contact signals and distinguishes C3 (dual-ledger
QP without governor) from C4 (the same QP with the governor enabled).
