#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: reproduce.sh NEW_OUTPUT_DIRECTORY' >&2
  exit 64
fi
POC_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
POC_DOC_ROOT=docs/kuka_poc_figures/20261016-41359f2
POC_PYTHON=${POC_PYTHON:-/Users/carl/code/contactSimulation/.venv/bin/python}
cd "$POC_REPO_ROOT"
PYTHONDONTWRITEBYTECODE=1 exec "$POC_PYTHON" scripts/kuka_poc_figures.py \
  --inputs "$POC_DOC_ROOT/inputs" --out "$1" --reviews "$POC_DOC_ROOT/reviews"
