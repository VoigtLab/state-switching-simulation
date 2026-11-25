#!/usr/bin/env bash
#
# Reproduce every model result in this repository from scratch.
#
#   bash run_all.sh
#
# Checks the environment against requirements.txt, then writes
# results/model_results.xlsx.  The workbook's own `validation` sheet records
# every check; this script exits non-zero if any of them fails.
set -euo pipefail

cd "$(dirname "$0")"

# Prefer a project-local virtual environment if one exists, so the pinned
# versions in requirements.txt are the ones that actually run.
if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

echo "=============================================================="
echo " Stochastic differentiation model - full reproduction"
echo " (expected runtime: about 3 minutes)"
echo "=============================================================="
echo
"$PYTHON" check_env.py
echo

"$PYTHON" make_results.py
echo

echo "=============================================================="
echo " Done.  Results in $(pwd)/results"
echo "=============================================================="
ls -1 results
