#!/usr/bin/env bash
# Run from any working directory after activating the gogn environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MPLBACKEND="${MPLBACKEND:-Agg}"
PYTHON="${PYTHON:-python}"
for coverage in good bad; do
  for sigma in 1e-2; do
    for norm_p in 0.5 1 1.5 2 2.5 3; do
      "$PYTHON" tests/GOGNTT-test.py "$coverage" "$sigma" "$norm_p"
      "$PYTHON" tests/plot-GOGNTT-test.py "$coverage" "$sigma" "$norm_p"
    done
  done
done
