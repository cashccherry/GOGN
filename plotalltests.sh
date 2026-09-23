#!/usr/bin/env bash
# Run from any working directory after activating the gogn environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MPLBACKEND="${MPLBACKEND:-Agg}"
PYTHON="${PYTHON:-python}"
# Plot the main waveform cases from already computed results.
for coverage in good bad good25 bad25; do
  for sigma in 1e-2 1e-1 1; do
    "$PYTHON" tests/plot-GOGN-test.py "$coverage" "$sigma"
  done
done
