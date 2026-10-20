#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${CAGRA_EXTERNAL_PREFIX:-$root/.external}"
if [[ -f "$prefix/env.sh" ]]; then
  source "$prefix/env.sh"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "${PYTHON:-python3}" -m cagra_chunksplice --mode external "$@" \
  --device cuda --external-backend native --require-native-validation
