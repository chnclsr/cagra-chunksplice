#!/usr/bin/env bash
# Update latency versus base size: ChunkSplice/Naive against full rebuilds and
# native cuVS-CAGRA extend/rebuild.
#
# Requires the native suite (scripts/build_external.sh) for the cuVS rows; the
# run fails instead of substituting the simulator. Each base receives one
# 256-vector chunk; three timed replays follow one untimed warmup. The
# PyTorch incremental rows use --insertion-mode bounded (fixed-budget beam
# candidates plus reverse merge over linked rows only) instead of the O(n)
# exact path; trailing "$@" args can still override it.
#
# Produces results/scaling_native.json with per-size latency samples, recall,
# fitted log-log exponents, speedups and the first measured crossover size.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
prefix="${CAGRA_EXTERNAL_PREFIX:-$root/.external}"
if [[ -f "$prefix/env.sh" ]]; then
  source "$prefix/env.sh"
fi
exec "${PYTHON:-python3}" -m cagra_chunksplice --mode scaling --device cuda \
  --scaling-bases 2048 8192 32768 131072 524288 --chunk-size 256 --repeats 3 \
  --insertion-mode bounded \
  --scaling-cagra --external-backend native "$@" \
  --output "${SCALING_OUTPUT:-results/scaling_native.json}"
