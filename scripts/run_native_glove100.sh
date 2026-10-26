#!/usr/bin/env bash
# Full-corpus native GloVe-100 external sweep with measured Pareto frontiers.
#
# Requires the native suite (see docs/external_baselines.md): native Linux or
# WSL2, an NVIDIA GPU, and the builds from scripts/build_external.sh. This is
# a heavy run: 1,183,514 GloVe vectors (base 983514 plus 4 inserts of 50000),
# 1,000 held-out angular queries, 3 replays, and the full per-method
# parameter sweep. Vectors and queries are unit-normalized by the loader.
#
# Produces results/native_glove100.json (curve_points, curves.frontiers,
# curves.matched_recall) plus the sibling results/native_glove100.pareto.svg,
# then validates native backend provenance. For a capacity check, run a
# labeled subset first via scripts/run_native_validation.sh with smaller
# --base/--chunks; never present a subset as full-corpus evidence.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
bash "$root/scripts/run_native_validation.sh" \
  --dataset glove-100 --download-datasets --external-sweep \
  --base 983514 --chunks 4 --chunk-size 50000 --queries 1000 --repeats 3 \
  --output results/native_glove100.json
"${PYTHON:-python3}" "$root/scripts/validate_external_report.py" results/native_glove100.json
