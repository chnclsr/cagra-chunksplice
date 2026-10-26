#!/usr/bin/env bash
# Full-corpus native SIFT1M external sweep with measured Pareto frontiers.
#
# Requires the native suite (see docs/external_baselines.md): native Linux or
# WSL2, an NVIDIA GPU, and the builds from scripts/build_external.sh. This is
# a heavy run: 1,000,000 SIFT vectors (base 800000 plus 4 inserts of 50000),
# 1,000 held-out queries, 3 replays, and the full per-method parameter sweep.
#
# Produces results/native_sift1m.json (curve_points, curves.frontiers,
# curves.matched_recall) plus the sibling results/native_sift1m.pareto.svg,
# then validates native backend provenance. For a capacity check, run a
# labeled subset first via scripts/run_native_validation.sh with smaller
# --base/--chunks; never present a subset as full-corpus evidence.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
bash "$root/scripts/run_native_validation.sh" \
  --dataset sift1m --download-datasets --external-sweep \
  --base 800000 --chunks 4 --chunk-size 50000 --queries 1000 --repeats 3 \
  --output results/native_sift1m.json
"${PYTHON:-python3}" "$root/scripts/validate_external_report.py" results/native_sift1m.json
