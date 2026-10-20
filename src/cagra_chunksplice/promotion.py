"""Fail-closed validation of measured native external comparisons."""
from __future__ import annotations

import math

EXTERNAL_METHODS = ("FAISS-HNSW", "MIRAGE", "cuVS-CAGRA")
BACKENDS = {"FAISS-HNSW": "faiss-hnsw-cpu", "MIRAGE": "faiss-mirage-cpu",
            "cuVS-CAGRA": "cuvs-cagra-cuda"}


def native_row(name, row):
    meta = row.get("metadata", {})
    device = str(meta.get("effective_device", ""))
    correct_device = device in ("cuda", "cuda:0") if name == "cuVS-CAGRA" else device == "cpu"
    version = meta.get("version")
    return (row.get("simulated") is False and meta.get("simulated") is False
            and row.get("backend") == BACKENDS[name] and meta.get("backend") == BACKENDS[name]
            and correct_device and isinstance(version, str)
            and version.lower() not in ("", "unknown", "mock", "simulated"))


def native_rows_complete(rows):
    return all(native_row(name, rows.get(name, {})) for name in EXTERNAL_METHODS)


def _finite(value, low=0, high=math.inf, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and low <= value <= high and (not positive or value > 0))


def validate_native_report(report):
    """Validate every measured row, not just a caller-provided completion flag."""
    errors = []
    parameters = report.get("parameters", {})
    if parameters.get("external_backend") != "native":
        errors.append("--external-backend native is required")
    if report.get("status") != "ok" or report.get("benchmark_completed") is not True:
        errors.append("benchmark did not complete successfully")
    if report.get("gpu_execution_requested") is not True:
        errors.append("CUDA execution is required for native CAGRA")
    if report.get("simulated") is not False:
        errors.append("external simulated must be false")
    if report.get("native_external_comparison_complete") is not True:
        errors.append("native_external_comparison_complete must be true")
    repeats = parameters.get("repeats")
    if type(repeats) is not int or repeats <= 0:
        errors.append("positive replay count is required")

    def check_row(name, row, n=None, samples=True):
        if not native_row(name, row):
            errors.append(name + ": native provenance is missing or inconsistent")
        if n is not None and row.get("metadata", {}).get("ntotal") != n:
            errors.append(name + ": index size does not match its visible prefix")
        for key in ("update_ms", "qps", "recall10"):
            if not _finite(row.get(key), high=1 if key == "recall10" else math.inf,
                           positive=key == "qps"):
                errors.append(name + ": invalid " + key)
        if samples:
            for key in ("update_samples_ms", "query_samples_ms", "recall10_samples"):
                values = row.get(key, [])
                if (not isinstance(values, list) or len(values) != repeats
                        or any(not _finite(v, high=1 if key == "recall10_samples" else math.inf,
                                           positive=key == "query_samples_ms") for v in values)):
                    errors.append(name + ": missing or invalid " + key)

    stages = report.get("stages", [])
    if not stages or len(stages) != parameters.get("chunks"):
        errors.append("every update stage must be present")
    for stage in stages:
        for name in EXTERNAL_METHODS:
            check_row(name, stage.get("methods", {}).get(name, {}), stage.get("n"))
    for name in EXTERNAL_METHODS:
        check_row(name, report.get("summary", {}).get(name, {}), samples=False)
    points = report.get("curve_points", [])
    if parameters.get("external_sweep") and not points:
        errors.append("sweep has no measured configurations")
    for point in points:
        name = point.get("method")
        if name in EXTERNAL_METHODS:
            check_row(name, point, samples=False)
            if len(point.get("stages", [])) != parameters.get("chunks"):
                errors.append("every configuration needs its measured stages")
            for stage in point.get("stages", []):
                check_row(name, stage.get("methods", {}).get(name, {}), stage.get("n"))
    if errors:
        raise ValueError("Native Promotion Gate failed: " + "; ".join(dict.fromkeys(errors)))
    return {"passed": True, "scope": "native backend validation; not evidence of superiority",
            "academic_dataset": report.get("dataset", {}).get("synthetic") is False}
