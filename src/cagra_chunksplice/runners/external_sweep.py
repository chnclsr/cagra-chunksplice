"""Fresh-index parameter sweeps using one immutable dataset/query stream."""
from __future__ import annotations

from copy import copy, deepcopy

from cagra_chunksplice.datasets import load_external_data
from cagra_chunksplice.pareto import parameter_grid, summarize_curves
from cagra_chunksplice.promotion import EXTERNAL_METHODS, native_row, native_rows_complete
from cagra_chunksplice.reporting import finish_report
from .external import benchmark_external


def benchmark_external_sweep(a, device):
    prepared = load_external_data(a, device)
    rng_state = prepared[3].get_state()
    report, points = None, []
    for config_id, (name, overrides) in enumerate(parameter_grid(a)):
        parameters = copy(a)
        for key, value in overrides.items():
            setattr(parameters, key, value)
        # Replay identical synthetic queries in every configuration.
        prepared[3].set_state(rng_state)
        result = benchmark_external(parameters, device, methods=(name,), prepared=prepared)
        if report is None:
            report = deepcopy(result)
            report["summary"] = {}
            for stage in report["stages"]:
                stage["methods"] = {}
        row = result["summary"][name]
        report["summary"].setdefault(name, row)
        for destination, source in zip(report["stages"], result["stages"]):
            destination["methods"].setdefault(name, source["methods"][name])
        points.append({"configuration_id": config_id, "method": name,
                       "parameters": overrides, "stages": result["stages"], **row})
    report["parameters"] = {key: str(value) if hasattr(value, "__fspath__") else value
                            for key, value in vars(a).items()}
    report["methods"] = list(report["summary"])
    report["experiment"] = "external-pareto"
    report["curve_points"] = points
    report["curves"] = summarize_curves(
        [{k: v for k, v in point.items() if k != "stages"} for point in points], a.recall_targets)
    report["native_external_comparison_complete"] = (
        native_rows_complete(report["summary"])
        and all(native_row(p["method"], p)
                for p in points if p["method"] in EXTERNAL_METHODS))
    report["simulated"] = any(p["simulated"] for p in points if p["method"] in EXTERNAL_METHODS)
    report["sweep_protocol"] = {
        "fresh_build_per_configuration": True, "shared_queries_and_truth": True,
        "point_stage": "final visible prefix", "build_time": "excluded",
        "stages": "first configuration per method; full final-stage samples in curve_points",
        "mirage_parameters": "S/R/iter change construction and insertion, not just search",
        "selection_bias": "same queries used for tuning and reporting; no held-out tuning claim",
    }
    return finish_report(report, device)
