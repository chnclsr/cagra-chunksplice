"""Dependency-free tests for the native gate, budgets, and measured frontiers."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cagra_chunksplice.cli import arguments
from cagra_chunksplice.pareto import frontier, summarize_curves, parameter_grid, curves_svg
from cagra_chunksplice.promotion import BACKENDS, validate_native_report


def native_report():
    rows = {}
    for name, backend in BACKENDS.items():
        rows[name] = {
            "backend": backend, "simulated": False, "update_ms": 1.0,
            "qps": 1000.0, "recall10": 0.9,
            "metadata": {"backend": backend, "simulated": False, "version": "1.10.0",
                         "effective_device": "cuda" if name == "cuVS-CAGRA" else "cpu", "ntotal": 40},
            "update_samples_ms": [1.0], "query_samples_ms": [4.0], "recall10_samples": [0.9],
        }
    return {"parameters": {"external_backend": "native", "repeats": 1, "chunks": 1},
            "status": "ok", "benchmark_completed": True, "simulated": False,
            "gpu_execution_requested": True, "native_external_comparison_complete": True,
            "stages": [{"n": 40, "methods": copy.deepcopy(rows)}], "summary": rows,
            "dataset": {"synthetic": True}}


class NativePromotionTests(unittest.TestCase):
    def test_complete_native_rows_pass_without_claiming_academic_data(self):
        result = validate_native_report(native_report())
        self.assertTrue(result["passed"])
        self.assertFalse(result["academic_dataset"])

    def test_completion_flag_cannot_hide_fallbacks_or_missing_data(self):
        for kind in ("summary", "stage"):
            for name in BACKENDS:
                for key in ("simulated", "metadata_simulated", "backend", "device", "version"):
                    with self.subTest(kind=kind, name=name, key=key):
                        report = native_report()
                        row = report["summary"][name] if kind == "summary" else report["stages"][0]["methods"][name]
                        if key == "simulated":
                            row[key] = True
                        elif key == "metadata_simulated":
                            row["metadata"]["simulated"] = True
                        elif key == "backend":
                            row[key] = "torch-exact"
                        elif key == "device":
                            row["metadata"]["effective_device"] = "cpu" if name == "cuVS-CAGRA" else "cuda"
                        else:
                            row["metadata"]["version"] = "mock"
                        with self.assertRaises(ValueError):
                            validate_native_report(report)

    def test_gate_requires_strict_mode_samples_and_consistent_size(self):
        mutations = [
            lambda r: r["parameters"].update(external_backend="auto"),
            lambda r: r.update(native_external_comparison_complete=False),
            lambda r: r.update(simulated=True),
            lambda r: r.update(benchmark_completed=False),
            lambda r: r.update(stages=[]),
            lambda r: r["stages"][0]["methods"]["MIRAGE"].update(query_samples_ms=[]),
            lambda r: r["stages"][0]["methods"]["MIRAGE"].update(recall10_samples=[float("nan")]),
            lambda r: r["stages"][0]["methods"]["MIRAGE"]["metadata"].update(ntotal=39),
        ]
        for mutate in mutations:
            report = native_report()
            mutate(report)
            with self.assertRaises(ValueError):
                validate_native_report(report)

    def test_every_sweep_configuration_is_checked(self):
        report = native_report()
        report["parameters"]["external_sweep"] = True
        point = {"method": "MIRAGE", **copy.deepcopy(report["summary"]["MIRAGE"]),
                 "stages": [{"n": 40, "methods": {"MIRAGE": copy.deepcopy(report["summary"]["MIRAGE"])}}]}
        report["curve_points"] = [point]
        self.assertTrue(validate_native_report(report)["passed"])
        point["stages"][0]["methods"]["MIRAGE"]["simulated"] = True
        with self.assertRaises(ValueError):
            validate_native_report(report)


class FrontierTests(unittest.TestCase):
    def point(self, recall, qps, simulated=False):
        return {"method": "MIRAGE", "recall10": recall, "qps": qps, "simulated": simulated}

    def test_dominated_points_and_unattainable_recall(self):
        points = [self.point(.7, 100), self.point(.8, 120), self.point(.9, 90),
                  self.point(.85, 80)]
        self.assertEqual(frontier(points), [points[1], points[2]])
        result = summarize_curves(points, [.8, .95])
        self.assertEqual(result["matched_recall"]["0.8"]["MIRAGE / native"], points[1])
        self.assertIsNone(result["matched_recall"]["0.95"]["MIRAGE / native"])

    def test_simulated_never_dominates_native_and_svg_uses_measured_points(self):
        result = summarize_curves([self.point(.8, 100), self.point(1.0, 10000, True)], [.9])
        self.assertEqual(len(result["frontiers"]), 2)
        self.assertIsNone(result["matched_recall"]["0.9"]["MIRAGE / native"])
        svg = curves_svg(result)
        self.assertIn("simulated", svg)
        self.assertEqual(svg.count("<circle "), 2)

    def test_invalid_measurements_are_rejected(self):
        for point in (self.point(float("nan"), 1), self.point(.8, 0), self.point(1.1, 2)):
            with self.assertRaises(ValueError):
                frontier([point])

    def test_budget_grid_and_cli_defaults(self):
        a = arguments(["--mode", "external", "--external-sweep"])
        grid = parameter_grid(a)
        self.assertEqual(len([p for n, p in grid if n == "MIRAGE"]), 8)
        self.assertEqual(len([p for n, p in grid if n == "cuVS-CAGRA"]), 12)
        self.assertEqual({p["hnsw_m"] for n, p in grid if n == "FAISS-HNSW"}, {16, 32})
        self.assertEqual(a.dataset, "synthetic")
        self.assertFalse(a.allow_synthetic_fallback)

    def test_invalid_cli_combinations_fail_before_runtime(self):
        import contextlib
        import io
        cases = [
            ["--require-native-validation"],
            ["--mode", "external", "--require-native-validation", "--external-backend", "auto"],
            ["--dataset", "sift1m"],
            ["--mode", "external", "--mirage-r", "0"],
            ["--mode", "external", "--recall-targets", "nan"],
            ["--mode", "external", "--external-sweep", "--cagra-itopk-values", "33"],
        ]
        for case in cases:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(case)


if __name__ == "__main__":
    unittest.main()
