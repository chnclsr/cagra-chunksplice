"""Small end-to-end CPU runs exercising all experiment modes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SMALL = ["--device", "cpu", "--base", "32", "--chunks", "2", "--chunk-size", "4",
         "--dim", "3", "--queries", "4", "--degree", "10", "--ef", "12",
         "--search-width", "2", "--steps", "2", "--query-batch", "4",
         "--distance-batch", "16", "--local-cap", "16", "--candidate-cap", "12",
         "--refine-rounds", "1", "--repeats", "1", "--splice-intra", "2"]


class ExperimentTests(unittest.TestCase):
    def run_cli(self, argv, cwd):
        result = subprocess.run(
            [sys.executable, "-m", "cagra_chunksplice", *argv],
            cwd=cwd, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_self_test_command(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(["--device", "cpu", "--self-test-only"], directory)
            self.assertIn("SELF-TEST PASS", result.stdout)

    def test_benchmark(self):
        self.check_mode("benchmark", "benchmark", "stages",
                        {"Rebuild-kNN", "Rebuild-RNG", "Naive", "ChunkSplice", "LocalRNG"})

    def test_sweep_with_remainder(self):
        self.check_mode("sweep", "chunk_sweep", "results",
                        {"Rebuild-kNN", "Rebuild-RNG", "Naive", "ChunkSplice"},
                        ["--sweep-total", "9", "--sweep-chunk-sizes", "4", "5"])

    def test_stream_checkpoints(self):
        self.check_mode("stream", "streaming", "history", {"Naive", "ChunkSplice"},
                        ["--chunks", "3", "--stream-eval-every", "10"])

    def test_stress_beam_angular_and_age_diagnostics(self):
        self.check_mode("stress", "long_horizon_stress", "history",
                        {"Rebuild-kNN", "Rebuild-RNG", "Naive", "ChunkSplice"},
                        ["--chunks", "6", "--stream-eval-every", "3",
                         "--insertion-mode", "beam", "--insertion-ef", "12",
                         "--insertion-steps", "1", "--workload", "high-lid",
                         "--query-distribution", "angular", "--age-queries", "2"])

    def test_stream_ood_with_global_baselines(self):
        self.check_mode("stream", "streaming", "history",
                        {"Rebuild-kNN", "Rebuild-RNG", "Naive", "ChunkSplice"},
                        ["--chunks", "3", "--stream-rebuild", "--query-distribution", "ood"])

    def test_scaling_against_full_rebuild(self):
        methods = {"ChunkSplice", "Naive", "Rebuild-kNN", "Rebuild-RNG",
                   "cuVS-CAGRA-extend", "cuVS-CAGRA-rebuild"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scaling.json"
            self.run_cli([*SMALL, "--mode", "scaling", "--scaling-bases", "32", "64",
                          "--scaling-rebuild-rng", "--scaling-cagra",
                          "--external-backend", "simulated", "--output", str(output)], directory)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["experiment"], "update_scaling")
            self.assertEqual([p["n"] for p in report["points"]], [32, 64])
            for point in report["points"]:
                self.assertEqual(set(point["methods"]), methods)
                for name, row in point["methods"].items():
                    self.assertEqual(len(row["update_samples_ms"]), 1)
                    self.assertTrue(0 <= row["recall10"] <= 1)
                    self.assertEqual(row["incremental"], name in {"ChunkSplice", "Naive", "cuVS-CAGRA-extend"})
            self.assertTrue(report["points"][0]["methods"]["cuVS-CAGRA-rebuild"]["metadata"]["simulated"])
            self.assertEqual(set(report["scaling_exponent"]), methods)
            self.assertEqual(set(report["speedup_vs_full_rebuild"]),
                             {"Rebuild-kNN", "Rebuild-RNG", "cuVS-CAGRA-rebuild"})

    def test_scaling_options_require_scaling_mode(self):
        result = subprocess.run(
            [sys.executable, "-m", "cagra_chunksplice", "--scaling-cagra"],
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 2)

    def check_mode(self, mode, experiment, rows_key, methods, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results" / "nested" / (mode + ".json")
            self.run_cli([*SMALL, "--mode", mode, *extra, "--output", str(output)], directory)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["experiment"], experiment)
            self.assertEqual(set(report["methods"]), methods)
            self.assertEqual(report["schema_version"], 2)
            self.assertIn("source_files_sha256", report)
            for row in report[rows_key]:
                self.assertEqual(set(row["methods"]), methods)
                for metrics in row["methods"].values():
                    self.assertTrue(all(t >= 0 for t in metrics["update_samples_ms"]))
                    if "recall10" in metrics:
                        self.assertTrue(0 <= metrics["recall10"] <= 1)
            for row in report[rows_key]:
                for record in row["methods"].values():
                    if "recall10" in record:
                        self.assertGreater(record["scc_count"], 0)
                        self.assertTrue(0 <= record["in_degree_gini"] <= 1)
                        self.assertIn("mean_bridge_length", record)
                        if mode in ("stream", "stress"):
                            self.assertIn("age_stratified_recall10", record)
                            self.assertIn("previous_checkpoint_edge_survival", record)
                            self.assertIn("base", record["age_stratified_recall10"])
            if mode == "stress":
                self.assertEqual(len(report["history"]), 6)
                self.assertEqual(set(report["history"][-1]["methods"]["Naive"]["age_stratified_recall10"]),
                                 {"base", "early", "middle", "recent"})
            if mode == "sweep":
                self.assertEqual([row["last_chunk_size"] for row in report["results"]], [1, 4])
                self.assertTrue(report["controls"]["same_ordered_data_queries_truth_entries"])
            if mode == "stream" and "--stream-rebuild" not in extra:
                self.assertEqual([row["evaluated"] for row in report["history"]], [True, False, True])
                for row in report["history"]:
                    for name, alias in (("Naive", "naive"), ("ChunkSplice", "splice")):
                        self.assertEqual(row[alias], row["methods"][name])
