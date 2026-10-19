"""Portable adapter contracts, native API mocks, and external CLI integration."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cagra_chunksplice.baselines import (
    CuVSCAGRA, FaissHNSW, GraphBaseline, IncrementalIndex, MirageChunkSplice,
)
from cagra_chunksplice.cli import arguments, main
from cagra_chunksplice.runners.external import METHODS, benchmark_external

try:
    import numpy as np
except ImportError:
    np = None


ROOT = Path(__file__).resolve().parents[1]
SMALL = ["--device", "cpu", "--base", "32", "--chunks", "3", "--chunk-size", "3",
         "--dim", "3", "--queries", "4", "--degree", "10", "--ef", "12",
         "--search-width", "2", "--steps", "2", "--query-batch", "4",
         "--distance-batch", "16", "--repeats", "2", "--splice-intra", "2",
         "--mirage-buffer-capacity", "4", "--external-backend", "simulated"]


def parameters():
    return arguments(["--mode", "external", *SMALL])


class ExternalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.a = parameters()
        gen = torch.Generator().manual_seed(12)
        self.base = torch.randn(32, 3, generator=gen)
        self.new = torch.randn(9, 3, generator=gen) + 20

    def indexes(self):
        return [GraphBaseline(self.a, "Naive"), GraphBaseline(self.a),
                FaissHNSW(backend="simulated"), CuVSCAGRA(self.a, backend="simulated"),
                MirageChunkSplice(self.a, buffer_capacity=4)]

    def test_common_contract_and_reset(self):
        for index in self.indexes():
            with self.subTest(index=type(index).__name__):
                self.assertIs(index.build(self.base), index)
                index.add(self.new[:3]).add(self.new[3:])
                self.assertEqual(index.ntotal, 41)
                found = index.search(self.new[:2], 10)
                self.assertEqual(found.shape, (2, 10))
                self.assertEqual(found.dtype, torch.long)
                self.assertTrue(bool(((found >= -1) & (found < 41)).all()))
                for row in found:
                    valid = row[row >= 0]
                    self.assertEqual(len(valid), len(valid.unique()))
                self.assertEqual(index.search(self.new[:0]).shape, (0, 10))
                wide = index.search(self.new[:1], 45)
                self.assertEqual(wide.shape, (1, 45))
                self.assertTrue(bool((wide[:, 41:] == -1).all()))
                index.add(self.new[:0])
                self.assertEqual(index.ntotal, 41)
                index.build(self.base)
                self.assertEqual(index.ntotal, 32)
                self.assertTrue(index.metadata()["simulated"])

    def test_invalid_input_and_configuration(self):
        for index in self.indexes():
            with self.subTest(index=type(index).__name__):
                with self.assertRaises(ValueError):
                    index.add(self.new)
                with self.assertRaises(ValueError):
                    index.search(self.new)
                for invalid in (torch.empty(0, 3), torch.zeros(4),
                                torch.full((2, 3), float("nan"))):
                    with self.assertRaises(ValueError):
                        index.build(invalid)
                index.build(self.base)
                for invalid in (torch.zeros(2, 4), torch.full((2, 3), float("inf"))):
                    with self.assertRaises(ValueError):
                        index.add(invalid)
                    with self.assertRaises(ValueError):
                        index.search(invalid)
                for k in (0, -1, 1.5, True):
                    with self.assertRaises(ValueError):
                        index.search(self.new, k)
        for factory in (FaissHNSW, lambda **kw: CuVSCAGRA(self.a, **kw)):
            with self.assertRaises(ValueError):
                factory(backend="unknown")
        with self.assertRaises(ValueError):
            MirageChunkSplice(self.a, buffer_capacity=0)

    def test_missing_native_dependencies_and_strict_mode(self):
        with patch.dict(sys.modules, {"faiss": None}):
            index = FaissHNSW().build(self.base).add(self.new)
            self.assertEqual(index.metadata()["backend"], "torch-exact")
            self.assertIn("ModuleNotFoundError", index.metadata()["fallback_reason"])
            self.assertTrue(torch.equal(index.search(self.new, 1).flatten(), torch.arange(32, 41)))
            with self.assertRaisesRegex(RuntimeError, "native backend unavailable"):
                FaissHNSW(backend="native").build(self.base)
        cagra = CuVSCAGRA(self.a).build(self.base).add(self.new)
        naive = GraphBaseline(self.a, "Naive").build(self.base).add(self.new)
        self.assertTrue(torch.equal(cagra.fallback.graph, naive.graph))
        self.assertIn("CUDA", cagra.metadata()["fallback_reason"])
        with self.assertRaisesRegex(RuntimeError, "native backend unavailable"):
            CuVSCAGRA(self.a, backend="native").build(self.base)

    def test_buffer_visibility_capacity_multiple_flushes_and_tail(self):
        index = MirageChunkSplice(self.a, buffer_capacity=4).build(self.base)
        index.add(self.new[:3])
        self.assertEqual((index.buffer_size, index.main.ntotal, index.flush_count), (3, 32, 0))
        self.assertTrue(torch.equal(index.search(self.new[:3], 1).flatten(), torch.arange(32, 35)))
        index.add(self.new[3:])
        self.assertEqual((index.buffer_size, index.main.ntotal, index.flush_count), (1, 40, 2))
        self.assertEqual(index.search(self.new[-1:], 1).item(), 40)
        index.flush().flush()
        self.assertEqual((index.buffer_size, index.main.ntotal, index.flush_count), (0, 41, 3))
        self.assertEqual(index.metadata()["merged_vectors"], 9)
        self.assertTrue(torch.equal(index.data, index.main.data))
        index.build(self.base)
        self.assertEqual((index.buffer_size, index.flush_count, index.merged_vectors), (0, 0, 0))

    @unittest.skipIf(np is None, "native API mocks require optional NumPy")
    def test_faiss_native_add_and_gpu_request_falls_back_to_cpu(self):
        class FakeHNSW:
            def __init__(self, dimension, m):
                self.hnsw = SimpleNamespace()
                self.data = np.empty((0, dimension), dtype="float32")
                self.add_sizes = []

            def add(self, batch):
                self.add_sizes.append(len(batch))
                self.data = np.concatenate((self.data, batch))

            def search(self, queries, k):
                distances = ((queries[:, None] - self.data[None]) ** 2).sum(-1)
                ids = distances.argsort(axis=1)[:, :k]
                return np.take_along_axis(distances, ids, axis=1), ids

        module = SimpleNamespace(IndexHNSWFlat=FakeHNSW, __version__="mock")
        with patch.dict(sys.modules, {"faiss": module}):
            index = FaissHNSW(backend="native").build(self.base).add(self.new)
            self.assertEqual(index.index.add_sizes, [32, 9])
            self.assertFalse(index.metadata()["simulated"])
            self.assertTrue(torch.equal(index.search(self.new, 1).flatten(), torch.arange(32, 41)))
            # Exercise device negotiation without requiring actual GPU hardware.
            gpu = FaissHNSW(device="cuda", backend="native")
            with patch.object(gpu, "_vectors", side_effect=lambda x: x):
                gpu.build(self.base)
            self.assertEqual(gpu.metadata()["effective_device"], "cpu")
            self.assertIn("CPU HNSW", gpu.metadata()["fallback_reason"])

    @unittest.skipIf(np is None, "native API mocks require optional NumPy")
    def test_cagra_native_legacy_and_concatenated_extension(self):
        for concatenated in (False, True):
            with self.subTest(concatenated=concatenated):
                calls = []

                def legacy(params, index, additional):
                    calls.append(("legacy", len(additional)))
                    return SimpleNamespace(data=np.concatenate((index.data, additional)))

                def modern(params, index, extended_dataset, new_start_row):
                    calls.append(("concatenated", new_start_row, len(extended_dataset)))
                    index.data = extended_dataset

                def search(params, index, queries, k):
                    distances = ((queries[:, None] - index.data[None]) ** 2).sum(-1)
                    ids = distances.argsort(axis=1)[:, :k]
                    return np.take_along_axis(distances, ids, axis=1), ids

                cp = SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray,
                                     concatenate=np.concatenate,
                                     cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
                cagra = SimpleNamespace(
                    extend=modern if concatenated else legacy,
                    build=lambda params, data: SimpleNamespace(data=data),
                    search=search, IndexParams=lambda **kw: kw,
                    SearchParams=lambda **kw: kw, ExtendParams=lambda **kw: kw)
                modules = {"cupy": cp, "cuvs": SimpleNamespace(__version__="mock"),
                           "cuvs.neighbors.cagra": cagra}
                index = CuVSCAGRA(self.a, device="cuda", backend="native")
                with patch.dict(sys.modules, modules), patch.object(index, "_vectors", side_effect=lambda x: x):
                    index.build(self.base)
                    index.device = torch.device("cpu")
                    index.add(self.new[:3]).add(self.new[3:])
                    found = index.search(self.new, 1)
                self.assertTrue(torch.equal(found.flatten(), torch.arange(32, 41)))
                self.assertFalse(index.metadata()["simulated"])
                self.assertEqual(len(calls), 2)
                if concatenated:
                    self.assertEqual(calls, [("concatenated", 32, 35), ("concatenated", 35, 41)])
                else:
                    self.assertEqual(calls, [("legacy", 3), ("legacy", 6)])


class ExternalRunnerTests(unittest.TestCase):
    def test_replays_share_data_and_do_not_duplicate_mutable_updates(self):
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            first = benchmark_external(parameters(), torch.device("cpu"))
            second = benchmark_external(parameters(), torch.device("cpu"))
        self.assertTrue(first["benchmark_completed"])
        self.assertEqual(first["methods"], list(METHODS))
        self.assertEqual(first["experiment"], "external")
        for stage, repeat in zip(first["stages"], second["stages"]):
            for name, record in stage["methods"].items():
                self.assertEqual(record["metadata"]["ntotal"], stage["n"])
                self.assertEqual(len(record["update_samples_ms"]), 2)
                self.assertTrue(all(t >= 0 for t in record["update_samples_ms"]))
                self.assertTrue(0 <= record["recall10"] <= 1)
                self.assertGreater(record["qps"], 0)
                self.assertEqual(record["recall10"], repeat["methods"][name]["recall10"])
        summary = first["summary"]["MIRAGE-ChunkSplice"]
        self.assertEqual(summary["metadata"]["buffer_size"], 0)
        self.assertEqual(summary["metadata"]["merged_vectors"], 9)
        self.assertEqual(summary["metadata"]["flush_count"], 3)
        self.assertEqual(first["summary"]["FAISS-HNSW"]["recall10"], 1.0)
        json.dumps(first, allow_nan=False)

    def test_external_argument_validation(self):
        for flag in ("--mirage-buffer-capacity", "--hnsw-m", "--hnsw-ef-construction"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(["--mode", "external", flag, "0"])

    def test_cli_and_experiment_entry_points(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            for command in (["-m", "cagra_chunksplice", "--mode", "external"],
                            [str(ROOT / "experiments" / "benchmark_external.py")]):
                output = Path(directory) / "nested" / "external.json"
                result = subprocess.run(
                    [sys.executable, *command, *SMALL, "--repeats", "1", "--output", str(output)],
                    cwd=directory, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("Mixed Recall@10", result.stdout)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(report["benchmark_completed"])
                self.assertEqual(set(report["summary"]), set(METHODS))

    def test_native_failure_keeps_completed_report(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "external.json"
            output.write_text('{"status": "previous"}', encoding="utf-8")
            with patch.dict(sys.modules, {"faiss": None}), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--mode", "external", *SMALL, "--external-backend", "native",
                             "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.read_text())["status"], "previous")
            failure = json.loads(output.with_suffix(".error.json").read_text())
            self.assertFalse(failure["benchmark_completed"])
