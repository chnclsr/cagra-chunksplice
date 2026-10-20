"""ANN cache, metric, fallback, and sweep replay regression tests."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cagra_chunksplice.cli import arguments
from cagra_chunksplice.datasets import load_external_data, _download
from cagra_chunksplice.runners.external_sweep import benchmark_external_sweep

try:
    import h5py
    import numpy as np
except ImportError:
    h5py = None

SMALL = ["--mode", "external", "--device", "cpu", "--external-backend", "simulated",
         "--base", "32", "--chunks", "1", "--chunk-size", "8", "--degree", "10",
         "--ef", "12", "--queries", "4", "--dim", "3", "--steps", "2", "--repeats", "1",
         "--search-width", "2", "--splice-intra", "2"]


class DatasetTests(unittest.TestCase):
    def test_missing_dataset_is_never_implicitly_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            a = arguments([*SMALL, "--dataset", "sift1m", "--dataset-cache", directory])
            with self.assertRaises(FileNotFoundError):
                load_external_data(a, torch.device("cpu"))
            a.allow_synthetic_fallback = True
            first = load_external_data(a, torch.device("cpu"))
            second = load_external_data(a, torch.device("cpu"))
            self.assertEqual(first[0].shape, (40, 128))
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(first[4]["synthetic"])
            self.assertEqual(first[4]["actual"], "synthetic-scaled-sift1m")

    @unittest.skipIf(h5py is None, "requires datasets extra")
    def test_cache_queries_normalization_and_per_prefix_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glove-100-angular.hdf5"
            with h5py.File(path, "w") as source:
                source.attrs["distance"] = "angular"
                source["train"] = np.random.default_rng(1).normal(size=(48, 100))
                source["test"] = np.random.default_rng(2).normal(size=(6, 100))
            a = arguments([*SMALL, "--dataset", "glove-100", "--dataset-cache", directory])
            with patch("urllib.request.urlopen", side_effect=AssertionError("cache must prevent download")):
                x, q, _, _, info = load_external_data(a, torch.device("cpu"))
            self.assertEqual(a.dim, 100)
            self.assertTrue(torch.allclose(x.norm(dim=1), torch.ones(40)))
            self.assertTrue(torch.allclose(q.norm(dim=1), torch.ones(4)))
            self.assertFalse(info["synthetic"])
            self.assertTrue(info["subset"])
            self.assertEqual(len(info["sha256"]), 64)
            path.write_bytes(b"corrupt")
            a.allow_synthetic_fallback = True
            with self.assertRaises(OSError):
                load_external_data(a, torch.device("cpu"))

    @unittest.skipIf(h5py is None, "requires datasets extra")
    def test_undeclared_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sift-128-euclidean.hdf5"
            with h5py.File(path, "w") as source:
                source["train"] = np.zeros((48, 128), dtype="float32")
                source["test"] = np.zeros((6, 128), dtype="float32")
            a = arguments([*SMALL, "--dataset", "sift1m", "--dataset-cache", directory])
            with self.assertRaisesRegex(ValueError, "distance metric"):
                load_external_data(a, torch.device("cpu"))

    def test_incomplete_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.hdf5"
            response = io.BytesIO(b"truncated")
            response.headers = {"Content-Length": "100"}
            with patch("urllib.request.urlopen", return_value=response):
                with self.assertRaises(OSError):
                    _download("https://example.test/data", path, lambda p: None)
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob("*.part")), [])

    def test_http_protocol_errors_become_os_errors(self):
        import http.client
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.hdf5"
            with patch("urllib.request.urlopen", side_effect=http.client.IncompleteRead(b"")):
                with self.assertRaises(OSError):
                    _download("https://example.test/data", path, lambda p: None)
            self.assertEqual(list(path.parent.glob("*")), [])

    def test_sweep_preserves_methods_samples_and_shared_truth(self):
        a = arguments([*SMALL, "--external-sweep", "--graph-ef-values", "12", "16",
                       "--hnsw-m-values", "8", "--hnsw-ef-search-values", "12",
                       "--mirage-s-values", "16", "--mirage-r-values", "4",
                       "--mirage-iter-values", "10", "--cagra-itopk-values", "32",
                       "--cagra-search-width-values", "1", "--cagra-iterations-values", "2"])
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            report = benchmark_external_sweep(a, torch.device("cpu"))
        self.assertEqual(len(report["curve_points"]), 7)
        self.assertEqual(len(report["summary"]), 5)
        self.assertEqual(len(report["stages"][0]["methods"]), 5)
        self.assertTrue(report["simulated"])
        self.assertFalse(report["native_external_comparison_complete"])
        self.assertEqual(len(report["curves"]["frontiers"]), 5)
        for point in report["curve_points"]:
            self.assertEqual(point["metadata"]["ntotal"], 40)
            self.assertEqual(len(point["stages"]), 1)
            if point["method"] in ("MIRAGE", "FAISS-HNSW"):
                self.assertEqual(point["recall10"], 1.0)


if __name__ == "__main__":
    unittest.main()
