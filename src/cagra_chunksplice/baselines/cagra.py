"""Optional cuVS CAGRA graph extension and a portable chunk-hole simulator."""
from __future__ import annotations

import importlib
import inspect

import torch

from .base import GraphBaseline, IncrementalIndex, check_backend, unavailable


class CuVSCAGRA(IncrementalIndex):
    """Native cuvs.neighbors.cagra build/extend/search when CUDA is available.

    The fallback reuses Naive's snapshot chunk-hole model, not native CAGRA
    kernels or a GPU timing model. Legacy and concatenated-dataset extend
    signatures are supported without substituting full index rebuilds.
    """

    def __init__(self, parameters, device="cpu", backend="auto"):
        super().__init__(device)
        check_backend(backend)
        self.a, self.requested_backend = parameters, backend
        self.fallback = None
        self.cp = self.cagra = self.index = self.gpu_data = None
        self.version = None

    def _array(self, vectors):
        if vectors.device.type == "cuda":
            return self.cp.from_dlpack(vectors)
        return self.cp.asarray(vectors.cpu().numpy())

    def _build(self, vectors):
        self.backend, self.simulated = "torch-cagra-chunk-hole", True
        self.fallback, self.fallback_reason = None, None
        self.cp = self.cagra = self.index = self.gpu_data = None
        self.version = None
        if self.requested_backend != "simulated":
            try:
                if self.device.type != "cuda":
                    raise RuntimeError("cuVS CAGRA requires a CUDA device")
                self.cp = importlib.import_module("cupy")
                self.cagra = importlib.import_module("cuvs.neighbors.cagra")
                cuvs = importlib.import_module("cuvs")
                self.version = getattr(cuvs, "__version__", "unknown")
                signature = inspect.signature(self.cagra.extend)
                self.concatenated_extend = "new_start_row" in signature.parameters
                self.gpu_data = self._array(vectors)
                params = self.cagra.IndexParams(
                    metric="sqeuclidean", graph_degree=self.a.degree,
                    intermediate_graph_degree=min(len(vectors) - 1, max(2 * self.a.degree, 32)))
                self.index = self.cagra.build(params, self.gpu_data)
                self.cp.cuda.get_current_stream().synchronize()
            except (ImportError, ModuleNotFoundError) as exc:
                unavailable(self, self.requested_backend, exc)
                self.cp = self.cagra = self.index = self.gpu_data = None
            except RuntimeError as exc:
                if "requires a CUDA device" in str(exc) or "CUDA" in str(exc):
                    unavailable(self, self.requested_backend, exc)
                    self.cp = self.cagra = self.index = self.gpu_data = None
                else:
                    raise
            else:
                self.backend, self.simulated = "cuvs-cagra-cuda", False
                return
        else:
            self.fallback_reason = "portable chunk-hole simulator explicitly requested"
        self.fallback = GraphBaseline(self.a, "Naive", self.device).build(vectors)

    def _add(self, vectors):
        if self.fallback is not None:
            self.fallback.add(vectors)
            return
        additional = self._array(vectors)
        params = self.cagra.ExtendParams(max_chunk_size=self.a.chunk_size)
        if self.concatenated_extend:
            extended = self.cp.concatenate((self.gpu_data, additional), axis=0)
            result = self.cagra.extend(params, self.index, extended, self.ntotal)
            self.gpu_data = extended  # Retain caller-owned storage for the index lifetime.
        else:
            result = self.cagra.extend(params, self.index, additional)
            self.gpu_data = self.cp.concatenate((self.gpu_data, additional), axis=0)
        if result is not None:
            self.index = result

    def _search(self, queries, k):
        if self.fallback is not None:
            return self.fallback.search(queries, k)
        params = self.cagra.SearchParams(
            itopk_size=max(32, ((max(k, self.a.ef) + 31) // 32) * 32),
            search_width=self.a.search_width, max_iterations=self.a.steps)
        _, neighbors = self.cagra.search(params, self.index, self._array(queries), k)
        neighbors = self.cp.asnumpy(self.cp.asarray(neighbors)).astype("int64")
        return torch.as_tensor(neighbors, dtype=torch.long, device=self.device)

    def synchronize(self):
        super().synchronize()
        if self.cp is not None:
            self.cp.cuda.get_current_stream().synchronize()

    def metadata(self):
        return {**super().metadata(), "version": self.version,
                "architecture": "gpu-graph-insertion",
                "implementation_architecture": "pytorch-graph-model" if self.simulated else "native-gpu-graph",
                "graph_degree": self.a.degree, "max_chunk_size": self.a.chunk_size,
                "search_parameters": {
                    "itopk_size": max(32, ((max(10, self.a.ef) + 31) // 32) * 32),
                    "search_width": self.a.search_width, "max_iterations": self.a.steps},
                "extend_api": ("concatenated" if self.concatenated_extend else "legacy")
                if not self.simulated else None,
                "definition": "Native cuVS extension or PyTorch snapshot-insertion mechanism simulator."}
