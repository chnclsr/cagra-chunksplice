"""CPU-only MIRAGE-ANNS baseline (PACMMOD 3(3), Art. 188, SIGMOD 2025).

The published C++/FAISS index uses a multi-layer hierarchy, NN-Descent base
construction and RNG-pruned incremental insertion. It is not a GPU index,
insertion buffer, or ChunkSplice variant. Upstream builds with
-DFAISS_ENABLE_GPU=OFF; see https://github.com/dsg-uwaterloo/mirage.

Native execution requires custom Python bindings exposing faiss.IndexMirage.
Stock faiss-cpu does not supply this class. The portable fallback below is
only an exact CPU functional reference, not a MIRAGE graph or timing model.
"""
from __future__ import annotations

import importlib

import torch

from .base import CPUIndex, check_backend, unavailable


class CPUMirage(CPUIndex):
    """Append directly through native CPU add, or use an exact CPU reference."""

    def __init__(self, device="cpu", backend="auto"):
        super().__init__(device)
        check_backend(backend)
        self.requested_backend = backend
        self.index = None
        self.version = None

    def _build(self, vectors):
        self.backend, self.simulated = "torch-exact-cpu", True
        self.index, self.version, self.fallback_reason = None, None, None
        self.native_threads = None
        if self.requested_backend == "simulated":
            self.fallback_reason = "exact CPU reference explicitly requested; no MIRAGE hierarchy"
            return
        try:
            faiss = importlib.import_module("faiss")
            if not hasattr(faiss, "IndexMirage"):
                raise ImportError("MIRAGE requires custom faiss.IndexMirage Python bindings; "
                                  "stock faiss-cpu is insufficient")
            index = faiss.IndexMirage(vectors.shape[1])
            index.add(vectors.numpy())
        except ImportError as exc:
            unavailable(self, self.requested_backend, exc)
            return
        self.index = index
        self.backend, self.simulated = "faiss-mirage-cpu", False
        self.version = getattr(faiss, "__version__", "unknown")
        self.native_threads = faiss.omp_get_max_threads() if hasattr(faiss, "omp_get_max_threads") else None

    def _add(self, vectors):
        if self.index is not None:
            self.index.add(vectors.numpy())

    def _search(self, queries, k):
        if self.index is None:
            return super()._search(queries, k)
        _, neighbors = self.index.search(queries.numpy(), k)
        return torch.as_tensor(neighbors, dtype=torch.long)

    def metadata(self):
        return {**super().metadata(), "version": self.version,
                "native_algorithm": {
                    "hierarchy": "multi-layer",
                    "base_construction": "NN-Descent with RNG pruning",
                    "incremental_update": "CPU multi-threaded add with RNG pruning",
                    "gpu_enabled": False,
                },
                "native_parameters": {key: getattr(self.index.mirage, key)
                                      for key in ("S", "R", "iter")
                                      if hasattr(self.index.mirage, key)}
                if self.index is not None else None,
                "reference": "https://doi.org/10.1145/3725325",
                "definition": "Native CPU MIRAGE-ANNS or exact CPU reference; "
                              "fallback does not implement MIRAGE hierarchy or model its performance."}
