"""Optional FAISS HNSW insertion with an explicitly labeled exact fallback."""
from __future__ import annotations

import importlib

import torch

from .base import IncrementalIndex, check_backend, unavailable


class FaissHNSW(IncrementalIndex):
    """Use IndexHNSWFlat.add, or an exact PyTorch reference if absent.

    FAISS HNSW is a CPU index. CUDA requests retain native CPU HNSW and include
    transfers in measurements; another GPU algorithm is never called HNSW.
    """

    def __init__(self, device="cpu", backend="auto", m=32, ef_construction=80, ef_search=64):
        super().__init__(device)
        check_backend(backend)
        if min(m, ef_construction, ef_search) <= 0:
            raise ValueError("HNSW parameters must be positive")
        self.requested_backend = backend
        self.m, self.ef_construction, self.ef_search = m, ef_construction, ef_search
        self.version = None
        self.index = None

    def _build(self, vectors):
        self.backend, self.simulated = "torch-exact", True
        self.index, self.fallback_reason, self.version = None, None, None
        if self.requested_backend == "simulated":
            self.fallback_reason = "portable exact reference explicitly requested"
            return
        try:
            faiss = importlib.import_module("faiss")
            index = faiss.IndexHNSWFlat(vectors.shape[1], self.m)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            index.add(vectors.cpu().numpy())
        except (ImportError, ModuleNotFoundError) as exc:
            unavailable(self, self.requested_backend, exc)
            return
        except RuntimeError as exc:
            if "could not find" in str(exc).lower() or "not available" in str(exc).lower():
                unavailable(self, self.requested_backend, exc)
                return
            raise
        self.index = index
        self.backend, self.simulated = "faiss-hnsw-cpu", False
        self.version = getattr(faiss, "__version__", "unknown")
        if self.device.type == "cuda":
            self.fallback_reason = "FAISS HNSW has no native GPU implementation; using CPU HNSW"

    def _add(self, vectors):
        if self.index is not None:
            self.index.add(vectors.cpu().numpy())

    def _search(self, queries, k):
        if self.index is None:
            return super()._search(queries, k)
        _, neighbors = self.index.search(queries.cpu().numpy(), k)
        return torch.as_tensor(neighbors, device=self.device, dtype=torch.long)

    def metadata(self):
        return {**super().metadata(), "version": self.version,
                "effective_device": "cpu" if self.index is not None else str(self.device),
                "m": self.m, "ef_construction": self.ef_construction, "ef_search": self.ef_search,
                "definition": "Native CPU HNSW or exact reference; fallback is not an HNSW performance model."}
