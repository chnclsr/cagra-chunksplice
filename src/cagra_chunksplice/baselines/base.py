"""Common append-only interface and portable reference implementations."""
from __future__ import annotations

import torch

from cagra_chunksplice.algorithms.chunk_splice import chunk_splice_extend
from cagra_chunksplice.algorithms.naive import naive_extend
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.graph import build_graph
from cagra_chunksplice.search import graph_search
from cagra_chunksplice.timing import synchronize


class IncrementalIndex:
    """Append-only squared-L2 index using insertion-order integer identifiers.

    build resets the index, add appends a batch, and search returns a long
    tensor of shape (queries, k) on the requested device, padded with -1.
    """

    backend = "torch-exact"
    simulated = True

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.data = None
        self.fallback_reason = None

    @property
    def ntotal(self):
        return 0 if self.data is None else len(self.data)

    def _vectors(self, vectors):
        vectors = torch.as_tensor(vectors, dtype=torch.float32, device=self.device)
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("vectors must have shape (n, dimension), dimension > 0")
        if not bool(torch.isfinite(vectors).all()):
            raise ValueError("vectors must be finite")
        return vectors.detach().contiguous()

    def build(self, vectors):
        vectors = self._vectors(vectors)
        if not len(vectors):
            raise ValueError("initial dataset must not be empty")
        self._build(vectors)
        self.data = vectors.clone()
        return self

    def add(self, vectors):
        if self.data is None:
            raise ValueError("build the index before adding vectors")
        vectors = self._vectors(vectors)
        if vectors.shape[1] != self.data.shape[1]:
            raise ValueError("vector dimensions do not match the index")
        if len(vectors):
            self._add(vectors)
            self.data = torch.cat((self.data, vectors))
        return self

    def search(self, queries, k=10):
        if self.data is None:
            raise ValueError("build the index before searching")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        queries = self._vectors(queries)
        if queries.shape[1] != self.data.shape[1]:
            raise ValueError("query dimensions do not match the index")
        result = torch.full((len(queries), k), -1, device=self.device, dtype=torch.long)
        if len(queries):
            take = min(k, self.ntotal)
            result[:, :take] = self._search(queries, take)
        return result

    def _build(self, vectors):
        pass

    def _add(self, vectors):
        pass

    def _search(self, queries, k):
        return exact_knn(queries, self.data, k, 256)

    def synchronize(self):
        synchronize(self.device)

    def metadata(self):
        return {"backend": self.backend, "simulated": self.simulated,
                "requested_device": str(self.device), "effective_device": str(self.device),
                "fallback_reason": self.fallback_reason, "ntotal": self.ntotal}


class GraphBaseline(IncrementalIndex):
    """Adapter for the existing Naive and ChunkSplice mechanism models."""

    def __init__(self, parameters, method="ChunkSplice", device="cpu"):
        super().__init__(device)
        if method not in ("Naive", "ChunkSplice"):
            raise ValueError("unknown graph method: " + method)
        self.a, self.method = parameters, method
        self.backend = "torch-" + method.lower()

    def _build(self, vectors):
        if len(vectors) <= self.a.degree:
            raise ValueError("initial graph requires more vectors than degree")
        self.graph = build_graph(vectors, self.a)
        gen = torch.Generator().manual_seed(self.a.seed)
        self.base_entries = torch.randperm(len(vectors), generator=gen)[:self.a.ef]

    def _add(self, vectors):
        active = torch.cat((self.data, vectors))
        extend = naive_extend if self.method == "Naive" else chunk_splice_extend
        self.graph = extend(active, self.graph, self.ntotal, len(active), self.a)

    def _search(self, queries, k):
        # The traversal retains at most ef candidates, even when k is larger.
        take = min(k, self.a.ef)
        if self.a.entry_policy == "old":
            entries = self.base_entries
        else:
            gen = torch.Generator().manual_seed(self.a.seed + self.ntotal)
            entries = torch.randperm(self.ntotal, generator=gen)[:self.a.ef]
        found = graph_search(self.data, self.graph, queries, entries.to(self.device), self.a, take)
        if take == k:
            return found
        result = torch.full((len(queries), k), -1, device=self.device, dtype=torch.long)
        result[:, :take] = found
        return result


def check_backend(backend):
    if backend not in ("auto", "native", "simulated"):
        raise ValueError("backend must be auto, native, or simulated")


def unavailable(index, backend, exc):
    """Fallback is selected only at build time, never during measured updates."""
    if backend == "native":
        raise RuntimeError("native backend unavailable: " + str(exc)) from exc
    index.fallback_reason = f"{type(exc).__name__}: {exc}"
