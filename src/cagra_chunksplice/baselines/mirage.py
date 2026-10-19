"""MIRAGE-inspired bounded insertion buffer with synchronous ChunkSplice merges.

This is the requested buffer experiment, not an implementation or timing model
of the published MIRAGE-ANNS system (doi:10.1145/3725325).
"""
from __future__ import annotations

import torch

from cagra_chunksplice.core import candidate_distances, exact_knn

from .base import GraphBaseline, IncrementalIndex


class MirageChunkSplice(IncrementalIndex):
    """Search committed graph and pending buffer; merge at a fixed capacity.

    Inserts are immediately visible. Full buffers flush synchronously, including
    oversized batches split at capacity. flush drains a partial buffer and
    preserves insertion-order IDs. No background thread or simulated sleeps.
    """

    backend = "torch-mirage-buffer-chunksplice"

    def __init__(self, parameters, device="cpu", buffer_capacity=256):
        super().__init__(device)
        if not isinstance(buffer_capacity, int) or buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be a positive integer")
        self.a, self.buffer_capacity = parameters, buffer_capacity
        self.main = GraphBaseline(parameters, "ChunkSplice", device)
        self.flush_count = self.merged_vectors = 0

    @property
    def buffer_size(self):
        return len(self.buffer) if self.data is not None else 0

    def _build(self, vectors):
        self.main.build(vectors)
        self.buffer = vectors.new_empty((0, vectors.shape[1]))
        self.flush_count = self.merged_vectors = 0

    def _add(self, vectors):
        offset = 0
        while offset < len(vectors):
            take = min(self.buffer_capacity - len(self.buffer), len(vectors) - offset)
            self.buffer = torch.cat((self.buffer, vectors[offset:offset + take]))
            offset += take
            if len(self.buffer) == self.buffer_capacity:
                self.flush()

    def flush(self):
        if self.data is None:
            raise ValueError("build the index before flushing")
        if len(self.buffer):
            count = len(self.buffer)
            self.main.add(self.buffer)
            self.buffer = self.buffer[:0]
            self.flush_count += 1
            self.merged_vectors += count
        return self

    def _search(self, queries, k):
        main_ids = self.main.search(queries, k)
        if not len(self.buffer):
            return main_ids
        pending = exact_knn(queries, self.buffer, min(k, len(self.buffer)),
                            self.a.distance_batch) + self.main.ntotal
        candidates = torch.cat((main_ids, pending), dim=1)
        distances = candidate_distances(queries, self.data, candidates)
        scores, order = distances.topk(k, largest=False, sorted=True)
        return candidates.gather(1, order).masked_fill(~scores.isfinite(), -1)

    def metadata(self):
        return {**super().metadata(), "buffer_capacity": self.buffer_capacity,
                "buffer_size": self.buffer_size, "flush_count": self.flush_count,
                "merged_vectors": self.merged_vectors, "committed_vectors": self.main.ntotal,
                "definition": "MIRAGE-inspired buffer simulator; ChunkSplice merge and exact pending-buffer search; not full MIRAGE-ANNS."}
