"""Exact graph construction and adjacency invariants."""
from __future__ import annotations

import torch
from cagra_chunksplice.core import exact_knn


def build_graph(x, a):
    return exact_knn(x, x, a.degree, a.distance_batch,
                     torch.arange(len(x), device=x.device))


def validate_graph(graph, n):
    assert graph.ndim == 2 and len(graph) == n
    assert bool(((graph >= -1) & (graph < n)).all()), "out-of-range edge"
    rows = torch.arange(n, device=graph.device)
    assert not bool((graph == rows[:, None]).any()), "self edge"
    ordered = graph.sort(dim=1).values
    assert not bool(((ordered[:, 1:] == ordered[:, :-1])
                     & (ordered[:, 1:] >= 0)).any()), "duplicate edge"
