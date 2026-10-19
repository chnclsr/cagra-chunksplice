"""Global graph construction and adjacency invariants."""
from __future__ import annotations

import torch
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.algorithms.pruning import rng_prune


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

def build_rng_graph(x, a):
    """Global kNN pool, greedy RNG pruning, reverse proposals, and re-pruning.

    This is a degree-capped, candidate-restricted directed RNG heuristic, not
    the exact relative neighborhood graph. Reverse proposals are merged before
    final pruning; degree capping does not guarantee reciprocal edges.
    No rejected candidates are backfilled. Timing includes the global pool.
    """
    n = len(x)
    if n <= 1:
        return torch.full((n, a.degree), -1, dtype=torch.long, device=x.device)
    pool = min(getattr(a, "rebuild_pool", 4 * a.degree), n - 1)
    if pool < min(a.degree, n - 1):
        raise ValueError("rebuild_pool must be >= degree")
    ids = torch.arange(n, device=x.device)
    candidates = exact_knn(x, x, pool, a.distance_batch, ids)
    forward = torch.full((n, a.degree), -1, dtype=torch.long, device=x.device)
    alpha = getattr(a, "rebuild_alpha", 1.0)
    for lo in range(0, n, a.distance_batch):
        hi = min(lo + a.distance_batch, n)
        forward[lo:hi] = rng_prune(x, ids[lo:hi], candidates[lo:hi],
                                   a.degree, alpha=alpha)
    # Ragged reverse lists avoid an N-by-N reverse adjacency allocation.
    forward_rows = forward.cpu().tolist()
    merged = [set(row) - {-1} for row in forward_rows]
    for source, row in enumerate(forward_rows):
        for target in row:
            if target >= 0:
                merged[target].add(source)
    result = torch.full_like(forward, -1)
    for lo in range(0, n, a.distance_batch):
        hi = min(lo + a.distance_batch, n)
        width = max(len(row) for row in merged[lo:hi])
        if not width:
            continue
        batch = torch.tensor([sorted(row) + [-1] * (width - len(row))
                              for row in merged[lo:hi]], device=x.device)
        result[lo:hi] = rng_prune(x, ids[lo:hi], batch, a.degree, alpha=alpha)
    return result
