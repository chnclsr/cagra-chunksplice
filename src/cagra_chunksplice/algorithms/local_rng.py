"""Bounded MIRAGE-inspired local graph refinement."""
from __future__ import annotations

import torch
from cagra_chunksplice.core import candidate_distances
from cagra_chunksplice.algorithms.naive import naive_extend
from cagra_chunksplice.algorithms.pruning import rng_prune
from cagra_chunksplice.core import squared_l2
from cagra_chunksplice.core import unique_candidates


def local_refine(x, graph, start, end, a):
    """Repair one chunk plus its bounded 1/2-hop old neighborhood on device.

    Select old vertices only from graph hops; if the neighborhood exceeds the
    cap, retain those geometrically closest to this chunk. All new rows survive
    this cap. Existing outgoing neighbors remain candidates even outside the
    local subgraph, preserving opportunities for boundary connectivity.
    """
    new_ids = torch.arange(start, end, device=x.device)
    frontier = new_ids
    parts = []
    for _ in range(a.hops):
        edges = graph[frontier].reshape(-1)
        frontier = torch.unique(edges[edges >= 0])
        parts.append(frontier)
    old = torch.unique(torch.cat(parts))
    old = old[old < start]
    old_limit = a.local_cap - len(new_ids)
    if len(old) > old_limit:
        nearest = torch.full((len(old),), float("inf"), device=x.device)
        for lo in range(0, len(new_ids), a.distance_batch):
            nearest = torch.minimum(nearest, squared_l2(
                x[new_ids[lo:lo + a.distance_batch]], x[old]).amin(dim=0))
        old = old[nearest.topk(old_limit, largest=False).indices]
    local = torch.cat((new_ids, old))
    take = min(a.candidate_cap, len(local) - 1)
    # Local GEMM also makes chunk peers available; no global kNN is run here.
    local_d = squared_l2(x[local], x[local])
    local_d.fill_diagonal_(float("inf"))
    local_neighbors = local[local_d.topk(take, largest=False).indices]
    for _ in range(a.refine_rounds):
        # Snapshot before parallel row updates: no order-dependent mutation.
        snapshot = graph.clone()
        for lo in range(0, len(local), a.distance_batch):
            hi = min(lo + a.distance_batch, len(local))
            rows = local[lo:hi]
            # Explicit neighbor exchange inside the bounded subgraph:
            # local nearest candidates plus their current graph neighbors.
            seeds = local_neighbors[lo:hi, :min(4, take)]
            exchange = snapshot[seeds].flatten(1)
            ids = torch.cat((snapshot[rows], local_neighbors[lo:hi], exchange), dim=1)
            # Bound RNG work, while keeping candidates selected from all sources.
            ids = unique_candidates(ids, rows)
            d = candidate_distances(x[rows], x, ids)
            cap = min(a.candidate_cap, ids.shape[1])
            vals, order = d.topk(cap, largest=False)
            ids = ids.gather(1, order).masked_fill(~vals.isfinite(), -1)
            # Cohort boundary is this chunk: earlier chunks count as old cohort.
            graph[rows] = rng_prune(x, rows, ids, a.degree, alpha=a.alpha,
                                    route=a.route,
                                    cohort_split=start if a.cohort_protect else None,
                                    backfill=a.backfill)
    return graph, {"local_vertices": len(local), "local_old_vertices": len(old),
                   "modified_row_limit": a.local_cap, "rng_alpha": a.alpha,
                   "rng_route": a.route, "cohort_protect": a.cohort_protect,
                   "rng_backfill": a.backfill}


def refined_extend(x, old_graph, start, end, a):
    graph = naive_extend(x, old_graph, start, end, a)
    return local_refine(x, graph, start, end, a)
