"""Budgeted intra-chunk splicing with old-cohort bridges."""
from __future__ import annotations

import torch
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.algorithms.naive import naive_extend
from cagra_chunksplice.algorithms.pruning import rng_prune


def chunk_splice_extend(x, old_graph, start, end, a):
    """Exact or beam Naive insertion plus a diverse splice for new rows only.

    Dense chunks are sinkholes when peers take most of the degree budget: a beam
    entering a microcluster spends its expansions on near-duplicate peers and has
    few exits (Rebuild's exact kNN shows the extreme). So the splice is budgeted:
    degree - splice_intra slots are reserved for new->old bridges (the Naive
    links that keep base navigation), and only splice_intra slots go to peers.
    One cohort-local rng_prune call over [Naive old links | intra kNN pool] picks
    both: peers cannot occlude bridges, and RNG spreads the few peer slots across
    directions instead of the nearest near-duplicates; backfill uses only available candidates; missing ANN bridges remain absent.
    Work is one O(b^2 d) GEMM plus one width (degree + splice_pool) prune over
    the chunk rows; old rows are exactly the Naive result.
    """
    graph = naive_extend(x, old_graph, start, end, a)
    b = end - start
    if b <= 1:
        return graph
    new = x[start:end]
    new_ids = torch.arange(start, end, device=x.device)
    k_intra = min(a.splice_intra, b - 1)
    pool = min(max(a.splice_pool, k_intra), b - 1)
    intra = exact_knn(new, new, pool, a.distance_batch,
                      torch.arange(b, device=x.device)) + start
    candidates = torch.cat((graph[start:end], intra), dim=1)
    route = a.degree - k_intra if start > 0 else 0
    graph[start:end] = rng_prune(x, new_ids, candidates, a.degree,
                                 alpha=a.splice_alpha, route=route,
                                 cohort_split=start, backfill=True)
    return graph
