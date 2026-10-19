"""Snapshot insertion with degree-capped reverse proposals."""
from __future__ import annotations

import torch
from cagra_chunksplice.core import candidate_distances
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.core import squared_l2


def naive_extend(x, old_graph, start, end, a):
    """Snapshot insertion: new->old only, with degree-capped old->new reverses.

    Reverse proposals are *only* reverses of chosen new->old edges. This avoids
    the trivial unreachable-new-nodes strawman while preserving the chunk hole.
    """
    new = x[start:end]
    links = exact_knn(new, x[:start], a.degree, a.distance_batch)
    graph = torch.cat((old_graph.clone(), links), dim=0)
    incoming = torch.zeros((start, end - start), dtype=torch.bool, device=x.device)
    cols = torch.arange(end - start, device=x.device).repeat_interleave(a.degree)
    incoming[links.reshape(-1), cols] = True
    for lo in range(0, start, a.distance_batch):
        hi = min(lo + a.distance_batch, start)
        reverse_d = squared_l2(x[lo:hi], new).masked_fill(
            ~incoming[lo:hi], float("inf"))
        take = min(a.degree, len(new))
        vals, idx = reverse_d.topk(take, largest=False)
        reverse_ids = (idx + start).masked_fill(~vals.isfinite(), -1)
        candidates = torch.cat((graph[lo:hi], reverse_ids), dim=1)
        distances = candidate_distances(x[lo:hi], x, candidates)
        vals, pos = distances.topk(a.degree, largest=False)
        graph[lo:hi] = candidates.gather(1, pos).masked_fill(~vals.isfinite(), -1)
    return graph
