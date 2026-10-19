"""Fixed-budget beam traversal over directed proximity graphs."""
from __future__ import annotations

import torch
from cagra_chunksplice.core import candidate_distances
from cagra_chunksplice.core import unique_candidates


def graph_search(x, graph, queries, entries, a, k=10):
    """Fixed-budget GPU beam traversal, no exact scan during query search.

    Common entry IDs, ef, expansion width and rounds for all methods.
    A per-query expanded bitmap prevents repeated expansion. Distances are
    evaluated only for beam/neighbor candidates, never for the whole database.
    """
    answers = []
    for lo in range(0, len(queries), a.query_batch):
        q = queries[lo:lo + a.query_batch]
        b = len(q)
        beam = entries[None, :].expand(b, -1).clone()
        scores = candidate_distances(q, x, beam)
        order = scores.argsort(dim=1)
        beam, scores = beam.gather(1, order), scores.gather(1, order)
        expanded = torch.zeros((b, len(x)), dtype=torch.bool, device=x.device)
        batch = torch.arange(b, device=x.device)[:, None]
        for _ in range(a.steps):
            done = expanded.gather(1, beam.clamp_min(0)) | (beam < 0)
            frontier_scores, pos = scores.masked_fill(done, float("inf")).topk(
                a.search_width, largest=False)
            frontier = beam.gather(1, pos)
            valid = frontier_scores.isfinite() & (frontier >= 0)
            batch_front = batch.expand_as(frontier)
            expanded[batch_front[valid], frontier[valid]] = True
            neighbors = graph[frontier.clamp_min(0)].masked_fill(
                ~valid[:, :, None], -1).flatten(1)
            ids = unique_candidates(torch.cat((beam, neighbors), dim=1))
            d = candidate_distances(q, x, ids)
            scores, pos = d.topk(a.ef, largest=False, sorted=True)
            beam = ids.gather(1, pos).masked_fill(~scores.isfinite(), -1)
        answers.append(beam[:, :k])
    return torch.cat(answers)
