"""Fixed-budget beam traversal over directed proximity graphs."""
from __future__ import annotations

import torch
from cagra_chunksplice.core import candidate_distances
from cagra_chunksplice.core import unique_candidates


def graph_search(x, graph, queries, entries, a, k=10):
    """Fixed-budget GPU beam traversal, no exact scan during query search.

    Common entry IDs, ef, expansion width and rounds for all methods.
    A fixed-size visited ledger (steps * search_width node IDs per query)
    prevents repeated expansion. Distances are evaluated only for
    beam/neighbor candidates, never for the whole database, so working memory
    and per-round bookkeeping are independent of the database size.

    Memory scaling: the boolean ``done`` mask built each round has shape
    ``(query_batch, ef, steps * search_width)``, so it is independent of the
    database size n and grows with ``query_batch * ef * steps * width``.
    Duplicate entry IDs are masked to -1 before scoring so they occupy no
    initial beam slots.
    """
    answers = []
    cap = a.steps * a.search_width
    for lo in range(0, len(queries), a.query_batch):
        q = queries[lo:lo + a.query_batch]
        b = len(q)
        # Duplicate entry IDs would waste beam slots: mask repeats to -1.
        beam = unique_candidates(entries[None, :].expand(b, -1).clone())
        scores = candidate_distances(q, x, beam)
        order = scores.argsort(dim=1)
        beam, scores = beam.gather(1, order), scores.gather(1, order)
        visited = torch.full((b, cap), -1, dtype=torch.long, device=x.device)
        for step in range(a.steps):
            done = (beam[:, :, None] == visited[:, None, :]).any(dim=-1) | (beam < 0)
            frontier_scores, pos = scores.masked_fill(done, float("inf")).topk(
                a.search_width, largest=False)
            frontier = beam.gather(1, pos)
            valid = frontier_scores.isfinite() & (frontier >= 0)
            visited[:, step * a.search_width:(step + 1) * a.search_width] = \
                frontier.masked_fill(~valid, -1)
            neighbors = graph[frontier.clamp_min(0)].masked_fill(
                ~valid[:, :, None], -1).flatten(1)
            ids = unique_candidates(torch.cat((beam, neighbors), dim=1))
            d = candidate_distances(q, x, ids)
            scores, pos = d.topk(a.ef, largest=False, sorted=True)
            beam = ids.gather(1, pos).masked_fill(~scores.isfinite(), -1)
        answers.append(beam[:, :k])
    return torch.cat(answers)
