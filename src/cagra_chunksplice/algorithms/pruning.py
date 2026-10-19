"""Directed greedy relative-neighborhood candidate pruning."""
from __future__ import annotations

import torch
from cagra_chunksplice.core import candidate_distances
from cagra_chunksplice.core import unique_candidates


def rng_prune(x, rows, candidates, degree, alpha=1.0, route=0, cohort_split=None,
              backfill=False):
    """Greedy relative-neighborhood heuristic, vectorized over source rows.

    Process q by increasing d(p,q). Reject q if an accepted r satisfies
    max(d(p,r), alpha*d(r,q)) < d(p,q), with unsquared L2 d. Squared distances
    use alpha^2 and preserve this strict inequality; alpha=1 is plain greedy RNG,
    alpha>1 (Vamana-style) occludes less and keeps longer edges.
    This is a candidate-restricted, directed greedy RNG, not the exact global RNG.

    cohort_split: ids < split are the old cohort, ids >= split the new cohort.
    If set, occlusion is cohort-local: r may reject q only if both share a cohort,
    so a dense new chunk cannot prune old routing edges (and vice versa).
    route: reserve up to this many slots for old-cohort candidates, preferring
    RNG survivors, then rejected old candidates by distance. Requires cohort_split.
    backfill: fill remaining empty slots with rejected candidates by distance.
    Defaults (alpha=1, route=0, no split, no backfill) give the legacy result:
    rejected edges stay rejected; sparse rows use -1 padding.
    """
    if route > 0 and cohort_split is None:
        raise ValueError("route reservation requires cohort_split")
    candidates = unique_candidates(candidates, rows)
    distances = candidate_distances(x[rows], x, candidates)
    # Stable: equal distances keep ascending-id order on every device.
    distances, order = distances.sort(dim=1, stable=True)
    candidates = candidates.gather(1, order)
    vectors = x[candidates.clamp_min(0)]
    norm = vectors.square().sum(-1)
    between = (norm[:, :, None] + norm[:, None, :]
               - 2.0 * torch.bmm(vectors, vectors.transpose(1, 2))).clamp_min_(0)
    between = between * (alpha * alpha)
    valid = distances.isfinite()
    if cohort_split is not None:
        newer = candidates >= cohort_split
        same_cohort = newer[:, :, None] == newer[:, None, :]
    blocked = ~valid
    width = candidates.shape[1]
    kept = torch.zeros_like(valid)
    # Uncapped pass: the first `degree` survivors equal those of a capped pass,
    # because occlusion of q depends only on survivors closer than q.
    for j in range(width):
        accept = ~blocked[:, j]
        kept[:, j] = accept
        # Both comparisons matter for equal-distance points.
        occludes = ((between[:, j, :] < distances)
                    & (distances[:, j, None] < distances))
        if cohort_split is not None:
            occludes &= same_cohort[:, j, :]
        blocked |= accept[:, None] & occludes
    # Slot priority: reserved route edges, then RNG survivors, then backfill.
    pos = torch.arange(width, device=x.device).expand_as(candidates)
    absent = 3 * width
    key = torch.full_like(pos, absent)
    if backfill:
        key = torch.where(valid, 2 * width + pos, key)
    key = torch.where(kept, width + pos, key)
    if route > 0:
        old_valid = valid & (candidates < cohort_split)
        old_key = torch.where(old_valid, (~kept).long() * width + pos,
                              torch.full_like(pos, absent))
        old_rank = old_key.argsort(dim=1).argsort(dim=1)
        key = torch.where(old_valid & (old_rank < route), pos, key)
    take = min(degree, width)
    vals, chosen = key.topk(take, largest=False)
    # Emit chosen edges in increasing distance; absent slots sort last.
    chosen = chosen.masked_fill(vals >= absent, width).sort(dim=1).values
    picked = candidates.gather(1, chosen.clamp_max(width - 1)).masked_fill(
        chosen >= width, -1)
    result = torch.full((len(rows), degree), -1, dtype=torch.long, device=x.device)
    result[:, :take] = picked
    return result
