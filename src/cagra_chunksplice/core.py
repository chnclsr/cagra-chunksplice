"""Blocked distance computation and candidate handling."""
from __future__ import annotations

import torch


def squared_l2(a, b):
    return (a.square().sum(-1, keepdim=True)
            + b.square().sum(-1).unsqueeze(0)
            - 2.0 * (a @ b.T)).clamp_min_(0)


def exact_knn(queries, database, k, block, self_ids=None):
    """No full NxN allocation; self_ids optionally masks exact database identity."""
    out = []
    for start in range(0, len(queries), block):
        d = squared_l2(queries[start:start + block], database)
        if self_ids is not None:
            ids = self_ids[start:start + block]
            d[torch.arange(len(d), device=d.device), ids] = float("inf")
        out.append(d.topk(k, largest=False, sorted=True).indices)
    return torch.cat(out)


def candidate_distances(centers, x, ids):
    vectors = x[ids.clamp_min(0)]
    d = (centers.square().sum(-1, keepdim=True)
         + vectors.square().sum(-1)
         - 2.0 * torch.bmm(vectors, centers.unsqueeze(-1)).squeeze(-1))
    return d.clamp_min_(0).masked_fill(ids < 0, float("inf"))


def unique_candidates(ids, self_ids=None):
    """Row-wise GPU dedup; -1 denotes an absent candidate."""
    ids = ids.sort(dim=1).values
    dup = torch.zeros_like(ids, dtype=torch.bool)
    dup[:, 1:] = ids[:, 1:] == ids[:, :-1]
    invalid = dup | (ids < 0)
    if self_ids is not None:
        invalid |= ids == self_ids[:, None]
    return ids.masked_fill(invalid, -1)
