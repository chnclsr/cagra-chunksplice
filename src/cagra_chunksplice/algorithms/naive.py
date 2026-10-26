"""Snapshot insertion with degree-capped reverse proposals."""
from __future__ import annotations

import torch
from cagra_chunksplice.core import candidate_distances
from cagra_chunksplice.algorithms.insertion import insertion_candidates
from cagra_chunksplice.core import squared_l2


def _sparse_reverse_update(x, graph, links, start, end, a):
    """Merge each new row only into the old rows it links to.

    Only old rows named by the new links are read and rewritten; every other
    old row keeps its exact previous adjacency, including slot order. At most
    ``(end - start) * degree`` old rows are touched, so for fixed chunk size,
    degree and beam budgets the reverse work is independent of n. Merging every
    proposal and keeping the closest ``degree`` neighbors reproduces the dense
    pass on the touched rows: the dense pass first keeps the ``degree`` closest
    proposals, and any proposal it drops is farther than ``degree`` retained
    proposals, so it cannot survive the final top-``degree`` merge either.
    New-row links are drawn from the old snapshot, so proposers (< start) and
    existing old neighbors (< start) share one cohort and cannot collide.
    """
    targeted = (links >= 0) & (links < start)
    b = end - start
    flat_targets = links.masked_fill(~targeted, -1).reshape(-1)
    flat_sources = (torch.arange(b, device=x.device).repeat_interleave(a.degree)
                    + start)
    keep = flat_targets >= 0
    flat_targets, flat_sources = flat_targets[keep], flat_sources[keep]
    if flat_targets.numel() == 0:
        return graph
    order = flat_targets.argsort(stable=True)
    flat_targets, flat_sources = flat_targets[order], flat_sources[order]
    affected, counts = torch.unique(flat_targets, return_counts=True)
    begins = counts.cumsum(0) - counts
    rows = torch.repeat_interleave(
        torch.arange(len(affected), device=x.device), counts)
    cols = torch.arange(flat_targets.numel(), device=x.device) - begins[rows]
    # One scalar size query packs the ragged proposer lists into a dense pad;
    # there is no per-row Python loop and no .tolist() host round-trips.
    widest = int(counts.max().item())
    proposers = torch.full((len(affected), widest), -1,
                           dtype=torch.long, device=x.device)
    proposers[rows, cols] = flat_sources
    candidates = torch.cat((graph[affected], proposers), dim=1)
    distances = candidate_distances(x[affected], x, candidates)
    vals, pos = distances.topk(a.degree, largest=False)
    graph[affected] = candidates.gather(1, pos).masked_fill(~vals.isfinite(), -1)
    return graph


def _shares_storage(view, other):
    return (view.shape == other.shape and view.stride() == other.stride()
            and view.data_ptr() == other.data_ptr())


def naive_extend(x, old_graph, start, end, a, out=None):
    """Snapshot insertion: new->old only, with degree-capped old->new reverses.

    Reverse proposals are *only* reverses of chosen new->old edges. This avoids
    the trivial unreachable-new-nodes strawman while preserving the chunk hole.

    When ``out`` is given it must have shape ``(end, degree)`` and is used as
    the result buffer: the old snapshot is copied once into ``out[:start]``,
    the new links are written to ``out[start:end]``, and only the old rows
    named by the new links are then rewritten. Reusing one caller-owned
    buffer across updates avoids a per-update allocation. The snapshot copy
    is a single contiguous O(n * degree) memcpy, which is NOT independent of
    n; it is skipped only when ``out[:start]`` already aliases ``old_graph``
    (same storage, shape and strides). In that aliased case the old rows are
    updated in place, so ``old_graph`` is mutated and re-running the same
    update on it is not a pure function; callers that need a pristine
    snapshot (e.g. repeated timing in ``runners/scaling.py``) must not
    alias and therefore pay and time the memcpy. All other update compute
    stays independent of n (see the per-mode notes below). The default
    ``out=None`` allocates the result with ``torch.cat`` and never mutates
    ``old_graph``.

    Complexity by insertion mode: ``exact`` runs an O(n b d) new-to-old scan
    plus a dense reverse pass over all n old rows; ``beam`` replaces the scan
    with a fixed-budget graph search but keeps the dense O(n) reverse pass;
    ``bounded`` uses the same beam search and additionally restricts the
    reverse merge to the old rows named by the new links (at most b * degree
    rows), so update work is independent of n for fixed chunk, degree and beam
    budgets. Untouched old rows keep their exact previous adjacency.
    """
    new = x[start:end]
    links = insertion_candidates(x, old_graph, start, end, a)
    if out is None:
        graph = torch.cat((old_graph, links), dim=0)  # cat allocates; no clone
    else:
        if tuple(out.shape) != (end, a.degree):
            raise ValueError(
                f"out must have shape ({end}, {a.degree}), "
                f"got {tuple(out.shape)}")
        if not _shares_storage(out[:start], old_graph):
            out[:start] = old_graph  # O(n * degree) memcpy
        out[start:end] = links
        graph = out
    if getattr(a, "insertion_mode", "exact") == "bounded":
        return _sparse_reverse_update(x, graph, links, start, end, a)
    incoming = torch.zeros((start, end - start), dtype=torch.bool, device=x.device)
    cols = torch.arange(end - start, device=x.device).repeat_interleave(a.degree)
    valid = links.reshape(-1) >= 0
    incoming[links.reshape(-1)[valid], cols[valid]] = True
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
