"""Shared exact or graph-based new-to-old candidate generation."""
from __future__ import annotations

from types import SimpleNamespace
import torch

from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.search import graph_search


def _unique_anchors(anchor_count, ef, gen):
    """Draw ``min(ef, anchor_count)`` distinct anchor IDs on the CPU.

    The generator is a CPU generator, so sampling stays on the CPU and the
    result is moved to the target device by the caller. Work is O(ef) for
    large pools (rejection top-up of duplicate draws); small pools use a
    permutation, which is bounded by ``4 * ef``.
    """
    want = min(ef, anchor_count)
    if anchor_count <= 4 * want:
        return torch.randperm(anchor_count, generator=gen)[:want]
    picked = torch.empty(0, dtype=torch.long)
    while len(picked) < want:
        draw = torch.randint(0, anchor_count, (want - len(picked),), generator=gen)
        picked = torch.cat((picked, draw)).unique()
    # unique() sorts; shuffle so the choice stays unbiased in beam order.
    return picked[torch.randperm(len(picked), generator=gen)]


def insertion_candidates(x, old_graph, start, end, a):
    """Search the immutable predecessor only, without exact fallback in beam modes.

    Fixed seeded base entries keep repeated timings reproducible. Independent
    insertion budgets permit candidate errors to accumulate along the stream.
    Unreachable or unfilled slots remain -1, rather than invoking an oracle.
    ``bounded`` uses the same fixed-budget beam search as ``beam``; it
    additionally selects the sparse targeted reverse path in ``naive_extend``,
    so the whole update is independent of n for fixed chunk, degree and beam
    budgets. ``beam`` keeps the legacy dense reverse scan over all old rows.
    """
    mode = getattr(a, "insertion_mode", "exact")
    if mode == "exact":
        return exact_knn(x[start:end], x[:start], a.degree, a.distance_batch)
    if mode not in ("beam", "bounded"):
        raise ValueError("insertion_mode must be exact, beam or bounded")
    ef = min(start, getattr(a, "insertion_ef", a.degree * 2))
    width = getattr(a, "insertion_search_width", 4)
    steps = getattr(a, "insertion_steps", 8)
    if ef < a.degree or not 1 <= width <= ef or steps < 1:
        raise ValueError("insertion requires ef >= degree, 1 <= width <= ef, steps >= 1")
    budget = SimpleNamespace(ef=ef, search_width=width, steps=steps,
                             query_batch=getattr(a, "query_batch", a.distance_batch))
    anchor_count = min(start, getattr(a, "base", start))
    gen = torch.Generator().manual_seed(getattr(a, "seed", 42))
    entries = _unique_anchors(anchor_count, ef, gen).to(x.device)
    return graph_search(x[:start], old_graph, x[start:end], entries, budget, k=a.degree)
