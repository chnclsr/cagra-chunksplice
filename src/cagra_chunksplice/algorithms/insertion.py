"""Shared exact or graph-based new-to-old candidate generation."""
from __future__ import annotations

from types import SimpleNamespace
import torch

from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.search import graph_search


def insertion_candidates(x, old_graph, start, end, a):
    """Search the immutable predecessor only, without exact fallback in beam mode.

    Fixed seeded base entries keep repeated timings reproducible. Independent
    insertion budgets permit candidate errors to accumulate along the stream.
    Unreachable or unfilled slots remain -1, rather than invoking an oracle.
    """
    mode = getattr(a, "insertion_mode", "exact")
    if mode == "exact":
        return exact_knn(x[start:end], x[:start], a.degree, a.distance_batch)
    if mode != "beam":
        raise ValueError("insertion_mode must be exact or beam")
    ef = min(start, getattr(a, "insertion_ef", a.degree * 2))
    width = getattr(a, "insertion_search_width", 4)
    steps = getattr(a, "insertion_steps", 8)
    if ef < a.degree or not 1 <= width <= ef or steps < 1:
        raise ValueError("insertion requires ef >= degree, 1 <= width <= ef, steps >= 1")
    budget = SimpleNamespace(ef=ef, search_width=width, steps=steps,
                             query_batch=getattr(a, "query_batch", a.distance_batch))
    anchor_count = min(start, getattr(a, "base", start))
    gen = torch.Generator().manual_seed(getattr(a, "seed", 42))
    # Padding retains a fixed beam shape even when the base has fewer than ef nodes.
    entries = torch.full((ef,), -1, dtype=torch.long, device=x.device)
    anchors = torch.randperm(anchor_count, generator=gen)[:ef].to(x.device)
    entries[:len(anchors)] = anchors
    return graph_search(x[:start], old_graph, x[start:end], entries, budget, k=a.degree)
