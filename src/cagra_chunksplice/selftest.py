"""Small mechanism checks also available in installed CLI builds."""
from __future__ import annotations

import torch
from cagra_chunksplice.graph import build_graph
from cagra_chunksplice.algorithms.chunk_splice import chunk_splice_extend
from cagra_chunksplice.metrics import connectivity_metrics
from cagra_chunksplice.data import evaluation_set
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.search import graph_search
from cagra_chunksplice.metrics import intra_edges
from cagra_chunksplice.algorithms.local_rng import local_refine
from cagra_chunksplice.algorithms.naive import naive_extend
from cagra_chunksplice.metrics import recall
from cagra_chunksplice.algorithms.pruning import rng_prune
from cagra_chunksplice.graph import validate_graph


def self_test(device):
    """Focused checks: identity masking, RNG lune, reverse reachability, search."""
    from types import SimpleNamespace
    a = SimpleNamespace(degree=2, distance_batch=4, candidate_cap=6, hops=2,
                        local_cap=10, refine_rounds=1, query_batch=4,
                        ef=4, search_width=2, steps=6, alpha=1.0, route=0,
                        cohort_protect=False, backfill=False,
                        splice_intra=1, splice_pool=4, splice_alpha=1.0)
    x = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [0., 1.],
                      [0.10, 0.02], [0.12, 0.03]], device=device)
    rows = torch.tensor([0], device=device)
    candidates = torch.tensor([[1, 2, 3, -1, 1]], device=device)
    r = rng_prune(x, rows, candidates, 3)
    assert 1 in r[0] and 3 in r[0] and 2 not in r[0], "RNG lune failure"
    assert r[0].tolist() == [1, 3, -1], "legacy RNG must keep -1 padding"
    # alpha=2: 4*d2(1,2)=4 is not < d2(0,2)=4, so 1 no longer occludes 2.
    r = rng_prune(x, rows, candidates, 3, alpha=2.0)
    assert sorted(r[0].tolist()) == [1, 2, 3], "alpha must relax occlusion"
    # Row 0 sees new 4 first; plain RNG lets 4 occlude 5 and old 1, 2.
    mixed_ids = torch.tensor([[4, 5, 1, 2]], device=device)
    assert rng_prune(x, rows, mixed_ids, 3).tolist() == [[4, -1, -1]]
    assert rng_prune(x, rows, mixed_ids, 3, backfill=True).tolist() == [[4, 5, 1]], \
        "backfill must fill empty slots by distance"
    # Cohort-local: new 4 cannot occlude old 1; old 1 still occludes old 2.
    r = rng_prune(x, rows, mixed_ids, 3, cohort_split=4)
    assert r.tolist() == [[4, 1, -1]], "cohort-local occlusion failure"
    r = rng_prune(x, rows, mixed_ids, 3, route=2, cohort_split=4)
    assert r.tolist() == [[4, 1, 2]], "route must backfill rejected old edges"
    r = rng_prune(x, rows, mixed_ids, 2, route=2, cohort_split=4)
    assert r.tolist() == [[1, 2]], "route slots must outrank new survivors"
    try:
        rng_prune(x, rows, mixed_ids, 2, route=1)
        raise AssertionError("route without cohort_split must fail")
    except ValueError:
        pass
    initial = build_graph(x[:4], a)
    g = naive_extend(x, initial, 4, 6, a)
    validate_graph(g, len(x))
    assert intra_edges(g, 4, 2) == 0
    assert bool((g[:4] >= 4).any()), "new nodes must have incoming old edges"
    before = g.clone()
    fixed, info = local_refine(x, g, 4, 6, a)
    validate_graph(fixed, len(x))
    assert intra_edges(fixed, 4, 2) > 0, "repair must restore chunk peers"
    assert not torch.equal(before, fixed) and info["local_vertices"] <= a.local_cap
    # local_refine must forward the LocalRNG knobs into rng_prune.
    knobs = SimpleNamespace(**{**vars(a), "alpha": 1.2, "route": 1,
                               "cohort_protect": True, "backfill": True})
    tuned, info = local_refine(x, before.clone(), 4, 6, knobs)
    validate_graph(tuned, len(x))
    assert (info["rng_alpha"], info["rng_route"], info["cohort_protect"],
            info["rng_backfill"]) == (1.2, 1, True, True)
    assert bool((tuned[4:6] >= 0).all()), "backfill must leave no empty slots"
    assert bool((tuned[4:6] < 4).any(dim=1).all()), "route must keep an old edge"
    spliced = chunk_splice_extend(x, initial, 4, 6, a)
    validate_graph(spliced, len(x))
    assert torch.equal(spliced[:4], before[:4]), "splice must not touch old rows"
    assert 5 in spliced[4] and 4 in spliced[5], "splice must link chunk peers"
    bridges = ((spliced[4:6] >= 0) & (spliced[4:6] < 4)).sum(dim=1)
    assert bool((bridges >= a.degree - a.splice_intra).all()), "splice must keep bridges"
    single = chunk_splice_extend(x[:5], initial, 4, 5, a)
    assert torch.equal(single, naive_extend(x[:5], initial, 4, 5, a))
    all_ids = torch.arange(len(x), device=device).expand(len(x), -1)
    complete = all_ids[all_ids != torch.arange(len(x), device=device)[:, None]]
    complete = complete.reshape(len(x), len(x) - 1)
    q = torch.tensor([[0.11, 0.025], [1.8, 0.1]], device=device)
    truth = exact_knn(q, x, 2, 2)
    pred = graph_search(x, complete, q, torch.arange(4, device=device), a, k=2)
    assert recall(pred, truth) == 1.0, "complete graph traversal must be exact"
    own = exact_knn(x, x, 2, 2, torch.arange(len(x), device=device))
    assert not bool((own == torch.arange(len(x), device=device)[:, None]).any())
    disconnected = torch.tensor([[1, -1], [0, -1], [3, -1],
                                 [2, -1], [0, -1]], device=device)
    connectivity = connectivity_metrics(
        disconnected, torch.tensor([0], device=device), base=2, latest_start=4)
    assert connectivity["reachable_nodes"] == 2
    assert connectivity["unreachable_nodes"] == 3
    assert connectivity["orphan_nodes"] == 1
    assert connectivity["latest_chunk_orphan_nodes"] == 1
    assert connectivity["inserted_reachable_fraction"] == 0.0
    # A disconnected cycle (2 <-> 3) is unreachable without being orphaned.
    a.base, a.queries, a.dim, a.workload = 4, 5, 2, "iid"
    centers = torch.tensor([[1000., 1000.], [2000., 2000.]])
    mixed, latest, old_count = evaluation_set(
        x.cpu(), centers, 2, 1, a, torch.Generator().manual_seed(7), device)
    assert len(mixed) == 5 and old_count == 2 and len(latest) == 3
    assert bool((latest > 1900).all()), "latest probes must use only newest centers"
    print("SELF-TEST PASS: kNN/self-mask, RNG, reverse edges, chunk hole, repair, splice, search, connectivity, cohorts",
          flush=True)
