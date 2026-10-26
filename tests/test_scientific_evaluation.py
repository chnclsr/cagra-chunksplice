"""Analytical fixtures for global baselines and failure-mode diagnostics."""
import time
from types import SimpleNamespace

import pytest
import torch

from cagra_chunksplice.algorithms.chunk_splice import chunk_splice_extend
from cagra_chunksplice.algorithms.insertion import insertion_candidates
from cagra_chunksplice.algorithms.naive import _sparse_reverse_update, naive_extend
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.data import age_query_sets, evaluation_set, generate_data
from cagra_chunksplice.graph import build_graph, build_rng_graph, validate_graph
from cagra_chunksplice.metrics import (
    EdgeDecayTracker, edge_length_metrics, hubness_metrics, scc_metrics)


def test_rng_rebuild_prunes_and_adds_reverse_proposals():
    x = torch.tensor([[0.], [1.], [3.], [10.]])
    a = SimpleNamespace(degree=2, distance_batch=1, rebuild_pool=2, rebuild_alpha=1.0)
    knn = build_graph(x, a)
    rng = build_rng_graph(x, a)
    validate_graph(rng, len(x))
    assert rng.tolist() == [[1, -1], [0, 2], [1, 3], [2, -1]]
    assert 3 not in knn[2] and 3 in rng[2]  # Reverse proposal outside forward pool.
    assert 2 in knn[0] and 2 not in rng[0]  # Lune occlusion without backfill.
    a.distance_batch = 4
    assert torch.equal(rng, build_rng_graph(x, a))
    assert scc_metrics(rng)["scc_count"] == 1
    assert build_rng_graph(x[:1], a).tolist() == [[-1, -1]]


@pytest.mark.parametrize("graph,count,largest,sinks", [
    ([[1], [2], [0]], 1, 1.0, 1),
    ([[1], [0], [3], [2], [-1]], 3, 0.4, 3),
    ([[1], [2], [-1]], 3, 1 / 3, 1),
    ([[-1], [-1]], 2, 0.5, 2),
])
def test_exact_scc_components(graph, count, largest, sinks):
    result = scc_metrics(torch.tensor(graph))
    assert result["scc_count"] == count
    assert result["largest_scc_fraction"] == pytest.approx(largest)
    assert result["sink_scc_count"] == sinks


def test_scc_long_chain_is_not_recursive():
    graph = torch.arange(1, 5001).reshape(-1, 1)
    graph[-1] = -1
    assert scc_metrics(graph)["scc_count"] == 5000
    assert scc_metrics(torch.empty((0, 2), dtype=torch.long))["largest_scc_fraction"] is None


def test_hubness_analytic_values_and_degenerate_graphs():
    regular = hubness_metrics(torch.tensor([[1], [2], [0]]))
    assert regular["in_degree_gini"] == regular["in_degree_skewness"] == 0
    # Directed star: in-degrees [3, 1, 0, 0], Gini = 0.625.
    star = hubness_metrics(torch.tensor([[1], [0], [0], [0]]))
    assert star["in_degree_gini"] == pytest.approx(0.625)
    assert star["in_degree_skewness"] > 0
    assert star["max_in_degree"] == 3
    empty_edges = hubness_metrics(torch.full((4, 2), -1))
    assert empty_edges["in_degree_gini"] == empty_edges["in_degree_skewness"] == 0
    assert hubness_metrics(torch.empty((0, 1), dtype=torch.long))["in_degree_gini"] is None


def test_bridge_lengths_and_checkpoint_decay():
    x = torch.tensor([[0.], [1.], [3.], [7.]])
    graph = torch.tensor([[1, -1], [0, 2], [1, 3], [2, -1]])
    metrics = edge_length_metrics(graph, x, base=2, chunk_size=1, block=1)
    assert metrics["bridge_edges"] == 4
    assert metrics["bridge_fraction"] == pytest.approx(2 / 3)
    assert metrics["bridge_slot_occupancy"] == 0.5
    assert metrics["mean_edge_length"] == pytest.approx(14 / 6)
    assert metrics["mean_bridge_length"] == 3
    tracker = EdgeDecayTracker(2, 1)
    tracker.observe(graph[:2].masked_fill(graph[:2] >= 2, -1), x[:2])
    first = tracker.observe(graph, x)
    assert first["initial_bridge_survival"] == 1
    changed = graph.clone()
    changed[2, 1] = -1
    second = tracker.observe(changed, x)
    assert second["initial_edge_survival"] == 1
    assert second["initial_bridge_survival"] == 0.75
    assert second["previous_checkpoint_edge_survival"] == pytest.approx(5 / 6)
    assert second["mean_bridge_length"] == pytest.approx(8 / 3)
    assert second["initial_bridge_reference_n"] == 4


def insertion_args(mode):
    return SimpleNamespace(degree=2, distance_batch=4, insertion_mode=mode,
                           insertion_ef=2, insertion_search_width=1, insertion_steps=1,
                           query_batch=2, seed=7, base=1,
                           splice_intra=1, splice_pool=2, splice_alpha=1.0)


def test_beam_insertion_has_no_oracle_fallback_or_invalid_reverse_edges():
    x = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
    previous = torch.full((6, 2), -1, dtype=torch.long)
    a = insertion_args("beam")
    links = insertion_candidates(x, previous, 6, 8, a)
    assert links.tolist() == [[0, -1], [0, -1]]
    assert not torch.equal(links, exact_knn(x[6:], x[:6], 2, 4))
    naive = naive_extend(x, previous, 6, 8, a)
    spliced = chunk_splice_extend(x, previous, 6, 8, a)
    validate_graph(naive, 8)
    validate_graph(spliced, 8)
    assert naive[5].tolist() == [-1, -1]  # -1 proposals must not target the last old row.
    assert torch.equal(spliced[:6], naive[:6])
    assert bool((previous == -1).all())
    assert spliced[6].tolist() == [0, 7] or spliced[6].tolist() == [7, 0]
    assert torch.equal(naive, naive_extend(x, previous, 6, 8, a))


def test_bounded_insertion_uses_beam_candidates_and_touches_only_linked_rows():
    x = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
    previous = torch.full((6, 2), -1, dtype=torch.long)
    # Unsorted by distance from row 3, with no incoming proposals below.
    previous[3] = torch.tensor([0, 5])
    a = insertion_args("bounded")
    links = insertion_candidates(x, previous, 6, 8, a)
    assert links.tolist() == [[0, -1], [0, -1]]
    assert torch.equal(links, insertion_candidates(x, previous, 6, 8,
                                                   insertion_args("beam")))
    dense = naive_extend(x, previous, 6, 8, insertion_args("beam"))
    bounded = naive_extend(x, previous, 6, 8, a)
    validate_graph(bounded, 8)
    assert torch.equal(bounded, naive_extend(x, previous, 6, 8, a))
    assert torch.equal(bounded[6:], dense[6:])
    assert torch.equal(bounded[0], dense[0])  # the only linked old row matches
    assert bounded[3].tolist() == [0, 5]  # untouched rows keep exact adjacency
    assert dense[3].tolist() == [5, 0]  # the dense pass re-sorts every old row
    touched = [row for row in range(6)
               if bounded[row].tolist() != previous[row].tolist()]
    assert set(touched) <= {0}
    assert len(touched) <= (8 - 6) * a.degree
    spliced = chunk_splice_extend(x, previous, 6, 8, a)
    validate_graph(spliced, 8)
    assert torch.equal(spliced[:6], bounded[:6])
    assert spliced[6].tolist() == [0, 7] or spliced[6].tolist() == [7, 0]


@pytest.mark.parametrize("mode", ["exact", "beam", "bounded"])
def test_insertion_modes_across_multiple_epochs(mode):
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(18, 4, generator=gen)
    a = insertion_args(mode)
    a.base, a.insertion_ef = 8, 4
    graphs = [build_graph(x[:8], a), build_graph(x[:8], a)]
    for start, end in ((8, 12), (12, 13), (13, 18)):
        for index, extend in enumerate((naive_extend, chunk_splice_extend)):
            predecessor = graphs[index]
            old = predecessor.clone()
            graphs[index] = extend(x[:end], predecessor, start, end, a)
            validate_graph(graphs[index], end)
            assert torch.equal(old, predecessor)


@pytest.mark.parametrize("distribution", ["in-distribution", "ood", "angular"])
def test_high_lid_queries_and_age_strata_are_seeded(distribution):
    a = SimpleNamespace(seed=17, base=12, dim=64, chunks=6, chunk_size=2,
                        degree=2, workload="high-lid", queries=6, age_queries=3,
                        query_distribution=distribution, ood_shift=10.0)
    x, centers, gen = generate_data(a, torch.device("cpu"))
    again, repeated, _ = generate_data(a, torch.device("cpu"))
    assert torch.equal(x, again) and torch.equal(centers, repeated)
    queries, latest, _ = evaluation_set(x, centers, 12, 2, a, gen, x.device)
    assert queries.shape == (6, 64) and latest.shape == (3, 64)
    strata = age_query_sets(x, 12, a)
    assert [source for _, source in strata.values()] == [[0, 12], [12, 16], [16, 20], [20, 24]]
    if distribution == "angular":
        assert torch.allclose(x.norm(dim=-1), torch.ones(len(x)), atol=1e-6)
        assert torch.allclose(queries.norm(dim=-1), torch.ones(6), atol=1e-6)
        cosine = (queries @ x.T).topk(4).indices
        assert torch.equal(exact_knn(queries, x, 4, 3), cosine)
    elif distribution == "ood":
        assert queries[:, 0].mean() > 7


def test_bounded_sparse_reverse_merge_matches_dense_under_oversubscription():
    degree = 2
    x = torch.tensor([[0.], [10.], [20.], [30.],
                      [0.1], [0.2], [0.3], [0.4]])
    old_graph = torch.tensor([[1, 2], [0, 3], [0, 3], [1, 2]])
    start, end = 4, 8
    # Every new row proposes old row 0: four proposers for two slots, plus
    # four proposals colliding on row 1 while rows 2 and 3 see nothing.
    links = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    a = SimpleNamespace(degree=degree)
    graph = torch.cat((old_graph.clone(), links), dim=0)
    result = _sparse_reverse_update(x, graph, links, start, end, a)
    assert result is graph
    # Brute-force dense reference over the same proposals.
    expected = torch.cat((old_graph.clone(), links), dim=0)
    incoming = [[] for _ in range(start)]
    for source, row in enumerate(links.tolist()):
        for target in row:
            if 0 <= target < start:
                incoming[target].append(source + start)
    assert len(incoming[0]) == 4 > degree
    assert len(incoming[1]) == 4 > degree
    for row_id in range(start):
        if not incoming[row_id]:
            continue
        pool = [c for c in expected[row_id].tolist() + incoming[row_id]
                if c >= 0]
        pool.sort(key=lambda c: float(((x[row_id] - x[c]) ** 2).sum()))
        expected[row_id] = torch.tensor((pool[:degree] + [-1] * degree)[:degree])
    assert torch.equal(result, expected)
    assert result[0].tolist() == [4, 5]  # the two closest new rows win
    assert torch.equal(result[2], old_graph[2])
    assert torch.equal(result[3], old_graph[3])


def test_bounded_matches_dense_when_proposals_collide_on_one_row():
    beam_args = insertion_args("beam")
    bounded_args = insertion_args("bounded")
    x = torch.cat((torch.arange(6, dtype=torch.float32).reshape(-1, 1),
                   torch.tensor([[0.1], [0.2], [0.3]])))
    previous = torch.full((6, 2), -1, dtype=torch.long)
    links = insertion_candidates(x, previous, 6, 9, bounded_args)
    assert links.tolist() == [[0, -1], [0, -1], [0, -1]]
    assert torch.equal(links, insertion_candidates(x, previous, 6, 9, beam_args))
    targets = [t for row in links.tolist() for t in row if t >= 0]
    assert targets.count(0) == 3 > bounded_args.degree
    dense = naive_extend(x, previous, 6, 9, beam_args)
    bounded = naive_extend(x, previous, 6, 9, bounded_args)
    validate_graph(bounded, 9)
    assert torch.equal(bounded[6:], dense[6:])
    assert torch.equal(bounded[0], dense[0])
    assert bounded[0].tolist() == [6, 7]
    for row_id in range(1, 6):
        assert torch.equal(bounded[row_id], previous[row_id])


def _out_fixture(mode):
    x = torch.arange(9, dtype=torch.float32).reshape(-1, 1)
    previous = torch.full((6, 2), -1, dtype=torch.long)
    return x, previous, insertion_args(mode)


@pytest.mark.parametrize("mode", ["exact", "beam", "bounded"])
def test_out_buffer_matches_default_allocation(mode):
    x, previous, a = _out_fixture(mode)
    snapshot = previous.clone()
    expected = naive_extend(x, previous, 6, 9, a)
    out = torch.empty((9, 2), dtype=torch.long)
    result = naive_extend(x, previous, 6, 9, a, out=out)
    assert result is out
    assert torch.equal(result, expected)
    assert torch.equal(previous, snapshot)  # snapshot never mutated


def test_out_aliasing_old_graph_skips_copy():
    x, previous, a = _out_fixture("bounded")
    expected = naive_extend(x, previous, 6, 9, a)
    out = torch.empty((9, 2), dtype=torch.long)
    out[:6] = previous
    naive_extend(x, out[:6], 6, 9, a, out=out)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("shape", [(8, 2), (9, 3), (10, 2)])
def test_out_shape_mismatch_raises(shape):
    x, previous, a = _out_fixture("bounded")
    with pytest.raises(ValueError, match="out must have shape"):
        naive_extend(x, previous, 6, 9, a, out=torch.empty(shape, dtype=torch.long))


def test_insertion_anchors_are_unique():
    from cagra_chunksplice.algorithms.insertion import _unique_anchors
    for count, ef in ((1000, 16), (20, 16), (5, 16)):
        ids = _unique_anchors(count, ef, torch.Generator().manual_seed(3))
        assert len(ids) == len(set(ids.tolist())) == min(ef, count)
        assert ids.max() < count


def test_graph_search_masks_duplicate_entries():
    from cagra_chunksplice.search import graph_search
    x = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
    graph = torch.tensor([[1], [0], [1], [2], [3], [4]])
    a = SimpleNamespace(ef=4, search_width=1, steps=2, query_batch=2)
    found = graph_search(x, graph, x[:2], torch.tensor([0, 0, 0, 0]), a, k=2)
    assert found[:, 0].tolist() == [0, 1]
    for row in found.tolist():
        valid = [i for i in row if i >= 0]
        assert len(valid) == len(set(valid))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ["beam", "bounded"])
def test_cuda_smoke_insertion(mode):
    x, previous, a = _out_fixture(mode)
    cpu = naive_extend(x, previous, 6, 9, a)
    gpu = naive_extend(x.cuda(), previous.cuda(), 6, 9, a,
                       out=torch.empty((9, 2), dtype=torch.long, device="cuda"))
    assert torch.equal(gpu.cpu(), cpu)


def test_bounded_update_time_does_not_grow_with_base_size():
    device = torch.device("cpu")
    gen = torch.Generator().manual_seed(1234)
    degree, chunk, dim = 8, 16, 8
    fastest = {}
    for n in (2048, 65536):
        a = SimpleNamespace(degree=degree, distance_batch=512,
                            insertion_mode="bounded", insertion_ef=16,
                            insertion_search_width=4, insertion_steps=4,
                            query_batch=16, seed=7, base=256)
        x = torch.randn(n + chunk, dim, generator=gen, device=device)
        old = ((torch.arange(n, device=device)[:, None]
                + torch.arange(1, degree + 1, device=device)[None, :]) % n)
        start, end = n, n + chunk

        def update():
            return naive_extend(x, old, start, end, a)

        graph = update()  # warmup outside the timed region
        samples = []
        for _ in range(5):
            began = time.perf_counter()
            graph = update()
            samples.append((time.perf_counter() - began) * 1000.0)
        validate_graph(graph, end)
        fastest[n] = min(samples)
    # A 32x larger base with an O(n) update would take ~32x longer; the
    # bounded path must stay far below linear growth.
    assert fastest[65536] < fastest[2048] * 8
