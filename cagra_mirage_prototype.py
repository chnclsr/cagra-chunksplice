#!/usr/bin/env python3
"""CAGRA chunk-hole simulation and MIRAGE-inspired local GPU graph repair.

Requires: Python >=3.10, PyTorch with CUDA support.
Run:
    python cagra_mirage_prototype.py
    python cagra_mirage_prototype.py --self-test-only
    python cagra_mirage_prototype.py --workload iid --seed 17
    python cagra_mirage_prototype.py --device cpu --base 512 --chunks 2 --chunk-size 64
    python cagra_mirage_prototype.py --verify-effect
    python cagra_mirage_prototype.py --mode sweep --sweep-total 4096 --sweep-chunk-sizes 64 256 1024 2048
    python cagra_mirage_prototype.py --mode stream --chunks 50 --chunk-size 64 --stream-eval-every 10

This is a PyTorch mechanism experiment, NOT an implementation/benchmark of cuVS
CAGRA or the full hierarchical MIRAGE index. Exact blocked kNN builds and exact
old-neighbor insertion isolate topology from approximate insertion-search error.
MIRAGE inspiration: neighbor exchange, reverse edges and greedy RNG pruning.
The bounded GPU local-subgraph adaptation here is our prototype design.

Sources:
https://docs.nvidia.com/cuvs/api-reference/c-api-neighbors-cagra
https://github.com/dsg-uwaterloo/mirage
https://uwspace.uwaterloo.ca/items/1bbb8dde-8b3f-4b8c-92fa-5bb25266895e

L2 squared distances, float32, bounded degree, -1 padding. All graph geometry,
candidate exchange, pruning and query traversal run on the selected torch device.
Python launches operations; these are not optimized fused CUDA kernels.
Construction has O(N^2 D) work with blocked O(distance_batch*N) scratch.
Local repair has O(L^2 D) work per round, L <= local_cap, not a global rebuild.
Search uses an O(query_batch*N) expanded bitmap, not global exact query scoring.
Small-problem timings can favor rebuilding; speedups are measured, never assumed.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path

torch = None


def arguments():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="CPU is an explicit debug option; no silent CUDA fallback.")
    p.add_argument("--base", type=int, default=2048)
    p.add_argument("--chunks", type=int, default=4,
                   help="Number of updates in benchmark/stream mode.")
    p.add_argument("--chunk-size", type=int, default=256,
                   help="Update size; in sweep, fixed data-generation block size.")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--queries", type=int, default=256)
    p.add_argument("--degree", type=int, default=24)
    p.add_argument("--ef", type=int, default=64)
    p.add_argument("--search-width", type=int, default=4)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--query-batch", type=int, default=64)
    p.add_argument("--distance-batch", type=int, default=256)
    p.add_argument("--local-cap", type=int, default=768)
    p.add_argument("--candidate-cap", type=int, default=64)
    p.add_argument("--hops", type=int, choices=(1, 2), default=2)
    p.add_argument("--refine-rounds", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--entry-policy", choices=("all", "old"), default="all",
                   help="Common random starts over all active nodes; old is a stable-anchor control.")
    p.add_argument("--workload", choices=("drift", "iid"), default="drift",
                   help="drift: temporal dense microclusters; iid: Gaussian control.")
    p.add_argument("--mode", choices=("benchmark", "sweep", "stream"), default="benchmark",
                   help="benchmark: standard run; sweep: chunk-size sweep; stream: long-horizon streaming.")
    p.add_argument("--sweep-chunk-sizes", type=int, nargs="+",
                   default=[64, 256, 1024, 2048],
                   help="Sweep partitions of one shared corpus, in nodes per update.")
    p.add_argument("--sweep-total", type=int, default=4096,
                   help="Fixed inserted-node count in sweep (replaces --chunks).")
    p.add_argument("--stream-eval-every", type=int, default=10,
                   help="Evaluate stream at first, last and every Nth update.")
    p.add_argument("--self-test-only", action="store_true")
    p.add_argument("--verify-effect", action="store_true",
                   help="Fail unless final recall erosion and recovery each exceed 0.02.")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).with_name("cagra_mirage_results.json"))
    a = p.parse_args()
    positive = ("base", "chunks", "chunk_size", "dim", "queries", "degree", "ef",
                "search_width", "steps", "query_batch", "distance_batch", "local_cap",
                "candidate_cap", "refine_rounds", "repeats", "sweep_total",
                "stream_eval_every")
    for key in positive:
        if getattr(a, key) <= 0:
            p.error("--" + key.replace("_", "-") + " must be positive")
    if not 10 <= a.degree < a.base:
        p.error("require 10 <= degree < base for Recall@10")
    if not 10 <= a.ef <= a.base or a.search_width > a.ef:
        p.error("require 10 <= ef <= base and search-width <= ef")
    if a.queries < 2 or a.dim < 2:
        p.error("queries and dim must be >= 2")
    if a.mode == "benchmark" and a.local_cap <= a.chunk_size:
        p.error("local-cap must exceed chunk-size to include old neighbors")
    if any(b <= 0 for b in a.sweep_chunk_sizes):
        p.error("sweep-chunk-sizes must be positive")
    if a.mode == "sweep" and max(a.sweep_chunk_sizes) > a.sweep_total:
        p.error("sweep-total must be >= every sweep chunk size")
    if a.verify_effect and a.mode != "benchmark":
        p.error("verify-effect is supported only in benchmark mode")
    if a.candidate_cap < a.degree:
        p.error("candidate-cap must be >= degree")
    return a


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


def build_graph(x, a):
    return exact_knn(x, x, a.degree, a.distance_batch,
                     torch.arange(len(x), device=x.device))


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


def rng_prune(x, rows, candidates, degree):
    """Greedy relative-neighborhood heuristic, vectorized over source rows.

    Process q by increasing d(p,q). Reject q if an accepted r satisfies
    max(d(p,r), d(r,q)) < d(p,q). Squared L2 preserves this strict inequality.
    This is a candidate-restricted, directed greedy RNG, not the exact global RNG.
    No kNN backfill: rejected edges stay rejected; sparse rows use -1 padding.
    """
    candidates = unique_candidates(candidates, rows)
    distances = candidate_distances(x[rows], x, candidates)
    distances, order = distances.sort(dim=1)
    candidates = candidates.gather(1, order)
    vectors = x[candidates.clamp_min(0)]
    norm = vectors.square().sum(-1)
    between = (norm[:, :, None] + norm[:, None, :]
               - 2.0 * torch.bmm(vectors, vectors.transpose(1, 2))).clamp_min_(0)
    blocked = ~distances.isfinite()
    count = torch.zeros(len(rows), dtype=torch.long, device=x.device)
    result = torch.full((len(rows), degree), -1, dtype=torch.long, device=x.device)
    row_index = torch.arange(len(rows), device=x.device)
    for j in range(candidates.shape[1]):
        accept = ~blocked[:, j] & (count < degree)
        result[row_index[accept], count[accept]] = candidates[accept, j]
        # Both comparisons matter for equal-distance points.
        occludes = ((between[:, j, :] < distances)
                    & (distances[:, j, None] < distances))
        blocked |= accept[:, None] & occludes
        count += accept.long()
    return result


def local_refine(x, graph, start, end, a):
    """Repair one chunk plus its bounded 1/2-hop old neighborhood on device.

    Select old vertices only from graph hops; if the neighborhood exceeds the
    cap, retain those geometrically closest to this chunk. All new rows survive
    this cap. Existing outgoing neighbors remain candidates even outside the
    local subgraph, preserving opportunities for boundary connectivity.
    """
    new_ids = torch.arange(start, end, device=x.device)
    frontier = new_ids
    parts = []
    for _ in range(a.hops):
        edges = graph[frontier].reshape(-1)
        frontier = torch.unique(edges[edges >= 0])
        parts.append(frontier)
    old = torch.unique(torch.cat(parts))
    old = old[old < start]
    old_limit = a.local_cap - len(new_ids)
    if len(old) > old_limit:
        nearest = torch.full((len(old),), float("inf"), device=x.device)
        for lo in range(0, len(new_ids), a.distance_batch):
            nearest = torch.minimum(nearest, squared_l2(
                x[new_ids[lo:lo + a.distance_batch]], x[old]).amin(dim=0))
        old = old[nearest.topk(old_limit, largest=False).indices]
    local = torch.cat((new_ids, old))
    take = min(a.candidate_cap, len(local) - 1)
    # Local GEMM also makes chunk peers available; no global kNN is run here.
    local_d = squared_l2(x[local], x[local])
    local_d.fill_diagonal_(float("inf"))
    local_neighbors = local[local_d.topk(take, largest=False).indices]
    for _ in range(a.refine_rounds):
        # Snapshot before parallel row updates: no order-dependent mutation.
        snapshot = graph.clone()
        for lo in range(0, len(local), a.distance_batch):
            hi = min(lo + a.distance_batch, len(local))
            rows = local[lo:hi]
            # Explicit neighbor exchange inside the bounded subgraph:
            # local nearest candidates plus their current graph neighbors.
            seeds = local_neighbors[lo:hi, :min(4, take)]
            exchange = snapshot[seeds].flatten(1)
            ids = torch.cat((snapshot[rows], local_neighbors[lo:hi], exchange), dim=1)
            # Bound RNG work, while keeping candidates selected from all sources.
            ids = unique_candidates(ids, rows)
            d = candidate_distances(x[rows], x, ids)
            cap = min(a.candidate_cap, ids.shape[1])
            vals, order = d.topk(cap, largest=False)
            ids = ids.gather(1, order).masked_fill(~vals.isfinite(), -1)
            graph[rows] = rng_prune(x, rows, ids, a.degree)
    return graph, {"local_vertices": len(local), "local_old_vertices": len(old),
                   "modified_row_limit": a.local_cap}


def refined_extend(x, old_graph, start, end, a):
    graph = naive_extend(x, old_graph, start, end, a)
    return local_refine(x, graph, start, end, a)


def chunk_splice_extend(x, old_graph, start, end, a):
    """Ablation: naive insertion plus exact intra-chunk kNN spliced into new rows.

    Old rows are identical to naive_extend; only new->new edges are added, with
    no RNG pruning and no old-neighborhood repair.
    """
    graph = naive_extend(x, old_graph, start, end, a)
    b = end - start
    if b <= 1:
        return graph
    new = x[start:end]
    new_ids = torch.arange(start, end, device=x.device)
    intra = exact_knn(new, new, min(a.degree, b - 1), a.distance_batch,
                      torch.arange(b, device=x.device)) + start
    curr_links = graph[start:end]
    candidates = unique_candidates(torch.cat([curr_links, intra], dim=1), new_ids)
    distances = candidate_distances(new, x, candidates)
    vals, pos = distances.topk(a.degree, largest=False)
    graph[start:end] = candidates.gather(1, pos).masked_fill(~vals.isfinite(), -1)
    return graph


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


def recall(predicted, truth):
    return float((predicted[:, :, None] == truth[:, None, :]).any(-1)
                 .float().mean().item())


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measured(fn, device, repeats):
    """Median synchronized wall time includes Python launch/allocation overhead."""
    # Same-shape warmup also applies to sweep/stream; fn must not mutate inputs.
    fn()
    samples = []
    result = None
    for _ in range(repeats):
        result = None
        synchronize(device)
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, statistics.median(samples), samples


def sample_summary(samples):
    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples)
    return {"min_ms": min(samples), "max_ms": max(samples),
            "stdev_ms": stdev, "coefficient_of_variation": stdev / mean if mean else 0.0}


def experiment_report(a, device, experiment, methods):
    torch.manual_seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "schema_version": 2, "status": "ok", "experiment": experiment,
        "implementation": "PyTorch CAGRA-hole simulation / MIRAGE-inspired",
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": sys.version, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "dtype": "float32", "distance": "squared_l2",
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "parameters": {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(a).items()},
        "methods": methods,
        "timing": {
            "statistic": "median wall milliseconds", "repeats": a.repeats,
            "warmup": "one full untimed call of each measured operation at its current shape",
            "cuda_synchronization": "selected device before and after each timed call",
            "includes": "Python launches, allocations, update/reverse/repair work",
            "excludes": "data generation/transfers, exact truth, graph validation/diagnostics",
            "qps": "mixed-cohort batched throughput; latest-chunk probes timed separately",
            "ordering": "method order rotated by stage or sweep configuration",
            "repeated_updates": "each repetition starts from the same immutable graph snapshot",
        },
        "query_cohorts": {
            "old": "noisy original base vectors",
            "inserted": "all inserted vectors/centers so far; new_query_recall10 is this cohort",
            "latest_chunk": "independent probes of newest chunk; separate from cumulative inserted",
            "old_count": a.queries // 2, "inserted_count": a.queries - a.queries // 2,
            "latest_chunk_count": a.queries - a.queries // 2,
            "drift_noise_std": 0.08, "iid_noise_std": 0.10,
            "comparison": "same-stage methods share queries, truth and search entries",
            "stage_zero": "base-only queries use a different distribution from later stages",
        },
        "scope": "Mechanism simulation, not cuVS CAGRA or full MIRAGE; "
                 "no universal speed/recall guarantee. Exact-kNN ties may vary by device.",
    }


def finish_report(report, device):
    synchronize(device)
    report["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["peak_cuda_allocated_mb"] = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None)
    return report


def evaluation_set(x_cpu, centers, inserted, latest_size, a, gen, device):
    mixed, old_count = query_set(x_cpu, centers, inserted, a, gen)
    count = a.queries - old_count
    lo = inserted - latest_size
    ids = torch.randint(lo, inserted, (count,), generator=gen)
    noise = 0.08 if a.workload == "drift" else 0.10
    latest = centers[ids] + noise * torch.randn(count, a.dim, generator=gen)
    return mixed.to(device), latest.to(device), old_count


def evaluate_graph(x, graph, queries, truth, latest, latest_truth, entries, old_count, a):
    pred, query_ms, samples = measured(
        lambda: graph_search(x, graph, queries, entries, a), x.device, a.repeats)
    latest_pred, latest_ms, latest_samples = measured(
        lambda: graph_search(x, graph, latest, entries, a), x.device, a.repeats)
    return {
        "query_ms": query_ms, "query_samples_ms": samples,
        "query_sample_summary": sample_summary(samples),
        "qps": len(queries) * 1000.0 / query_ms,
        "recall10": recall(pred, truth),
        "old_query_recall10": recall(pred[:old_count], truth[:old_count]),
        "new_query_recall10": recall(pred[old_count:], truth[old_count:]),
        "latest_chunk_query_recall10": recall(latest_pred, latest_truth),
        "latest_chunk_query_ms": latest_ms, "latest_chunk_query_samples_ms": latest_samples,
        "latest_chunk_query_sample_summary": sample_summary(latest_samples),
    }


def connectivity_metrics(graph, entries, base, latest_start):
    """Exact directed reachability; an orphan has zero incoming edges.

    A disconnected cycle can have no orphans yet be unreachable from base anchors.
    These untimed diagnostics deliberately ignore the search expansion budget.
    """
    n = len(graph)
    reached = torch.zeros(n, dtype=torch.bool, device=graph.device)
    frontier = torch.unique(entries)
    reached[frontier] = True
    while frontier.numel():
        edges = graph[frontier].flatten()
        targets = torch.unique(edges[edges >= 0])
        frontier = targets[~reached[targets]]
        reached[frontier] = True
    edges = graph[graph >= 0]
    indegree = torch.bincount(edges, minlength=n)
    orphan = indegree == 0
    return {
        "reachable_nodes": int(reached.sum().item()),
        "unreachable_nodes": int((~reached).sum().item()),
        "reachable_fraction": float(reached.float().mean().item()),
        "inserted_reachable_fraction": float(reached[base:].float().mean().item()),
        "latest_chunk_reachable_fraction": float(reached[latest_start:].float().mean().item()),
        "orphan_nodes": int(orphan.sum().item()),
        "orphan_fraction": float(orphan.float().mean().item()),
        "inserted_orphan_nodes": int(orphan[base:].sum().item()),
        "latest_chunk_orphan_nodes": int(orphan[latest_start:].sum().item()),
        "mean_in_degree": float(indegree.float().mean().item()),
    }


def validate_graph(graph, n):
    assert graph.ndim == 2 and len(graph) == n
    assert bool(((graph >= -1) & (graph < n)).all()), "out-of-range edge"
    rows = torch.arange(n, device=graph.device)
    assert not bool((graph == rows[:, None]).any()), "self edge"
    ordered = graph.sort(dim=1).values
    assert not bool(((ordered[:, 1:] == ordered[:, :-1])
                     & (ordered[:, 1:] >= 0)).any()), "duplicate edge"


def intra_edges(graph, base, chunk_size):
    rows = torch.arange(len(graph), device=graph.device)
    same = ((rows[:, None] >= base) & (graph >= base)
            & ((rows[:, None] - base) // chunk_size == (graph - base) // chunk_size))
    return int(same.sum().item())


def structural_metrics(graph, x, start, end, base, a, intra_truth):
    neighbors = graph[start:end]
    wanted = (intra_truth >= start) & (intra_truth < end)
    present = (intra_truth[:, :, None] == neighbors[:, None, :]).any(-1)
    count = int(wanted.sum().item())
    return {
        "intra_chunk_edges_all": intra_edges(graph, base, a.chunk_size),
        "mean_out_degree": float((graph >= 0).sum(-1).float().mean().item()),
        "latest_chunk_gt10_intra_demand": float(wanted.float().mean().item()),
        "latest_chunk_gt10_intra_edge_coverage":
            float((wanted & present).sum().item() / count) if count else None,
        "old_to_new_edges": int(((graph[:start] >= start)
                                  & (graph[:start] < end)).sum().item()),
    }


def generate_data(a, device):
    # Generate deterministically on CPU, then transfer; not part of update timing.
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    base = torch.randn(a.base, a.dim, generator=gen)
    chunks, centers = [], []
    for _ in range(a.chunks):
        if a.workload == "iid":
            points = torch.randn(a.chunk_size, a.dim, generator=gen)
            query_centers = points
        else:
            # Temporal locality: groups of ~2*degree near existing anchors.
            groups = min(a.base, max(1, a.chunk_size // (2 * a.degree)))
            anchors = torch.randperm(a.base, generator=gen)[:groups]
            c = base[anchors] + 0.10 * torch.randn(groups, a.dim, generator=gen)
            assignments = torch.arange(a.chunk_size) % groups
            query_centers = c[assignments]
            points = query_centers + 0.08 * torch.randn(
                a.chunk_size, a.dim, generator=gen)
        chunks.append(points)
        centers.append(query_centers)
    x = torch.cat([base] + chunks).to(device)
    return x, torch.cat(centers), gen


def query_set(x_cpu, new_centers, inserted, a, gen):
    n_old_q = a.queries // 2
    ids = torch.randint(a.base, (n_old_q,), generator=gen)
    old = x_cpu[ids] + 0.10 * torch.randn(n_old_q, a.dim, generator=gen)
    ids = torch.randint(inserted, (a.queries - n_old_q,), generator=gen)
    if a.workload == "drift":
        fresh = new_centers[ids] + 0.08 * torch.randn(
            a.queries - n_old_q, a.dim, generator=gen)
    else:
        fresh = new_centers[ids] + 0.10 * torch.randn(
            a.queries - n_old_q, a.dim, generator=gen)
    return torch.cat((old, fresh)), n_old_q


def self_test(device):
    """Focused checks: identity masking, RNG lune, reverse reachability, search."""
    from types import SimpleNamespace
    a = SimpleNamespace(degree=2, distance_batch=4, candidate_cap=6, hops=2,
                        local_cap=10, refine_rounds=1, query_batch=4,
                        ef=4, search_width=2, steps=6)
    x = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [0., 1.],
                      [0.10, 0.02], [0.12, 0.03]], device=device)
    rows = torch.tensor([0], device=device)
    candidates = torch.tensor([[1, 2, 3, -1, 1]], device=device)
    r = rng_prune(x, rows, candidates, 3)
    assert 1 in r[0] and 3 in r[0] and 2 not in r[0], "RNG lune failure"
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
    spliced = chunk_splice_extend(x, initial, 4, 6, a)
    validate_graph(spliced, len(x))
    assert torch.equal(spliced[:4], before[:4]), "splice must not touch old rows"
    assert 5 in spliced[4] and 4 in spliced[5], "splice must link chunk peers"
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


def benchmark(a, device):
    names = ["Rebuild", "Naive", "ChunkSplice", "LocalRNG"]
    report = experiment_report(a, device, "benchmark", names)
    x, new_centers, gen = generate_data(a, device)
    x_cpu = x.cpu()
    entries = torch.randperm(a.base, generator=gen)[:a.ef].to(device)
    initial, initial_ms, initial_samples = measured(
        lambda: build_graph(x[:a.base], a), device, a.repeats)
    validate_graph(initial, a.base)
    graphs = {name: initial.clone() for name in ("Rebuild", "Naive", "ChunkSplice", "LocalRNG")}
    report.update({
        "initial_build_ms": initial_ms, "initial_build_samples_ms": initial_samples,
        "initial_build_sample_summary": sample_summary(initial_samples),
        "not_applicable_parameters": ["sweep_total", "sweep_chunk_sizes", "stream_eval_every"],
        "stages": [],
    })
    # Stage zero provides a fixed-distribution baseline for the original data.
    base_q = x_cpu[torch.randint(a.base, (a.queries,), generator=gen)]
    base_q = (base_q + 0.1 * torch.randn(base_q.shape, generator=gen)).to(device)
    base_truth = exact_knn(base_q, x[:a.base], 10, a.distance_batch)
    pred, base_query_ms, base_samples = measured(
        lambda: graph_search(x[:a.base], initial, base_q, entries, a),
        device, a.repeats)
    report["base_query"] = {"query_ms": base_query_ms, "query_samples_ms": base_samples,
                            "recall10": recall(pred, base_truth),
                            "qps": a.queries * 1000.0 / base_query_ms}
    print(f"Device={device}, GPU={report['environment']['gpu']}, "
          f"torch={torch.__version__}, workload={a.workload}", flush=True)
    print(f"Initial exact kNN: {initial_ms:.3f} ms; "
          f"base Recall@10={report['base_query']['recall10']:.4f}", flush=True)
    print("stage N method       update_ms recall10 old_R10 new_R10 latest_R10 QPS       "
          "intra_edges intra_cov degree local", flush=True)
    for stage in range(1, a.chunks + 1):
        start = a.base + (stage - 1) * a.chunk_size
        end = start + a.chunk_size
        active = x[:end]
        queries, latest, old_count = evaluation_set(
            x_cpu, new_centers, stage * a.chunk_size, a.chunk_size, a, gen, device)
        stage_entries = (torch.randperm(end, generator=gen)[:a.ef].to(device)
                         if a.entry_policy == "all" else entries)
        truth = exact_knn(queries, active, 10, a.distance_batch)
        latest_truth = exact_knn(latest, active, 10, a.distance_batch)
        intra_truth = exact_knn(active[start:end], active, 10, a.distance_batch,
                                torch.arange(start, end, device=device))
        records = {}
        # Rotate method order to reduce systematic clock/thermal ordering bias.
        names = ["Rebuild", "Naive", "ChunkSplice", "LocalRNG"]
        shift = (stage - 1) % len(names)
        order = names[shift:] + names[:shift]
        for name in order:
            prev = graphs[name]
            info = {}
            if name == "Rebuild":
                fn = lambda: build_graph(active, a)
            elif name == "Naive":
                fn = lambda: naive_extend(active, prev, start, end, a)
            elif name == "ChunkSplice":
                fn = lambda: chunk_splice_extend(active, prev, start, end, a)
            else:
                fn = lambda: refined_extend(active, prev, start, end, a)
            built, update_ms, update_samples = measured(fn, device, a.repeats)
            if name == "LocalRNG":
                built, info = built
            graphs[name] = built
            validate_graph(built, end)
            if name == "Naive":
                assert intra_edges(built, a.base, a.chunk_size) == 0, "chunk invariant"
            metrics = structural_metrics(built, active, start, end, a.base, a,
                                         intra_truth)
            records[name] = {
                "update_ms": update_ms, "update_samples_ms": update_samples,
                "update_sample_summary": sample_summary(update_samples),
                **evaluate_graph(active, built, queries, truth, latest, latest_truth,
                                 stage_entries, old_count, a),
                **metrics, **info,
            }
        for name in names:
            r = records[name]
            cov = r["latest_chunk_gt10_intra_edge_coverage"]
            cov_text = f"{cov:.3f}" if cov is not None else "n/a"
            print(f"{stage:>5} {end:>5} {name:<11} {r['update_ms']:>9.3f} "
                  f"{r['recall10']:>8.4f} {r['old_query_recall10']:>7.4f} "
                  f"{r['new_query_recall10']:>7.4f} "
                  f"{r['latest_chunk_query_recall10']:>10.4f} {r['qps']:>9.1f} "
                  f"{r['intra_chunk_edges_all']:>11} {cov_text:>9} "
                  f"{r['mean_out_degree']:>6.2f} "
                  f"{r.get('local_vertices', 0):>5}", flush=True)
        erosion = records["Rebuild"]["recall10"] - records["Naive"]["recall10"]
        recovery = records["LocalRNG"]["recall10"] - records["Naive"]["recall10"]
        splice = records["ChunkSplice"]["recall10"] - records["Naive"]["recall10"]
        report["stages"].append({"stage": stage, "n": end, "methods": records,
                                  "method_order": order,
                                  "search_entry_ids": stage_entries.cpu().tolist(),
                                  "latest_query_source_range": [start, end],
                                  "erosion_vs_rebuild": erosion,
                                  "recovery_vs_naive": recovery,
                                  "splice_recovery_vs_naive": splice,
                                  "rng_gain_vs_splice": recovery - splice})
    last = report["stages"][-1]
    restored = last["methods"]["LocalRNG"]["intra_chunk_edges_all"] > 0
    demonstrated = (last["erosion_vs_rebuild"] > 0.02
                    and last["recovery_vs_naive"] > 0.02 and restored)
    report["effect_demonstrated"] = demonstrated
    print(f"Final erosion vs rebuild: {last['erosion_vs_rebuild']:+.4f}; "
          f"local recovery vs naive: {last['recovery_vs_naive']:+.4f}; "
          f"splice recovery vs naive: {last['splice_recovery_vs_naive']:+.4f}; "
          f"RNG gain vs splice: {last['rng_gain_vs_splice']:+.4f}", flush=True)
    print("EFFECT " + ("DEMONSTRATED" if demonstrated else "NOT DEMONSTRATED")
          + " (empirical, this workload/budget only)", flush=True)
    return finish_report(report, device)


def run_sweep(a, device):
    """Vary only insertion partitioning of one fixed ordered corpus."""
    names = ["Rebuild", "Naive", "ChunkSplice"]
    report = experiment_report(a, device, "chunk_sweep", names)
    # Generate once with a fixed block size, never with the swept partition size.
    data_args = copy.copy(a)
    data_args.chunks = (a.sweep_total + a.chunk_size - 1) // a.chunk_size
    x, centers, gen = generate_data(data_args, device)
    x = x[:a.base + a.sweep_total]
    centers = centers[:a.sweep_total]
    x_cpu = x.cpu()
    entries = torch.randperm(a.base, generator=gen)[:a.ef].to(device)
    stage_entries = (torch.randperm(len(x), generator=gen)[:a.ef].to(device)
                     if a.entry_policy == "all" else entries)
    # Shared tail lies inside the final chunk of every partition, including remainders.
    tail_size = min(a.sweep_total % b or b for b in a.sweep_chunk_sizes)
    queries, latest, old_count = evaluation_set(
        x_cpu, centers, a.sweep_total, tail_size, a, gen, device)
    truth = exact_knn(queries, x, 10, a.distance_batch)
    latest_truth = exact_knn(latest, x, 10, a.distance_batch)
    initial, initial_ms, initial_samples = measured(
        lambda: build_graph(x[:a.base], a), device, a.repeats)
    validate_graph(initial, a.base)
    report.update({
        "initial_build_ms": initial_ms, "initial_build_samples_ms": initial_samples,
        "initial_build_sample_summary": sample_summary(initial_samples),
        "controls": {
            "total_inserted": a.sweep_total, "final_n": len(x),
            "data_generation_chunk_size": a.chunk_size,
            "same_ordered_data_queries_truth_entries": True,
            "latest_query_source_range": [len(x) - tail_size, len(x)],
            "latest_probe": "fixed shared tail within every partition's last chunk",
            "search_entry_ids": stage_entries.cpu().tolist(),
            "update_time_scope": "entire insertion trajectory, excluding initial build",
        },
        "not_applicable_parameters": ["chunks", "local_cap", "candidate_cap",
                                      "hops", "refine_rounds", "stream_eval_every"],
        "results": [],
    })
    print(f"SWEEP: fixed N={len(x)}, inserted={a.sweep_total}, "
          f"generation block={a.chunk_size}", flush=True)
    print("chunk method       total_update_ms recall10 inserted_R10 latest_R10 QPS", flush=True)
    for index, b in enumerate(a.sweep_chunk_sizes):
        config = copy.copy(a)
        config.chunk_size = b
        chunks = (a.sweep_total + b - 1) // b
        latest_start = a.base + (chunks - 1) * b
        intra_truth = exact_knn(x[latest_start:], x, 10, a.distance_batch,
                                torch.arange(latest_start, len(x), device=device))
        result = {"chunk_size": b, "chunks": chunks, "n": len(x),
                  "last_chunk_size": len(x) - latest_start, "methods": {}}
        shift = index % len(names)
        order = names[shift:] + names[:shift]
        result["method_order"] = order
        for name in order:
            def trajectory():
                graph = initial
                for start in range(a.base, len(x), b):
                    end = min(start + b, len(x))
                    active = x[:end]
                    if name == "Rebuild":
                        graph = build_graph(active, config)
                    elif name == "Naive":
                        graph = naive_extend(active, graph, start, end, config)
                    else:
                        graph = chunk_splice_extend(active, graph, start, end, config)
                return graph

            graph, total_ms, samples = measured(trajectory, device, a.repeats)
            validate_graph(graph, len(x))
            if name == "Naive":
                assert intra_edges(graph, a.base, b) == 0, "chunk invariant"
            metrics = structural_metrics(graph, x, latest_start, len(x), a.base,
                                         config, intra_truth)
            record = {
                "update_ms": total_ms, "update_samples_ms": samples,
                "update_sample_summary": sample_summary(samples),
                "amortized_update_ms_per_node": total_ms / a.sweep_total,
                **evaluate_graph(x, graph, queries, truth, latest, latest_truth,
                                 stage_entries, old_count, a),
                **metrics,
            }
            record["intra_edges"] = record["intra_chunk_edges_all"]
            result["methods"][name] = record
        for name in names:
            r = result["methods"][name]
            print(f"{b:>5} {name:<11} {r['update_ms']:>15.3f} {r['recall10']:>8.4f} "
                  f"{r['new_query_recall10']:>12.4f} "
                  f"{r['latest_chunk_query_recall10']:>10.4f} {r['qps']:>9.1f}", flush=True)
        report["results"].append(result)
    return finish_report(report, device)


def run_stream(a, device):
    """Stream CLI-sized updates; retain all timings and checkpoint diagnostics."""
    names = ["Naive", "ChunkSplice"]
    report = experiment_report(a, device, "streaming", names)
    b = a.chunk_size
    x, centers, gen = generate_data(a, device)
    x_cpu = x.cpu()
    entries = torch.randperm(a.base, generator=gen)[:a.ef].to(device)
    initial, initial_ms, initial_samples = measured(
        lambda: build_graph(x[:a.base], a), device, a.repeats)
    validate_graph(initial, a.base)
    graphs = {name: initial.clone() for name in names}
    report.update({
        "initial_build_ms": initial_ms, "initial_build_samples_ms": initial_samples,
        "initial_build_sample_summary": sample_summary(initial_samples),
        "effective_chunks": a.chunks, "effective_chunk_size": b,
        "connectivity": {
            "definition": "exact directed traversal from fixed original-base anchors",
            "anchor_ids": entries.cpu().tolist(),
            "independent_of_entry_policy": True,
            "orphan_definition": "zero incoming edges; distinct from unreachable nodes",
            "evaluated": "first, last, and every stream_eval_every updates",
        },
        "not_applicable_parameters": ["local_cap", "candidate_cap", "hops",
                                      "refine_rounds", "sweep_total", "sweep_chunk_sizes"],
        "history": [],
    })
    print(f"STREAM: chunks={a.chunks}, chunk_size={b}, repeats={a.repeats}", flush=True)
    print("chunk N method       update_ms recall10 inserted_R10 latest_R10 reach% orphans", flush=True)
    for step in range(1, a.chunks + 1):
        start = a.base + (step - 1) * b
        end = start + b
        active = x[:end]
        evaluate = step in (1, a.chunks) or step % a.stream_eval_every == 0
        row = {"chunk": step, "n": end, "evaluated": evaluate, "methods": {}}
        shift = (step - 1) % len(names)
        order = names[shift:] + names[:shift]
        row["method_order"] = order
        for name in order:
            prev = graphs[name]
            fn = (lambda: naive_extend(active, prev, start, end, a)) if name == "Naive" else (
                  lambda: chunk_splice_extend(active, prev, start, end, a))
            built, update_ms, samples = measured(fn, device, a.repeats)
            validate_graph(built, end)
            graphs[name] = built
            row["methods"][name] = {
                "update_ms": update_ms, "update_samples_ms": samples,
                "update_sample_summary": sample_summary(samples),
            }
        if evaluate:
            queries, latest, old_count = evaluation_set(
                x_cpu, centers, step * b, b, a, gen, device)
            truth = exact_knn(queries, active, 10, a.distance_batch)
            latest_truth = exact_knn(latest, active, 10, a.distance_batch)
            intra_truth = exact_knn(active[start:end], active, 10, a.distance_batch,
                                    torch.arange(start, end, device=device))
            stage_entries = (torch.randperm(end, generator=gen)[:a.ef].to(device)
                             if a.entry_policy == "all" else entries)
            row["search_entry_ids"] = stage_entries.cpu().tolist()
            row["latest_query_source_range"] = [start, end]
            for name in order:
                graph = graphs[name]
                r = row["methods"][name]
                r.update(evaluate_graph(active, graph, queries, truth, latest, latest_truth,
                                        stage_entries, old_count, a))
                r.update(structural_metrics(graph, active, start, end, a.base, a, intra_truth))
                r.update(connectivity_metrics(graph, entries, a.base, start))
                print(f"{step:>5} {end:>5} {name:<11} {r['update_ms']:>9.3f} "
                      f"{r['recall10']:>8.4f} {r['new_query_recall10']:>12.4f} "
                      f"{r['latest_chunk_query_recall10']:>10.4f} "
                      f"{100 * r['reachable_fraction']:>6.1f} {r['orphan_nodes']:>7}", flush=True)
        # Keep the original history names available to existing report readers.
        for name, alias in (("Naive", "naive"), ("ChunkSplice", "splice")):
            r = row["methods"][name]
            r["ms"] = r["update_ms"]
            if evaluate:
                r.update(recall=r["recall10"], new_recall=r["new_query_recall10"],
                         degree=r["mean_out_degree"])
            row[alias] = r
        report["history"].append(row)
    report["total_update_ms"] = {
        name: sum(row["methods"][name]["update_ms"] for row in report["history"])
        for name in names}
    report["timing"]["stream_total"] = "sum of per-stage medians; not an end-to-end sample"
    return finish_report(report, device)


def save_report(path, report):
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def main():
    global torch
    a = arguments()
    try:
        import torch as torch_module
        torch = torch_module
        if a.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Install a CUDA-enabled PyTorch build "
                               "and compatible NVIDIA driver. --device cpu is debug only.")
        device = torch.device(a.device)
        with torch.inference_mode():
            self_test(device)
            if a.self_test_only:
                return 0
            if a.mode == "sweep":
                report = run_sweep(a, device)
            elif a.mode == "stream":
                report = run_stream(a, device)
            else:
                report = benchmark(a, device)
        save_report(a.output, report)
        print("JSON: " + str(a.output.resolve()), flush=True)
        if a.verify_effect and a.mode == "benchmark" and not report.get("effect_demonstrated", False):
            print("Requested effect threshold not met; inspect measured results.",
                  file=sys.stderr)
            return 2
        return 0
    except (ImportError, OSError, RuntimeError, AssertionError, ValueError) as exc:
        error = {
            "schema_version": 2, "status": "runtime_error", "benchmark_completed": False,
            "mode": a.mode,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parameters": {k: str(v) if isinstance(v, Path) else v
                           for k, v in vars(a).items()},
            "error_type": type(exc).__name__, "error": str(exc),
            "python": sys.version, "requested_device": a.device,
            "torch_distribution_versions": [],
        }
        error["torch_distribution_versions"] = [
            d.version for d in importlib.metadata.distributions()
            if d.metadata.get("Name", "").lower() == "torch"]
        # Keep completed results intact; failures have a separate artifact.
        path = a.output.with_name(a.output.stem + ".error.json")
        save_report(path, error)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("No benchmark results claimed. Diagnostic JSON: " + str(path.resolve()),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
