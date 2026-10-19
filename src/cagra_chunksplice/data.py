"""Seeded synthetic workloads and shared query cohorts."""
from __future__ import annotations

import torch


def generate_data(a, device):
    # Generate deterministically on CPU, then transfer; not part of update timing.
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    base = torch.randn(a.base, a.dim, generator=gen)
    chunks, centers = [], []
    for _ in range(a.chunks):
        if a.workload in ("iid", "high-lid"):
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
    x = torch.cat([base] + chunks)
    centers = torch.cat(centers)
    if getattr(a, "query_distribution", "in-distribution") == "angular":
        x = torch.nn.functional.normalize(x, dim=-1)
        centers = torch.nn.functional.normalize(centers, dim=-1)
    return x.to(device), centers, gen


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
    return transform_queries(torch.cat((old, fresh)), a), n_old_q


def evaluation_set(x_cpu, centers, inserted, latest_size, a, gen, device):
    mixed, old_count = query_set(x_cpu, centers, inserted, a, gen)
    count = a.queries - old_count
    lo = inserted - latest_size
    ids = torch.randint(lo, inserted, (count,), generator=gen)
    noise = 0.08 if a.workload == "drift" else 0.10
    latest = centers[ids] + noise * torch.randn(count, a.dim, generator=gen)
    return mixed.to(device), transform_queries(latest, a).to(device), old_count


def transform_queries(queries, a):
    """Translate OOD probes or normalize angular probes without changing truth.

    Angular runs normalize both corpus and queries; squared L2 then yields the
    same ranking as cosine distance. OOD translation uses a fixed coordinate
    direction so its magnitude and interpretation remain reproducible.
    """
    distribution = getattr(a, "query_distribution", "in-distribution")
    if distribution == "ood":
        queries = queries.clone()
        queries[:, 0] += getattr(a, "ood_shift", 4.0)
    elif distribution == "angular":
        queries = torch.nn.functional.normalize(queries, dim=-1)
    return queries


def age_query_sets(x_cpu, inserted, a):
    """Source-age strata: base, early, middle, recent insertion thirds.

    Strata describe source vectors, not necessarily the ages of exact neighbors.
    Query generation has its own seed and cannot alter subsequent insertions.
    """
    epochs = inserted // a.chunk_size
    boundaries = [0, epochs // 3, 2 * epochs // 3, epochs]
    ranges = [("base", 0, a.base)]
    for i, label in enumerate(("early", "middle", "recent")):
        lo = a.base + boundaries[i] * a.chunk_size
        hi = a.base + boundaries[i + 1] * a.chunk_size
        if hi > lo:
            ranges.append((label, lo, hi))
    gen = torch.Generator().manual_seed(a.seed + 104729)
    count = getattr(a, "age_queries", 32)
    probes = {}
    for label, lo, hi in ranges:
        ids = torch.randint(lo, hi, (count,), generator=gen)
        q = x_cpu[ids] + 0.1 * torch.randn(count, a.dim, generator=gen)
        probes[label] = (transform_queries(q, a), [lo, hi])
    return probes
