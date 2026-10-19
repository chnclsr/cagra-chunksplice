"""Seeded synthetic workloads and shared query cohorts."""
from __future__ import annotations

import torch


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


def evaluation_set(x_cpu, centers, inserted, latest_size, a, gen, device):
    mixed, old_count = query_set(x_cpu, centers, inserted, a, gen)
    count = a.queries - old_count
    lo = inserted - latest_size
    ids = torch.randint(lo, inserted, (count,), generator=gen)
    noise = 0.08 if a.workload == "drift" else 0.10
    latest = centers[ids] + noise * torch.randn(count, a.dim, generator=gen)
    return mixed.to(device), latest.to(device), old_count
