"""Recall, topology and directed connectivity diagnostics."""
from __future__ import annotations

import torch


def recall(predicted, truth):
    return float((predicted[:, :, None] == truth[:, None, :]).any(-1)
                 .float().mean().item())


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
