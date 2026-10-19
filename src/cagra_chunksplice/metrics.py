"""Recall, topology and directed connectivity diagnostics."""
from __future__ import annotations

import torch


def recall(predicted, truth):
    return float((predicted[:, :, None] == truth[:, None, :]).any(-1)
                 .float().mean().item())


def _fraction(values):
    return float(values.double().mean().item()) if values.numel() else None


def scc_metrics(graph):
    """Iterative Kosaraju traversal, O(V+E), including isolated vertices.

    CPU traversal is untimed. A sink SCC has no edges to another SCC; multiple
    SCCs can be reachable from anchors without being mutually reachable.
    """
    adjacency = [[v for v in row if v >= 0] for row in graph.cpu().tolist()]
    n = len(adjacency)
    reverse = [[] for _ in range(n)]
    for u, row in enumerate(adjacency):
        for v in row:
            reverse[v].append(u)
    seen, order = set(), []
    for root in range(n):
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(adjacency[root]))]
        while stack:
            u, edges = stack[-1]
            v = next(edges, None)
            if v is None:
                order.append(u)
                stack.pop()
            elif v not in seen:
                seen.add(v)
                stack.append((v, iter(adjacency[v])))
    labels = [-1] * n
    sizes = []
    for root in reversed(order):
        if labels[root] >= 0:
            continue
        label = len(sizes)
        labels[root] = label
        stack, size = [root], 0
        while stack:
            u = stack.pop()
            size += 1
            for v in reverse[u]:
                if labels[v] < 0:
                    labels[v] = label
                    stack.append(v)
        sizes.append(size)
    sinks = [True] * len(sizes)
    for u, row in enumerate(adjacency):
        for v in row:
            if labels[u] != labels[v]:
                sinks[labels[u]] = False
    return {
        "scc_count": len(sizes),
        "largest_scc_fraction": max(sizes, default=0) / n if n else None,
        "sink_scc_count": sum(sinks),
        "largest_sink_scc_fraction": max(
            (size for size, sink in zip(sizes, sinks) if sink), default=0) / n if n else None,
    }


def hubness_metrics(graph):
    """Population in-degree Gini and standardized third central moment.

    Zero-edge and constant-degree graphs have Gini/skewness zero. Empty graphs
    report None, keeping the JSON finite without implying an observation.
    """
    n = len(graph)
    if not n:
        return {"in_degree_gini": None, "in_degree_skewness": None,
                "max_in_degree": 0}
    degree = torch.bincount(graph[graph >= 0], minlength=n).double()
    ordered = degree.sort().values
    ranks = torch.arange(1, n + 1, device=graph.device, dtype=torch.float64)
    total = degree.sum()
    gini = float(((2 * ranks - n - 1) * ordered).sum() / (n * total)) if total > 0 else 0.0
    centered = degree - degree.mean()
    variance = centered.square().mean()
    skew = float(centered.pow(3).mean() / variance.pow(1.5)) if variance > 0 else 0.0
    return {"in_degree_gini": gini, "in_degree_skewness": skew,
            "max_in_degree": int(degree.max())}


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
        **scc_metrics(graph),
        **hubness_metrics(graph),
        "reachable_nodes": int(reached.sum().item()),
        "unreachable_nodes": int((~reached).sum().item()),
        "reachable_fraction": _fraction(reached),
        "inserted_reachable_fraction": _fraction(reached[base:]),
        "latest_chunk_reachable_fraction": _fraction(reached[latest_start:]),
        "orphan_nodes": int(orphan.sum().item()),
        "orphan_fraction": _fraction(orphan),
        "inserted_orphan_nodes": int(orphan[base:].sum().item()),
        "latest_chunk_orphan_nodes": int(orphan[latest_start:].sum().item()),
        "mean_in_degree": _fraction(indegree),
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
        **edge_length_metrics(graph, x, base, a.chunk_size, a.distance_batch),
        "intra_chunk_edges_all": intra_edges(graph, base, a.chunk_size),
        "mean_out_degree": float((graph >= 0).sum(-1).float().mean().item()),
        "latest_chunk_gt10_intra_demand": float(wanted.float().mean().item()),
        "latest_chunk_gt10_intra_edge_coverage":
            float((wanted & present).sum().item() / count) if count else None,
        "latest_chunk_outgoing_bridge_fraction": _fraction(neighbors[neighbors >= 0] < start),
        "latest_chunk_bridge_slot_occupancy": _fraction((neighbors >= 0) & (neighbors < start)),
        "latest_chunk_zero_exit_fraction": _fraction(~((neighbors >= 0) & (neighbors < start)).any(dim=1)),
        "old_to_new_edges": int(((graph[:start] >= start)
                                  & (graph[:start] < end)).sum().item()),
    }


def cohort_ids(n, base, chunk_size, device):
    """Base cohort 0; insertion epochs 1, 2, ... ."""
    rows = torch.arange(n, device=device)
    return torch.where(rows < base, 0, 1 + (rows - base) // chunk_size)


def edge_length_metrics(graph, x, base, chunk_size, block=256):
    """Euclidean edge lengths and cross-epoch bridge occupancy.

    A bridge joins different insertion epochs (the base is epoch 0). Occupancy
    uses all allocated slots; bridge_fraction uses valid edges only. These are
    cohort bridges, not graph-theoretic cut edges.
    """
    cohorts = cohort_ids(len(graph), base, chunk_size, graph.device)
    lengths, bridge_lengths = [], []
    older_count = newer_count = base_count = 0
    for lo in range(0, len(graph), block):
        neighbors = graph[lo:lo + block]
        valid = neighbors >= 0
        source = cohorts[lo:lo + block, None]
        target = cohorts[neighbors.clamp_min(0)]
        bridge = valid & (source != target)
        distance = (x[lo:lo + block, None] - x[neighbors.clamp_min(0)]).norm(dim=-1)
        lengths.append(distance[valid])
        bridge_lengths.append(distance[bridge])
        older_count += int((valid & (target < source)).sum())
        newer_count += int((valid & (target > source)).sum())
        base_count += int((valid & (source > 0) & (target == 0)).sum())
    lengths = torch.cat(lengths) if lengths else x.new_empty(0)
    bridges = torch.cat(bridge_lengths) if bridge_lengths else x.new_empty(0)
    edges, count = len(lengths), len(bridges)
    return {
        "bridge_edges": count,
        "bridge_fraction": count / edges if edges else None,
        "bridge_slot_occupancy": count / graph.numel() if graph.numel() else None,
        "edges_to_older_cohort": older_count,
        "edges_to_newer_cohort": newer_count,
        "inserted_to_base_edges": base_count,
        "mean_edge_length": _fraction(lengths),
        "p90_edge_length": float(torch.quantile(lengths.float(), 0.9)) if edges else None,
        "mean_bridge_length": _fraction(bridges),
    }


class EdgeDecayTracker:
    """Survival of directed edges between evaluated checkpoints.

    The first edge snapshot is fixed; bridges use the first nonempty snapshot. Surviving endpoints are restricted to vertices
    present in the reference snapshot, so corpus growth cannot inflate survival.
    A deleted edge that later reappears counts as present at the checkpoint.
    """

    def __init__(self, base, chunk_size):
        self.base = base
        self.chunk_size = chunk_size
        self.initial_edges = self.initial_bridges = None
        self.previous_edges = self.previous_bridges = None
        self.initial_mean_length = None
        self.initial_bridge_length = None
        self.initial_bridge_reference_n = None

    def observe(self, graph, x, block=256):
        metrics = edge_length_metrics(graph, x, self.base, self.chunk_size, block)
        cohorts = cohort_ids(len(graph), self.base, self.chunk_size, graph.device).cpu().tolist()
        edges = {(u, v) for u, row in enumerate(graph.cpu().tolist()) for v in row if v >= 0}
        bridges = {(u, v) for u, v in edges if cohorts[u] != cohorts[v]}
        if self.initial_edges is None:
            self.initial_edges, self.initial_bridges = edges, bridges
            self.previous_edges, self.previous_bridges = edges, bridges
            self.initial_mean_length = metrics["mean_edge_length"]
        if not self.initial_bridges and bridges:
            self.initial_bridges = bridges
            self.initial_bridge_length = metrics["mean_bridge_length"]
            self.initial_bridge_reference_n = len(graph)
        metrics["initial_bridge_reference_n"] = self.initial_bridge_reference_n
        initial_bridge, current_bridge = self.initial_bridge_length, metrics["mean_bridge_length"]
        metrics["mean_bridge_length_ratio_to_initial"] = (
            current_bridge / initial_bridge if initial_bridge and current_bridge is not None else None)
        for label, reference, reference_bridges in (
                ("initial", self.initial_edges, self.initial_bridges),
                ("previous_checkpoint", self.previous_edges, self.previous_bridges)):
            metrics[label + "_edge_survival"] = (
                len(reference & edges) / len(reference) if reference else None)
            metrics[label + "_bridge_survival"] = (
                len(reference_bridges & bridges) / len(reference_bridges) if reference_bridges else None)
        initial, current = self.initial_mean_length, metrics["mean_edge_length"]
        metrics["mean_edge_length_ratio_to_initial"] = current / initial if initial and current is not None else None
        self.previous_edges, self.previous_bridges = edges, bridges
        return metrics
