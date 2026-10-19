"""Streaming updates and checkpoint evaluation."""
from __future__ import annotations

import torch
from cagra_chunksplice.graph import build_graph, build_rng_graph
from cagra_chunksplice.algorithms.chunk_splice import chunk_splice_extend
from cagra_chunksplice.metrics import connectivity_metrics
from cagra_chunksplice.evaluation import evaluate_graph
from cagra_chunksplice.data import evaluation_set, age_query_sets
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.reporting import experiment_report
from cagra_chunksplice.reporting import finish_report
from cagra_chunksplice.data import generate_data
from cagra_chunksplice.timing import measured
from cagra_chunksplice.algorithms.naive import naive_extend
from cagra_chunksplice.timing import sample_summary
from cagra_chunksplice.metrics import structural_metrics
from cagra_chunksplice.graph import validate_graph
from cagra_chunksplice.metrics import EdgeDecayTracker, recall
from cagra_chunksplice.search import graph_search


def run_stream(a, device):
    """Stream CLI-sized updates; retain all timings and checkpoint diagnostics."""
    names = ["Naive", "ChunkSplice"]
    stress = getattr(a, "mode", "stream") == "stress"
    if stress or getattr(a, "stream_rebuild", False):
        names = ["Rebuild-kNN", "Rebuild-RNG"] + names
    report = experiment_report(a, device, "long_horizon_stress" if stress else "streaming", names)
    b = a.chunk_size
    x, centers, gen = generate_data(a, device)
    x_cpu = x.cpu()
    entries = torch.randperm(a.base, generator=gen)[:a.ef].to(device)
    initial, initial_ms, initial_samples = measured(
        lambda: build_graph(x[:a.base], a), device, a.repeats)
    validate_graph(initial, a.base)
    graphs = {name: initial.clone() for name in names}
    trackers = {name: EdgeDecayTracker(a.base, b) for name in names}
    for tracker in trackers.values():
        tracker.observe(initial, x[:a.base], a.distance_batch)
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
            if name == "Rebuild-kNN":
                fn = lambda: build_graph(active, a)
            elif name == "Rebuild-RNG":
                fn = lambda: build_rng_graph(active, a)
            elif name == "Naive":
                fn = lambda: naive_extend(active, prev, start, end, a)
            else:
                fn = lambda: chunk_splice_extend(active, prev, start, end, a)
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
            age_sets = {}
            for label, (probes, source_range) in age_query_sets(x_cpu, step * b, a).items():
                probes = probes.to(device)
                age_sets[label] = (probes, exact_knn(probes, active, 10, a.distance_batch), source_range)
            row["age_query_source_ranges"] = {label: values[2] for label, values in age_sets.items()}
            for name in order:
                graph = graphs[name]
                r = row["methods"][name]
                r.update(evaluate_graph(active, graph, queries, truth, latest, latest_truth,
                                        stage_entries, old_count, a))
                r.update(structural_metrics(graph, active, start, end, a.base, a, intra_truth))
                r.update(connectivity_metrics(graph, entries, a.base, start))
                r.update(trackers[name].observe(graph, active, a.distance_batch))
                r["age_stratified_recall10"] = {
                    label: recall(graph_search(active, graph, probes, stage_entries, a), exact)
                    for label, (probes, exact, _) in age_sets.items()}
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
