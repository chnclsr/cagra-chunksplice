"""Controlled insertion-partition sweep driver."""
from __future__ import annotations

import copy
import torch
from cagra_chunksplice.graph import build_graph, build_rng_graph
from cagra_chunksplice.metrics import connectivity_metrics
from cagra_chunksplice.algorithms.chunk_splice import chunk_splice_extend
from cagra_chunksplice.evaluation import evaluate_graph
from cagra_chunksplice.data import evaluation_set
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.reporting import experiment_report
from cagra_chunksplice.reporting import finish_report
from cagra_chunksplice.data import generate_data
from cagra_chunksplice.metrics import intra_edges
from cagra_chunksplice.timing import measured
from cagra_chunksplice.algorithms.naive import naive_extend
from cagra_chunksplice.timing import sample_summary
from cagra_chunksplice.metrics import structural_metrics
from cagra_chunksplice.graph import validate_graph


def run_sweep(a, device):
    """Vary only insertion partitioning of one fixed ordered corpus."""
    names = ["Rebuild-kNN", "Rebuild-RNG", "Naive", "ChunkSplice"]
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
                    if name == "Rebuild-kNN":
                        graph = build_graph(active, config)
                    elif name == "Rebuild-RNG":
                        graph = build_rng_graph(active, config)
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
                **connectivity_metrics(graph, entries, a.base, latest_start),
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
