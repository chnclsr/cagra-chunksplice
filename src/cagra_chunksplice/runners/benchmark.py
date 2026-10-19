"""Four-method incremental benchmark driver."""
from __future__ import annotations

import torch
from cagra_chunksplice.graph import build_graph, build_rng_graph
from cagra_chunksplice.metrics import connectivity_metrics
from cagra_chunksplice.algorithms.chunk_splice import chunk_splice_extend
from cagra_chunksplice.evaluation import evaluate_graph
from cagra_chunksplice.data import evaluation_set, transform_queries
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.reporting import experiment_report
from cagra_chunksplice.reporting import finish_report
from cagra_chunksplice.data import generate_data
from cagra_chunksplice.search import graph_search
from cagra_chunksplice.metrics import intra_edges
from cagra_chunksplice.timing import measured
from cagra_chunksplice.algorithms.naive import naive_extend
from cagra_chunksplice.metrics import recall
from cagra_chunksplice.algorithms.local_rng import refined_extend
from cagra_chunksplice.timing import sample_summary
from cagra_chunksplice.metrics import structural_metrics
from cagra_chunksplice.graph import validate_graph


def benchmark(a, device):
    names = ["Rebuild-kNN", "Rebuild-RNG", "Naive", "ChunkSplice", "LocalRNG"]
    report = experiment_report(a, device, "benchmark", names)
    x, new_centers, gen = generate_data(a, device)
    x_cpu = x.cpu()
    entries = torch.randperm(a.base, generator=gen)[:a.ef].to(device)
    initial, initial_ms, initial_samples = measured(
        lambda: build_graph(x[:a.base], a), device, a.repeats)
    validate_graph(initial, a.base)
    graphs = {name: initial.clone() for name in names}
    report.update({
        "initial_build_ms": initial_ms, "initial_build_samples_ms": initial_samples,
        "initial_build_sample_summary": sample_summary(initial_samples),
        "not_applicable_parameters": ["sweep_total", "sweep_chunk_sizes", "stream_eval_every"],
        "stages": [],
    })
    # Stage zero provides a fixed-distribution baseline for the original data.
    base_q = x_cpu[torch.randint(a.base, (a.queries,), generator=gen)]
    base_q = transform_queries(base_q + 0.1 * torch.randn(base_q.shape, generator=gen), a).to(device)
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
        names = ["Rebuild-kNN", "Rebuild-RNG", "Naive", "ChunkSplice", "LocalRNG"]
        shift = (stage - 1) % len(names)
        order = names[shift:] + names[:shift]
        for name in order:
            prev = graphs[name]
            info = {}
            if name == "Rebuild-kNN":
                fn = lambda: build_graph(active, a)
            elif name == "Rebuild-RNG":
                fn = lambda: build_rng_graph(active, a)
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
                **connectivity_metrics(built, entries, a.base, start),
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
        erosion = records["Rebuild-kNN"]["recall10"] - records["Naive"]["recall10"]
        recovery = records["LocalRNG"]["recall10"] - records["Naive"]["recall10"]
        splice = records["ChunkSplice"]["recall10"] - records["Naive"]["recall10"]
        report["stages"].append({"stage": stage, "n": end, "methods": records,
                                  "method_order": order,
                                  "search_entry_ids": stage_entries.cpu().tolist(),
                                  "latest_query_source_range": [start, end],
                                  "erosion_vs_rebuild": erosion,
                                  "erosion_vs_rebuild_rng": records["Rebuild-RNG"]["recall10"] - records["Naive"]["recall10"],
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
