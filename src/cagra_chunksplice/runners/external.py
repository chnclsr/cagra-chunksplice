"""Comparative incremental benchmark with explicit native/simulator provenance."""
from __future__ import annotations

import statistics
import time

from cagra_chunksplice.baselines import CuVSCAGRA, FaissHNSW, GraphBaseline, CPUMirage
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.data import query_set
from cagra_chunksplice.datasets import load_external_data
from cagra_chunksplice.promotion import EXTERNAL_METHODS, native_row, native_rows_complete
from cagra_chunksplice.metrics import recall
from cagra_chunksplice.reporting import experiment_report, finish_report
from cagra_chunksplice.timing import sample_summary


METHODS = ("Naive", "ChunkSplice", "FAISS-HNSW", "cuVS-CAGRA", "MIRAGE")


def create_indexes(a, device, methods=METHODS):
    """Return only requested fresh indexes; imports happen during build."""
    factories = {
        "Naive": lambda: GraphBaseline(a, "Naive", device),
        "ChunkSplice": lambda: GraphBaseline(a, "ChunkSplice", device),
        "FAISS-HNSW": lambda: FaissHNSW(device, a.external_backend, a.hnsw_m,
                                      a.hnsw_ef_construction, a.ef),
        "cuVS-CAGRA": lambda: CuVSCAGRA(a, device, a.external_backend),
        "MIRAGE": lambda: CPUMirage(device, a.external_backend,
                                   s=a.mirage_s, r=a.mirage_r, iterations=a.mirage_iter),
    }
    return {name: factories[name]() for name in methods}


def _timed(index, operation):
    index.synchronize()
    start = time.perf_counter()
    value = operation()
    index.synchronize()
    return value, (time.perf_counter() - start) * 1000.0


def benchmark_external(a, device, *, methods=METHODS, prepared=None):
    prepared = prepared or load_external_data(a, device)
    x, heldout_queries, centers, gen, dataset_info = prepared
    report = experiment_report(a, device, "external", list(methods))
    report.update({
        "implementation": "GPU dynamic graph insertion versus CPU multi-threaded hierarchical insertion",
        "architecture_groups": {
            "cpu-hierarchical": ["FAISS-HNSW", "MIRAGE"],
            "gpu-graph-insertion": ["Naive", "ChunkSplice", "cuVS-CAGRA"],
        },
        "gpu_execution_requested": device.type == "cuda",
        "scope": "Simulator latency measures this implementation only; it does not estimate "
                 "external CPU or GPU performance. CPU debug runs do not measure GPU insertion. "
                 "Native and simulated results must be distinguished.",
        "method_definitions": {
            "Naive": "Existing PyTorch snapshot-insertion chunk-hole model.",
            "ChunkSplice": "Existing PyTorch budgeted intra-chunk graph repair.",
            "FAISS-HNSW": "Native CPU IndexHNSWFlat.add, or explicitly labeled exact fallback.",
            "cuVS-CAGRA": "Native cuVS build/extend/search, or Naive chunk-hole simulator.",
            "MIRAGE": "CPU-only multi-layer MIRAGE-ANNS: NN-Descent base and RNG-pruned add; or exact CPU reference.",
        },
        "timing": {
            "statistic": "median wall milliseconds across independent stream replays",
            "repeats": a.repeats,
            "warmup": "one complete untimed stream replay using fresh indexes",
            "repeated_updates": "each measured replay builds fresh indexes; each batch is added once",
            "includes": "adapter validation, allocations, host/device transfers and insertion",
            "excludes": "initial build, data/query generation, exact truth and metric computation",
            "cuda_synchronization": "PyTorch and native CuPy streams before and after operations",
            "ordering": "method order rotated by stage and replay",
            "qps": "mixed-cohort batch size divided by median end-to-end query seconds",
            "summary_update_ms": "median total update time per replay divided by update count",
            "summary_recall_qps": "final-stage mixed recall and QPS",
        },
        "controls": {
            "same_ordered_data_queries_truth": True,
            "graph_entries": "seeded by active size; old policy uses fixed seeded base entries; "
                             "native CPU/GPU indexes use their own traversal; exact fallbacks have no graph",
            "optional_backend_policy": a.external_backend,
        },
        "stages": [],
    })
    # These diagnostics belong to the original graph-only experiment.
    report.pop("diagnostic_definitions", None)
    report["query_cohorts"].update({
        "comparison": "all methods share queries and exact truth at each stage",
        "latest_chunk": "not separately evaluated in external mode",
        "latest_chunk_count": 0,
        "stage_zero": "not evaluated in external mode",
    })
    report["dataset"] = dataset_info
    if heldout_queries is not None:
        report["query_cohorts"] = {
            "comparison": "shared held-out queries and exact per-prefix truth",
            "distribution": dataset_info["metric"], "old_count": 0, "inserted_count": 0,
            "latest_chunk_count": 0, "source_age_cohorts": "not defined for held-out academic queries",
        }
    x_cpu = x.cpu()
    checkpoints = []
    for stage in range(1, a.chunks + 1):
        end = a.base + stage * a.chunk_size
        queries = heldout_queries
        if queries is None:
            queries, _ = query_set(x_cpu, centers, stage * a.chunk_size, a, gen)
        queries = queries.to(device)
        truth = exact_knn(queries, x[:end], 10, a.distance_batch)
        checkpoints.append((queries, truth))
        report["stages"].append({
            "stage": stage, "n": end,
            "methods": {name: {"update_samples_ms": [], "query_samples_ms": [],
                                "recall10_samples": []} for name in methods},
        })

    backend_identities = {}
    for trial in range(-1, a.repeats):
        indexes = create_indexes(a, device, methods)
        for index in indexes.values():
            index.build(x[:a.base])
            index.synchronize()
        for offset, (queries, truth) in enumerate(checkpoints):
            start = a.base + offset * a.chunk_size
            end = start + a.chunk_size
            shift = (offset + trial + 1) % len(methods)
            for name in methods[shift:] + methods[:shift]:
                index = indexes[name]

                def update():
                    index.add(x[start:end])

                _, update_ms = _timed(index, update)
                prediction, query_ms = _timed(index, lambda: index.search(queries, 10))
                if index.ntotal != end:
                    raise AssertionError("index did not append exactly one update batch")
                metadata = index.metadata()
                if a.external_backend == "native" and name in EXTERNAL_METHODS:
                    if not native_row(name, {"metadata": metadata, **metadata}):
                        raise RuntimeError("native backend returned invalid provenance: " + name)
                identity = (metadata["backend"], metadata.get("version"), metadata["simulated"],
                            metadata["effective_device"])
                if name in backend_identities and backend_identities[name] != identity:
                    raise RuntimeError("backend changed between replays: " + name)
                backend_identities[name] = identity
                if trial < 0:
                    continue
                row = report["stages"][offset]["methods"][name]
                row["update_samples_ms"].append(update_ms)
                row["query_samples_ms"].append(query_ms)
                row["recall10_samples"].append(recall(prediction, truth))
                row["metadata"] = metadata
        del indexes

    print("stage method                SIM device backend                          Update Latency (ms) Mixed Recall@10 QPS",
          flush=True)
    for stage in report["stages"]:
        for name, record in stage["methods"].items():
            record.update({
                "update_ms": statistics.median(record["update_samples_ms"]),
                "update_sample_summary": sample_summary(record["update_samples_ms"]),
                "query_ms": statistics.median(record["query_samples_ms"]),
                "recall10": statistics.mean(record["recall10_samples"]),
                "query_sample_summary": sample_summary(record["query_samples_ms"]),
                "backend": record["metadata"]["backend"],
                "simulated": record["metadata"]["simulated"],
            })
            record["qps"] = a.queries * 1000.0 / max(record["query_ms"], 1e-12)
            sim_col = "YES" if record["simulated"] else "NO "
            print(f"{stage['stage']:>5} {name:<21} {sim_col} {record['metadata']['effective_device']:<6} {record['backend']:<32} "
                  f"{record['update_ms']:>19.3f} {record['recall10']:>15.4f} {record['qps']:>10.1f}",
                  flush=True)
    report["summary"] = {}
    for name in methods:
        totals = [sum(stage["methods"][name]["update_samples_ms"][trial]
                      for stage in report["stages"]) for trial in range(a.repeats)]
        last = report["stages"][-1]["methods"][name]
        report["summary"][name] = {
            "update_ms": statistics.median(totals) / a.chunks,
            "total_update_samples_ms": totals,
            "recall10": last["recall10"], "qps": last["qps"],
            "backend": last["backend"], "simulated": last["simulated"],
            "metadata": last["metadata"],
        }
    report["native_external_comparison_complete"] = (
        native_rows_complete(report["summary"])
        and all(native_rows_complete(stage["methods"]) for stage in report["stages"]))
    report["simulated"] = any(report["summary"][name]["simulated"]
                              for name in methods if name in EXTERNAL_METHODS)
    report["simulated_scope"] = "external backends only; Naive/ChunkSplice remain PyTorch models"
    report["benchmark_completed"] = True
    return finish_report(report, device)
