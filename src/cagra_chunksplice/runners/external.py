"""Comparative incremental benchmark with explicit native/simulator provenance."""
from __future__ import annotations

import statistics
import time

from cagra_chunksplice.baselines import CuVSCAGRA, FaissHNSW, GraphBaseline, MirageChunkSplice
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.data import generate_data, query_set
from cagra_chunksplice.metrics import recall
from cagra_chunksplice.reporting import experiment_report, finish_report
from cagra_chunksplice.timing import sample_summary


METHODS = ("Naive", "ChunkSplice", "FAISS-HNSW", "cuVS-CAGRA", "MIRAGE-ChunkSplice")


def create_indexes(a, device):
    """Return fresh indexes; optional libraries are loaded only during build."""
    return {
        "Naive": GraphBaseline(a, "Naive", device),
        "ChunkSplice": GraphBaseline(a, "ChunkSplice", device),
        "FAISS-HNSW": FaissHNSW(device, a.external_backend, a.hnsw_m,
                                a.hnsw_ef_construction, a.ef),
        "cuVS-CAGRA": CuVSCAGRA(a, device, a.external_backend),
        "MIRAGE-ChunkSplice": MirageChunkSplice(a, device, a.mirage_buffer_capacity),
    }


def _timed(index, operation):
    index.synchronize()
    start = time.perf_counter()
    value = operation()
    index.synchronize()
    return value, (time.perf_counter() - start) * 1000.0


def benchmark_external(a, device):
    report = experiment_report(a, device, "external", list(METHODS))
    report.update({
        "implementation": "Optional native external indexes and labeled PyTorch simulators",
        "scope": "Simulator latency measures this implementation only; it does not estimate "
                 "external GPU performance. Native and simulated results must be distinguished.",
        "method_definitions": {
            "Naive": "Existing PyTorch snapshot-insertion chunk-hole model.",
            "ChunkSplice": "Existing PyTorch budgeted intra-chunk graph repair.",
            "FAISS-HNSW": "Native CPU IndexHNSWFlat.add, or explicitly labeled exact fallback.",
            "cuVS-CAGRA": "Native cuVS build/extend/search, or Naive chunk-hole simulator.",
            "MIRAGE-ChunkSplice": "MIRAGE-inspired buffer simulator with synchronous ChunkSplice merges.",
        },
        "timing": {
            "statistic": "median wall milliseconds across independent stream replays",
            "repeats": a.repeats,
            "warmup": "one complete untimed stream replay using fresh indexes",
            "repeated_updates": "each measured replay builds fresh indexes; each batch is added once",
            "includes": "adapter validation, allocations, transfers, insertion and synchronous merges",
            "excludes": "initial build, data/query generation, exact truth and metric computation",
            "cuda_synchronization": "PyTorch and native CuPy streams before and after operations",
            "final_buffer_drain": "included in the last update latency before final queries",
            "ordering": "method order rotated by stage and replay",
            "qps": "mixed-cohort batch size divided by median end-to-end query seconds",
            "summary_update_ms": "median total update time per replay divided by update count",
            "summary_recall_qps": "final-stage mixed recall and QPS",
        },
        "controls": {
            "same_ordered_data_queries_truth": True,
            "graph_entries": "seeded by active size; old policy uses fixed seeded base entries; "
                             "MIRAGE graph uses committed size; native indexes use their own traversal",
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
    x, centers, gen = generate_data(a, device)
    x_cpu = x.cpu()
    checkpoints = []
    for stage in range(1, a.chunks + 1):
        end = a.base + stage * a.chunk_size
        queries, _ = query_set(x_cpu, centers, stage * a.chunk_size, a, gen)
        queries = queries.to(device)
        truth = exact_knn(queries, x[:end], 10, a.distance_batch)
        checkpoints.append((queries, truth))
        report["stages"].append({
            "stage": stage, "n": end,
            "methods": {name: {"update_samples_ms": [], "query_samples_ms": [],
                                "recall10_samples": []} for name in METHODS},
        })

    backend_identities = {}
    for trial in range(-1, a.repeats):
        indexes = create_indexes(a, device)
        for index in indexes.values():
            index.build(x[:a.base])
            index.synchronize()
        for offset, (queries, truth) in enumerate(checkpoints):
            start = a.base + offset * a.chunk_size
            end = start + a.chunk_size
            shift = (offset + trial + 1) % len(METHODS)
            for name in METHODS[shift:] + METHODS[:shift]:
                index = indexes[name]

                def update():
                    index.add(x[start:end])
                    if offset == a.chunks - 1 and isinstance(index, MirageChunkSplice):
                        index.flush()

                _, update_ms = _timed(index, update)
                prediction, query_ms = _timed(index, lambda: index.search(queries, 10))
                if index.ntotal != end:
                    raise AssertionError("index did not append exactly one update batch")
                if trial < 0:
                    continue
                metadata = index.metadata()
                identity = (metadata["backend"], metadata.get("version"))
                if name in backend_identities and backend_identities[name] != identity:
                    raise RuntimeError("backend changed between replays: " + name)
                backend_identities[name] = identity
                row = report["stages"][offset]["methods"][name]
                row["update_samples_ms"].append(update_ms)
                row["query_samples_ms"].append(query_ms)
                row["recall10_samples"].append(recall(prediction, truth))
                row["metadata"] = metadata
        del indexes

    print("stage method                SIM backend                          Update Latency (ms) Mixed Recall@10 QPS",
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
            print(f"{stage['stage']:>5} {name:<21} {sim_col} {record['backend']:<32} "
                  f"{record['update_ms']:>19.3f} {record['recall10']:>15.4f} {record['qps']:>10.1f}",
                  flush=True)
    report["summary"] = {}
    for name in METHODS:
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
    report["benchmark_completed"] = True
    return finish_report(report, device)
