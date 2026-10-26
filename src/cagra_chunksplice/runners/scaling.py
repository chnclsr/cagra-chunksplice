"""Update latency versus index size: incremental insertion against full rebuilds.

Each configuration inserts one chunk of fixed size b into a base of n vectors.
Incremental methods (ChunkSplice, Naive, native cuVS-CAGRA extend) touch the
new rows plus bounded reverse work; full rebuilds reconstruct all n + b rows.
The PyTorch incremental methods reuse one preallocated ``out=`` result buffer
per base, so no per-update allocation is timed; with
``--insertion-mode bounded`` the reverse merge additionally rewrites at most
b * degree linked old rows. Update latency is also the freshness delay: the
time from batch arrival until the batch is searchable. Each timed update
still includes one O(n * degree) memcpy of the initial graph into that buffer
(see ``update_memcpy`` in the report). Fitted exponents and
crossovers are measured, never interpolated beyond the recorded sizes.
"""
from __future__ import annotations

import copy
import math
import statistics
import time

import torch
from cagra_chunksplice.algorithms.chunk_splice import chunk_splice_extend
from cagra_chunksplice.algorithms.naive import naive_extend
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.data import evaluation_set, generate_data
from cagra_chunksplice.evaluation import evaluate_graph
from cagra_chunksplice.graph import build_graph, build_rng_graph, validate_graph
from cagra_chunksplice.metrics import recall
from cagra_chunksplice.reporting import experiment_report, finish_report
from cagra_chunksplice.timing import measured, sample_summary


INCREMENTAL = ("ChunkSplice", "Naive", "cuVS-CAGRA-extend")


def name_is_rebuild(name):
    return name.startswith("Rebuild") or name.endswith("rebuild")


def log_log_slope(sizes, values):
    """Least-squares exponent p in latency ~ n^p over the measured sizes."""
    if len(sizes) < 2:
        return None
    xs = [math.log(n) for n in sizes]
    ys = [math.log(v) for v in values]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else None


def _cagra_samples(local, device, x, start, end, rebuild):
    """Fresh native index per replay: rebuild times build(n + b); extend times add(b)."""
    from cagra_chunksplice.baselines import CuVSCAGRA
    samples, index = [], None
    for replay in range(local.repeats + 1):
        index = CuVSCAGRA(local, device, local.external_backend)
        if not rebuild:
            index.build(x[:start])
        index.synchronize()
        began = time.perf_counter()
        if rebuild:
            index.build(x[:end])
        else:
            index.add(x[start:end])
        index.synchronize()
        if replay:  # Replay 0 is the untimed warmup.
            samples.append((time.perf_counter() - began) * 1000.0)
    if index.ntotal != end:
        raise RuntimeError(f"cuVS-CAGRA holds {index.ntotal} vectors, expected {end}")
    return index, statistics.median(samples), samples


def run_scaling(a, device):
    names = ["ChunkSplice", "Naive", "Rebuild-kNN"]
    if a.scaling_rebuild_rng:
        names.append("Rebuild-RNG")
    if a.scaling_cagra:
        names += ["cuVS-CAGRA-extend", "cuVS-CAGRA-rebuild"]
    report = experiment_report(a, device, "update_scaling", names)
    b = a.chunk_size
    report.update({
        "scope": "One fixed-size chunk inserted into growing bases. PyTorch methods are "
                 "mechanism models; cuVS-CAGRA rows are native only when metadata says "
                 "simulated: false. Exponents and crossovers cover measured sizes only.",
        "freshness_definition": "update latency: batch arrival until the batch is searchable",
        "scaling_bases": list(a.scaling_bases), "effective_chunk_size": b,
        "insertion_mode": getattr(a, "insertion_mode", "exact"),
        "update_buffer": ("ChunkSplice/Naive reuse one preallocated out= buffer "
                          "per base: no per-update allocation is timed, and with "
                          "insertion_mode=bounded the reverse merge rewrites at "
                          "most chunk_size * degree linked old rows"),
        "update_memcpy": ("Each timed ChunkSplice/Naive update copies the shared "
                          "initial graph into the out= buffer (one contiguous "
                          "O(n * degree) memcpy, included in update_ms). It is "
                          "not skipped here because the initial graph must stay "
                          "pristine across repeats; the copy is skipped only when "
                          "out[:start] aliases old_graph, which mutates old rows "
                          "in place. Bounded-mode compute is independent of n, "
                          "but measured update_ms still contains this memcpy."),
        "not_applicable_parameters": ["chunks", "local_cap", "candidate_cap", "hops",
                                      "refine_rounds", "sweep_total", "sweep_chunk_sizes"],
        "points": [],
    })
    print(f"SCALING: bases={list(a.scaling_bases)}, chunk_size={b}, repeats={a.repeats}", flush=True)
    print("      n method              update_ms  ms/new_vec recall10 new_R10", flush=True)
    for position, n in enumerate(a.scaling_bases):
        local = copy.copy(a)
        local.base, local.chunks = n, 1
        x, centers, gen = generate_data(local, device)
        x_cpu = x.cpu()
        start, end = n, n + b
        initial = build_graph(x[:start], local)  # Shared and untimed for incremental methods.
        # One caller-owned result buffer per base: timed updates copy the
        # snapshot into it once and never allocate; the shared `initial`
        # tensor itself is never mutated.
        scratch = torch.empty((end, initial.shape[1]), dtype=initial.dtype,
                              device=device)
        queries, latest, old_count = evaluation_set(x_cpu, centers, b, b, local, gen, device)
        truth = exact_knn(queries, x, 10, a.distance_batch)
        latest_truth = exact_knn(latest, x, 10, a.distance_batch)
        entries = torch.randperm(end, generator=gen)[:a.ef].to(device)
        point = {"n": n, "active": end, "methods": {}}
        order = names[position % len(names):] + names[:position % len(names)]
        point["method_order"] = order
        for name in order:
            if name.startswith("cuVS-CAGRA"):
                index, update_ms, samples = _cagra_samples(
                    local, device, x, start, end, name.endswith("rebuild"))
                predicted = index.search(queries, 10)
                index.synchronize()
                row = {"recall10": recall(predicted, truth),
                       "old_query_recall10": recall(predicted[:old_count], truth[:old_count]),
                       "new_query_recall10": recall(predicted[old_count:], truth[old_count:]),
                       "latest_chunk_query_recall10": recall(index.search(latest, 10), latest_truth),
                       "metadata": index.metadata()}
            else:
                if name == "ChunkSplice":
                    fn = lambda: chunk_splice_extend(x, initial, start, end, local,
                                                     out=scratch)
                elif name == "Naive":
                    fn = lambda: naive_extend(x, initial, start, end, local,
                                              out=scratch)
                elif name == "Rebuild-kNN":
                    fn = lambda: build_graph(x, local)
                else:
                    fn = lambda: build_rng_graph(x, local)
                graph, update_ms, samples = measured(fn, device, a.repeats)
                validate_graph(graph, end)
                row = evaluate_graph(x, graph, queries, truth, latest, latest_truth,
                                     entries, old_count, local)
            row.update({"update_ms": update_ms, "update_samples_ms": samples,
                        "update_sample_summary": sample_summary(samples),
                        "update_ms_per_new_vector": update_ms / b,
                        "incremental": name in INCREMENTAL})
            point["methods"][name] = row
            print(f"{n:>7} {name:<19} {update_ms:>9.3f} {update_ms / b:>11.5f} "
                  f"{row['recall10']:>8.4f} {row['new_query_recall10']:>7.4f}", flush=True)
        report["points"].append(point)
    sizes = [p["n"] for p in report["points"]]
    report["scaling_exponent"] = {
        name: log_log_slope(sizes, [p["methods"][name]["update_ms"] for p in report["points"]])
        for name in names}
    report["speedup_vs_full_rebuild"] = {
        rebuild: {name: [p["methods"][rebuild]["update_ms"] / p["methods"][name]["update_ms"]
                         for p in report["points"]]
                  for name in names if name in INCREMENTAL}
        for rebuild in names if name_is_rebuild(rebuild)}
    report["first_measured_crossover_n"] = {
        rebuild: {name: next((p["n"] for p in report["points"]
                              if p["methods"][name]["update_ms"] < p["methods"][rebuild]["update_ms"]), None)
                  for name in names if name in INCREMENTAL}
        for rebuild in names if name_is_rebuild(rebuild)}
    report["timing"]["cuvs"] = ("fresh native index per replay after one untimed warmup replay; "
                                "extend excludes the base build, rebuild times build(n + b)")
    return finish_report(report, device)
