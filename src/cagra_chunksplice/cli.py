"""CAGRA chunk-hole simulation and MIRAGE-inspired graph repair.

Default modes are PyTorch mechanism studies. External mode adds optional native
CPU FAISS-HNSW/MIRAGE and GPU cuVS adapters and explicitly labeled portable simulators.
Use --device cpu for debugging, --mode sweep for controlled chunk partitions,
and --mode stream for checkpointed updates. CUDA is the default device.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import sys
from pathlib import Path


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="CPU is an explicit debug option; no silent CUDA fallback.")
    p.add_argument("--base", type=int, default=2048)
    p.add_argument("--chunks", type=int, default=None,
                   help="Number of updates; default 4, or 100 in stress mode.")
    p.add_argument("--chunk-size", type=int, default=256,
                   help="Update size; in sweep, fixed data-generation block size.")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--queries", type=int, default=256)
    p.add_argument("--degree", type=int, default=24)
    p.add_argument("--ef", type=int, default=64)
    p.add_argument("--search-width", type=int, default=4)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--query-batch", type=int, default=64)
    p.add_argument("--distance-batch", type=int, default=256)
    p.add_argument("--local-cap", type=int, default=768)
    p.add_argument("--candidate-cap", type=int, default=64)
    p.add_argument("--hops", type=int, choices=(1, 2), default=2)
    p.add_argument("--refine-rounds", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="LocalRNG occlusion slack (>=1); 1 is plain greedy RNG.")
    p.add_argument("--route", type=int, default=0,
                   help="LocalRNG slots reserved per row for old-cohort routing edges.")
    p.add_argument("--cohort-protect", action="store_true",
                   help="LocalRNG cohort-local occlusion: old/new nodes never occlude each other.")
    p.add_argument("--backfill", action="store_true",
                   help="LocalRNG fills empty slots with RNG-rejected candidates by distance.")
    p.add_argument("--splice-intra", type=int, default=6,
                   help="ChunkSplice slots per new row for same-chunk peers; rest stay new->old bridges.")
    p.add_argument("--splice-pool", type=int, default=24,
                   help="ChunkSplice nearest-peer pool that RNG picks the peer slots from.")
    p.add_argument("--splice-alpha", type=float, default=1.0,
                   help="ChunkSplice RNG occlusion slack among peers (>=1).")
    p.add_argument("--rebuild-pool", type=int, default=96,
                   help="Global Rebuild-RNG exact candidate pool per row, >= degree.")
    p.add_argument("--rebuild-alpha", type=float, default=1.0,
                   help="Rebuild-RNG occlusion slack; no backfill.")
    p.add_argument("--insertion-mode", choices=("exact", "beam"), default="exact",
                   help="New-to-old candidate source; beam has no exact fallback.")
    p.add_argument("--insertion-ef", type=int, default=48)
    p.add_argument("--insertion-search-width", type=int, default=4)
    p.add_argument("--insertion-steps", type=int, default=8)
    p.add_argument("--query-distribution", choices=("in-distribution", "ood", "angular"),
                   default="in-distribution")
    p.add_argument("--ood-shift", type=float, default=4.0,
                   help="OOD query translation magnitude along coordinate 0.")
    p.add_argument("--age-queries", type=int, default=32,
                   help="Queries per source-age stratum at stream checkpoints.")
    p.add_argument("--stream-rebuild", action="store_true",
                   help="Include both global baselines in stream mode (always on in stress mode).")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--entry-policy", choices=("all", "old"), default="all",
                   help="Common random starts over all active nodes; old is a stable-anchor control.")
    p.add_argument("--workload", choices=("drift", "iid", "high-lid"), default="drift",
                   help="drift: temporal dense microclusters; iid: Gaussian control; high-lid: full-dimensional isotropic Gaussian (use --dim 128 or higher).")
    p.add_argument("--mode", choices=("benchmark", "sweep", "stream", "stress", "external"), default="benchmark",
                   help="benchmark: standard run; sweep: chunk-size sweep; stream: checkpointed streaming; stress: streaming with global baselines (default 100 epochs); external: native/simulated index comparison.")
    p.add_argument("--external-backend", choices=("auto", "native", "simulated"), default="auto",
                   help="External indexes: auto selects native or labeled fallback; native requires libraries/hardware; simulated is portable.")
    p.add_argument("--dataset", choices=("synthetic", "sift1m", "glove-100"), default="synthetic")
    p.add_argument("--dataset-cache", type=Path, default=Path("~/.cache/cagra-chunksplice/datasets"))
    p.add_argument("--download-datasets", action="store_true")
    p.add_argument("--allow-synthetic-fallback", action="store_true",
                   help="Explicitly substitute missing academic data; always label it synthetic.")
    p.add_argument("--require-native-validation", action="store_true",
                   help="Fail unless every measured external backend passes the native Promotion Gate.")
    p.add_argument("--external-sweep", action="store_true",
                   help="Rebuild/replay each configuration and export measured Pareto curves.")
    p.add_argument("--mirage-s", type=int, default=32)
    p.add_argument("--mirage-r", type=int, default=4)
    p.add_argument("--mirage-iter", type=int, default=15)
    p.add_argument("--graph-ef-values", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--hnsw-m-values", type=int, nargs="+", default=[16, 32])
    p.add_argument("--hnsw-ef-search-values", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--mirage-s-values", type=int, nargs="+", default=[16, 32])
    p.add_argument("--mirage-r-values", type=int, nargs="+", default=[4, 8])
    p.add_argument("--mirage-iter-values", type=int, nargs="+", default=[10, 15])
    p.add_argument("--cagra-itopk-values", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--cagra-search-width-values", type=int, nargs="+", default=[1, 4])
    p.add_argument("--cagra-iterations-values", type=int, nargs="+", default=[8, 16])
    p.add_argument("--recall-targets", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--hnsw-ef-construction", type=int, default=80)
    p.add_argument("--sweep-chunk-sizes", type=int, nargs="+",
                   default=[64, 256, 1024, 2048],
                   help="Sweep partitions of one shared corpus, in nodes per update.")
    p.add_argument("--sweep-total", type=int, default=4096,
                   help="Fixed inserted-node count in sweep (replaces --chunks).")
    p.add_argument("--stream-eval-every", type=int, default=10,
                   help="Evaluate stream at first, last and every Nth update.")
    p.add_argument("--self-test-only", action="store_true")
    p.add_argument("--verify-effect", action="store_true",
                   help="Fail unless final recall erosion and recovery each exceed 0.02.")
    p.add_argument("--output", type=Path,
                   default=None)
    a = p.parse_args(argv)
    if a.mode != "external" and (a.dataset != "synthetic" or a.external_sweep or a.require_native_validation):
        p.error("datasets, external-sweep and require-native-validation require external mode")
    if a.dataset != "synthetic" and a.query_distribution != "in-distribution":
        p.error("academic datasets use their registered metric and held-out queries")
    if a.external_sweep and min(a.graph_ef_values) < a.search_width:
        p.error("graph search-width must not exceed any graph-ef-values budget")
    if a.require_native_validation and (a.external_backend != "native" or a.device != "cuda"):
        p.error("require-native-validation requires --external-backend native --device cuda")
    for key in ("mirage_s", "mirage_r", "mirage_iter"):
        if getattr(a, key) <= 0:
            p.error(key.replace("_", "-") + " must be positive")
    for key in ("graph_ef_values", "hnsw_m_values", "hnsw_ef_search_values",
                "mirage_s_values", "mirage_r_values", "mirage_iter_values",
                "cagra_itopk_values", "cagra_search_width_values", "cagra_iterations_values"):
        if any(v <= 0 for v in getattr(a, key)):
            p.error(key.replace("_", "-") + " must be positive")
    if any(not 0 < v <= 1 for v in a.recall_targets):
        p.error("recall-targets must be finite and in (0, 1]")
    if a.external_sweep:
        if any(not 10 <= ef <= a.base for ef in a.graph_ef_values):
            p.error("graph-ef-values must be in [10, base]")
        if any(ef < 10 or ef % 32 for ef in a.cagra_itopk_values):
            p.error("cagra-itopk-values must be multiples of 32 and >= 10")
        if max(a.cagra_search_width_values) > min(a.cagra_itopk_values):
            p.error("CAGRA search width cannot exceed itopk budget")
    if a.output is None:
        a.output = Path("results/external.json" if a.mode == "external"
                        else "results/cagra_mirage_results.json")
    if a.chunks is None:
        a.chunks = 100 if a.mode == "stress" else 4
    positive = ("base", "chunks", "chunk_size", "dim", "queries", "degree", "ef",
                "search_width", "steps", "query_batch", "distance_batch", "local_cap",
                "candidate_cap", "refine_rounds", "repeats", "sweep_total",
                "stream_eval_every", "rebuild_pool", "insertion_ef",
                "insertion_search_width", "insertion_steps", "age_queries",
                "hnsw_m", "hnsw_ef_construction")
    for key in positive:
        if getattr(a, key) <= 0:
            p.error("--" + key.replace("_", "-") + " must be positive")
    if a.rebuild_pool < a.degree:
        p.error("rebuild-pool must be >= degree")
    if a.insertion_mode == "beam" and (a.insertion_ef < a.degree or
                                       a.insertion_search_width > min(a.insertion_ef, a.base)):
        p.error("beam insertion requires insertion-ef >= degree and insertion-search-width <= min(insertion-ef, base)")
    if not a.rebuild_alpha >= 1 or a.rebuild_alpha == float("inf"):
        p.error("rebuild-alpha must be finite and >= 1")
    if not 0 <= a.ood_shift < float("inf"):
        p.error("ood-shift must be finite and nonnegative")
    if not 10 <= a.degree < a.base:
        p.error("require 10 <= degree < base for Recall@10")
    if not 10 <= a.ef <= a.base or a.search_width > a.ef:
        p.error("require 10 <= ef <= base and search-width <= ef")
    if a.queries < 2 or a.dim < 2:
        p.error("queries and dim must be >= 2")
    if a.mode == "benchmark" and a.local_cap <= a.chunk_size:
        p.error("local-cap must exceed chunk-size to include old neighbors")
    if any(b <= 0 for b in a.sweep_chunk_sizes):
        p.error("sweep-chunk-sizes must be positive")
    if a.mode == "sweep" and max(a.sweep_chunk_sizes) > a.sweep_total:
        p.error("sweep-total must be >= every sweep chunk size")
    if a.verify_effect and a.mode != "benchmark":
        p.error("verify-effect is supported only in benchmark mode")
    if a.candidate_cap < a.degree:
        p.error("candidate-cap must be >= degree")
    if not a.alpha >= 1.0 or a.alpha == float("inf"):
        p.error("alpha must be finite and >= 1")
    if not 1 <= a.splice_intra < a.degree or a.splice_pool <= 0:
        p.error("require 1 <= splice-intra < degree and positive splice-pool")
    if not a.splice_alpha >= 1.0 or a.splice_alpha == float("inf"):
        p.error("splice-alpha must be finite and >= 1")
    if not 0 <= a.route <= a.degree:
        p.error("require 0 <= route <= degree")
    if a.route and not a.cohort_protect:
        p.error("route requires --cohort-protect to define the old cohort")
    return a



def main(argv=None):
    a = arguments(argv)
    try:
        import torch
        from cagra_chunksplice.reporting import save_report
        from cagra_chunksplice.selftest import self_test
        from cagra_chunksplice.runners.benchmark import benchmark
        from cagra_chunksplice.runners.sweep import run_sweep
        from cagra_chunksplice.runners.stream import run_stream
        if a.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Install a CUDA-enabled PyTorch build "
                               "and compatible NVIDIA driver. --device cpu is debug only.")
        device = torch.device(a.device)
        with torch.inference_mode():
            self_test(device)
            if a.self_test_only:
                return 0
            if a.mode == "external":
                from cagra_chunksplice.runners.external import benchmark_external
                if a.external_sweep:
                    from cagra_chunksplice.runners.external_sweep import benchmark_external_sweep
                    report = benchmark_external_sweep(a, device)
                else:
                    report = benchmark_external(a, device)
            elif a.mode == "sweep":
                report = run_sweep(a, device)
            elif a.mode in ("stream", "stress"):
                report = run_stream(a, device)
            else:
                report = benchmark(a, device)
        if a.mode == "external":
            from cagra_chunksplice.promotion import validate_native_report
            if a.external_backend == "native" or a.require_native_validation:
                report["promotion_gate"] = validate_native_report(report)
        save_report(a.output, report)
        if a.mode == "external" and a.external_sweep:
            from cagra_chunksplice.pareto import curves_svg
            a.output.with_suffix(".pareto.svg").write_text(curves_svg(report["curves"]), encoding="utf-8")
        print("JSON: " + str(a.output.resolve()), flush=True)
        if a.verify_effect and a.mode == "benchmark" and not report.get("effect_demonstrated", False):
            print("Requested effect threshold not met; inspect measured results.",
                  file=sys.stderr)
            return 2
        return 0
    except (ImportError, OSError, RuntimeError, AssertionError, ValueError) as exc:
        error = {
            "schema_version": 2, "status": "runtime_error", "benchmark_completed": False,
            "mode": a.mode,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parameters": {k: str(v) if isinstance(v, Path) else v
                           for k, v in vars(a).items()},
            "error_type": type(exc).__name__, "error": str(exc),
            "python": sys.version, "requested_device": a.device,
            "torch_distribution_versions": [],
        }
        error["torch_distribution_versions"] = [
            d.version for d in importlib.metadata.distributions()
            if d.metadata.get("Name", "").lower() == "torch"]
        # Keep completed results intact; failures have a separate artifact.
        path = a.output.with_name(a.output.stem + ".error.json")
        from cagra_chunksplice.reporting import save_report
        save_report(path, error)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("No benchmark results claimed. Diagnostic JSON: " + str(path.resolve()),
              file=sys.stderr)
        return 1

