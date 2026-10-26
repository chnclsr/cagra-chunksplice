"""Experiment provenance and JSON persistence."""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path


def experiment_report(a, device, experiment, methods):
    import torch
    from cagra_chunksplice.timing import synchronize

    torch.manual_seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "schema_version": 2, "status": "ok", "experiment": experiment,
        "implementation": "PyTorch CAGRA-hole simulation / MIRAGE-inspired",
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **source_provenance(),
        "environment": {
            "python": sys.version, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "dtype": "float32", "distance": "squared_l2",
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "parameters": {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(a).items()},
        "methods": methods,
        "method_definitions": {
            "Rebuild-kNN": "Global unpruned exact kNN, degree capped.",
            "Rebuild-RNG": "Global exact kNN pool; greedy directed RNG; reverse proposals; RNG re-pruning; no backfill.",
            "incremental_initialization": "Shared unpruned exact kNN base graph; update timings exclude initial build.",
            "insertion_candidates": getattr(a, "insertion_mode", "exact"),
            "insertion_entries": "Fixed seeded original-base anchors; no exact fallback in beam/bounded modes; bounded additionally restricts reverse proposals to linked old rows.",
        },
        "diagnostic_definitions": {
            "scc": "Exact iterative Kosaraju; largest fraction over all active vertices.",
            "hubness": "Population in-degree Gini and skewness, including zero-degree nodes.",
            "bridge": "Directed edge across insertion epochs; original base is epoch 0.",
            "edge_length": "Euclidean L2, including unit-normalized vectors in angular runs.",
            "decay_reference": "Edges: initial base graph; bridges: first nonempty evaluated snapshot; previous checkpoint retained separately. No continuous-lifetime claim.",
            "age_strata": "Query source: base and early/middle/recent insertion-epoch thirds; exact truth over all active nodes.",
            "high_lid": "Isotropic Gaussian in all requested dimensions; intrinsic dimension is not empirically estimated.",
        },
        "timing": {
            "statistic": "median wall milliseconds", "repeats": a.repeats,
            "warmup": "one full untimed call of each measured operation at its current shape",
            "cuda_synchronization": "selected device before and after each timed call",
            "includes": "Python launches, allocations, update/reverse/repair work",
            "excludes": "data generation/transfers, exact truth, graph validation/diagnostics",
            "qps": "mixed-cohort batched throughput; latest-chunk probes timed separately",
            "ordering": "method order rotated by stage or sweep configuration",
            "repeated_updates": "each repetition starts from the same immutable graph snapshot",
        },
        "query_cohorts": {
            "distribution": getattr(a, "query_distribution", "in-distribution"),
            "ood_shift": getattr(a, "ood_shift", 4.0),
            "angular_normalization": "Both corpus and queries are unit normalized; L2 and cosine rankings agree.",
            "old": "noisy original base vectors",
            "inserted": "all inserted vectors/centers so far; new_query_recall10 is this cohort",
            "latest_chunk": "independent probes of newest chunk; separate from cumulative inserted",
            "old_count": a.queries // 2, "inserted_count": a.queries - a.queries // 2,
            "latest_chunk_count": a.queries - a.queries // 2,
            "drift_noise_std": 0.08, "iid_noise_std": 0.10,
            "comparison": "same-stage methods share queries, truth and search entries",
            "stage_zero": "base-only queries use a different distribution from later stages",
        },
        "scope": "Mechanism simulation, not cuVS CAGRA or full MIRAGE; "
                 "no universal speed/recall guarantee. Exact-kNN ties may vary by device.",
    }


def finish_report(report, device):
    import torch
    from cagra_chunksplice.timing import synchronize

    synchronize(device)
    report["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["peak_cuda_allocated_mb"] = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None)
    return report


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def source_provenance():
    """Hash all package sources by relative path, independent of checkout location."""
    package = Path(__file__).resolve().parent
    files = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(package.rglob("*.py"))}
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "source_sha256": hashlib.sha256(manifest).hexdigest(),
        "source_sha256_scope": "sha256 of canonical JSON source_files_sha256 manifest",
        "source_files_sha256": files,
    }
