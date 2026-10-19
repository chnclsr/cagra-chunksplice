"""Synchronized wall-clock measurements and sample summaries."""
from __future__ import annotations

import statistics
import time
import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measured(fn, device, repeats):
    """Median synchronized wall time includes Python launch/allocation overhead."""
    # Same-shape warmup also applies to sweep/stream; fn must not mutate inputs.
    fn()
    samples = []
    result = None
    for _ in range(repeats):
        result = None
        synchronize(device)
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, statistics.median(samples), samples


def sample_summary(samples):
    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples)
    return {"min_ms": min(samples), "max_ms": max(samples),
            "stdev_ms": stdev, "coefficient_of_variation": stdev / mean if mean else 0.0}
