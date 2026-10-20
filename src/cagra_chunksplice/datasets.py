"""Cached ANN-Benchmarks datasets with explicit synthetic substitutes."""
from __future__ import annotations

import hashlib
import http.client
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import torch

from .data import generate_data

DATASETS = {
    "sift1m": ("sift-128-euclidean", 128, 1000000, "euclidean"),
    "glove-100": ("glove-100-angular", 100, 1183514, "angular"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path, rows, queries, dimension, metric):
    import h5py
    with h5py.File(path, "r") as source:
        train, test = source["train"], source["test"]
        if (len(train.shape) != 2 or len(test.shape) != 2
                or train.shape[1] != dimension or test.shape[1] != dimension):
            raise ValueError("dataset matrices have unexpected dimensions")
        if len(train) < rows or len(test) < queries:
            raise ValueError("dataset is smaller than the requested corpus/query count")
        # ANN-Benchmarks files declare their metric; never assume it.
        actual_metric = source.attrs.get("distance")
        if actual_metric is None:
            raise ValueError("dataset does not declare its distance metric")
        if isinstance(actual_metric, bytes):
            actual_metric = actual_metric.decode()
        if actual_metric != metric:
            raise ValueError("dataset distance does not match its registered metric")
        x = torch.from_numpy(train[:rows].astype("float32"))
        q = torch.from_numpy(test[:queries].astype("float32"))
        counts = (int(len(train)), int(len(test)))
    if not bool(torch.isfinite(x).all() and torch.isfinite(q).all()):
        raise ValueError("dataset contains nonfinite vectors")
    if metric == "angular":
        if bool((x.norm(dim=1) == 0).any() or (q.norm(dim=1) == 0).any()):
            raise ValueError("angular dataset contains zero vectors")
        x = torch.nn.functional.normalize(x, dim=1)
        q = torch.nn.functional.normalize(q, dim=1)
    return x, q, counts


def _download(url, path, validate):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        try:
            with os.fdopen(fd, "wb") as target, urllib.request.urlopen(url, timeout=60) as response:
                expected = response.headers.get("Content-Length")
                received = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    target.write(block)
                    received += len(block)
                if expected is not None and received != int(expected):
                    raise OSError("incomplete dataset download")
        except http.client.HTTPException as exc:
            # e.g. IncompleteRead; report like other transfer failures.
            raise OSError("dataset download failed: " + repr(exc)) from exc
        validate(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_external_data(a, device):
    """Return corpus, held-out queries (or None), synthetic centers/RNG, provenance.

    Academic corpora are the first ``rows`` train vectors in file order,
    inserted in a deterministic seeded permutation; subsets are therefore file
    prefixes, not uniform random samples of the full train set. Truth must be recomputed for each visible prefix; full-dataset neighbor IDs
    in the HDF5 file cannot be reused after partitioning.
    """
    name = getattr(a, "dataset", "synthetic")
    if name == "synthetic":
        x, centers, gen = generate_data(a, device)
        return x, None, centers, gen, {
            "requested": name, "actual": name, "synthetic": True,
            "rows": len(x), "dimension": a.dim, "seed": a.seed,
            "metric": "angular" if a.query_distribution == "angular" else "squared_l2",
        }
    slug, dimension, full_rows, metric = DATASETS[name]
    rows = a.base + a.chunks * a.chunk_size
    path = Path(a.dataset_cache).expanduser() / (slug + ".hdf5")
    url = "https://ann-benchmarks.com/" + slug + ".hdf5"
    validator = lambda candidate: _read(candidate, rows, a.queries, dimension, metric)
    reason = None
    if not path.exists() and a.download_datasets:
        # Fail before a large download if the reader dependency is missing.
        import h5py  # noqa: F401
        try:
            _download(url, path, validator)
        except (OSError, urllib.error.URLError) as exc:
            reason = str(exc)
    if path.exists():
        # Corrupt caches and dependency errors are errors, never synthetic data.
        x, q, counts = validator(path)
        a.dim = dimension
        gen = torch.Generator().manual_seed(a.seed)
        x = x[torch.randperm(len(x), generator=gen)]
        info = {
            "requested": name, "actual": slug, "synthetic": False,
            "cache_path": str(path.resolve()), "source_url": url,
            "sha256": file_sha256(path), "rows": rows, "dimension": dimension,
            "available_train_rows": counts[0], "available_query_rows": counts[1],
            "full_academic_corpus": rows == full_rows and counts[0] == full_rows,
            "subset": rows != full_rows, "seed": a.seed,
            "subset_selection": f"first {rows} train vectors in file order",
            "insertion_order": "seeded permutation of the selected rows",
            "queries_selected": f"first {a.queries} test vectors in file order",
            "metric": metric, "normalization": "unit_l2" if metric == "angular" else None,
            "query_protocol": "held-out test queries; exact truth over each visible prefix",
        }
        return x.to(device), q.to(device), None, gen, info
    if not a.allow_synthetic_fallback:
        raise FileNotFoundError(
            f"Dataset missing: {path}. Use --download-datasets or explicitly "
            "--allow-synthetic-fallback. " + (reason or ""))
    a.dim = dimension
    # Preserve requested scale, dimension and metric, never the academic label.
    from copy import copy
    synthetic = copy(a)
    synthetic.workload = "iid"
    synthetic.query_distribution = "angular" if metric == "angular" else "in-distribution"
    x, centers, gen = generate_data(synthetic, device)
    q = torch.randn(a.queries, dimension, generator=gen)
    if metric == "angular":
        q = torch.nn.functional.normalize(q, dim=1)
    return x, q.to(device), centers, gen, {
        "requested": name, "actual": "synthetic-scaled-" + name, "synthetic": True,
        "fallback_reason": reason or "dataset not cached; downloading disabled",
        "rows": rows, "reference_train_rows": full_rows, "dimension": dimension,
        "seed": a.seed, "metric": metric, "full_academic_corpus": False,
        "query_protocol": "independent synthetic queries; not academic measurements",
    }
