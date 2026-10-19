# Development and migration

## Module ownership

`src/cagra_chunksplice` is the installable package. `core` owns distances and
candidates; `graph` owns construction and invariants; `algorithms` owns insertion
and repair; `search` owns traversal. `data`, `metrics`, `timing`, `evaluation` and
`reporting` support the experiment `runners`.

`experiments/` holds entry scripts rather than algorithm copies. `tests/` uses
standard-library unittest. Packaged `selftest` checks work without a checkout.

Algorithm operation order, validation, distance formulas, RNG pruning, random
generation and timing regions are preserved. No algorithmic speedup is claimed.

## Install and test

```console
python -m pip install -e .
python -m unittest discover -s tests -v
python -m cagra_chunksplice --device cpu --self-test-only
python -m cagra_chunksplice --device cuda --self-test-only
```

The CUDA unit check is skipped only when CUDA is unavailable; CPU mechanism and
experiment tests must run. Explicit CUDA CLI requests still fail without CUDA.
PyTorch import failures are test failures.

```console
python -m pip install build
python -m build
```

Install the wheel in a fresh environment, then check `cagra-chunksplice --help`
and `python -m cagra_chunksplice --device cpu --self-test-only` outside the checkout.

## CLI and Python migration

Old: `python cagra_mirage_prototype.py ...`

New: `python -m cagra_chunksplice ...` or `cagra-chunksplice ...`

All old options and validation remain. Python callers import the owning module:

```python
from cagra_chunksplice.algorithms.chunk_splice import chunk_splice_extend
from cagra_chunksplice.core import exact_knn
from cagra_chunksplice.graph import build_graph, validate_graph
```

The compatibility wrapper preserves checkout CLI execution, not every old module
global. Install before running `experiments/` scripts. Default output now lives
under `results/` relative to the caller's working directory. Success and error
reports create missing parent directories.

## Source provenance

The pre-refactor working source SHA-256 was `8e8c782fffbb87d6cc1ad532c42a3a14fbf02baed0ba467bcbe9e8a0230dad70`.
Historical sweep/stream outputs have their own digests and need not match it.

New reports retain schema version 2 and add `source_files_sha256`, mapping
package-relative Python paths to SHA-256 hashes. `source_sha256` hashes the
UTF-8 canonical JSON mapping (`sort_keys=True`, `separators=(",", ":")`);
`source_sha256_scope` states this definition. Algorithms, CLI, evaluation,
runners and self-checks are all covered.

Historical artifact bytes are unchanged. Their parameters, embedded paths,
timestamps and digests are not rewritten to match the new layout.

The previous README mixed multiple result generations. [results.md](results.md)
uses exact checked-in JSON values and per-artifact digests. The architecture
SVG moved to `docs/assets/` without content changes.
