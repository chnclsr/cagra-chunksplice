# External index comparisons

External mode compares Naive, ChunkSplice, FAISS-HNSW, cuVS-CAGRA, and
MIRAGE-ChunkSplice on the same seeded insertion stream and mixed query cohorts.
It reports update latency in milliseconds, Mixed Recall@10, and queries per
second (QPS), with samples and backend provenance in JSON.

Install the package as described in the project README, then run a portable
comparison without FAISS, cuVS, CuPy, or CUDA:

```console
python experiments/benchmark_external.py --device cpu --external-backend simulated --base 128 --chunks 3 --chunk-size 16 --dim 16 --queries 32 --mirage-buffer-capacity 32 --output results/external_cpu.json
```

The equivalent installed entry point is
`python -m cagra_chunksplice --mode external ...`.
CPU and GPU timings describe different configurations and should be reported
separately.

## Backend selection

| Method | Native implementation | Portable implementation |
| --- | --- | --- |
| Naive | None | Existing PyTorch snapshot-insertion model. |
| ChunkSplice | None | Existing PyTorch budgeted graph repair. |
| FAISS-HNSW | CPU `IndexHNSWFlat`, incremental `add`. | Exact blocked PyTorch search with append-only storage. |
| cuVS-CAGRA | CUDA `cagra.build`, `extend`, and `search`. | Naive snapshot chunk-hole model with approximate graph search. |
| MIRAGE-ChunkSplice | None | Bounded pending buffer, exact buffer search, and synchronous ChunkSplice merges. |

`--external-backend auto` selects each external backend when its initial build
succeeds; otherwise it records the fallback and reason. `simulated` forces
portable implementations. `native` requires native FAISS and cuVS; missing
libraries or hardware cause a diagnostic error report. MIRAGE-ChunkSplice
remains a simulator in all modes. Backends never change during timed updates.
A failed native update fails the run rather than silently changing algorithms.

FAISS is optional. A CPU build can be installed separately with
`python -m pip install faiss-cpu`; see the
[FAISS installation reference](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss)
for supported configurations. HNSW is a CPU index: CUDA requests retain native
CPU HNSW, record `effective_device: cpu`, and include transfers in timing.
See [FAISS index types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
and [GPU implementations](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU).
The exact fallback is a functional reference, not an HNSW graph or performance
model. Its recall and timings must not be attributed to FAISS.

Native CAGRA uses the optional Python `cuvs.neighbors.cagra` bindings and CuPy,
rather than a command-line executable. Use a compatible CUDA environment
following the [cuVS installation guide](https://docs.nvidia.com/cuvs/user-guide/getting-started/).
The adapter supports the legacy extension returning an index and the newer
extension consuming concatenated old/new vectors plus `new_start_row`.
It retains caller-owned device data and synchronizes native work. The
[cuVS API](https://docs.nvidia.com/cuvs/api-reference/python-api-neighbors-cagra)
defines the native semantics. The portable model does not reproduce the
native kernels, graph optimization, or GPU throughput.

## Buffer simulation

MIRAGE-ChunkSplice appends each incoming small batch to a bounded buffer.
Queries search the committed ChunkSplice graph and scan the pending buffer,
then combine candidates by squared L2 distance. Pending vectors are immediately
visible and keep their insertion-order IDs.

At `--mirage-buffer-capacity` vectors, the simulator synchronously inserts the
buffer into the main graph using ChunkSplice and clears it. Oversized inputs
are split at the capacity boundary; one update can trigger multiple merges.
An explicit `flush()` drains a partial buffer. Reports include committed and
buffered vector counts, flush count, and merged-vector count. The runner drains
the final partial buffer inside the last update timer, before final queries.

This is an experimental buffer design inspired by the requested mixed
incremental/refinement comparison. It is not a reproduction of MIRAGE-ANNS or
a claim that its published algorithm uses these buffer mechanics. The
[upstream implementation](https://github.com/dsg-uwaterloo/mirage) is a C++
FAISS-based index; see also the [paper](https://doi.org/10.1145/3725325).
The simulator models neither GPU scheduling nor background merge overlap,
and uses real execution times without invented delays or calibrated estimates.

## Measurement protocol and interpretation

1. Generate one deterministic corpus and one old/inserted mixed query set per
   stage. Recompute exact Recall@10 truth over all visible vectors.
2. Warm up with one complete stream replay. Build fresh indexes for each
   measured replay (`--repeats`); add every batch exactly once.
3. Rotate method order by stage and replay. Synchronize PyTorch and native CuPy
   streams around wall-clock measurements.
4. Include adapter validation, memory allocation, transfers, insertion, and
   synchronous merges in update latency. Exclude initial builds, query
   generation, exact truth, and recall computation.
5. Report stage-wise median update/query times and mean recall across replays.
   QPS is mixed query count divided by median query time. The summary uses
   median total update time divided by update count, and final-stage recall/QPS.

Native indexes use their own construction and traversal budgets. Graph models
use the existing degree, insertion, and beam parameters; MIRAGE searches its
committed graph separately from the buffer. These comparisons share workload
and exact truth, but do not establish equal search work or equal recall.
Native HNSW construction can vary with parallel execution; a fixed data seed
does not guarantee identical native graphs across platforms or library versions.

Every stage and summary identifies `backend`, `simulated`, effective device,
fallback reason, and native version where available. Preserve those labels
when publishing a comparison. Simulator measurements demonstrate functional
and mechanism behavior only, and cannot establish superiority over native
external indexes. Historical result artifacts are not regenerated by this mode.

## Python interface and tests

All adapters expose `build(vectors)`, `add(vectors)`, `search(queries, k=10)`,
`synchronize()`, `metadata()`, and `ntotal`. Build resets state; add preserves
insertion-order IDs. Search returns a device-local long tensor padded with -1
when fewer candidates are available. Inputs must be finite matrices with a
consistent dimension. Empty updates and query batches are supported after build.

```python
from cagra_chunksplice.baselines import FaissHNSW

index = FaissHNSW(device="cpu", backend="auto").build(base_vectors)
index.add(new_vectors)
neighbors = index.search(queries, k=10)
print(index.metadata())
```

Graph, CAGRA, and MIRAGE adapters also accept the namespace returned by
`cagra_chunksplice.cli.arguments`. HNSW exposes `--hnsw-m`,
`--hnsw-ef-construction`, and shared `--ef` for search.

```console
python -m pytest tests/test_external_baselines.py -q
```

Tests cover portable contracts, immediate buffer visibility, capacity and tail
merges, deterministic replay, missing dependencies, strict-native failure,
FAISS/CAGRA API mocks, JSON output, and both CLI entry points. Native API mocks
require NumPy and do not replace validation on actual CUDA/cuVS hardware.
