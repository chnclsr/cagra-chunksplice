# External index comparisons

External mode compares **GPU dynamic graph insertion** (Naive, ChunkSplice,
cuVS-CAGRA) with **CPU multi-threaded hierarchical insertion** (FAISS-HNSW,
MIRAGE) on the same seeded stream and mixed query cohorts. Reports contain
update latency, Mixed Recall@10, QPS, raw samples, and backend/device provenance.
The architectural groups describe the intended algorithms; implementation labels
and effective devices describe what actually ran.

## Verified MIRAGE-ANNS architecture

[MIRAGE-ANNS](https://doi.org/10.1145/3725325), SIGMOD 2025,
PACMMOD 3(3), Article 188, is entirely CPU. Its
[official C++ implementation](https://github.com/dsg-uwaterloo/mirage)
extends FAISS and explicitly builds with `-DFAISS_ENABLE_GPU=OFF`.
It combines a multi-layer hierarchy with NN-Descent base-layer construction
and RNG pruning, including incremental add. It does not buffer updates into
ChunkSplice merges. The author's
[thesis, Chapters 3 and 4](https://dspacemainprd01.lib.uwaterloo.ca/server/api/core/bitstreams/57c37957-4d87-441d-b49a-739e97fc36a7/content)
describes CPU experiments on Intel Xeon Platinum 8380 with 128 threads and
1 TB DDR4 RAM. Those are upstream evaluation settings, not local measurements.

## Backend selection

| Method | Architecture | Native implementation | Portable implementation |
| --- | --- | --- | --- |
| Naive | GPU graph insertion model | None | PyTorch snapshot attachment and capped reverse edges. |
| ChunkSplice | GPU graph insertion model | None | PyTorch degree budgeting and cohort-local RNG pruning. |
| FAISS-HNSW | CPU hierarchical | CPU `IndexHNSWFlat.add`. | Exact CPU search with append-only storage; no HNSW hierarchy. |
| MIRAGE | CPU hierarchical | CPU `IndexMirage.add` through custom bindings. | Exact CPU search with append-only storage; no MIRAGE hierarchy. |
| cuVS-CAGRA | GPU graph insertion | CUDA `cagra.build/extend/search`. | PyTorch Naive chunk-hole model. |

`--device` selects the device for the GPU-oriented graph models and cuVS.
Both CPU adapters keep storage, validation, insertion and search on CPU even
when CUDA is requested. They return IDs on the requested device for the shared
evaluation interface. Transfers are included in end-to-end timing.

`--external-backend auto` selects available native external implementations
at build time and records a fallback reason otherwise. `simulated` forces
portable references. `native` requires native FAISS-HNSW, MIRAGE bindings and
CUDA/cuVS; missing dependencies fail the run. Native update errors fail rather
than silently changing algorithms. Naive and ChunkSplice remain PyTorch models.

Native MIRAGE requires a custom FAISS Python module exposing `IndexMirage`
with NumPy-compatible `add/search` and its `mirage` configuration object.
**Stock `faiss-cpu` does not include MIRAGE.** The upstream documented build
also uses `-DFAISS_ENABLE_PYTHON=OFF`, so simply following that build does not
create Python bindings. The optional adapter is a binding integration point;
its API is covered by mocks, not proof of a working upstream Python build.
It delegates directly to native add and records native S/R/iter defaults.
There is no pending buffer, capacity setting or final drain.

HNSW can be installed separately with `python -m pip install faiss-cpu`;
see [FAISS installation](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss).
Native CAGRA requires compatible CUDA, CuPy and cuVS Python bindings:
[installation](https://docs.nvidia.com/cuvs/user-guide/getting-started/) and
[CAGRA API](https://docs.nvidia.com/cuvs/api-reference/python-api-neighbors-cagra).
The adapter supports legacy extension and concatenated-dataset extension with
`new_start_row`, retains caller-owned storage and synchronizes native work.

## Run the comparison

Install the package as described in the project README. For GPU insertion
versus CPU insertion, with native selection where available:

```console
python experiments/benchmark_external.py --device cuda --external-backend auto --base 512 --chunks 3 --chunk-size 64 --dim 32 --queries 64 --output results/external.json
```

For a portable CPU check without FAISS, custom MIRAGE, cuVS or CUDA:

```console
python experiments/benchmark_external.py --device cpu --external-backend simulated --base 128 --chunks 3 --chunk-size 16 --dim 16 --queries 32 --output results/external_cpu.json
```

The installed equivalent is `python -m cagra_chunksplice --mode external ...`.
External mode defaults to `results/external.json`; other modes retain their
previous output default. Keep CPU debug and GPU runs in separate files.
The obsolete `MIRAGE-ChunkSplice` adapter and `--mirage-buffer-capacity`
option were removed because their buffer semantics do not describe MIRAGE.

## Measurement protocol

1. Generate one corpus and one mixed base/inserted query set per stage, with
   exact Recall@10 truth over all visible vectors.
2. Warm up with a full untimed replay. Build fresh indexes for each measured
   replay; append each batch once. Rotate method order by stage and replay.
3. Synchronize PyTorch and native CuPy streams around wall-clock operations.
   Include validation, allocation, transfers and insertion; exclude initial
   build, data/query generation, exact truth and recall computation.
4. Report median update/query milliseconds and mean recall across replays.
   QPS uses query count divided by median query seconds. Summary update latency
   is median total replay update time divided by update count; summary
   recall/QPS uses the final stage.

Native indexes retain their own construction/search budgets. Portable graph
models share degree and beam controls; exact fallbacks have no graph. Common
data and truth do not imply equal recall or equal search work. Native
multi-threaded graph construction may vary across replays despite fixed data.
CPU thread-pool limits are reported as `cpu_threads`, using native
`omp_get_max_threads()` when exposed and `torch.get_num_threads()` for CPU
references. These are configured limits, not observed concurrent utilization.
Use the runtime's thread configuration before launching to control them.

Each row records `architecture`, `implementation_architecture`,
`effective_device`, `requested_device`, `backend`, `simulated`, fallback
reason and native version. `native_external_comparison_complete` is false
unless all three external backends are native. `gpu_execution_requested`
distinguishes CPU debugging from requested GPU execution.

Exact CPU references validate append/search behavior only. Their recall and
timings cannot be attributed to MIRAGE or HNSW. The cuVS fallback cannot
establish native CAGRA throughput. No artificial delays, calibrated latency
models or synthetic speedups are used.

## Local execution status (2026-10-20)

[Completed CUDA benchmark](../results/external_validation.json) records a
successful run on an NVIDIA GeForce RTX 4070 SUPER with `status: "ok"` and
`benchmark_completed: true`. It uses a base of 2048 vectors, four updates of
256 vectors, 32 dimensions, 256 queries and one measured replay after warmup.
The artifact contains measured latency, recall, QPS and backend provenance;
it is not a failure log.

With `--external-backend auto`, FAISS-HNSW and MIRAGE used exact CPU fallbacks
(`torch-exact` and `torch-exact-cpu`) because FAISS was unavailable. cuVS-CAGRA
used the CUDA PyTorch chunk-hole simulator (`torch-cagra-chunk-hole`) because
CuPy was unavailable. Naive and ChunkSplice also ran as CUDA PyTorch models.
Accordingly, `native_external_comparison_complete: false`: these measurements
describe the fallback implementations, not native HNSW, MIRAGE or CAGRA builds.
A completed comparison using all three native external backends remains pending.

The separate [GPU-run diagnostic](../results/external.error.json) and
[CPU-run diagnostic](../results/external_cpu.error.json) preserve earlier
`_ctypes` DLL import failures. Those diagnostics do not describe the successful
CUDA benchmark, and that benchmark alone does not establish a passing test suite.


## Python interface and tests

All adapters expose `build`, `add`, `search`, `synchronize`, `metadata`
and `ntotal`. Build resets state, add preserves insertion-order IDs, and
search returns long IDs padded with -1. Inputs must be finite matrices of a
consistent dimension. Empty batches are supported after build.

```python
from cagra_chunksplice.baselines import CPUMirage

index = CPUMirage(device="cpu", backend="auto").build(base_vectors)
index.add(new_vectors)
neighbors = index.search(queries, k=10)
print(index.metadata())
```

```console
python -m pytest tests/test_external_baselines.py -q
python -m pytest -q
```

Tests cover contracts, immediate insertion visibility, CPU-only storage under
CUDA requests, native MIRAGE/HNSW/CAGRA API mocks, strict failure, replay,
architecture provenance and both CLI entry points. Mocks do not replace
validation with actual native libraries and GPU hardware.
