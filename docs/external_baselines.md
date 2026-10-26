# External index comparisons

External mode compares **GPU dynamic graph insertion** (Naive, ChunkSplice,
cuVS-CAGRA) with **CPU multi-threaded hierarchical insertion** (FAISS-HNSW,
MIRAGE) on the same seeded stream and mixed query cohorts. Reports contain
update latency, Mixed Recall@10, QPS, raw samples, and backend/device provenance.
The architectural groups describe the intended algorithms; implementation labels
and effective devices describe what actually ran.

## External Baselines Benchmark Status & Promotion Gate

This matrix describes the evidence currently recorded in
[`results/external_validation.json`](../results/external_validation.json):
a completed all-native external comparison on synthetic data (2026-10-22,
RTX 4070 SUPER). External numbers remain excluded from the primary Key
Experimental Results tables in the README until every promotion requirement
below is satisfied; the synthetic run passes the automated native gate but
does not satisfy the academic-dataset requirement.

| Baseline | Native implementation and expected backend | Recorded implementation | Status and promotion decision |
| --- | --- | --- | --- |
| cuVS-CAGRA | CUDA `cagra.build/extend/search`; `cuvs-cagra-cuda`. | Native CUDA build/extend/search; `cuvs-cagra-cuda` (cuVS 26.10.00), `simulated: false`. | Native GPU run recorded; primary-table promotion still requires SIFT1M/GloVe sweeps. |
| FAISS-HNSW | Compiled CPU FAISS `IndexHNSWFlat.add/search`; `faiss-hnsw-cpu`. | Native CPU HNSW add/search; `faiss-hnsw-cpu` (FAISS 1.15.1, 14 OpenMP threads), `simulated: false`. | Native CPU run recorded; primary-table promotion still requires SIFT1M/GloVe sweeps. |
| MIRAGE | Compiled upstream CPU `IndexMirage.add/search`, via C ABI bridge or custom bindings; `faiss-mirage-cpu`. | Native CPU MIRAGE add/search via the upstream C ABI bridge; `faiss-mirage-cpu` (`6749e9c`, S=32 R=4 iter=15), `simulated: false`. | Native CPU run recorded; primary-table promotion still requires SIFT1M/GloVe sweeps. |

| Gate | Required evidence for promotion into the primary Key Experimental Results tables |
| --- | --- |
| Native binaries and execution | All three external methods run actual compiled native build/add/search on the intended devices: CPU for FAISS-HNSW and MIRAGE, CUDA for cuVS-CAGRA. Record library versions, upstream revisions and build configuration. Require `simulated: false`, `native_external_comparison_complete: true`, successful completion, valid metrics and per-replay samples at every update and swept configuration. Imports, API mocks and setup smoke tests do not satisfy this gate. |
| Real SIFT1M and GloVe datasets | Provide measured sweeps on both SIFT1M (128-dimensional Euclidean) and GloVe-100 (100-dimensional angular). Archive dataset source, SHA-256, row/query counts, seed, metric, normalization and split provenance. Use held-out test queries and recompute exact truth over each visible prefix. Label subsets explicitly; full-corpus claims require `full_academic_corpus: true`. Synthetic or synthetic-scaled substitutes cannot qualify as academic dataset evidence. |
| Measured Pareto frontiers | Sweep construction/insertion and search parameters with fresh indexes and identical corpus/query streams for all compared methods. Retain all measured configurations, recall/QPS frontiers, update latency and replay samples. At each common recall target, compare the fastest observed point with recall at least that target; mark unattained targets as unavailable. Never interpolate missing performance or merge native and simulated frontiers. |
| Reproducibility and review | Archive successful native validation JSON, `.pareto.svg`, the build manifest, package/source provenance, hardware, thread settings, timing protocol and complete sweep grids. Review both dataset runs and frontier coverage before adding numbers to the primary tables. Preserve the CPU/GPU architectural distinction and PyTorch model labels; gate passage is not evidence of superiority. |

The automated `promotion_gate.passed` result validates native backend
provenance and measured-row integrity, including stage sizes of
`base + stage * chunk_size`. With `--external-sweep`, it also requires every
configured grid point of each external method to be measured. It does **not**
require real datasets, full-corpus coverage, an external sweep, or a grid wide
enough to support a frontier claim.
Its `academic_dataset` field only reports whether the dataset is nonsynthetic.
If the gate fails, the measured report is written to `<output-stem>.error.json`
with `status: "promotion_gate_failed"`, the promoted output path is not written,
and the command exits with status 1.
Thus a native synthetic smoke run can pass the automated check while remaining
ineligible for the primary tables. The publication gate above is stricter and
requires explicit review of both academic dataset sweeps.

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
| MIRAGE | CPU hierarchical | CPU `IndexMirage.add` through the upstream C ABI bridge or custom bindings. | Exact CPU search with append-only storage; no MIRAGE hierarchy. |
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

Native MIRAGE can use the compiled C ABI bridge in `native/mirage_bridge`,
selected by `CAGRA_MIRAGE_LIBRARY`, or a custom FAISS Python module exposing
`IndexMirage` with NumPy-compatible `add/search` and its `mirage` configuration
object. **Stock `faiss-cpu` does not include MIRAGE.** The setup script builds
upstream MIRAGE with `-DFAISS_ENABLE_PYTHON=OFF` and builds the separate bridge;
it does not require adding MIRAGE to the stock FAISS Python module. Both paths
delegate directly to native add and record S/R/iter settings. There is no
pending buffer, capacity setting or final drain. Shipped integration code and
mock coverage do not prove a successful native build or measured native run.

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

## Native setup and validation

Run these Bash scripts from the repository root in an activated Python 3.10+
environment on native Linux or Linux under WSL2. Windows Python and WSL1 are
unsupported for this suite. Provide a visible NVIDIA GPU/driver, a compatible
CUDA-enabled PyTorch installation, and build tools (`git`, CMake, a C++
compiler, SWIG, and the BLAS/LAPACK/OpenMP development dependencies needed by
FAISS). Match CUDA, CuPy and cuVS versions to the actual environment.

Probe first; the probe makes no installation changes. Inspect the planned build
before installing into the active Python environment:

```bash
bash scripts/build_external.sh --probe
bash scripts/build_external.sh --dry-run --cuda-major 12
bash scripts/build_external.sh --cuda-major 12
```

Use `--cuda-major 13` only for a compatible CUDA 13 environment; the setup
rejects a mismatch with detected `nvcc`. Set `PYTHON` if the active interpreter
is not named `python3`. The setup does not install system packages or enable
Windows features. By default it builds CPU FAISS with Python bindings, builds
CPU MIRAGE plus the separate C ABI bridge, installs cuVS/CuPy wheels and the
project's `[test,datasets]` extras, and exercises native build/add/search.
`--source-cuvs` selects an upstream cuVS source build and requires `nvcc`.

Build files default to `.external/`; setup writes `env.sh` with
`CAGRA_MIRAGE_LIBRARY` and `build-manifest.json` with environment, resolved
source revisions and smoke-test status. For reproducible publication builds,
pin `--faiss-ref` and `--mirage-ref` to exact commits and pin
`--cuvs-version` to an exact wheel release, or `--cuvs-ref` for a source build.
`--mirage-ref` defaults to `6749e9c`, the upstream revision of the recorded
validation and the base of the local patch series; the cuVS version wildcard
is not an immutable pin.

After checking out MIRAGE, setup applies every patch in
[`native/mirage_bridge/patches/series`](../native/mirage_bridge/patches/series)
(currently the `IndexMirage::add` double-insert fix) before building, refuses a
patch exported against a different revision, and accepts an existing checkout
only if its sole local changes are those patches. A missing patch file stops
setup; export it once with `bash scripts/export_mirage_patch.sh /root/mirage`
(see the [patch README](../native/mirage_bridge/patches/README.md)). The bridge
is built as `libmirage_bridge.so`, the name recorded in
`results/external_validation.json`. Patch names and SHA-256 digests are compiled
into the bridge (`cs_mirage_patches`, reported as MIRAGE `metadata.patches`)
and written to `build-manifest.json` (`mirage_patches`). Independently,
`cs_mirage_add` fails if `ntotal` does not grow by exactly the batch size.
CPU FAISS and MIRAGE default to `--faiss-opt-level generic`, a portable build
without AVX2/AVX-512 kernels that understates CPU baseline throughput. For
performance comparisons, pass the highest level the CPU supports
(`avx2` or `avx512`); the manifest records the chosen level.
A successful setup smoke test still leaves the full promotion gate pending.

The validation wrapper sources `.external/env.sh` when present and forces
`--device cuda --external-backend native --require-native-validation`.
Missing native dependencies fail rather than select fallbacks. If setup used
`--prefix /path/to/build`, export `CAGRA_EXTERNAL_PREFIX=/path/to/build`
before validation. The wrapper defaults `OMP_NUM_THREADS=8` and
`OPENBLAS_NUM_THREADS=1` unless already set; record any overrides.

A small native integration check uses synthetic data and cannot qualify for
primary-table promotion:

```bash
bash scripts/run_native_validation.sh --base 512 --chunks 3 --chunk-size 64 --dim 32 --queries 64 --repeats 3 --output results/native_smoke.json
```

For the academic dataset sweeps, use the pinned wrappers (each runs the
full-corpus native sweep with Pareto frontiers, then validates native
provenance):

```bash
bash scripts/run_native_sift1m.sh
bash scripts/run_native_glove100.sh
```

The equivalent explicit commands are:

```bash
bash scripts/run_native_validation.sh --dataset sift1m --download-datasets --external-sweep --base 800000 --chunks 4 --chunk-size 50000 --queries 1000 --repeats 3 --output results/native_sift1m.json
bash scripts/run_native_validation.sh --dataset glove-100 --download-datasets --external-sweep --base 983514 --chunks 4 --chunk-size 50000 --queries 1000 --repeats 3 --output results/native_glove100.json
python scripts/validate_external_report.py results/native_sift1m.json
python scripts/validate_external_report.py results/native_glove100.json
```

Expected Pareto and provenance artifacts per dataset (written automatically
by `--external-sweep` and the validation wrapper):

| Dataset | Sweep report | Pareto frontier plot | Validation |
| --- | --- | --- | --- |
| SIFT1M | `results/native_sift1m.json` (`curve_points`, `curves.frontiers`, `curves.matched_recall`) | `results/native_sift1m.pareto.svg` | `python scripts/validate_external_report.py results/native_sift1m.json` |
| GloVe-100 | `results/native_glove100.json` (same curve fields) | `results/native_glove100.pareto.svg` | `python scripts/validate_external_report.py results/native_glove100.json` |

A promoted dataset run must additionally record `dataset.synthetic: false`,
`full_academic_corpus: true`, and the dataset SHA-256 with corpus/query
counts in its report.

These commands request the full registered corpora: 1,000,000 SIFT1M vectors
and 1,183,514 GloVe vectors, with 1,000 held-out queries. They are not claims
that these runs have completed or fit a particular GPU. Exact ground truth,
PyTorch graph models and repeated fresh-index builds can require substantial
memory and runtime. Start with explicitly labeled subsets to check capacity;
subset validation must not be presented as full-corpus evidence.

Datasets are cached under `~/.cache/cagra-chunksplice/datasets`; override with
`--dataset-cache`. Downloads use ANN-Benchmarks HDF5 files
`sift-128-euclidean.hdf5` and `glove-100-angular.hdf5`. GloVe vectors and queries
are unit-normalized. A subset is the first `base + chunks * chunk_size` train
vectors in file order, inserted in a seeded permutation; it is a file prefix,
not a uniform random sample. Queries are the first `--queries` test vectors.
The file must declare its `distance` attribute. Missing data fails unless `--allow-synthetic-fallback`
is explicitly supplied; do not enable that option for promotion runs. Check
`dataset.synthetic: false`, the actual dataset name, hash and corpus counts
in each report.

`--external-sweep` rebuilds and replays these default grids:

| Method | Swept parameters and defaults |
| --- | --- |
| Naive and ChunkSplice | `--graph-ef-values 32 64 128`. |
| FAISS-HNSW | `--hnsw-m-values 16 32` and `--hnsw-ef-search-values 32 64 128`; `--hnsw-ef-construction` remains fixed unless explicitly changed. |
| MIRAGE | `--mirage-s-values 16 32`, `--mirage-r-values 4 8`, `--mirage-iter-values 10 15`; these change construction and insertion, not only search. |
| cuVS-CAGRA | `--cagra-itopk-values 32 64 128`, `--cagra-search-width-values 1 4`, `--cagra-iterations-values 8 16`. |

`--recall-targets` defaults to `0.8 0.9 0.95`. JSON contains
`curve_points` (including each configuration's measured stages),
`curves.frontiers`, and `curves.matched_recall`; a sibling `.pareto.svg`
plots observed frontiers. Update latency remains a per-point metric, not an
axis of the recall/QPS frontier. The same queries tune and report these
curves, so they do not establish held-out tuning performance. Native and
simulated results remain separate; Naive and ChunkSplice remain PyTorch models.

## Measurement protocol

1. Generate one corpus and one mixed base/inserted query set per stage, with
   exact Recall@10 truth over all visible vectors. Academic datasets instead
   reuse the same held-out test queries at every stage, with truth recomputed
   over each visible prefix.
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

## Local execution status (2026-10-22)

[Completed native validation](../results/external_validation.json) records a
successful all-native run on an NVIDIA GeForce RTX 4070 SUPER with
`status: "ok"`, `benchmark_completed: true`,
`native_external_comparison_complete: true` and a passed promotion gate
(`academic_dataset: false`). It uses a base of 2048 synthetic vectors, four
updates of 256 vectors, 32 dimensions, 256 queries and three measured replays
after one untimed warmup replay. The artifact contains measured latency,
recall, QPS, replay samples and backend provenance; it is not a failure log.
This run supersedes the earlier fallback-only runs.

FAISS-HNSW ran native CPU HNSW (`faiss-hnsw-cpu`, 1.15.1, M=32,
ef_construction=80, ef_search=64, 14 OpenMP threads). cuVS-CAGRA ran native
CUDA CAGRA (`cuvs-cagra-cuda`, 26.10.00, degree 24, itopk 64, width 4,
16 iterations). MIRAGE ran the native CPU C ABI bridge (`faiss-mirage-cpu`,
upstream `6749e9c`, S=32 R=4 iter=15). All three record `simulated: false`
with native devices (CUDA for CAGRA, CPU for HNSW/MIRAGE). Naive and
ChunkSplice ran as CUDA PyTorch mechanism models (`simulated: true`).

Final-stage measurements (n=3072; update is median total replay update time
divided by update count; recall and QPS use the final stage):

| Method | Backend | Update (ms) | Mixed Recall@10 | Mixed QPS |
| --- | --- | --- | --- | --- |
| Naive | `torch-naive` | 6.500646499262075 | 0.9183593988418579 | 5103.278394050177 |
| ChunkSplice | `torch-chunksplice` | 10.626298249917454 | 0.975390613079071 | 5013.386820077181 |
| FAISS-HNSW | `faiss-hnsw-cpu` | 13.999516750118346 | 0.9957031607627869 | 254931.07401390868 |
| cuVS-CAGRA | `cuvs-cagra-cuda` | 6.495410249044653 | 0.9959635337193807 | 82412.25832658686 |
| MIRAGE | `faiss-mirage-cpu` | 6.888773000355286 | 0.970703125 | 418112.09162927506 |

Mixed Recall@10 by stage (n = 2304, 2560, 2816, 3072):

| Method | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| --- | --- | --- | --- | --- |
| Naive | 0.965624988079071 | 0.9410156607627869 | 0.918749988079071 | 0.9183593988418579 |
| ChunkSplice | 0.991406261920929 | 0.992968738079071 | 0.984375 | 0.975390613079071 |
| FAISS-HNSW | 0.9996094107627869 | 0.9996094107627869 | 0.999218761920929 | 0.9957031607627869 |
| cuVS-CAGRA | 0.9996093908945719 | 0.998828113079071 | 0.9990885655085245 | 0.9959635337193807 |
| MIRAGE | 0.9630208412806193 | 0.964843769868215 | 0.9709635376930237 | 0.970703125 |

Median per-update latency by stage in milliseconds (Naive, ChunkSplice,
FAISS-HNSW, cuVS-CAGRA, MIRAGE): stage 1: 5.264890998660121,
8.733132999623194, 11.568597998120822, 5.170344000362093,
6.674734002444893; stage 2: 7.41693099917029, 9.24384100289899,
15.17877399965073, 5.806007997307461, 6.651823001448065; stage 3:
7.43282899929909, 10.375724999903468, 16.357479002181208,
5.491651001648279, 7.604325001011603; stage 4: 6.653923999692779,
10.632434998115059, 13.734186002693605, 5.319371997757116,
6.866628998977831.

The local GPU-run diagnostic `results/external.error.json` (ignored by Git
and not published) records `RuntimeError: native MIRAGE did not append the
requested vectors` (`status: "runtime_error"`, `benchmark_completed: false`,
2026-10-22T14:22:00.397503+00:00). That is a native MIRAGE append runtime
failure, not a `_ctypes`/DLL import failure. A local `results/external_cpu.error.json`
is a separate CPU-run diagnostic, likewise ignored by Git and not published.
Neither diagnostic describes the successful
[completed native validation](../results/external_validation.json), and that
successful run alone does not establish a passing test suite.

The append failure was resolved by fixing the double-insert bug in upstream
MIRAGE `IndexMirage::add` in `faiss/IndexMIRAGE.cpp` on WSL, rebuilding, and
copying the resulting `libmirage_bridge.so` into place. The subsequent
`results/external_validation.json` run (started 2026-10-22T14:36:21.131331+00:00,
`library_path: /root/mirage/build/libmirage_bridge.so`) validates every
`metadata.ntotal` against its stage size (2304, 2560, 2816, 3072), including
the native `faiss-mirage-cpu` backend, with `benchmark_completed: true` and
`native_external_comparison_complete: true`.

That run used a hand-built bridge, so its MIRAGE metadata has no `patches`
field. The fix is now part of the build through the patch series described in
[native setup](#native-setup-and-validation); runs built by
`scripts/setup_external.py` record the applied patches and their digests.


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
