# cagra-chunksplice

**A controlled PyTorch study of intra-chunk connectivity in incremental proximity graphs.**

### Before and after insertion

| Stage | New-vector adjacency | Existing vectors |
| --- | --- | --- |
| Before: `Naive` | Select neighbors only from the preceding snapshot. No same-chunk edges. | Degree-capped reverse proposals connect earlier rows to new vectors. |
| After: `ChunkSplice` | Add a small RNG-pruned peer budget while reserving old-cohort bridges. | For the same input graph and update, earlier rows equal the Naive result. Repeated trajectories can diverge. |

The project separates reusable graph algorithms from experiment runners, tests
and recorded results. Default modes study a topology mechanism. External mode
adds optional native CPU FAISS-HNSW/MIRAGE and GPU cuVS-CAGRA adapters, with
explicitly labeled fallbacks when native backends are unavailable.

In the supplied sweep, ChunkSplice improves mixed and inserted-cohort recall
relative to Naive while lowering original-base recall. Update cost is higher.
See [recorded results](docs/results.md) and
[methodology, limitations and upstream references](docs/methodology.md).

## Problem formulation: The intra-chunk disconnectivity crisis

In graph-based vector search, including GPU cuVS/CAGRA and CPU MIRAGE-ANNS, proximity graph index construction requires establishing navigable, small-world topologies with bounded out-degrees (typically $R \in [16, 64]$). While static graph construction leverages global k-nearest-neighbor (kNN) descent followed by edge pruning (e.g., Relative Neighborhood Graph / RNG criteria), dynamic and streaming environments require incremental updates without full graph reconstruction.

### The Naive snapshot projection failure mode

Standard incremental attachment ("Naive") operates under a snapshot projection paradigm:
1. **Unidirectional Attachment**: When a new chunk of vectors `C_new` arrives, each vector `u` in `C_new` independently selects its `R` outgoing neighbors strictly from the existing base snapshot `V_old`.
2. **Reverse Edge Injection**: To avoid sinkhole nodes, earlier nodes in `V_old` receive reverse edge proposals from `C_new`, capped at degree `R`.
3. **The Intra-Chunk Void**: Crucially, **no edges are formed among vectors within `C_new`** in either direction. The induced subgraph on `C_new` is completely edge-free (`naive.py:14`).

### Structural pathology in clustered workloads

While 2-hop traversals (`u -> w -> v` through an intermediate base node `w` in `V_old`) exist in theory, dense semantic cohorts induce severe topological failure:
- **Boundary Node Saturation**: When `C_new` forms a tight semantic cluster, all vectors in `C_new` attach to the same small set of boundary base nodes `B` in `V_old`.
- **In-Degree Starvation & Orphan Influx**: Because each node in `B` can retain at most `R` reverse edges, proposals compete and are evicted (`naive.py:30-33`). Consequently, `C_new` receives at most `|B| * R << |C_new|` total entry edges. A large fraction of new nodes remain with in-degree 0 (unreachable unless selected as search start seeds).
- **Beam Backtracking & Step Exhaustion**: Within the cluster, greedy beam search cannot execute local monotone gradient descent. From any node in `C_new`, all out-edges point back to older, more distant base nodes. The beam is forced to cycle through boundary nodes `B`, rapidly exhausting the search step budget (`L_steps`) and degrading recall.

## Proposed solution: ChunkSplice degree budgeting

**ChunkSplice** addresses intra-chunk disconnectivity through **degree-budget partitioning** and **cohort-local Relative Neighborhood Graph (RNG) pruning**. In the recorded stream, it substantially reduces inserted-node disconnectivity while reserving outgoing bridges to the base corpus; full graph connectivity is not established.

```
Total Out-Degree Budget: R (e.g., R = 24)
┌──────────────────────────────────────┬─────────────────────────┐
│ R_bridge = 18 outgoing bridge edges  │ R_intra = 6 peer edges  │
│ Reserved for base snapshot (V_old)   │ Cohort-local RNG pruned │
└──────────────────────────────────────┴─────────────────────────┘
```

### Algorithmic mechanisms

1. **Explicit Edge Partitioning**: For every vector `u` in `C_new`, out-degree capacity `R` is strictly split into:
   - **Bridge edges (`R_bridge`)**: Outgoing connections directed into the existing base graph `V_old`, reserving capacity for routes to older vectors.
   - **Peer edges (`R_intra`)**: Dedicated intra-chunk connections connecting vectors within `C_new`.
2. **Cohort-Local Relative Neighborhood Graph (RNG) Pruning**:
   - Intra-chunk edges are not chosen via naive kNN (which would create redundant, highly collinear cliques).
   - A candidate pool of size `R_pool` within `C_new` is evaluated against Gabriel / RNG criteria: an edge `(u, v)` is pruned if there exists a mutual neighbor `w` in `C_new` such that:
     `max(dist(u, w), dist(v, w)) < dist(u, v)`
   - This enforces diverse, angularly distributed intra-cluster paths while strictly bounding local clustering overhead to `O(|C_new|^2)` on GPU tensors.
3. **Preserved Reverse Proposals**: Reverse edge proposals to `V_old` use the same degree-capped process as Naive. Retained edges can provide entry routes into newly appended cohorts, but do not guarantee reachability of every new vector.

## Comparative analysis and related work

| Paradigm | Update Mechanism | Update Latency | Mixed Recall@10 | Latest Chunk Recall | Navigability / Graph Health |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Rebuild (kNN / RNG)** | Global rebuild of $V_{old} \cup \mathcal{C}_{new}$ from scratch | Workload-dependent; recorded Rebuild is faster at this scale | Moderate to High (0.65-0.90) | High (0.83-0.99) | Global reconstruction; connectivity depends on construction and pruning |
| **Naive Attachment** | Snapshot $V_{old}$ lookup + capped reverse edges | Fast (low update ms) | Low (0.72-0.88) | Degraded (0.88-0.99) | Zero intra-chunk edges; clustered workloads can lose reachability |
| **LocalRNG (Two-hop repair)** | Multi-hop neighbor exchange across entire active graph | Extremely Slow (up to 40x higher latency) | High | High | Excessive GPU global memory random accesses |
| **ChunkSplice (Ours)** | Degree budget partitioning ($R_{bridge} + R_{intra}$) + cohort-local RNG | Recorded sweep: $<0.5$ ms/vector; higher update cost than Naive | 0.734-0.934 in the sweep; 0.970 at stream end | 0.992-0.998 shared-tail sweep recall; 0.999 at stream end | Stream-end inserted reachability 0.998046875; residual unreachable nodes remain |

### Key trade-offs

- **GPU graph insertion (cuVS-CAGRA, Naive, ChunkSplice)**: cuVS provides native CUDA graph extension. Naive and ChunkSplice are PyTorch mechanism models; ChunkSplice adds batched, cohort-local RNG peer pruning. CPU debugging measures the models on CPU.
- **CPU hierarchical insertion (FAISS-HNSW, MIRAGE)**: Both use multi-threaded CPU implementations. MIRAGE-ANNS is entirely CPU: C++ on FAISS, multi-layer hierarchy, NN-Descent base construction, and RNG pruning during incremental add. The official build disables GPU support with `-DFAISS_ENABLE_GPU=OFF`. It is neither a GPU method nor a buffered ChunkSplice variant. See the [official MIRAGE repository](https://github.com/dsg-uwaterloo/mirage) and [SIGMOD 2025 paper, PACMMOD 3(3), Art. 188](https://doi.org/10.1145/3725325).
- **Comparison scope**: External mode compares GPU dynamic graph updates with CPU multi-threaded insertion on shared data and queries. Exact CPU fallbacks provide a functional reference and cannot establish native MIRAGE or HNSW performance.


## Key scientific contributions

1. **Characterization of Intra-Chunk Disconnectivity**: The controlled snapshot-projection experiments support missing intra-chunk edges as a contributor to recall loss in the tested clustered workloads; they do not establish it as the primary bottleneck across dynamic vector indexes.
2. **Degree-Budget Partitioning Architecture**: A degree split ($R_{bridge}, R_{intra}$) reserves old-cohort bridges and adds peer connections. In the recorded stream, final inserted-cohort reachability is **0.998046875**, compared with **0.865234375** for Naive. This is an empirical improvement, not a guarantee of bidirectional connectivity.
3. **Cohort-Local Parallel RNG Pruning**: A GPU-native batched pruning algorithm that selects angularly diversified peer edges without requiring global graph traversals.
4. **Observed Recall and Connectivity Trade-offs**: In the supplied experiments on an NVIDIA RTX 4070 SUPER, ChunkSplice:
   - Improves mixed and inserted-cohort recall in all four sweep partitions, with lower original-base recall and higher update cost than Naive (see the tables below).
   - Reaches **1.0** inserted-cohort recall and **4919.841181206764 QPS** at the final checkpoint of the four-update stream. Inserted orphans fall from **138** with Naive to **2** with ChunkSplice, substantially suppressing disconnectivity in this workload. Batch updates take **4.725700000562938-6.062199998268625 ms** for 256 vectors, so update cost is sub-millisecond **per vector**, not per batch.
5. **Open-Source Reproducible Diagnostic Framework**: Implementation of advanced graph health metrics including Tarjan Strongly Connected Components (SCC), in-degree Gini/hubness analysis, and long-horizon edge decay tracking.

## Key experimental results

Detailed benchmark logs and full floating-point values are preserved in [docs/results.md](docs/results.md) and [results/](results/).

### External Baselines Benchmark Status & Promotion Gate

The primary Key Experimental Results tables below contain controlled PyTorch
measurements. External baseline numbers may enter those tables only after all
promotion requirements below are met. A successful fallback run is functional
validation, not a native performance comparison.

| External baseline | Native execution required | Recorded status in `results/external_validation.json` | Primary-table eligibility |
| --- | --- | --- | --- |
| cuVS-CAGRA | Actual CUDA `cagra.build/extend/search` binaries. | Simulated: CUDA PyTorch chunk-hole model (`torch-cagra-chunk-hole`); CuPy unavailable. | Blocked pending native dataset sweeps. |
| FAISS-HNSW | Compiled CPU FAISS `IndexHNSWFlat.add/search`. | Simulated: exact CPU reference (`torch-exact`); FAISS unavailable. | Blocked pending native dataset sweeps. |
| MIRAGE | Compiled upstream CPU `IndexMirage.add/search` through the C ABI bridge or custom bindings. | Simulated: exact CPU reference (`torch-exact-cpu`); FAISS unavailable. | Blocked pending native dataset sweeps. |

| Promotion requirement | Evidence required before publishing external numbers in the primary tables |
| --- | --- |
| Native binaries | All three external methods must execute native build, incremental add and search, with recorded versions/revisions, correct CPU/CUDA devices, `simulated: false`, and `native_external_comparison_complete: true`. Imports, mocks and setup smoke tests alone are insufficient. |
| SIFT1M and GloVe | Measured runs on real SIFT1M (`sift1m`) and GloVe-100 (`glove-100`) with dataset hashes, explicit corpus/query counts and split/metric provenance. Full-corpus claims require `full_academic_corpus: true`; subsets must be labeled. Synthetic substitutes cannot satisfy this gate. |
| Pareto frontier sweeps | Fresh-index parameter sweeps for all compared methods on shared data and queries, retaining measured recall/QPS frontiers, per-point update latency and replay samples. Compare the fastest measured points meeting the same recall target; report unattained targets without interpolation. |
| Reproducible evidence | Successful native report validation plus archived JSON, frontier SVG, build manifest, hardware/software and thread settings. Review dataset coverage and sweep completeness before promotion; a native validation pass alone does not establish superiority or publication eligibility. |

Run the native path from the repository root in an activated Linux/WSL2 Python
environment with an NVIDIA GPU and compatible CUDA dependencies:

```bash
bash scripts/build_external.sh --probe
bash scripts/build_external.sh --cuda-major 12
bash scripts/run_native_validation.sh --dataset sift1m --download-datasets --external-sweep --base 800000 --chunks 4 --chunk-size 50000 --queries 1000 --repeats 3 --output results/native_sift1m.json
bash scripts/run_native_validation.sh --dataset glove-100 --download-datasets --external-sweep --base 983514 --chunks 4 --chunk-size 50000 --queries 1000 --repeats 3 --output results/native_glove100.json
```

Select the CUDA major matching the environment, and pin upstream revisions for
publication runs. These full-corpus sweeps require substantial compute and
memory; they are commands, not recorded results or runtime estimates. See
[native setup, sweep controls and gate limitations](docs/external_baselines.md#native-setup-and-validation)
before running.

### Controlled chunk-size sweep (2048-dim, degree 24)

| Chunk size | Updates | Method | Mixed Recall@10 | Inserted R@10 | Shared-tail R@10 | Update trajectory (ms) | Mixed QPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **64** | 64 | Naive | 0.728 | 0.802 | 0.992 | 622.0 ms | 2356.1 |
| **64** | 64 | **ChunkSplice** | **0.734** | **0.835** | **0.992** | 1880.2 ms | 2561.2 |
| **256** | 16 | Naive | 0.889 | 0.873 | 0.895 | 292.1 ms | 2340.4 |
| **256** | 16 | **ChunkSplice** | **0.925** | **0.964** | **0.998** | 527.8 ms | 2291.7 |
| **1024** | 4 | Naive | 0.881 | 0.858 | 0.905 | 72.0 ms | 2201.1 |
| **1024** | 4 | **ChunkSplice** | **0.929** | **0.959** | **0.998** | 126.9 ms | 2204.7 |
| **2048** | 2 | Naive | 0.886 | 0.862 | 0.908 | 40.9 ms | 2223.3 |
| **2048** | 2 | **ChunkSplice** | **0.934** | **0.965** | **0.998** | 70.1 ms | 2243.2 |

### Streaming experiment (4 consecutive updates of 256 vectors)

| Step | Method | Mixed Recall@10 | Inserted R@10 | Newest Chunk R@10 | Query Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Update 1** | Naive | 0.964 | 0.943 | 0.941 | 3862.8 QPS |
| **Update 1** | **ChunkSplice** | **0.992** | **1.000** | **1.000** | **4870.2 QPS** |
| **Update 4** | Naive | 0.927 | 0.887 | 0.883 | 4505.8 QPS |
| **Update 4** | **ChunkSplice** | **0.970** | **1.000** | **0.999** | **4919.8 QPS** |

## Install and run

Requires Python 3.10+ and PyTorch 2.0+. Install a suitable PyTorch build using
the [official installation instructions](https://pytorch.org/get-started/locally/).
CUDA is the default device; CPU debugging requires explicit selection.

From the repository directory:

```console
python -m venv .venv
```

| Shell | Activate |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

```console
python -m pip install -e ".[test]"
python -m cagra_chunksplice --help
python -m cagra_chunksplice --device cpu --self-test-only
python -m pytest -q
```

The installed `cagra-chunksplice` command is equivalent to `python -m cagra_chunksplice`:

```console
cagra-chunksplice --device cpu --base 512 --chunks 2 --chunk-size 64 --output results/cpu_debug.json
cagra-chunksplice --device cuda --self-test-only
```

Historical `python cagra_mirage_prototype.py ...` commands remain available through
a small compatibility wrapper. Install the package for library imports.

## What is included

| Component | Purpose |
| --- | --- |
| Global rebuilds | Unpruned Rebuild-kNN and RNG-pruned Rebuild-RNG with reverse proposals; blocked exact candidate generation. |
| Naive insertion | Exact or approximate beam selection from the preceding snapshot, with capped reverse edges. |
| ChunkSplice | Peer slots, cohort-local RNG pruning and reserved old bridges. |
| LocalRNG | Bounded one/two-hop repair, neighbor exchange and optional backfill. |
| Beam search | Shared starts, beam capacity, expansion width and step budget. |
| Evaluation | Recall by source age, SCCs, hubness, reachability, bridge retention and edge lengths. |
| Stress workloads | 100-500 epochs, high-dimensional Gaussian data, OOD probes and angular queries. |
| Reporting | Synchronized samples, environment, parameters and package source hashes. |
| Tests | Mechanisms, graph invariants, CLI/errors and end-to-end experiments. |
| External baselines | CPU FAISS-HNSW/MIRAGE versus GPU cuVS-CAGRA and PyTorch Naive/ChunkSplice; explicit backend/device provenance. |

## Repository layout

```text
.
├── src/cagra_chunksplice/
│   ├── baselines/        # CPU hierarchical and GPU graph insertion adapters
│   ├── algorithms/       # Naive, ChunkSplice, LocalRNG and RNG pruning
│   ├── core.py           # Distances, exact kNN and candidates
│   ├── graph.py          # Construction and adjacency invariants
│   ├── search.py         # Fixed-budget beam traversal
│   ├── data.py           # Seeded workloads and query cohorts
│   ├── metrics.py        # Recall and connectivity diagnostics
│   ├── timing.py         # Synchronized measurements
│   ├── evaluation.py     # Shared query evaluation
│   ├── reporting.py      # JSON and source provenance
│   ├── runners/          # Benchmark, sweep and stream orchestration
│   ├── selftest.py        # Checks shipped with the CLI
│   └── cli.py            # Argument validation and dispatch
├── experiments/          # Thin benchmark/sweep/stream entry scripts
├── tests/                # Unit and end-to-end regression tests
├── results/              # Historical JSON evidence
├── docs/
│   ├── assets/architecture.svg
│   ├── methodology.md
│   ├── results.md
│   └── development.md
├── cagra_mirage_prototype.py  # Historical CLI compatibility wrapper
├── pyproject.toml        # Package metadata, dependency and CLI entry
├── LICENSE
└── CITATION.cff
```

The separation of library, experiments and tests follows broad research-library
conventions illustrated by [FAISS](https://github.com/facebookresearch/faiss),
[cuVS](https://github.com/NVIDIA/cuvs), [MIRAGE-ANNS](https://github.com/dsg-uwaterloo/mirage)
and [DiskANN](https://github.com/microsoft/DiskANN).

## Architecture

![Architecture and experimental workflow](docs/assets/architecture.svg)

One Python process launches PyTorch tensor operations on the selected device.
Data generation, ground truth, validation and diagnostics stay outside the
update/search timers. These operations are not fused CUDA kernels.

## Run experiments

All modes accept the shared CLI options; `--help` lists every option.
Scripts under `experiments/` use the installed package and select a default mode.

```console
python experiments/benchmark.py --device cpu --base 512 --chunks 2 --chunk-size 64 --output results/benchmark_debug.json
python experiments/sweep.py --sweep-total 4096 --sweep-chunk-sizes 64 256 1024 2048 --output results/sweep_reproduced.json
python experiments/stream.py --chunks 50 --chunk-size 64 --stream-eval-every 10 --output results/stream_long.json
```

| Mode | Methods | Purpose |
| --- | --- | --- |
| `benchmark` | Rebuild-kNN, Rebuild-RNG, Naive, ChunkSplice, LocalRNG | Compare trajectories and query cohorts. |
| `sweep` | Rebuild-kNN, Rebuild-RNG, Naive, ChunkSplice | Vary insertion partitions of one shared corpus. |
| `stream` | Naive, ChunkSplice; optional rebuilds | Retain update timings and evaluate checkpoints. |
| `stress` | Rebuild-kNN, Rebuild-RNG, Naive, ChunkSplice | Evaluate long horizons; 100 epochs by default. |
| `external` | CPU FAISS-HNSW, MIRAGE; GPU cuVS-CAGRA, Naive, ChunkSplice | Compare architectures with native/fallback labels. |

Use `--insertion-mode beam` to study accumulated candidate errors and
`--workload high-lid --dim 128` for a high-dimensional Gaussian control.
`--query-distribution ood` shifts queries away from the indexed distribution;
`angular` normalizes both queries and data for cosine-equivalent rankings.
[Scientific evaluation controls and metric definitions](experiments/README.md#long-horizon-and-failure-mode-evaluation)
describe 100-500 epoch runs, shared controls, and diagnostic interpretation.
New reports call the historical Rebuild method Rebuild-kNN; recorded results
retain their original labels and values.

For the external comparison (native backends selected when available):

```console
python experiments/benchmark_external.py --device cuda --external-backend auto --base 512 --chunks 3 --chunk-size 64 --dim 32 --queries 64 --output results/external.json
```

Use `--device cpu --external-backend simulated` for a portable check.
FAISS-HNSW and MIRAGE always compute on CPU, including their exact fallbacks.
Native MIRAGE requires the compiled upstream C ABI bridge or custom Python
bindings; stock `faiss-cpu` does not include it. Reports preserve the actual
device, CPU thread counts, backend and
fallback reason. See [external baseline details](docs/external_baselines.md)
for reproducible commands and run status. The [completed CUDA benchmark](results/external_validation.json)
records `status: "ok"` and `benchmark_completed: true` on an RTX 4070 SUPER.
HNSW and MIRAGE used exact CPU fallbacks; cuVS-CAGRA used the CUDA PyTorch
chunk-hole simulator. Thus `native_external_comparison_complete: false`:
the measured results describe these implementations, not native external builds.
Earlier `_ctypes` failure diagnostics are separate `*.error.json` files.

Output defaults to `results/cagra_mirage_results.json` in the current working
directory (`results/external.json` in external mode). Missing parents are created. Use a separate `--output` for new runs
to preserve historical evidence. Failures write `<output-stem>.error.json`
without replacing successful output.

Exit codes: `0` success, `1` caught runtime failure, `2` invalid arguments or an
unmet benchmark-only `--verify-effect` threshold. That threshold is empirical,
not an algorithmic guarantee.

ChunkSplice defaults reserve six of 24 slots for peers, draw from a peer pool
of 24 and use alpha 1.0. Controls: `--splice-intra`, `--splice-pool`,
`--splice-alpha`. LocalRNG independently exposes `--alpha`, `--cohort-protect`,
`--route` and `--backfill`.

[Development and migration](docs/development.md) covers module ownership,
tests and source-hash semantics. [Recorded results](docs/results.md) retain
full numeric precision and link to the unchanged JSON artifacts.

## License and citation

Original contributions use the [MIT License](LICENSE); retain copyright and
permission notices. Third-party works retain their own licenses and intellectual
property. For scholarly use, cite [CITATION.cff](CITATION.cff), the exact source
revision/digest and relevant [upstream works](docs/methodology.md#references).
Academic citation is an attribution expectation, not an added MIT condition.
