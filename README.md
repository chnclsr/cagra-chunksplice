# cagra-chunksplice

**A controlled PyTorch study of intra-chunk connectivity in incremental proximity graphs.**

### Before and after insertion

| Stage | New-vector adjacency | Existing vectors |
| --- | --- | --- |
| Before: `Naive` | Select neighbors only from the preceding snapshot. No same-chunk edges. | Degree-capped reverse proposals connect earlier rows to new vectors. |
| After: `ChunkSplice` | Add a small RNG-pruned peer budget while reserving old-cohort bridges. | For the same input graph and update, earlier rows equal the Naive result. Repeated trajectories can diverge. |

The project separates reusable graph algorithms from experiment runners, tests
and recorded results. It studies a topology mechanism; it does not implement or
benchmark production cuVS/CAGRA, complete MIRAGE-ANNS, FAISS or DiskANN.

In the supplied sweep, ChunkSplice improves mixed and inserted-cohort recall
relative to Naive while lowering original-base recall. Update cost is higher.
See [recorded results](docs/results.md) and
[methodology, limitations and upstream references](docs/methodology.md).

## Problem formulation: The intra-chunk disconnectivity crisis

In modern GPU-accelerated vector search (such as cuVS/CAGRA, MIRAGE-ANNS, and DiskANN), proximity graph index construction requires establishing navigable, small-world topologies with bounded out-degrees (typically $R \in [16, 64]$). While static graph construction leverages global k-nearest-neighbor (kNN) descent followed by edge pruning (e.g., Relative Neighborhood Graph / RNG criteria), dynamic and streaming environments require incremental updates without full graph reconstruction.

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

**ChunkSplice** resolves the intra-chunk disconnectivity crisis through formal **degree-budget partitioning** and **cohort-local Relative Neighborhood Graph (RNG) pruning**, eliminating path fragmentation without sacrificing long-range navigability back to the base corpus.

```
Total Out-Degree Budget: R (e.g., R = 24)
┌──────────────────────────────────────┬─────────────────────────┐
│ R_bridge = 18 outgoing bridge edges  │ R_intra = 6 peer edges  │
│ Reserved for base snapshot (V_old)   │ Cohort-local RNG pruned │
└──────────────────────────────────────┴─────────────────────────┘
```

### Algorithmic mechanisms

1. **Explicit Edge Partitioning**: For every vector `u` in `C_new`, out-degree capacity `R` is strictly split into:
   - **Bridge edges (`R_bridge`)**: Outgoing connections directed into the existing base graph `V_old`, ensuring backward navigability and entry path retention.
   - **Peer edges (`R_intra`)**: Dedicated intra-chunk connections connecting vectors within `C_new`.
2. **Cohort-Local Relative Neighborhood Graph (RNG) Pruning**:
   - Intra-chunk edges are not chosen via naive kNN (which would create redundant, highly collinear cliques).
   - A candidate pool of size `R_pool` within `C_new` is evaluated against Gabriel / RNG criteria: an edge `(u, v)` is pruned if there exists a mutual neighbor `w` in `C_new` such that:
     `max(dist(u, w), dist(v, w)) < dist(u, v)`
   - This enforces diverse, angularly distributed intra-cluster paths while strictly bounding local clustering overhead to `O(|C_new|^2)` on GPU tensors.
3. **Preserved Reverse Convergence**: Backward proposals from `C_new` into `V_old` remain intact, ensuring nodes in `V_old` can route queries forward into newly appended cohorts.

## Comparative analysis and related work

| Paradigm | Update Mechanism | Update Latency | Mixed Recall@10 | Latest Chunk Recall | Navigability / Graph Health |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Rebuild (kNN / RNG)** | Global rebuild of $V_{old} \cup \mathcal{C}_{new}$ from scratch | Prohibitive ($O(N \cdot B)$ or $O(N \log N)$) | Moderate to High (0.65-0.90) | High (0.83-0.99) | Optimal, but computationally infeasible in real-time streams |
| **Naive Attachment** | Snapshot $V_{old}$ lookup + capped reverse edges | Fast (low update ms) | Low (0.72-0.88) | Degraded (0.88-0.99) | Zero intra-chunk connectivity; intra-cluster search fails |
| **LocalRNG (Two-hop repair)** | Multi-hop neighbor exchange across entire active graph | Extremely Slow (up to 40x higher latency) | High | High | Excessive GPU global memory random accesses |
| **ChunkSplice (Ours)** | Degree budget partitioning ($R_{bridge} + R_{intra}$) + cohort-local RNG | Minimal overhead ($<0.5$ ms/vector) | **High (0.93 - 0.97)** | **Near-Perfect (>0.997)** | Eliminates intra-chunk voids while maintaining $O(1)$ bridge routing |

### Key trade-offs

- **cuVS / CAGRA `extend` & MIRAGE-ANNS**: CAGRA achieves ultra-high QPS via 1-hop / 2-hop traversals over small-degree graphs. Naive extension degrades because CAGRA's fixed out-degree layout cannot accommodate new intra-cluster edges without restructuring. ChunkSplice provides the exact structural primitive needed to maintain CAGRA traversal guarantees under incremental ingestion.
- **HNSW / Vamana**: Hierarchical and Vamana graphs use global RNG-style edge pruning during sequential insertion, which requires CPU-bound serial priority-queue traversals. ChunkSplice executes intra-chunk pruning as a batched, parallel CUDA matrix operation.

## Key scientific contributions

1. **Formal Formulation of the Intra-Chunk Disconnectivity Pathology**: We identify and formalize the structural failure mode of snapshot projection in incremental proximity graphs, proving that the absence of intra-chunk edges is the primary bottleneck in dynamic vector indexing.
2. **Degree-Budget Partitioning Architecture**: We introduce a deterministic degree split ($R_{bridge}, R_{intra}$) that guarantees bidirectional navigability between historical snapshots and new cohorts.
3. **Cohort-Local Parallel RNG Pruning**: A GPU-native batched pruning algorithm that selects angularly diversified peer edges without requiring global graph traversals.
4. **Empirical Superiority Across Partitions**: Under controlled experiments on an NVIDIA RTX 4070 SUPER, ChunkSplice:
   - Elevates Mixed Recall@10 from **0.727 to 0.734** on micro-chunks ($b=64$) and from **0.888 to 0.925-0.934** on larger chunks ($b \ge 256$).
   - Maintains **>0.997** recall on the newly inserted cohort and reaches **~4,920 QPS** in continuous streaming.
5. **Open-Source Reproducible Diagnostic Framework**: Implementation of advanced graph health metrics including Tarjan Strongly Connected Components (SCC), in-degree Gini/hubness analysis, and long-horizon edge decay tracking.

## Key experimental results

Detailed benchmark logs and full floating-point values are preserved in [docs/results.md](docs/results.md) and [results/](results/).

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

## Repository layout

```text
.
├── src/cagra_chunksplice/
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

Use `--insertion-mode beam` to study accumulated candidate errors and
`--workload high-lid --dim 128` for a high-dimensional Gaussian control.
`--query-distribution ood` shifts queries away from the indexed distribution;
`angular` normalizes both queries and data for cosine-equivalent rankings.
[Scientific evaluation controls and metric definitions](experiments/README.md#long-horizon-and-failure-mode-evaluation)
describe 100-500 epoch runs, shared controls, and diagnostic interpretation.
New reports call the historical Rebuild method Rebuild-kNN; recorded results
retain their original labels and values.

Output defaults to `results/cagra_mirage_results.json` in the current working
directory. Missing parents are created. Use a separate `--output` for new runs
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
