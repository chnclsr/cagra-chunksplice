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
python -m pip install -e .
python -m cagra_chunksplice --help
python -m cagra_chunksplice --device cpu --self-test-only
python -m unittest discover -s tests -v
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
| Exact blocked kNN | Shared topology reference and ground truth without a full global distance matrix. |
| Naive insertion | Snapshot new-to-old selection and capped reverse edges. |
| ChunkSplice | Peer slots, cohort-local RNG pruning and reserved old bridges. |
| LocalRNG | Bounded one/two-hop repair, neighbor exchange and optional backfill. |
| Beam search | Shared starts, beam capacity, expansion width and step budget. |
| Evaluation | Recall by cohort, graph structure, stream reachability and orphan counts. |
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
| `benchmark` | Rebuild, Naive, ChunkSplice, LocalRNG | Compare trajectories and query cohorts. |
| `sweep` | Rebuild, Naive, ChunkSplice | Vary insertion partitions of one shared corpus. |
| `stream` | Naive, ChunkSplice | Retain update timings and evaluate checkpoints. |

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
