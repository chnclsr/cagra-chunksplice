# Experiment entry points

Install with `python -m pip install -e .`. Each script delegates to the packaged
CLI and selects the matching mode:

```console
python experiments/benchmark.py --device cpu --base 512 --chunks 2 --chunk-size 64 --output results/benchmark_debug.json
python experiments/sweep.py --sweep-total 4096 --sweep-chunk-sizes 64 256 1024 2048 --output results/sweep_reproduced.json
python experiments/stream.py --chunks 4 --chunk-size 256 --output results/stream_reproduced.json
python experiments/scaling.py --scaling-bases 2048 8192 32768 131072 --output results/scaling_pytorch.json
```

`scaling.py` inserts one `--chunk-size` update into each base and times
ChunkSplice and Naive against a full Rebuild-kNN (`--scaling-rebuild-rng` adds
Rebuild-RNG; `--scaling-cagra` adds cuVS-CAGRA `extend` and full `build`,
using `--external-backend`). It honors `--insertion-mode`: run once with the
default `exact` mode and once with `bounded`, keep the outputs in separate
files (for example `results/scaling_pytorch.json` and
`results/scaling_bounded.json`), and compare the fitted `scaling_exponent`
values (expected about 1 for exact, about 0 for bounded, about 2 for
Rebuild-kNN). No scaling run has been recorded yet; do not quote a crossover
until `first_measured_crossover_n` is measured. See [streaming
analysis](../docs/streaming_analysis.md).

CUDA is the default; add `--device cpu` for debugging.
`python -m cagra_chunksplice --mode <mode> ...` and `cagra-chunksplice --mode <mode> ...`
use the same runners. `--help` lists shared options. CPU debug timings are
not comparable to historical GPU measurements.

## Long-horizon and failure-mode evaluation

Run 100-500 insertion epochs with graph-based insertion candidates:

```console
python experiments/stress.py --chunks 100 --chunk-size 32 --workload high-lid --dim 128 --insertion-mode beam --insertion-ef 48 --insertion-steps 4 --stream-eval-every 10 --output results/stress_high_lid.json
python experiments/stress.py --chunks 500 --chunk-size 16 --query-distribution angular --insertion-mode beam --output results/stress_angular.json
python experiments/stream.py --chunks 500 --chunk-size 16 --query-distribution ood --ood-shift 4 --insertion-mode beam --output results/stress_ood.json
```

Stress mode defaults to 100 epochs and includes Rebuild-kNN, Rebuild-RNG, Naive,
and ChunkSplice at every update. Stream mode retains only the two incremental
methods unless `--stream-rebuild` is supplied. The latter can reduce the cost of
500-epoch exploratory runs. There is no algorithmic epoch limit. Global exact
builds and truth still scale with the active corpus; these are reference
experiments rather than production-throughput measurements.

Rebuild-kNN is the former Rebuild method, renamed in new reports to distinguish
it from Rebuild-RNG. Historical result artifacts remain unchanged. Rebuild-RNG
constructs a global exact nearest-neighbor pool (`--rebuild-pool`, default 96),
applies greedy RNG pruning, merges reversed surviving edges, and re-prunes to
the common degree cap with `--rebuild-alpha` (default 1). Rejected candidates
are not backfilled. This is a directed, candidate-restricted heuristic, not the
exact global relative neighborhood graph or a guaranteed strongly connected
index. Reverse proposals can be removed by the final degree cap. All methods
start from the same exact kNN base snapshot; update times exclude that initial
build. Rebuild-RNG timings include pool construction and host reverse merging.

The default `--insertion-mode exact` retains the original oracle insertion
control. Beam mode uses the existing graph search on the immutable predecessor,
with fixed seeded original-base entries and independent `--insertion-ef`,
`--insertion-search-width`, and `--insertion-steps` budgets. It never replaces
missed candidates with exact neighbors. Missing candidates remain absent.
Bounded mode uses the same beam candidate search and additionally restricts
reverse proposals to old rows named by the new links (at most b * degree
rows), so update work is independent of n for fixed chunk, degree and beam
budgets; it is the mode to use for size-independent update claims and for the
`--mode scaling` bounded curve.
ChunkSplice still computes its within-chunk peer pool exactly; LocalRNG retains
its bounded exact local repair. Candidate errors therefore affect later graph
snapshots rather than being reset by a global insertion oracle.

`--workload high-lid --dim 128` generates full-dimensional isotropic Gaussian
points; this is a synthetic high-dimensional control, not an empirical LID
estimate. `--query-distribution ood` translates all queries along coordinate 0
by `--ood-shift`. `--query-distribution angular` normalizes both corpus and
queries to unit length, so squared L2 and cosine distance induce the same
ranking. Query truth is always recomputed against the complete active corpus.

At the first, last, and every `--stream-eval-every` checkpoint, reports include:

- Exact iterative Kosaraju SCC counts, largest SCC fraction, sink SCC counts,
  and largest sink SCC fraction. Sink SCCs alone do not establish a search trap.
- Population in-degree Gini and skewness, maximum in-degree, reachability, and
  orphan counts. Constant-degree graphs have zero Gini and skewness.
- Euclidean edge-length means and 90th percentiles, cross-epoch bridge counts,
  valid-edge fractions, slot occupancy, and older/newer cohort edge counts.
  A cohort bridge is not a graph-theoretic cut edge.
- Directed edge retention relative to the initial base graph and the preceding
  evaluated checkpoint. Bridges use the first nonempty evaluated bridge
  snapshot as their initial reference, whose vertex count is recorded.
  Retention measures presence at checkpoints, not uninterrupted edge lifetime.
  Length ratios compare current population means with reference means.
- Recall@10 for query sources from the base and early, middle, and recent
  insertion-epoch thirds (`--age-queries` probes per nonempty stratum).
  Source ranges are recorded. Exact neighbors may belong to other ages,
  particularly for OOD queries. These are cross-sectional strata, not fixed
  longitudinal query panels.

All diagnostics and exact truths are outside update/search timings. Checkpoint
history permits analysis of bridge saturation, topology fragmentation, and
hubness alongside age-related recall loss. No numerical outcome is presumed.
