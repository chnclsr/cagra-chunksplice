# Streaming updates: where ChunkSplice wins, where it does not

This document compares ChunkSplice with the other update strategies using only
recorded artifacts in [results/](../results/). Every number below is copied
from those JSON files; nothing was rerun or estimated to write it. Claims that
the recorded data cannot support are listed as open questions together with
the experiment that settles them.

## Update strategies compared

| Strategy | Representative | What an update does | Asymptotic update work (fixed chunk b, degree R, dim d, n old vectors) |
| --- | --- | --- | --- |
| Full rebuild | Rebuild-kNN, Rebuild-RNG, `cagra.build` on all vectors | Rebuild the whole graph over n + b vectors | O((n + b)^2 d) for exact kNN construction |
| Snapshot insertion (`exact`) | Naive, default `--insertion-mode exact` | New rows link to old rows only; capped reverse edges over all old rows | O(n b d) exact candidates + O(n b d) reverse scan |
| Snapshot insertion (`beam`) | Naive, `--insertion-mode beam` | Beam candidates; dense reverse pass still visits every old row | Bounded candidate search + O(n) reverse scan |
| Snapshot insertion (`bounded`) | Naive, `--insertion-mode bounded` | Beam candidates; reverse merge touches only old rows named by the new links (at most b * R rows) | O(1) in n for fixed chunk b, degree R and beam budgets |
| Graph repair | LocalRNG | Naive plus one/two-hop neighbour exchange over the active graph | Naive plus repair over touched neighbourhoods (multi-hop gathers) |
| **ChunkSplice** | `chunk_splice_extend` | Naive plus R_intra RNG-pruned peer edges inside the new chunk | Naive (in the selected insertion mode) plus O(b^2 d) peer kNN plus O(b (R + pool)^2) pruning; size-independent in n only with `bounded` |
| Native GPU extension | cuVS-CAGRA `extend` | Native CUDA insertion into the existing graph | Library-internal; measured only |
| CPU hierarchical | FAISS-HNSW, MIRAGE | Native multi-threaded CPU insertion | Library-internal; measured only |

With the default `exact` insertion mode, ChunkSplice's update is **linear in
n for a fixed chunk**: the exact new-to-old candidate search and the
reverse-proposal pass in `naive_extend` both scan every old row. `beam` mode
removes the exact candidate scan but the reverse pass still visits all old
rows. `--insertion-mode bounded` closes the remaining gap: beam search is
mandatory (there is no exact fallback) and reverse proposals merge only into
old rows named by the new links, so update work no longer grows with n. The
bounded path is implemented and covered by regression tests; an empirical
latency-vs-n curve for it is still unmeasured (see [Settling the open
questions](#settling-the-open-questions)).

## Evidence 1: ChunkSplice versus graph-degrading insertion (Naive)

Recorded stream (`stream_results.json`, base 2048, four updates of 256, RTX
4070 SUPER). This is the regime the method targets: incremental insertion that
leaves the new chunk internally disconnected.

| Metric at update 4 (n = 3072) | Naive | ChunkSplice |
| --- | --- | --- |
| Mixed Recall@10 | 0.926953136920929 | **0.9703125357627869** |
| Inserted-cohort Recall@10 | 0.88671875 | **1.0** |
| Newest-chunk Recall@10 | 0.8828125 | **0.999218761920929** |
| Original-base Recall@10 | **0.9671875238418579** | 0.940625011920929 |
| Inserted reachable fraction | 0.865234375 | **0.998046875** |
| Inserted orphans (in-degree 0) | 138 | **2** |
| Mixed QPS | 4505.773021663925 | **4919.841181206764** |
| Sum of four update medians (ms) | 23.83209999970859 | **22.562599999218946** |

In this stream ChunkSplice makes new data immediately findable (newest-chunk
recall 0.999 against 0.883) at no extra total update cost. The price is lower
original-base recall (0.941 against 0.967).

Controlled sweep (`sweep_results.json`, 4096 inserted vectors split into
chunks of 64 to 2048): ChunkSplice has higher mixed and inserted-cohort recall
than Naive in all four partitions, and shared-tail recall of 0.998 against
0.895-0.908 for chunks of 256 or more. Its trajectory update time is 1.7-3.0x
Naive's (for example 527.8 ms against 292.1 ms at chunk 256).

## Evidence 2: ChunkSplice versus graph repair (LocalRNG)

Benchmark (`cagra_mirage_results.json`, base 2048, four updates of 256):

| Stage | ChunkSplice update (ms) | LocalRNG update (ms) | Speedup | ChunkSplice R@10 | LocalRNG R@10 |
| --- | --- | --- | --- | --- | --- |
| 1 | 4.5662999982596375 | 44.12939999747323 | 9.7x | 0.9906250238418579 | 0.9917969107627869 |
| 2 | 12.941599998157471 | 124.3426000000909 | 9.6x | 0.987109363079071 | 0.989062488079071 |
| 3 | 13.886000000638887 | 112.23439999957918 | 8.1x | 0.940234363079071 | 0.9886718988418579 |
| 4 | 16.014000000723172 | 112.01530000107596 | 7.0x | 0.92578125 | 0.9820312857627869 |

ChunkSplice delivers LocalRNG-level recall at roughly a tenth of the update
cost for the first two updates. From stage 3 on, LocalRNG's global repair keeps
recall about 0.05 higher; ChunkSplice does not repair old rows.

## Evidence 3: native external baselines

All-native validation (`external_validation.json`, synthetic, base 2048, four
updates of 256, dim 32; HNSW/MIRAGE native CPU, cuVS-CAGRA native CUDA,
Naive/ChunkSplice PyTorch models):

| Method | Update (ms) | Mixed R@10, stages 1-4 | Final QPS |
| --- | --- | --- | --- |
| ChunkSplice (PyTorch model) | 10.626298249917454 | 0.991 / 0.993 / 0.984 / 0.975 | 5013.386820077181 |
| MIRAGE (native CPU) | 6.888773000355286 | 0.963 / 0.965 / 0.971 / 0.971 | 418112.09162927506 |
| FAISS-HNSW (native CPU) | 13.999516750118346 | 1.000 / 1.000 / 0.999 / 0.996 | 254931.07401390868 |
| cuVS-CAGRA extend (native CUDA) | 6.495410249044653 | 1.000 / 0.999 / 0.999 / 0.996 | 82412.25832658686 |

- ChunkSplice has higher mixed recall than native MIRAGE at every stage, and a
  lower update latency than native FAISS-HNSW.
- Native cuVS-CAGRA `extend` is both faster to update (6.5 ms against 10.6 ms)
  and more accurate (0.996 against 0.975) at this scale. cuVS does not require
  a full rebuild for appends; its `extend` API was used here.
- QPS compares a Python-launched PyTorch traversal with native C++/CUDA
  kernels and says nothing about the graph topology.
- The run is synthetic and small; it passes the native gate, not the
  SIFT1M/GloVe promotion requirement.

### cuVS-CAGRA `extend` versus ChunkSplice: architectural difference

There is no open point about what was compared; the two sides differ in kind,
not just in milliseconds:

- **cuVS-CAGRA `extend` is a native in-library update path, not a rebuild.**
  The adapter calls CUDA `cagra.build` once and then `cagra.extend` per batch
  (legacy or concatenated-dataset signature with `new_start_row`, configured
  by `ExtendParams(max_chunk_size=chunk_size)`), retaining caller-owned GPU
  storage across appends (`src/cagra_chunksplice/baselines/cagra.py`). No full
  rebuild runs on append; the library updates the existing graph with its own
  internal CUDA candidate search and kernels.
- **ChunkSplice is a PyTorch mechanism model, not a production index.**
  Python launches one tensor operation at a time (blocked exact or beam
  candidate search, degree budgeting, cohort-local RNG pruning); there are no
  fused CUDA graph-index kernels. Its purpose is to isolate the intra-chunk
  topology mechanism, and its timings include Python launch, allocation and
  synchronization overhead.
- **Reading the recorded gap.** At n = 3072 the native path updates in 6.5 ms
  against 10.6 ms for the PyTorch model, with higher mixed recall (0.996
  against 0.975). That is the expected signature of native kernels against a
  modeling harness at small scale; it does not show that degree budgeting is
  useless, only that this prototype cannot compete with native code on raw
  update throughput.
- **When to prefer which.** Use native cuVS-CAGRA `extend` for production GPU
  serving throughput. Use ChunkSplice (preferably `bounded` insertion) as a
  portable, inspectable model of intra-chunk connectivity: its peer-budget and
  bridge-reservation behavior, not its wall clock, is the contribution. The
  QPS column compares a Python-launched PyTorch traversal with native C++/CUDA
  kernels and says nothing about graph topology either way.

## Evidence 4: full rebuild at the recorded scale

At n of at most 6144, full rebuild is **faster** than ChunkSplice on the GPU:
sweep trajectory 23.5 ms against 70.1 ms at chunk 2048, and 6.7 ms against
16.0 ms at benchmark stage 4. The dense exact-kNN GEMM is efficient at these
sizes. Recall is mixed: unpruned Rebuild-kNN reaches only 0.645 mixed recall
in the sweep (ChunkSplice: 0.734-0.934), but 0.961 against 0.926 at
benchmark stage 4.

## Summary of supported claims

| Claim | Status |
| --- | --- |
| ChunkSplice removes the intra-chunk hole of snapshot insertion: new data is reachable and findable immediately | Supported (stream, sweep) |
| ChunkSplice matches graph repair recall at about 10x lower update cost for early updates | Supported for stages 1-2; recall gap opens later |
| ChunkSplice beats native MIRAGE recall on the recorded synthetic stream | Supported at n of at most 3072 |
| ChunkSplice update latency is O(1) or bounded independently of n | **Implemented, empirically unmeasured**: `--insertion-mode bounded` (mandatory beam candidates plus reverse merge over at most b * R linked rows) is size-independent by construction and covered by regression tests; no latency-vs-n curve has been recorded for it yet |
| ChunkSplice updates faster than full rebuild | **Not supported at recorded sizes**; rebuild is faster up to n = 6144. Asymptotically O(n b) against O(n^2); the crossover is unmeasured |
| ChunkSplice beats cuVS-CAGRA in streaming | **Not supported**: native `extend` was faster and more accurate in the only native run; the two are different kinds of artifact (native CUDA update path versus PyTorch mechanism model), so no wall-clock comparison can establish a topology advantage |
| ChunkSplice is the best method overall | **Not supported** by any recorded artifact |

## Settling the open questions

`--mode scaling` measures one fixed-size update against growing bases for
ChunkSplice, Naive, Rebuild-kNN, optional Rebuild-RNG, and native cuVS-CAGRA
`extend` and full `build`:

```bash
bash scripts/run_scaling.sh                       # native, bases 2048 .. 524288 (bounded PyTorch path)
python experiments/scaling.py --device cuda --scaling-bases 2048 8192 32768 131072 \
  --output results/scaling_pytorch.json           # PyTorch methods only
python experiments/scaling.py --device cuda --insertion-mode bounded \
  --scaling-bases 2048 8192 32768 131072 \
  --output results/scaling_bounded.json           # bounded PyTorch path only
```

The report records per-size latency samples and recall,
`scaling_exponent` (log-log slope over the measured sizes),
`speedup_vs_full_rebuild` and `first_measured_crossover_n`. The expected
exponents from the complexity table are about 1 for exact-mode ChunkSplice,
about 0 for bounded-mode ChunkSplice and about 2 for Rebuild-kNN; the run
decides whether and where ChunkSplice overtakes full rebuild, and how it
compares with native cuVS-CAGRA `extend`. The scaling runner honors
`--insertion-mode`, so run it once with the default and once with `bounded`
and compare the fitted exponents. Note that each timed ChunkSplice/Naive update
copies the initial graph into the reusable `out=` buffer (an O(n * degree)
memcpy, recorded as `update_memcpy` in the report), so measured `update_ms`
includes that copy even when the compute is independent of n. `naive_extend`
skips the copy only if `out[:start]` aliases `old_graph`, which mutates the
old rows in place. The beam search's boolean visited mask is independent of n
and scales with `query_batch * ef * steps * search_width`. Results belong in this document only after
the run completes.
