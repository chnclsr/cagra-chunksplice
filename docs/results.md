# Recorded experiment results

The JSON files in [results/](../results/) are the authoritative records.
Tables below copy their numeric fields using Python's full float representation,
without decimal rounding. No experiment was rerun to generate this document.
These historical measurements do not validate the refactored package.

## Provenance

Sweep and stream record different historical single-file source digests.
New package runs hash all Python sources; see [development](development.md).

| Artifact | Recorded source SHA-256 | Started (UTC) | Completed (UTC) |
| --- | --- | --- | --- |
| [sweep_results.json](../results/sweep_results.json) | 9b203948613f4734f65e906d36d11ec41b6210d7cabd119c79d7e9b09cf9053c | 2026-10-19T13:04:46.582725+00:00 | 2026-10-19T13:05:11.679631+00:00 |
| [stream_results.json](../results/stream_results.json) | ff932e66b873a5ae62b718154ee050f763ca4f67d26407bc54eeec1feb3be8d8 | 2026-10-19T12:59:30.522871+00:00 | 2026-10-19T12:59:32.222127+00:00 |
| [cagra_mirage_results.json](../results/cagra_mirage_results.json) | cda7cb1d0323ea936e0792117f7bc31b1a3e3a19f7b83038552da9047deb0e35 | 2026-10-19T12:52:52.977124+00:00 | 2026-10-19T12:53:05.400244+00:00 |
| [external_validation.json](../results/external_validation.json) | d1830af7c7515bab42dff11fea441674421d3d8bfcdc51f568bbd7a70160b6e1 | 2026-10-22T14:36:21.131331+00:00 | 2026-10-22T14:36:25.590272+00:00 |

## Controlled chunk-size sweep

A shared ordered corpus, queries, ground truth and search entries; only insertion partitioning changes.

| Chunk size | Updates | Method | Trajectory update (ms) | Update ms / node | Mixed query (ms) | Mixed QPS |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 64 | Rebuild | 279.39229999901727 | 0.06821101074194758 | 57.13390000164509 | 4480.702349964362 |
| 64 | 64 | Naive | 621.9703999995545 | 0.15184824218739124 | 108.65470000135247 | 2356.087679564837 |
| 64 | 64 | ChunkSplice | 1880.235900000116 | 0.45904196777346584 | 99.9547999999777 | 2561.1576432553225 |
| 256 | 16 | Naive | 292.12310000002617 | 0.07131911621094389 | 109.38399999940884 | 2340.3788488388022 |
| 256 | 16 | ChunkSplice | 527.8120000002673 | 0.12886035156256526 | 111.70979999951669 | 2291.652120056679 |
| 256 | 16 | Rebuild | 150.11039999808418 | 0.03664804687453227 | 111.58029999933206 | 2294.311809535666 |
| 1024 | 4 | ChunkSplice | 126.94390000251587 | 0.030992163086551727 | 116.11600000105682 | 2204.691859844208 |
| 1024 | 4 | Rebuild | 40.34060000049067 | 0.009848779296994792 | 112.49209999732557 | 2275.7153613994783 |
| 1024 | 4 | Naive | 72.03149999986636 | 0.017585815429654872 | 116.30710000099498 | 2201.069410189146 |
| 2048 | 2 | Rebuild | 23.522899999079527 | 0.005742895507587775 | 66.44349999987753 | 3852.8975746381793 |
| 2048 | 2 | Naive | 40.91130000233534 | 0.00998811035213265 | 115.14299999907962 | 2223.3223035881147 |
| 2048 | 2 | ChunkSplice | 70.11650000276859 | 0.017118286133488425 | 114.12310000014259 | 2243.191781503308 |

| Chunk size | Method | Mixed R@10 | Original-base R@10 | Inserted R@10 | Shared-tail R@10 |
| --- | --- | --- | --- | --- | --- |
| 64 | Rebuild | 0.645312488079071 | 0.56640625 | 0.7242187857627869 | 0.8304687738418579 |
| 64 | Naive | 0.727734386920929 | 0.653124988079071 | 0.8023437857627869 | 0.9921875 |
| 64 | ChunkSplice | 0.734375 | 0.633593738079071 | 0.835156261920929 | 0.9921875 |
| 256 | Naive | 0.888671875 | 0.9046875238418579 | 0.8726562857627869 | 0.895312488079071 |
| 256 | ChunkSplice | 0.925000011920929 | 0.885937511920929 | 0.964062511920929 | 0.9976562857627869 |
| 256 | Rebuild | 0.645312488079071 | 0.56640625 | 0.7242187857627869 | 0.8304687738418579 |
| 1024 | ChunkSplice | 0.928515613079071 | 0.897656261920929 | 0.9593750238418579 | 0.9976562857627869 |
| 1024 | Rebuild | 0.645312488079071 | 0.56640625 | 0.7242187857627869 | 0.8304687738418579 |
| 1024 | Naive | 0.8812500238418579 | 0.9046875238418579 | 0.8578125238418579 | 0.905468761920929 |
| 2048 | Rebuild | 0.645312488079071 | 0.56640625 | 0.7242187857627869 | 0.8304687738418579 |
| 2048 | Naive | 0.8863281607627869 | 0.910937488079071 | 0.8617187738418579 | 0.9078125357627869 |
| 2048 | ChunkSplice | 0.934374988079071 | 0.9039062857627869 | 0.96484375 | 0.9976562857627869 |

| Chunk size | Method | Intra edges | Latest intra demand | Latest intra coverage | Earlier-to-latest edges | Mean out-degree |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | Rebuild | 22771 | 0.22343750298023224 | 1.0 | 1436 | 24.0 |
| 64 | Naive | 0 | 0.22343750298023224 | 0.0 | 1199 | 24.0 |
| 64 | ChunkSplice | 20831 | 0.22343750298023224 | 0.993006993006993 | 1023 | 24.0 |
| 256 | Naive | 0 | 0.99609375 | 0.0 | 1275 | 24.0 |
| 256 | ChunkSplice | 24576 | 0.99609375 | 0.5388235294117647 | 1069 | 24.0 |
| 256 | Rebuild | 97684 | 0.99609375 | 1.0 | 1017 | 24.0 |
| 1024 | ChunkSplice | 24576 | 0.9966797232627869 | 0.5384087791495199 | 4210 | 24.0 |
| 1024 | Rebuild | 97684 | 0.9966797232627869 | 1.0 | 5055 | 24.0 |
| 1024 | Naive | 0 | 0.9966797232627869 | 0.0 | 4924 | 24.0 |
| 2048 | Rebuild | 97684 | 0.9955078363418579 | 1.0 | 9751 | 24.0 |
| 2048 | Naive | 0 | 0.9955078363418579 | 0.0 | 9637 | 24.0 |
| 2048 | ChunkSplice | 24576 | 0.9955078363418579 | 0.5390916225230528 | 8408 | 24.0 |

ChunkSplice's mixed and inserted-cohort recall exceed Naive's in all four
recorded partitions; original-base recall is lower. Rebuild is faster than
both incremental paths at this scale. These are conditional observations.

## Streaming experiment

The recorded stream contains four updates of 256 vectors. Quality is evaluated
at updates 1 and 4. Totals sum update medians; they are not end-to-end samples.

| Update | Active vectors | Method | Update (ms) | Evaluated |
| --- | --- | --- | --- | --- |
| 1 | 2304 | Naive | 9.447799999179551 | True |
| 1 | 2304 | ChunkSplice | 4.725700000562938 | True |
| 2 | 2560 | ChunkSplice | 5.957900000794325 | False |
| 2 | 2560 | Naive | 4.400300000270363 | False |
| 3 | 2816 | Naive | 4.651199997169897 | False |
| 3 | 2816 | ChunkSplice | 5.816799999593059 | False |
| 4 | 3072 | ChunkSplice | 6.062199998268625 | True |
| 4 | 3072 | Naive | 5.332800003088778 | True |

| Method | Sum of update medians (ms) |
| --- | --- |
| Naive | 23.83209999970859 |
| ChunkSplice | 22.562599999218946 |

| Update | Method | Mixed R@10 | Original-base R@10 | Inserted R@10 | Newest R@10 | Mixed QPS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Naive | 0.9644531607627869 | 0.9859375357627869 | 0.9429687857627869 | 0.940625011920929 | 3862.815721079536 |
| 1 | ChunkSplice | 0.9921875 | 0.984375 | 1.0 | 1.0 | 4870.188548635637 |
| 4 | ChunkSplice | 0.9703125357627869 | 0.940625011920929 | 1.0 | 0.999218761920929 | 4919.841181206764 |
| 4 | Naive | 0.926953136920929 | 0.9671875238418579 | 0.88671875 | 0.8828125 | 4505.773021663925 |

| Update | Method | Reachable | Unreachable | Inserted reachable fraction | Orphans | Inserted orphans | Newest orphans |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Naive | 2178 | 126 | 0.92578125 | 114 | 19 | 19 |
| 1 | ChunkSplice | 2197 | 107 | 1.0 | 95 | 0 | 0 |
| 4 | ChunkSplice | 2960 | 112 | 0.998046875 | 97 | 2 | 1 |
| 4 | Naive | 2830 | 242 | 0.865234375 | 230 | 138 | 27 |

At the final checkpoint, ChunkSplice has higher mixed and inserted-cohort
recall and lower original-base recall than Naive. Full reachability of inserted
vectors does not imply full graph connectivity. JSON contains raw samples,
newest-query timing, structural diagnostics and complete environments.

## Benchmark experiment (cagra_mirage_results.json)

`cagra_mirage_results.json` records a PyTorch mechanism-simulation benchmark
(`status: "ok"`, `experiment: "benchmark"`), not native cuVS CAGRA or full
MIRAGE. Configuration: `device: cuda`, base 2048, 4 chunks of 256 vectors,
dim 32, 256 queries, degree 24, ef 64, search width 4, 16 steps, 3 repeats,
seed 42, entry policy `all`, workload `drift`. Environment: RTX 4070 SUPER,
`torch 2.11.0+cu128`, CUDA 12.8, `Windows-11`, float32, squared L2.
Initial build took 3.048899998248089 ms; the base query scored
recall10 0.991015613079071 at 62.438199998723576 ms (4100.054133611049 QPS).

| Stage | n | Method | Update (ms) | Mixed query (ms) | Mixed QPS | Mixed R@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2304 | Rebuild | 1.6620999995211605 | 71.90020000052755 | 3560.490791376403 | 0.989453136920929 |
| 1 | 2304 | Naive | 4.908500002784422 | 51.354799998080125 | 4984.928380785641 | 0.964062511920929 |
| 1 | 2304 | ChunkSplice | 4.5662999982596375 | 48.55739999766229 | 5272.110945238515 | 0.9906250238418579 |
| 1 | 2304 | LocalRNG | 44.12939999747323 | 111.32319999887841 | 2299.6105034941434 | 0.9917969107627869 |
| 2 | 2560 | Naive | 11.334200000419514 | 112.0359999986249 | 2284.9798279405018 | 0.9371094107627869 |
| 2 | 2560 | ChunkSplice | 12.941599998157471 | 113.3903999980248 | 2257.6867177861564 | 0.987109363079071 |
| 2 | 2560 | LocalRNG | 124.3426000000909 | 116.14399999962188 | 2204.1603526728322 | 0.989062488079071 |
| 2 | 2560 | Rebuild | 3.918900001735892 | 111.48440000033588 | 2296.285399564681 | 0.986328125 |
| 3 | 2816 | ChunkSplice | 13.886000000638887 | 112.0072000012442 | 2285.567356358844 | 0.940234363079071 |
| 3 | 2816 | LocalRNG | 112.23439999957918 | 119.54820000028121 | 2141.395688093989 | 0.9886718988418579 |
| 3 | 2816 | Rebuild | 4.070899998623645 | 117.7733000004082 | 2173.667546032188 | 0.9390625357627869 |
| 3 | 2816 | Naive | 13.757500000792788 | 124.83010000141803 | 2050.787430251934 | 0.92578125 |
| 4 | 3072 | LocalRNG | 112.01530000107596 | 117.03829999896698 | 2187.3181685162854 | 0.9820312857627869 |
| 4 | 3072 | Rebuild | 6.705499999952735 | 113.49479999989853 | 2255.6099486516464 | 0.9613281488418579 |
| 4 | 3072 | Naive | 14.35219999984838 | 113.82649999723071 | 2249.036911494496 | 0.928906261920929 |
| 4 | 3072 | ChunkSplice | 16.014000000723172 | 114.43709999730345 | 2237.0367652276427 | 0.92578125 |

| Stage | Method | Original-base R@10 | Inserted R@10 | Latest-chunk R@10 |
| --- | --- | --- | --- | --- |
| 1 | Rebuild | 0.9789062738418579 | 1.0 | 1.0 |
| 1 | Naive | 0.983593761920929 | 0.944531261920929 | 0.9296875 |
| 1 | ChunkSplice | 0.981249988079071 | 1.0 | 1.0 |
| 1 | LocalRNG | 0.983593761920929 | 1.0 | 1.0 |
| 2 | Naive | 0.971875011920929 | 0.90234375 | 0.81640625 |
| 2 | ChunkSplice | 0.9742187857627869 | 1.0 | 1.0 |
| 2 | LocalRNG | 0.9781250357627869 | 1.0 | 1.0 |
| 2 | Rebuild | 0.97265625 | 1.0 | 1.0 |
| 3 | ChunkSplice | 0.9429687857627869 | 0.9375 | 1.0 |
| 3 | LocalRNG | 0.977343738079071 | 1.0 | 1.0 |
| 3 | Rebuild | 0.940625011920929 | 0.9375 | 1.0 |
| 3 | Naive | 0.9703125357627869 | 0.8812500238418579 | 0.859375 |
| 4 | LocalRNG | 0.964062511920929 | 1.0 | 1.0 |
| 4 | Rebuild | 0.9242187738418579 | 0.9984375238418579 | 1.0 |
| 4 | Naive | 0.9585937857627869 | 0.899218738079071 | 0.8609375357627869 |
| 4 | ChunkSplice | 0.9453125 | 0.90625 | 0.8359375 |

Timing uses median wall milliseconds across 3 repeats after one untimed
warmup call per shape, with device synchronization around each timed call.
QPS is mixed-cohort batched throughput. See the JSON for per-sample values,
latest-chunk timing, graph diagnostics and the full environment.

## Other artifacts

[external_validation.json](../results/external_validation.json) is the
completed all-native external comparison; see
[external baselines](external_baselines.md#local-execution-status-2026-10-22)
for its status table, measurement protocol and the MIRAGE append fix.
A local `cagra_mirage_results.error.json` is an unsuccessful diagnostic, not
completed evidence. Error files are ignored by Git.
Historical JSON bytes were preserved during the refactor.
