# Experimental methodology

## Scope and scientific provenance

This is a **mechanism simulation implemented with PyTorch**, not an implementation, modification, or performance benchmark of NVIDIA cuVS/CAGRA, the complete MIRAGE-ANNS index, or HNSW. Exact blocked k-nearest-neighbor construction and exact new-to-old neighbor selection deliberately remove approximate insertion-search error from the comparison. Python launches tensor operations; the implementation does not provide fused CUDA graph-index kernels.

The upstream intellectual contributions are explicitly acknowledged:

- **NVIDIA cuVS and CAGRA** provide the GPU graph-indexing context. CAGRA's graph construction and search algorithms belong to their authors and NVIDIA contributors. The official cuVS API documents graph extension; this prototype's insertion model is not evidence about the behavior of that implementation. See the [NVIDIA cuVS CAGRA API](https://docs.nvidia.com/cuvs/api-reference/c-api-neighbors-cagra), [cuVS source](https://github.com/NVIDIA/cuvs), and [CAGRA paper](https://doi.org/10.1109/ICDE60146.2024.00323).
- **MIRAGE-ANNS**, by Sairaj Voruganti and M. Tamer Özsu, motivates the combination of incremental construction and refinement. The optional `LocalRNG` path adapts neighbor exchange and a relative-neighborhood pruning heuristic to a bounded local tensor computation. It does not reproduce MIRAGE's complete hierarchical design. See the [upstream implementation](https://github.com/dsg-uwaterloo/mirage), [author's thesis](https://hdl.handle.net/10012/21582), and [paper](https://doi.org/10.1145/3725325).
- **HNSW**, by Yury A. Malkov and Dmitry A. Yashunin, is foundational prior work on incremental navigable proximity graphs and neighbor selection. This repository neither implements its hierarchy nor measures an HNSW baseline. See the [original paper](https://arxiv.org/abs/1603.09320).

**Bibliographic clarification:** the nearest-neighbor work relevant to this source is MIRAGE-ANNS, published in *Proceedings of the ACM on Management of Data*, 3(3), 2025, in the SIGMOD research track. The [official SIGMOD proceedings listing](https://2025.sigmod.org/toc-3-3.html) identifies its title and authors. It should not be attributed to ACM SIGCOMM or *ACM Transactions on Computer Systems* (TOCS); those venue labels do not identify the cited ANN work.


### Representation and computational costs

Vectors use `float32`; adjacency uses integer IDs, fixed row capacity, and `-1` padding. Self-loops, duplicate valid neighbors, and out-of-range IDs are rejected by graph validation.

Let `N` be the active population, `D` the dimension, `B` the incoming chunk size, and `L` the LocalRNG neighborhood size.

| Operation | Principal computational or memory cost |
| --- | --- |
| Exact graph construction | Quadratic distance work, `O(N²D)`; distance blocks use `O(distance_batch × N)` scratch instead of one full distance matrix. |
| Naive insertion | New-to-old distance work of order `O(B × N_old × D)`, reverse-edge processing, and graph copying. The implementation also materializes an `N_old × B` Boolean proposal mask. |
| ChunkSplice addition | Intra-chunk distance work of order `O(B²D)`, plus candidate merge and selection; this is additional work on top of Naive insertion. |
| LocalRNG refinement | Local pairwise geometry of order `O(L²D)`, plus exchange and candidate pruning. Candidate-pair pruning scratch scales with row-block size and the square of candidate count. Graph snapshots are still cloned globally. |
| Search | Candidate-based distance evaluation with an `O(query_batch × N)` expanded-node bitmap, bounded beam capacity, expansion width, and rounds. |

The LocalRNG cap limits the repair neighborhood, not the total cost or memory of the insertion procedure. It must exceed the incoming chunk size in benchmark mode.


## Measurement protocol

### Timing methodology

Each operation receives one complete untimed warmup at its current shape, followed by three timed repetitions. CUDA is synchronized on the selected device immediately before and after each timed call. Reported latency is the median wall time in milliseconds, including Python launches, allocations, and update/reverse-edge/repair work. Data generation, transfers, exact ground truth, validation, and diagnostics are excluded.

Repeated updates start from the same immutable graph snapshot. Method order rotates by stage or sweep configuration. Sweep `update_ms` measures the entire insertion trajectory, excluding initial construction. Stream `total_update_ms` is a **sum of per-update medians**, not an independently measured end-to-end latency. QPS is mixed-cohort batched throughput, calculated as `256 × 1000 / query_ms`; it is not single-query latency.

Raw samples and population standard deviation, coefficient of variation, minimum, and maximum remain available in the linked JSON. Three repetitions do not establish confidence intervals. For example, the 64-vector Naive sweep query samples are `[62.665699999342905, 108.62289999931818, 114.5500999991782]` ms, with coefficient of variation `0.2433689507320059`. Final stream ChunkSplice mixed-query samples are `[82.82890000009502, 60.99940000058268, 46.938999999838416]` ms, with coefficient of variation `0.23220925540155238`; timing comparisons should account for this dispersion.

### Metric definitions

- **Recall@10 (R@10):** mean fraction of each query's exact top-10 IDs recovered by graph traversal. Ground truth is computed over all active vectors.
- **Mixed cohort:** 128 noisy original-base probes plus 128 probes drawn from all inserted centers so far. `new_query_recall10` describes this cumulative inserted cohort, not only the latest chunk.
- **Newest-data cohort:** 128 separate probes. In the sweep these use the fixed shared tail `[6080, 6144)`, which lies in every partition's final chunk. In the stream they use the newest complete update.
- **Intra-chunk edges:** directed edges whose endpoints belong to the same inserted chunk, counted over all updates. These are not counts of all edges among inserted vectors.
- **Intra demand:** fraction of the newest chunk's vectors' exact top-10 neighbors that lie inside that chunk, with self-neighbors excluded.
- **Intra coverage:** fraction of those demanded same-chunk neighbors explicitly retained in the outgoing adjacency. A value of `1` is conditional coverage, not complete graph connectivity.
- **Earlier-to-newest edges:** `old_to_new_edges`, from every vector preceding the latest chunk into that chunk; “old” here includes previously inserted vectors.
- **Reachability:** exact directed traversal from 64 fixed original-base anchors, independent of the search entry policy and search budget.
- **Orphan:** a vertex with zero incoming edges. A disconnected cycle can be unreachable without containing an orphan.


## Limitations and validity

1. **Experimental abstraction.** Exact snapshot insertion isolates a specific topology mechanism. The observations do not establish a defect in cuVS, reproduce its extension path, or compare this prototype against production CAGRA.
2. **Restricted evidence.** The supplied results use one synthetic drift workload, one seed, one GPU, and a small active population. They contain neither real embedding datasets nor multi-seed uncertainty estimates.
3. **Short stream.** Four updates cannot establish long-term stability. Larger streams are supported but need separately recorded evaluation.
4. **Search-entry dependence.** The recorded `all` policy can start inside an inserted component. High recall can coexist with unreachability from original-base anchors; the independent reachability diagnostic addresses a different question.
5. **Connectivity versus navigation.** Budgeting same-chunk edges reserves old-cohort bridges but still changes traversal. Neither perfect intra-edge coverage nor exact kNN construction guarantees optimal finite-budget search.
6. **Timing scale.** Python launch, graph-copy, allocation, and synchronization overhead matter at these sizes. There is no demonstrated asymptotic speedup, production throughput claim, or concurrency guarantee.
7. **Memory scale.** Blocked distances avoid one full global distance matrix, but reverse proposals, global graph snapshots, candidate-pair tensors, and the expanded bitmap remain material allocations.
8. **Reproducibility limits.** Seeding does not imply bitwise determinism; deterministic algorithms are disabled, and exact-distance ties may resolve differently across devices or software versions.
9. **Unimplemented systems features.** No deletions, persistence of a deployable index, concurrent query/update coordination, multi-GPU execution, quantization, production service, or HNSW hierarchy is provided.
10. **Separate experimental paths.** The MIRAGE-inspired LocalRNG path is present in benchmark mode but absent from the supplied sweep and stream. No LocalRNG result is inferred from them.


## References

1. Hiroyuki Ootomo, Akira Naruse, Corey Nolet, Ray Wang, Tamas Feher, and Yong Wang. 2024. **CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs.** In *2024 IEEE 40th International Conference on Data Engineering (ICDE)*, 4236–4247. [DOI: 10.1109/ICDE60146.2024.00323](https://doi.org/10.1109/ICDE60146.2024.00323). [Author manuscript](https://arxiv.org/abs/2308.15136). Bibliographic data follow [NVIDIA's official references](https://docs.nvidia.com/cuvs/user-guide/references).
2. NVIDIA and cuVS contributors. **cuVS: GPU-accelerated vector search and clustering.** [Source repository](https://github.com/NVIDIA/cuvs). [CAGRA API and extension interface](https://docs.nvidia.com/cuvs/api-reference/c-api-neighbors-cagra).
3. Sairaj Voruganti and M. Tamer Özsu. 2025. **MIRAGE-ANNS: Mixed Approach Graph-based Indexing for Approximate Nearest Neighbor Search.** *Proceedings of the ACM on Management of Data* 3(3), Article 188, 1–27; SIGMOD 2025 research track. [DOI: 10.1145/3725325](https://doi.org/10.1145/3725325). [Official proceedings](https://2025.sigmod.org/toc-3-3.html). [Upstream implementation](https://github.com/dsg-uwaterloo/mirage).
4. Sairaj Voruganti. 2025. **MIRAGE-ANNS: Mixed Approach Graph-based Indexing for Approximate Nearest Neighbour Search.** Thesis, University of Waterloo. [Institutional record](https://hdl.handle.net/10012/21582). The thesis is a separate artifact from the coauthored journal article.
5. Yury A. Malkov and Dmitry A. Yashunin. **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.** *IEEE Transactions on Pattern Analysis and Machine Intelligence* 42(4), 824–836, 2020; first published online in 2018. [DOI: 10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473). [Author manuscript, initially submitted in 2016](https://arxiv.org/abs/1603.09320).
