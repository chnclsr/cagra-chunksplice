# Local patches to upstream MIRAGE

`scripts/setup_external.py` applies the patches listed in [`series`](series),
in order, to the upstream checkout before building MIRAGE and the C ABI bridge.

| Patch | Upstream base | File | Purpose |
| --- | --- | --- | --- |
| `0001-indexmirage-add-single-insert.patch` | `6749e9c` ([dsg-uwaterloo/mirage](https://github.com/dsg-uwaterloo/mirage)) | `faiss/IndexMIRAGE.cpp` | `IndexMirage::add` inserted incremental batches twice, so `ntotal` did not grow by exactly the batch size. |

## Provenance

- Symptom: `results/external.error.json` (2026-10-22T14:22:00Z, local and
  ignored by Git) failed with `RuntimeError: native MIRAGE did not append the
  requested vectors`, raised by the `ntotal` check in
  `src/cagra_chunksplice/baselines/mirage_bridge.py`.
- Fix: applied by hand to `/root/mirage` (upstream `6749e9c`) on WSL2, rebuilt,
  and loaded as `/root/mirage/build/libmirage_bridge.so`.
- Evidence: `results/external_validation.json` (2026-10-22T14:36:21Z) records
  `ntotal` 2304, 2560, 2816 and 3072 for `faiss-mirage-cpu` at every stage.
- That run predates this patch series. Its MIRAGE metadata has no `patches`
  field; runs built with `setup_external.py` record each applied patch with its
  SHA-256 in the bridge metadata (`cs_mirage_patches`) and in
  `.external/build-manifest.json`.

## Exporting the patch

The patch content lives in the WSL tree that holds the fix. Export it once and
commit it:

```bash
bash scripts/export_mirage_patch.sh /root/mirage
```

The script checks that only `faiss/IndexMIRAGE.cpp` differs from `6749e9c` and
that the patch applies to a pristine checkout. Until the `.patch` file exists,
`setup_external.py` stops before building MIRAGE instead of silently producing
an unpatched bridge.

## Guard

Independently of the patch, `bridge.cpp` checks after every
`IndexMirage::add` that `ntotal` grew by exactly `n`, and fails with a message
naming this directory otherwise. An unpatched build therefore cannot produce
silent duplicate rows.
