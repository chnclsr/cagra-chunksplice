# Experiment artifacts

- `cagra_mirage_results.json`: historical benchmark.
- `sweep_results.json`: controlled insertion-partition sweep.
- `stream_results.json`: four-update stream.
- `*.error.json`: unsuccessful diagnostics, ignored by Git.

Historical JSON bytes, embedded paths and digests are retained verbatim.
Embedded paths describe the original execution context; they are not repository
links. See [recorded results](../docs/results.md).

Use `--output results/<run-name>.json` for new runs. Package runs include all
Python source hashes and an aggregate `source_sha256`; historical single-file
digests are not directly comparable to this aggregate digest.
