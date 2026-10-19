# Experiment entry points

Install with `python -m pip install -e .`. Each script delegates to the packaged
CLI and selects the matching mode:

```console
python experiments/benchmark.py --device cpu --base 512 --chunks 2 --chunk-size 64 --output results/benchmark_debug.json
python experiments/sweep.py --sweep-total 4096 --sweep-chunk-sizes 64 256 1024 2048 --output results/sweep_reproduced.json
python experiments/stream.py --chunks 4 --chunk-size 256 --output results/stream_reproduced.json
```

CUDA is the default; add `--device cpu` for debugging.
`python -m cagra_chunksplice --mode <mode> ...` and `cagra-chunksplice --mode <mode> ...`
use the same runners. `--help` lists shared options. CPU debug timings are
not comparable to historical GPU measurements.
