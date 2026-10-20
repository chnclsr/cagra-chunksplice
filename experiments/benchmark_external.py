#!/usr/bin/env python3
"""Compare GPU dynamic graph insertion with CPU hierarchical add.

MIRAGE-ANNS and FAISS-HNSW always compute on CPU. --device selects the
PyTorch graph models/cuVS device; CPU runs are portable checks, not GPU results.
External reports default to results/external.json."""
import sys
from cagra_chunksplice.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--mode", "external", *sys.argv[1:]]))
