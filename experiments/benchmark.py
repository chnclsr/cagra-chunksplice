#!/usr/bin/env python3
"""Run the benchmark experiment after installing the package."""
import sys
from cagra_chunksplice.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--mode", "benchmark", *sys.argv[1:]]))
