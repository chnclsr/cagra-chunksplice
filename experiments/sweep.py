#!/usr/bin/env python3
"""Run the sweep experiment after installing the package."""
import sys
from cagra_chunksplice.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--mode", "sweep", *sys.argv[1:]]))
