#!/usr/bin/env python3
"""Run external index comparisons after installing the package."""
import sys
from cagra_chunksplice.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--mode", "external", *sys.argv[1:]]))
