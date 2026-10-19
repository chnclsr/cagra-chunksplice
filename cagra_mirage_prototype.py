#!/usr/bin/env python3
"""Compatibility entry point. Prefer python -m cagra_chunksplice after installation."""
from pathlib import Path
import sys

# Keep historical checkout commands usable before an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cagra_chunksplice.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
