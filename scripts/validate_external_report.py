#!/usr/bin/env python3
"""Independently check a saved external report without loading torch or CUDA."""
import argparse
import importlib.util
import json
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "src/cagra_chunksplice/promotion.py"
spec = importlib.util.spec_from_file_location("promotion", source)
promotion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promotion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("report must be a JSON object")
        result = promotion.validate_native_report(report)
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result))
