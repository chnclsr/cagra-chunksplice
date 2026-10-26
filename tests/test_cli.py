"""Argument validation and command entry points; no torch import required."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cagra_chunksplice.cli import arguments, main


ROOT = Path(__file__).resolve().parents[1]


class ArgumentTests(unittest.TestCase):
    def test_defaults(self):
        args = arguments([])
        self.assertEqual(args.mode, "benchmark")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.output, Path("results/cagra_mirage_results.json"))

    def test_bounded_insertion_mode_is_accepted(self):
        args = arguments(["--insertion-mode", "bounded"])
        self.assertEqual(args.insertion_mode, "bounded")

    def test_modes_and_local_rng_options(self):
        for mode in ("benchmark", "sweep", "stream", "stress"):
            with self.subTest(mode=mode):
                args = arguments(["--mode", mode, "--alpha", "1.2", "--route", "4",
                                  "--cohort-protect", "--backfill"])
                self.assertEqual(args.mode, mode)
                self.assertEqual(args.route, 4)
                self.assertTrue(args.backfill)

    def test_invalid_parameters(self):
        cases = [
            ["--base", "0"], ["--rebuild-pool", "1"], ["--rebuild-alpha", "nan"],
            ["--ood-shift", "inf"], ["--age-queries", "0"],
            ["--insertion-mode", "beam", "--insertion-ef", "10"],
            ["--insertion-mode", "beam", "--insertion-search-width", "100"],
            ["--insertion-mode", "bounded", "--insertion-ef", "10"],
            ["--insertion-mode", "bounded", "--insertion-search-width", "100"],
 ["--degree", "9"], ["--ef", "9"],
            ["--alpha", "nan"], ["--splice-alpha", "inf"],
            ["--route", "1"], ["--local-cap", "256"],
            ["--splice-intra", "24"], ["--sweep-chunk-sizes", "0"],
            ["--mode", "sweep", "--sweep-total", "8"],
            ["--mode", "stream", "--verify-effect"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    arguments(argv)
                self.assertEqual(caught.exception.code, 2)

    def test_stress_default_horizon_and_override(self):
        self.assertEqual(arguments(["--mode", "stress"]).chunks, 100)
        self.assertEqual(arguments(["--mode", "stress", "--chunks", "500"]).chunks, 500)
        self.assertEqual(arguments(["--mode", "stress", "--chunks=3"]).chunks, 3)

    def test_help_does_not_require_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as caught:
                    main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--splice-intra", output.getvalue())

    def test_missing_torch_preserves_successful_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "run.json"
            output.parent.mkdir()
            output.write_text('{"status":"ok"}', encoding="utf-8")
            with patch.dict(sys.modules, {"torch": None}):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = main(["--device", "cpu", "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.read_text())["status"], "ok")
            error = json.loads(output.with_suffix(".error.json").read_text(encoding="utf-8"))
            self.assertFalse(error["benchmark_completed"])
            self.assertEqual(error["error_type"], "ModuleNotFoundError")

    def test_module_legacy_and_experiment_help(self):
        commands = [
            ["-m", "cagra_chunksplice"], [str(ROOT / "cagra_mirage_prototype.py")],
            *[[str(ROOT / "experiments" / (name + ".py"))]
              for name in ("benchmark", "sweep", "stream", "stress")],
        ]
        # Also exercise module resolution from outside the checkout.
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run(
                        [sys.executable, *command, "--help"], cwd=directory, env=env,
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--mode", result.stdout)
