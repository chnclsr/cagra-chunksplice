"""JSON persistence and complete-package provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cagra_chunksplice.reporting import save_report, source_provenance


class ReportingTests(unittest.TestCase):
    def test_creates_nested_output_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "run.json"
            value = {"status": "ok", "value": [1, 2.5, None]}
            save_report(path, value)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), value)
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_report(Path(directory) / "run.json", {"value": float("nan")})

    def test_provenance_hashes_all_package_sources(self):
        report = source_provenance()
        files = report["source_files_sha256"]
        self.assertIn("algorithms/chunk_splice.py", files)
        self.assertIn("runners/sweep.py", files)
        self.assertIn("cli.py", files)
        package = Path(__file__).resolve().parents[1] / "src" / "cagra_chunksplice"
        self.assertEqual(set(files), {p.relative_to(package).as_posix()
                                    for p in package.rglob("*.py")})
        for name, digest in files.items():
            self.assertEqual(digest, hashlib.sha256((package / name).read_bytes()).hexdigest())
        manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(report["source_sha256"], hashlib.sha256(manifest).hexdigest())
        self.assertEqual(report, source_provenance())
