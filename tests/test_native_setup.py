"""Native setup bookkeeping that runs without compilers, GPUs or network."""
import importlib.util
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_external", ROOT / "scripts/setup_external.py")
setup_external = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_external)


class NativeSetupTests(unittest.TestCase):
    def test_series_targets_pinned_mirage_revision(self):
        series = setup_external.read_series()
        self.assertEqual([path.name for path, _ in series], ["0001-indexmirage-add-single-insert.patch"])
        self.assertEqual({revision for _, revision in series}, {setup_external.MIRAGE_DEFAULT_REF})

    def test_library_name_matches_cmake_output_and_recorded_run(self):
        cmake = (ROOT / "native/mirage_bridge/CMakeLists.txt").read_text(encoding="utf-8")
        output = re.search(r"OUTPUT_NAME (\w+)", cmake).group(1)
        self.assertEqual(setup_external.MIRAGE_LIBRARY, "lib" + output + ".so")
        recorded = (ROOT / "results/external_validation.json").read_text(encoding="utf-8")
        self.assertIn("/" + setup_external.MIRAGE_LIBRARY, recorded)

    def test_missing_patch_is_fatal_with_export_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "series").write_text("# comment\n0001-x.patch abc1234\n", encoding="utf-8")
            series = setup_external.read_series(directory)
            with self.assertRaisesRegex(RuntimeError, "export_mirage_patch.sh"):
                setup_external.patch_records(series)
            patch = directory / "0001-x.patch"
            patch.write_text("--- a/faiss/IndexMIRAGE.cpp\n+++ b/faiss/IndexMIRAGE.cpp\n", encoding="utf-8")
            record, = setup_external.patch_records(series)
            self.assertEqual((record["name"], record["base_revision"]), ("0001-x.patch", "abc1234"))
            self.assertEqual(len(record["sha256"]), 64)
            self.assertEqual(setup_external.patch_files(patch), {"faiss/IndexMIRAGE.cpp"})

    def test_malformed_series_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "series").write_text("only-a-name.patch\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                setup_external.read_series(directory)


if __name__ == "__main__":
    unittest.main()
