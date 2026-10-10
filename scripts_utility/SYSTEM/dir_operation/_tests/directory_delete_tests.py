"""Проверки удаления папок через API PySM на временных данных."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pysm_lib import pysm_operations as operations

SCRIPT = Path(__file__).resolve().parents[1] / "run_dir_operation.py"
spec = importlib.util.spec_from_file_location("dir_operation_script", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class DirectoryDeleteTests(unittest.TestCase):
    """Удаление дерева, ограничения и передача параметров из скрипта."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logger = patch.object(operations.pysm_context, "log_html")
        self.logger.start()
        self.addCleanup(self.logger.stop)

    def delete(self, source):
        return operations.perform_directory_operation(
            str(source), None, "delete", "skip", 1, False, ["*.jpg"]
        )

    def test_deletes_entire_tree_ignoring_filters(self):
        source = self.root / "source"
        (source / "nested" / "empty").mkdir(parents=True)
        (source / "nested" / "keep.txt").write_text("test")
        sibling = self.root / "sibling.txt"
        sibling.write_text("preserved")
        self.assertEqual(self.delete(source), 0)
        self.assertFalse(source.exists())
        self.assertEqual(sibling.read_text(), "preserved")

    def test_deletes_empty_directory(self):
        source = self.root / "empty"
        source.mkdir()
        self.assertEqual(self.delete(source), 0)
        self.assertFalse(source.exists())

    def test_rejects_invalid_sources_without_deleting(self):
        file = self.root / "file.txt"
        file.write_text("preserved")
        for source in ("", "   ", self.root / "missing", file, self.root.anchor):
            with self.subTest(source=source), patch.object(operations.shutil, "rmtree") as remove:
                self.assertEqual(self.delete(source), 1)
                remove.assert_not_called()
        self.assertTrue(file.exists())

    def test_reports_deletion_error(self):
        source = self.root / "source"
        source.mkdir()
        with patch.object(operations.shutil, "rmtree", side_effect=PermissionError("denied")):
            self.assertEqual(self.delete(source), 1)
        self.assertTrue(source.exists())

    def test_rejects_reparse_point(self):
        source = self.root / "source"
        source.mkdir()
        with patch.object(Path, "lstat", return_value=argparse.Namespace(st_file_attributes=0x400)), patch.object(operations.shutil, "rmtree") as remove:
            self.assertEqual(self.delete(source), 1)
            remove.assert_not_called()

    def test_script_allows_delete_without_destination(self):
        config = argparse.Namespace(do_source_dir="source", do_dest_dir=None,
            do_mode="delete", do_on_conflict="skip", all_threads=1,
            do_copy_base_folder=False, do_include=["*"])
        with patch.object(script, "get_config", return_value=config), patch.object(operations, "perform_directory_operation", return_value=0) as perform:
            with self.assertRaises(SystemExit) as result:
                script.main()
            self.assertEqual(result.exception.code, 0)
            self.assertIsNone(perform.call_args.kwargs["dest_dir_str"])
        config.do_mode = "copy"
        with patch.object(script, "get_config", return_value=config), patch.object(operations, "perform_directory_operation") as perform:
            with self.assertRaises(SystemExit) as result:
                script.main()
            self.assertEqual(result.exception.code, 1)
            perform.assert_not_called()


if __name__ == "__main__":
    unittest.main()
