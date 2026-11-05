"""Проверки удаления файлов по маске через API PySM на временных данных."""
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
    """Фильтрация файлов, ограничения и передача параметров из скрипта."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recycle_dir = self.root / "test_recycle_bin"
        self.recycle_dir.mkdir()
        self.recycled = []

        def recycle(path):
            destination = self.recycle_dir / str(len(self.recycled))
            path.rename(destination)
            self.recycled.append((path, destination))

        self.recycle_patch = patch.object(operations, "_recycle_path", side_effect=recycle)
        self.recycle_patch.start()
        self.addCleanup(self.recycle_patch.stop)
        self.logger = patch.object(operations.pysm_context, "log_html")
        self.log_html = self.logger.start()
        self.addCleanup(self.logger.stop)

    def delete(self, source):
        return operations.perform_directory_operation(
            str(source), None, "delete", "skip", 1, False, ["*.jpg"]
        )

    def test_deletes_only_matching_files_recursively(self):
        source = self.root / "source"
        (source / "nested" / "empty").mkdir(parents=True)
        (source / "nested" / "keep.txt").write_text("test")
        (source / "nested" / "remove.jpg").write_text("test")
        (source / "remove.jpg").write_text("test")
        (source / "keep.idml").write_text("test")
        sibling = self.root / "sibling.txt"
        sibling.write_text("preserved")
        self.assertEqual(self.delete(source), 0)
        self.assertFalse((source / "nested" / "empty").exists())
        self.assertTrue((source / "nested" / "keep.txt").exists())
        self.assertTrue((source / "keep.idml").exists())
        self.assertFalse((source / "nested" / "remove.jpg").exists())
        self.assertFalse((source / "remove.jpg").exists())
        self.assertEqual(sibling.read_text(), "preserved")

    def test_deletes_empty_directory(self):
        source = self.root / "empty"
        source.mkdir()
        self.assertEqual(self.delete(source), 0)
        self.assertFalse(source.exists())

    def test_deletes_source_when_matching_files_are_the_only_files(self):
        source = self.root / "source"
        (source / "nested" / "empty").mkdir(parents=True)
        (source / "nested" / "file.jpg").write_text("test")
        self.assertEqual(self.delete(source), 0)
        self.assertFalse(source.exists())

    def test_logs_every_removed_directory_on_its_own_line(self):
        source = self.root / "source"
        nested = source / "nested & more"
        leaf = nested / "leaf"
        leaf.mkdir(parents=True)
        (leaf / "file.jpg").write_text("test")

        self.assertEqual(self.delete(source), 0)

        folder_logs = [call.args[0] for call in self.log_html.call_args_list
                       if "Папка удалена:" in call.args[0]]
        self.assertEqual(3, len(folder_logs))
        for log, directory in zip(folder_logs, (leaf, nested, source)):
            self.assertIn(str(directory).replace("&", "&amp;"), log)
            self.assertEqual(1, log.count("Папка удалена:"))
            self.assertTrue(log.startswith("<div "))
            self.assertTrue(log.endswith("</div>"))

    def test_does_not_log_preserved_directories(self):
        source = self.root / "source"
        source.mkdir()
        (source / "keep.txt").write_text("test")
        (source / "remove.jpg").write_text("test")

        self.assertEqual(self.delete(source), 0)

        messages = [call.args[0] for call in self.log_html.call_args_list]
        self.assertTrue(any("Отправлено в корзину файлов: 1" in msg for msg in messages))
        self.assertFalse(any("Папка удалена:" in msg or "Папка сохранена:" in msg
                             for msg in messages))

    def test_logs_already_removed_directories_if_later_removal_fails(self):
        source = self.root / "source"
        nested = source / "nested"
        nested.mkdir(parents=True)
        recycle = operations._recycle_path

        def fail_on_source(path):
            if path == source:
                raise PermissionError("denied")
            recycle(path)

        with patch.object(operations, "_recycle_path", side_effect=fail_on_source):
            self.assertEqual(self.delete(source), 1)

        messages = [call.args[0] for call in self.log_html.call_args_list]
        self.assertTrue(any("Папка удалена:" in msg and str(nested) in msg for msg in messages))
        self.assertTrue(any("Ошибка отправки в корзину" in msg for msg in messages))
        self.assertFalse(any("Папка удалена:" in msg and str(source) + "</div>" in msg
                             for msg in messages))

    def test_preserves_source_with_nonmatching_file_in_subdirectory(self):
        source = self.root / "source"
        (source / "nested").mkdir(parents=True)
        (source / "nested" / "keep.txt").write_text("test")
        (source / "file.jpg").write_text("test")
        self.assertEqual(self.delete(source), 0)
        self.assertTrue((source / "nested" / "keep.txt").exists())
        self.assertTrue(source.is_dir())

    def test_rejects_invalid_sources_without_deleting(self):
        file = self.root / "file.txt"
        file.write_text("preserved")
        for source in ("", "   ", self.root / "missing", file, self.root.anchor):
            with self.subTest(source=source), patch.object(operations, "_recycle_path") as remove:
                self.assertEqual(self.delete(source), 1)
                remove.assert_not_called()
        self.assertTrue(file.exists())

    def test_reports_deletion_error(self):
        source = self.root / "source"
        source.mkdir()
        (source / "file.jpg").write_text("test")
        with patch.object(operations, "_recycle_path", side_effect=PermissionError("denied")):
            self.assertEqual(self.delete(source), 1)
        self.assertTrue(source.exists())

    def test_reports_error_removing_empty_directory(self):
        source = self.root / "source"
        source.mkdir()
        with patch.object(operations, "_recycle_path", side_effect=PermissionError("denied")):
            self.assertEqual(self.delete(source), 1)
        self.assertTrue(source.exists())

    def test_recycle_failure_never_uses_permanent_delete(self):
        source = self.root / "source"
        source.mkdir()
        file = source / "file.jpg"
        file.write_text("keep")
        with patch.object(operations, "_recycle_path", side_effect=OSError("unavailable")), \
             patch.object(Path, "unlink") as unlink, \
             patch.object(operations.shutil, "rmtree") as rmtree:
            self.assertEqual(self.delete(source), 1)
        unlink.assert_not_called()
        rmtree.assert_not_called()
        self.assertEqual(file.read_text(), "keep")

    def test_rejects_empty_and_path_patterns(self):
        source = self.root / "source"
        source.mkdir()
        keep = source / "keep.jpg"
        keep.write_text("test")
        for patterns in ([], [""], ["../*.jpg"], ["C:*.jpg"], ["*.jpg", "../*"]):
            with self.subTest(patterns=patterns):
                self.assertEqual(operations.perform_directory_operation(
                    str(source), None, "delete", "skip", 1, False, patterns
                ), 1)
                self.assertTrue(keep.exists())

    def test_explicit_idml_mask_preserves_other_files(self):
        source = self.root / "source"
        source.mkdir()
        (source / "folder.idml").mkdir()
        (source / "file.idml").write_text("idml")
        (source / "other.indd").write_text("keep")
        self.assertEqual(operations.perform_directory_operation(
            str(source), None, "delete", "skip", 1, False, ["*.idml"]
        ), 0)
        self.assertTrue(source.is_dir())
        self.assertFalse((source / "file.idml").exists())
        self.assertEqual((source / "other.indd").read_text(), "keep")
        self.assertFalse((source / "folder.idml").exists())

    def test_matching_directory_symlink_does_not_delete_target(self):
        source = self.root / "source"
        external = self.root / "external"
        source.mkdir()
        external.mkdir()
        protected = external / "file.jpg"
        protected.write_text("keep")
        try:
            (source / "link").symlink_to(external, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Символические ссылки недоступны: {error}")
        self.assertEqual(self.delete(source), 0)
        self.assertEqual(protected.read_text(), "keep")

    def test_nested_reparse_directory_is_not_traversed(self):
        source = self.root / "source"
        junction = source / "junction"
        junction.mkdir(parents=True)
        protected = junction / "file.jpg"
        protected.write_text("keep")
        original_lstat = Path.lstat

        def fake_lstat(path, *args, **kwargs):
            result = original_lstat(path, *args, **kwargs)
            if path == junction:
                return argparse.Namespace(st_mode=result.st_mode, st_file_attributes=0x400)
            return result

        with patch.object(Path, "lstat", fake_lstat):
            self.assertEqual(self.delete(source), 0)
        self.assertEqual(protected.read_text(), "keep")

    def test_rejects_reparse_point(self):
        source = self.root / "source"
        source.mkdir()
        with patch.object(Path, "lstat", return_value=argparse.Namespace(st_file_attributes=0x400)), patch.object(operations, "_recycle_path") as remove:
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
