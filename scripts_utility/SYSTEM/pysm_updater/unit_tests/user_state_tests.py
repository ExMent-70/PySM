import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts_utility.SYSTEM.pysm_updater.run_pysm_updater import (
    analyze_local_state,
    create_force_system_archive,
    is_user_state_path,
    restore_user_state_after_failed_merge,
    save_user_state_copies,
)


class DummyLogger:
    """Minimal logger stub for archive tests without PySM console output."""

    def __init__(self, report_dir: Path):
        self.log_path = report_dir / "update_test.log"
        self.timestamp = "20260830_104512"
        self.messages = []

    def section(self, *_args, **_kwargs):
        pass

    def kv_line(self, *_args, **_kwargs):
        pass

    def write(self, *_args, **_kwargs):
        pass

    def write_html(self, *_args, **_kwargs):
        pass

    def write_file(self, *_args, **_kwargs):
        pass

    def icon_line(self, message, *_args, **_kwargs):
        self.messages.append(message)


class UserStatePathTests(unittest.TestCase):
    def test_only_builtin_settings_and_collections_are_user_state(self):
        self.assertTrue(is_user_state_path("config.toml"))
        self.assertTrue(is_user_state_path("script_collections/Example.context.json"))
        self.assertFalse(is_user_state_path("themes/default/style.qss"))
        self.assertFalse(is_user_state_path("pysm_lib/app_controller.py"))

    @patch("scripts_utility.SYSTEM.pysm_updater.run_pysm_updater.path_matches_remote", return_value=False)
    @patch("scripts_utility.SYSTEM.pysm_updater.run_pysm_updater.remote_blob", return_value="remote-blob")
    @patch("scripts_utility.SYSTEM.pysm_updater.run_pysm_updater.get_status_entries")
    def test_user_state_intersection_does_not_become_code_conflict(self, get_status_entries, _remote_blob, _matches_remote):
        get_status_entries.return_value = [
            {"xy": " M", "path": "config.toml", "paths": ["config.toml"], "display": "Изменен локально: config.toml"},
            {
                "xy": " D",
                "path": "script_collections/Example.pysmc",
                "paths": ["script_collections/Example.pysmc"],
                "display": "Удален локально: script_collections/Example.pysmc",
            },
            {
                "xy": " M",
                "path": "pysm_lib/app_controller.py",
                "paths": ["pysm_lib/app_controller.py"],
                "display": "Изменен локально: pysm_lib/app_controller.py",
            },
        ]
        plan = {
            "changed_paths": ["config.toml", "script_collections/Example.pysmc", "pysm_lib/app_controller.py"],
            "added_paths": [],
        }

        state = analyze_local_state(Path("."), Path("."), "origin/main", plan)

        self.assertEqual(state["user_state_conflicts"], [
            "Изменен локально: config.toml",
            "Удален локально: script_collections/Example.pysmc",
        ])
        self.assertEqual(state["missing_tracked"], [])
        self.assertEqual(state["conflicts"], ["Изменен локально: pysm_lib/app_controller.py"])
        self.assertTrue(state["has_blocking_conflicts"])


class UserStateCopyTests(unittest.TestCase):
    def test_copies_use_one_update_prefix_and_keep_collection_extensions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_dir = Path(temporary_directory)
            report_dir = target_dir / "_OUTPUT" / "pysm_updater"
            report_dir.mkdir(parents=True)
            logger = DummyLogger(report_dir)
            config_path = target_dir / "config.toml"
            collection_dir = target_dir / "script_collections"
            collection_file = collection_dir / "Example.pysmc"
            context_file = collection_dir / "Example.context.json"
            config_path.write_text("local = true\n", encoding="utf-8")
            collection_dir.mkdir(parents=True)
            collection_file.write_text("local collection\n", encoding="utf-8")
            context_file.write_text("local context\n", encoding="utf-8")

            saved_state = save_user_state_copies(
                target_dir,
                logger,
                [
                    {"paths": ["config.toml"]},
                    {"paths": ["script_collections/Example.pysmc"]},
                    {"paths": ["script_collections/Example.context.json"]},
                ],
            )

            copied_paths = {item["source_path"]: Path(item["copy_path"]).name for item in saved_state["copied_files"]}
            self.assertEqual(copied_paths["config.toml"], "user_20260830_104512_config.toml")
            self.assertEqual(copied_paths["script_collections/Example.pysmc"], "user_20260830_104512_Example.pysmc")
            self.assertEqual(copied_paths["script_collections/Example.context.json"], "user_20260830_104512_Example.context.json")
            self.assertEqual((target_dir / saved_state["copied_files"][0]["copy_path"]).read_text(encoding="utf-8"), "local = true\n")
            self.assertIn(
                "После обновления сравните исходный файл с сохраненной копией и вручную перенесите нужные настройки.",
                logger.messages,
            )

    def test_failed_merge_restores_local_files_and_deletions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_dir = Path(temporary_directory)
            report_dir = target_dir / "_OUTPUT" / "pysm_updater"
            report_dir.mkdir(parents=True)
            logger = DummyLogger(report_dir)
            config_path = target_dir / "config.toml"
            deleted_collection_path = target_dir / "script_collections" / "Example.pysmc"
            config_path.write_text("local = true\n", encoding="utf-8")

            saved_state = save_user_state_copies(
                target_dir,
                logger,
                [
                    {"paths": ["config.toml"]},
                    {"paths": ["script_collections/Example.pysmc"]},
                ],
            )

            config_path.write_text("remote = true\n", encoding="utf-8")
            deleted_collection_path.parent.mkdir(parents=True, exist_ok=True)
            deleted_collection_path.write_text("remote collection\n", encoding="utf-8")
            restore_user_state_after_failed_merge(target_dir, saved_state, logger)

            self.assertEqual(config_path.read_text(encoding="utf-8"), "local = true\n")
            self.assertFalse(deleted_collection_path.exists())

    @patch("scripts_utility.SYSTEM.pysm_updater.run_pysm_updater.get_tracked_file_paths")
    def test_force_system_archive_contains_only_tracked_files(self, get_tracked_file_paths):
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_dir = Path(temporary_directory)
            report_dir = target_dir / "_OUTPUT" / "pysm_updater"
            report_dir.mkdir(parents=True)
            logger = DummyLogger(report_dir)
            tracked_path = target_dir / "pysm_lib" / "app.py"
            tracked_path.parent.mkdir(parents=True)
            tracked_path.write_text("tracked\n", encoding="utf-8")
            (target_dir / "local_notes.txt").write_text("untracked\n", encoding="utf-8")
            get_tracked_file_paths.return_value = ["pysm_lib/app.py"]

            archive_path = create_force_system_archive(Path("git.exe"), target_dir, logger)

            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.namelist(), ["pysm_lib/app.py"])


if __name__ == "__main__":
    unittest.main()
