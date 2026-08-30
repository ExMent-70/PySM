import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts_utility.SYSTEM.pysm_updater.run_pysm_updater import (
    analyze_local_state,
    create_force_system_archive,
    create_user_state_archive,
    is_user_state_path,
    restore_user_state_archive,
)


class DummyLogger:
    """Minimal logger stub for archive tests without PySM console output."""

    def __init__(self, report_dir: Path):
        self.log_path = report_dir / "update_test.log"

    def section(self, *_args, **_kwargs):
        pass

    def kv_line(self, *_args, **_kwargs):
        pass

    def write(self, *_args, **_kwargs):
        pass

    def icon_line(self, *_args, **_kwargs):
        pass


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


class UserStateArchiveTests(unittest.TestCase):
    def test_archive_restores_changed_file_and_local_deletion(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_dir = Path(temporary_directory)
            report_dir = target_dir / "_OUTPUT" / "pysm_updater"
            report_dir.mkdir(parents=True)
            logger = DummyLogger(report_dir)
            config_path = target_dir / "config.toml"
            deleted_collection_path = target_dir / "script_collections" / "Example.pysmc"
            config_path.write_text("local = true\n", encoding="utf-8")

            archive_info = create_user_state_archive(
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
            restore_user_state_archive(target_dir, archive_info, logger)

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
