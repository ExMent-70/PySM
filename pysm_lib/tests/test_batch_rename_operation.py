"""Тесты сортировки и построения плана пакетного переименования."""

from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from pysm_lib import pysm_operations


class TqdmStub:
    """Минимальная замена tqdm для детерминированных тестов."""

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else ()

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    @staticmethod
    def write(*args, **kwargs) -> None:
        return None


class BatchRenameOperationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_context = tempfile.TemporaryDirectory()
        self.source_dir = pathlib.Path(self.temp_dir_context.name)
        self.html_messages: list[str] = []

    def tearDown(self) -> None:
        self.temp_dir_context.cleanup()

    def _run_operation(
        self,
        *,
        template: str = "RAW_%index%%ext%",
        sort_method: str = "modified_time",
        on_conflict: str = "error",
        dry_run: bool = False,
    ) -> int:
        with (
            patch.object(pysm_operations, "tqdm", TqdmStub),
            patch.object(
                pysm_operations.pysm_context,
                "log_html",
                side_effect=self.html_messages.append,
            ),
            patch.object(pysm_operations.pysm_context, "log_link"),
        ):
            return pysm_operations.perform_batch_rename_operation(
                source_dir_str=str(self.source_dir),
                include_patterns=["*.CR3"],
                rename_template=template,
                start_index=1,
                index_digits=4,
                on_conflict=on_conflict,
                dry_run=dry_run,
                sort_method=sort_method,
                lowercase_extension=True,
                sanitize_filename=True,
            )

    def _create_file(self, name: str, content: str) -> pathlib.Path:
        path = self.source_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_modified_time_controls_final_index_order(self) -> None:
        newest = self._create_file("first.CR3", "newest")
        oldest = self._create_file("second.CR3", "oldest")
        middle = self._create_file("third.CR3", "middle")
        base_ns = 1_700_000_000_000_000_000
        os.utime(newest, ns=(base_ns + 300, base_ns + 300))
        os.utime(oldest, ns=(base_ns + 100, base_ns + 100))
        os.utime(middle, ns=(base_ns + 200, base_ns + 200))

        result = self._run_operation(sort_method="modified_time")

        self.assertEqual(result, 0)
        self.assertEqual(
            [
                (self.source_dir / "RAW_0001.cr3").read_text(encoding="utf-8"),
                (self.source_dir / "RAW_0002.cr3").read_text(encoding="utf-8"),
                (self.source_dir / "RAW_0003.cr3").read_text(encoding="utf-8"),
            ],
            ["oldest", "middle", "newest"],
        )

    def test_created_time_controls_final_index_order(self) -> None:
        files = [
            self._create_file("third.CR3", "third"),
            self._create_file("first.CR3", "first"),
            self._create_file("second.CR3", "second"),
        ]
        expected_contents = [
            path.read_text(encoding="utf-8")
            for path in sorted(
                files,
                key=lambda path: (path.stat().st_ctime_ns, path.name.lower()),
            )
        ]

        result = self._run_operation(sort_method="created_time")

        self.assertEqual(result, 0)
        actual_contents = [
            (self.source_dir / f"RAW_{index:04d}.cr3").read_text(encoding="utf-8")
            for index in range(1, 4)
        ]
        self.assertEqual(actual_contents, expected_contents)

    def test_dry_run_plan_does_not_probe_each_target_path(self) -> None:
        self._create_file("first.CR3", "first")
        self._create_file("second.CR3", "second")

        with (
            patch.object(
                pathlib.Path,
                "resolve",
                side_effect=AssertionError(
                    "Path.resolve() must not build the rename plan"
                ),
            ),
            patch.object(
                pathlib.Path,
                "exists",
                side_effect=AssertionError(
                    "Path.exists() must not probe every planned target"
                ),
            ),
        ):
            result = self._run_operation(dry_run=True)

        self.assertEqual(result, 0)

    def test_directory_snapshot_detects_external_target_conflict(self) -> None:
        self._create_file("first.CR3", "first")
        (self.source_dir / "RAW_0001.cr3").mkdir()

        result = self._run_operation(
            sort_method="name",
            on_conflict="error",
            dry_run=True,
        )

        self.assertEqual(result, 1)
        self.assertIn("План переименования содержит ошибки", "\n".join(self.html_messages))

    def test_rename_conflicts_use_distinct_assigned_targets(self) -> None:
        self._create_file("first.CR3", "first")
        self._create_file("second.CR3", "second")
        self._create_file("third.CR3", "third")

        result = self._run_operation(
            template="same%ext%",
            sort_method="name",
            on_conflict="rename",
            dry_run=True,
        )

        self.assertEqual(result, 0)
        preview = "\n".join(self.html_messages)
        self.assertIn("same.cr3", preview)
        self.assertIn("same (1).cr3", preview)
        self.assertIn("same (2).cr3", preview)


if __name__ == "__main__":
    unittest.main()
