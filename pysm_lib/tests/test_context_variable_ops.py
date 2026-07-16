"""Contract tests for shared context variable operations."""

from __future__ import annotations

import unittest
from typing import Any, Dict, Optional

from pysm_lib.context_variable_ops import (
    context_value_exists,
    copy_context_value,
    read_context_value,
    remove_context_value,
    write_context_value,
)
from pysm_lib.pysm_context import PySMContext


class InMemoryContext(PySMContext):
    """PySMContext test double that never touches disk or argv."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._context_file_path = None
        self._raw_context_data_cache = data or {}
        self._is_dirty = False

    def _initialize(self):
        return None

    def _send_ipc_update(self, action: str, **kwargs):
        return None

    def commit(self) -> None:
        self._is_dirty = False


class ContextVariableOpsTests(unittest.TestCase):
    def test_write_and_read_top_level_value_preserves_type(self) -> None:
        context = InMemoryContext()

        write_context_value(context, "output_path", "D:/out", var_type="dir_path")
        result = read_context_value(context, "output_path")

        self.assertTrue(result.exists)
        self.assertTrue(result.is_top_level)
        self.assertEqual(result.value, "D:/out")
        self.assertEqual(result.var_type, "dir_path")

    def test_write_and_read_dotted_value_updates_json_parent(self) -> None:
        context = InMemoryContext(
            {
                "project": {
                    "type": "json",
                    "value": {"name": "Old"},
                    "description": "",
                    "read_only": False,
                    "choices": None,
                }
            }
        )

        write_context_value(context, "project.name", "New")
        result = read_context_value(context, "project.name")

        self.assertTrue(result.exists)
        self.assertFalse(result.is_top_level)
        self.assertEqual(result.value, "New")
        self.assertEqual(context.get("project")["name"], "New")

    def test_copy_preserves_top_level_source_type(self) -> None:
        context = InMemoryContext(
            {
                "source_path": {
                    "type": "file_path",
                    "value": "D:/input.txt",
                    "description": "",
                    "read_only": False,
                    "choices": None,
                }
            }
        )

        copied = copy_context_value(context, "source_path", "target_path")

        self.assertTrue(copied.exists)
        self.assertEqual(context.get_variable("target_path")["type"], "file_path")
        self.assertEqual(context.get("target_path"), "D:/input.txt")

    def test_copy_dotted_value_to_dotted_target(self) -> None:
        context = InMemoryContext(
            {
                "project": {
                    "type": "json",
                    "value": {"name": "Album"},
                    "description": "",
                    "read_only": False,
                    "choices": None,
                }
            }
        )

        copied = copy_context_value(context, "project.name", "project.backup_name")

        self.assertTrue(copied.exists)
        self.assertEqual(context.get("project")["backup_name"], "Album")

    def test_remove_dotted_value(self) -> None:
        context = InMemoryContext(
            {
                "project": {
                    "type": "json",
                    "value": {"name": "Album", "counter": 2},
                    "description": "",
                    "read_only": False,
                    "choices": None,
                }
            }
        )

        remove_context_value(context, "project.name")

        self.assertFalse(context_value_exists(context, "project.name"))
        self.assertEqual(context.get("project"), {"counter": 2})


if __name__ == "__main__":
    unittest.main()
