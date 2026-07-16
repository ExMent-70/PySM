"""Contract tests for the public visual JSON editor API."""

from __future__ import annotations

import os
import unittest
from typing import Any, Dict, Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from pysm_lib.pysm_context import PySMContext
from pysm_lib.pysm_json_editor import (
    JsonEditor,
    JsonEditorStatus,
    create_json_editor,
    edit_json_variable,
    load_schema,
    validate_variable,
)


class InMemoryContext(PySMContext):
    """PySM context test double that never reads argv or writes files."""

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


def make_context() -> InMemoryContext:
    return InMemoryContext(
        {
            "project": {
                "type": "json",
                "value": {
                    "settings": {"quality": 90, "watermark": False},
                    "name": "Album",
                },
                "description": "Project settings",
                "read_only": False,
                "choices": None,
            },
            "project__schema": {
                "type": "json",
                "value": {
                    "version": 1,
                    "fields": {
                        "name": {
                            "label": "Название",
                            "widget": "string",
                        }
                    },
                },
                "description": "",
                "read_only": False,
                "choices": None,
            },
        }
    )


class JsonEditorApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, JsonEditor):
                widget.force_close = True
                widget.close()
        self.app.processEvents()

    def test_validation_supports_top_level_and_dotted_json_values(self) -> None:
        context = make_context()

        top_metadata, top_value = validate_variable("project", context)
        nested_metadata, nested_value = validate_variable("project.settings", context)

        self.assertEqual(top_metadata["type"], "json")
        self.assertEqual(top_value["name"], "Album")
        self.assertEqual(nested_metadata["type"], "json")
        self.assertEqual(nested_value, {"quality": 90, "watermark": False})

    def test_schema_is_loaded_for_target_variable(self) -> None:
        schema = load_schema("project", make_context())

        self.assertIsNotNone(schema)
        self.assertEqual(schema["fields"]["name"]["label"], "Название")

    def test_create_editor_saves_through_supplied_context(self) -> None:
        context = make_context()
        editor = create_json_editor("project", context=context, title="Test")
        results = []
        editor.finished.connect(results.append)

        name_node = next(child for child in editor.model.root.children if child.key == "name")
        name_node.value = "Updated"
        editor.save()

        self.assertEqual(context.get("project")["name"], "Updated")
        self.assertEqual(results[0].status, JsonEditorStatus.SAVED)
        self.assertTrue(results[0].changed)
        results[0].value["name"] = "Caller mutation"
        self.assertEqual(context.get("project")["name"], "Updated")

    def test_create_editor_saves_dotted_json_value(self) -> None:
        context = make_context()
        editor = create_json_editor("project.settings", context=context)

        quality_node = next(
            child for child in editor.model.root.children if child.key == "quality"
        )
        quality_node.value = 95
        editor.save()

        self.assertEqual(context.get_structured("project.settings.quality"), 95)
        self.assertEqual(editor.result.status, JsonEditorStatus.SAVED)

    def test_read_only_json_value_is_rejected(self) -> None:
        context = make_context()
        context._raw_context_data_cache["project"]["read_only"] = True

        with self.assertRaisesRegex(ValueError, "защищена от записи"):
            create_json_editor("project", context=context)

    def test_blocking_api_uses_local_event_loop_and_returns_cancel(self) -> None:
        context = make_context()

        def cancel_open_editor() -> None:
            editor = next(
                widget
                for widget in QApplication.topLevelWidgets()
                if isinstance(widget, JsonEditor)
            )
            editor.cancel()

        QTimer.singleShot(0, cancel_open_editor)
        result = edit_json_variable("project", context=context, apply_theme=False)

        self.assertEqual(result.status, JsonEditorStatus.CANCELLED)
        self.assertFalse(result.saved)
        self.assertEqual(context.get("project")["name"], "Album")


if __name__ == "__main__":
    unittest.main()
