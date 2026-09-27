"""Проверки независимого экспорта файлов активного рабочего процесса."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pysm_lib.app_controller import AppController
from pysm_lib.gui.main_window import MainWindow
from pysm_lib.models import (
    ContextVariableModel,
    ScriptRootModel,
    ScriptSetsCollectionModel,
)
from pysm_lib.set_manager import SetManager


class CollectionFileExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manager = SetManager(self.root / "collections")
        self.active_path = self.root / "active" / "Active.pysmc"
        self.manager.current_collection_model = ScriptSetsCollectionModel(
            collection_name="Active",
            description="Unsaved description",
            script_roots=[ScriptRootModel(path=str(self.root / "other" / "scripts"))],
            context_data={
                "student_ids": ContextVariableModel(
                    type="list", value=["S001", "S002"]
                )
            },
        )
        self.manager.current_collection_file_path = self.active_path
        self.manager._set_dirty(True)

    def _assert_active_state_unchanged(self) -> None:
        self.assertEqual(self.manager.current_collection_file_path, self.active_path)
        self.assertEqual(self.manager.current_collection_model.collection_name, "Active")
        self.assertEqual(
            self.manager.current_collection_model.description,
            "Unsaved description",
        )
        self.assertEqual(
            self.manager.current_collection_model.script_roots[0].path,
            str(self.root / "other" / "scripts"),
        )
        self.assertTrue(self.manager.is_dirty)

    def test_export_collection_writes_only_pysmc_and_preserves_active_state(self) -> None:
        target = self.root / "other" / "Transferred.pysmc"

        self.assertTrue(self.manager.export_collection_to_file(target))

        exported = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(exported["collection_name"], "Active")
        self.assertEqual(exported["description"], "Unsaved description")
        self.assertEqual(exported["script_roots"][0]["path"], "scripts")
        self.assertNotIn("context_data", exported)
        self.assertFalse(target.with_suffix(".context.json").exists())
        self._assert_active_state_unchanged()

    def test_export_context_writes_only_context_and_preserves_active_state(self) -> None:
        target = self.root / "other" / "Transferred.context.json"

        self.assertTrue(self.manager.export_context_to_file(target))

        exported = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(exported["student_ids"]["value"], ["S001", "S002"])
        self.assertFalse((target.parent / "Transferred.pysmc").exists())
        self._assert_active_state_unchanged()

    def test_empty_context_export_creates_empty_json_object(self) -> None:
        self.manager.current_collection_model.context_data = {}
        target = self.root / "empty.context.json"

        self.assertTrue(self.manager.export_context_to_file(target))

        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {})
        self._assert_active_state_unchanged()

    def test_required_suffix_is_added_once(self) -> None:
        self.assertEqual(
            MainWindow._with_required_suffix("D:/Export/Copy", ".pysmc"),
            Path("D:/Export/Copy.pysmc"),
        )
        self.assertEqual(
            MainWindow._with_required_suffix(
                "D:/Export/Copy.context.json", ".context.json"
            ),
            Path("D:/Export/Copy.context.json"),
        )

    def test_collection_dialog_exports_only_selected_pysmc(self) -> None:
        controller = SimpleNamespace(
            current_collection_file_path=self.active_path,
            set_manager=self.manager,
            export_current_collection_requested_by_gui=Mock(return_value=True),
            export_current_context_requested_by_gui=Mock(),
        )
        window = SimpleNamespace(
            controller=controller,
            locale_manager=SimpleNamespace(get=Mock(return_value="localized")),
            _collection_export_start_dir=lambda: self.active_path.parent,
            _with_required_suffix=MainWindow._with_required_suffix,
        )
        target_without_suffix = self.root / "other" / "Transferred"

        with patch(
            "pysm_lib.gui.main_window.QFileDialog.getSaveFileName",
            return_value=(str(target_without_suffix), ""),
        ):
            result = MainWindow._on_export_collection(window)

        self.assertTrue(result)
        controller.export_current_collection_requested_by_gui.assert_called_once_with(
            target_without_suffix.with_suffix(".pysmc")
        )
        controller.export_current_context_requested_by_gui.assert_not_called()

    def test_context_dialog_exports_only_selected_context(self) -> None:
        controller = SimpleNamespace(
            current_collection_file_path=self.active_path,
            set_manager=self.manager,
            export_current_collection_requested_by_gui=Mock(),
            export_current_context_requested_by_gui=Mock(return_value=True),
        )
        window = SimpleNamespace(
            controller=controller,
            locale_manager=SimpleNamespace(get=Mock(return_value="localized")),
            _collection_export_start_dir=lambda: self.active_path.parent,
            _with_required_suffix=MainWindow._with_required_suffix,
        )
        target_without_suffix = self.root / "other" / "Transferred"

        with patch(
            "pysm_lib.gui.main_window.QFileDialog.getSaveFileName",
            return_value=(str(target_without_suffix), ""),
        ):
            result = MainWindow._on_export_context(window)

        self.assertTrue(result)
        controller.export_current_context_requested_by_gui.assert_called_once_with(
            Path(f"{target_without_suffix}.context.json")
        )
        controller.export_current_collection_requested_by_gui.assert_not_called()

    def test_controller_rejects_export_over_active_files(self) -> None:
        controller = SimpleNamespace(
            current_collection_file_path=self.active_path,
            set_manager=self.manager,
            _show_active_file_export_error=Mock(),
            _is_active_collection_file=None,
        )
        controller._is_active_collection_file = lambda target, context: (
            AppController._is_active_collection_file(
                controller, target, context=context
            )
        )

        self.assertFalse(
            AppController.export_current_collection_requested_by_gui(
                controller, self.active_path
            )
        )
        self.assertFalse(
            AppController.export_current_context_requested_by_gui(
                controller, self.active_path.with_suffix(".context.json")
            )
        )
        self.assertEqual(controller._show_active_file_export_error.call_count, 2)
        self.assertFalse(self.active_path.exists())
        self._assert_active_state_unchanged()

    def test_failed_export_does_not_change_active_state(self) -> None:
        target = self.root / "failed" / "Export.pysmc"
        with patch.object(
            self.manager,
            "_atomic_write_json",
            side_effect=OSError("disk full"),
        ):
            self.assertFalse(self.manager.export_collection_to_file(target))

        self.assertFalse(target.exists())
        self._assert_active_state_unchanged()


if __name__ == "__main__":
    unittest.main()
