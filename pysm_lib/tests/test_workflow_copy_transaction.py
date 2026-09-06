"""Регрессии сохранения копии и восстановления исходного процесса без GUI."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

from pysm_lib.app_controller import AppController
from pysm_lib.models import ContextVariableModel, ScriptSetsCollectionModel
from pysm_lib.set_manager import SetManager


class WorkflowCopyTransactionTests(unittest.TestCase):
    """Запускает реальную цепочку контроллера на временных проектных файлах."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source_file = self.root / "Source.pysmc"
        self.source_file.write_bytes(b"original workflow")
        manager = SetManager(self.root)
        manager.current_collection_model = ScriptSetsCollectionModel(
            collection_name="Source with unsaved edits",
            context_data={
                "wf_psd_path": ContextVariableModel(type="dir_path", value=str(self.root)),
                "wf_session_name": ContextVariableModel(type="string", value="Source"),
            },
        )
        manager.current_collection_file_path = self.source_file
        manager._set_dirty(True)
        self.source = manager.current_collection_model.model_copy(deep=True)
        self.project_dir = self.root / "Copy"
        self.project_file = self.project_dir / "Copy.pysmc"
        self.context_file = self.project_dir / "Copy.context.json"
        controller = SimpleNamespace(
            set_manager=manager,
            current_collection_file_path=self.source_file,
            current_orchestrator=None,
            selected_set_node_id="source-selection",
            selected_set_node_model=None,
            config_manager=SimpleNamespace(
                config=SimpleNamespace(workflow_copy=SimpleNamespace(
                    reset_context_variables=["wf_session_name"],
                )),
                last_used_sets_collection_file=str(self.source_file),
                reload_workflow_copy_config=Mock(return_value=True),
                save_config=Mock(return_value=True),
            ),
            locale_manager=SimpleNamespace(get=lambda key, **kwargs: key),
            _update_suggested_save_dir=Mock(),
            _log_welcome_message=Mock(),
            _log_collection_properties=Mock(),
            refresh_available_scripts_list=Mock(),
            _request_collection_view_update=Mock(),
            clear_console_request=Mock(),
            log_message_to_console=Mock(),
            collection_dirty_state_changed=Mock(),
            status_message_updated=Mock(),
            _show_collection_copy_save_error=Mock(),
            _confirm_new_collection_save=Mock(return_value=True),
            _select_new_collection_target=Mock(return_value=(
                "Copy", self.project_dir, self.project_file,
            )),
            _copy_project_paths=AppController._copy_project_paths,
            get_main_window=lambda: None,
        )
        controller.set_active_script_set_node = lambda node_id: setattr(
            controller, "selected_set_node_id", node_id,
        )
        for method in (
            "new_collection_from_template_requested_by_gui",
            "_create_and_save_collection_copy",
            "_save_new_collection_in_context_root",
            "_restore_collection_after_copy_cancel",
            "save_current_collection_requested_by_gui",
            "update_collection_context",
        ):
            setattr(controller, method, MethodType(getattr(AppController, method), controller))
        self.controller = controller

    def assert_source_restored(self):
        controller = self.controller
        self.assertEqual(controller.set_manager.current_collection_model, self.source)
        self.assertEqual(controller.current_collection_file_path, self.source_file)
        self.assertEqual(controller.set_manager.current_collection_file_path, self.source_file)
        self.assertTrue(controller.set_manager.is_dirty)
        self.assertEqual(controller.selected_set_node_id, "source-selection")
        self.assertEqual(controller.config_manager.last_used_sets_collection_file, str(self.source_file))
        self.assertEqual(self.source_file.read_bytes(), b"original workflow")

    def test_message_cancel_restores_source_even_without_reset_matches(self):
        for names in ([], ["missing.field"], ["wf_session_name"]):
            with self.subTest(names=names):
                self.controller.config_manager.config.workflow_copy.reset_context_variables = names
                self.controller._confirm_new_collection_save.reset_mock()
                self.controller._confirm_new_collection_save.return_value = False
                self.controller.new_collection_from_template_requested_by_gui()
                self.controller._confirm_new_collection_save.assert_called_once()
                self.assert_source_restored()
                self.assertFalse(self.project_dir.exists())

    def test_file_dialog_cancel_restores_source(self):
        self.controller._select_new_collection_target.return_value = None
        self.controller.new_collection_from_template_requested_by_gui()
        self.assert_source_restored()
        self.assertFalse(self.project_dir.exists())

    def test_invalid_root_restores_source(self):
        context = self.controller.set_manager.current_collection_model.context_data
        context["wf_psd_path"].value = str(self.root / "missing")
        self.source = self.controller.set_manager.current_collection_model.model_copy(deep=True)
        self.controller.new_collection_from_template_requested_by_gui()
        self.assert_source_restored()
        self.controller._show_collection_copy_save_error.assert_called_once()

    def test_directory_creation_failure_restores_source(self):
        self.project_dir.write_bytes(b"not a directory")
        self.controller.new_collection_from_template_requested_by_gui()
        self.assert_source_restored()
        self.assertEqual(self.project_dir.read_bytes(), b"not a directory")

    def test_unexpected_exception_restores_source(self):
        self.controller._select_new_collection_target.side_effect = OSError("dialog error")
        self.controller.new_collection_from_template_requested_by_gui()
        self.assert_source_restored()

    def test_success_without_resets_creates_saved_copy(self):
        self.controller.config_manager.config.workflow_copy.reset_context_variables = []
        self.controller.new_collection_from_template_requested_by_gui()
        self.controller._confirm_new_collection_save.assert_called_once()
        self.assertEqual(self.controller.current_collection_file_path, self.project_file)
        self.assertFalse(self.controller.set_manager.is_dirty)
        saved = json.loads(self.project_file.read_text(encoding="utf-8"))
        context = json.loads(self.context_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["collection_name"], "Copy")
        self.assertEqual(context["wf_session_name"]["value"], "Copy")
        self.assertEqual(context["wf_psd_path"]["value"], str(self.root))
        self.assertEqual(self.source_file.read_bytes(), b"original workflow")

    def test_existing_folder_and_context_are_allowed(self):
        self.project_dir.mkdir()
        unrelated = self.project_dir / "photo.txt"
        unrelated.write_bytes(b"keep")
        self.context_file.write_bytes(b"previous context")
        self.controller.new_collection_from_template_requested_by_gui()
        self.assertEqual(self.controller.current_collection_file_path, self.project_file)
        self.assertEqual(unrelated.read_bytes(), b"keep")
        context = json.loads(self.context_file.read_text(encoding="utf-8"))
        self.assertEqual(context["wf_session_name"]["value"], "Copy")
        self.assertFalse(list(self.project_dir.glob(".pysm-copy-*")))

    def test_existing_workflow_is_never_overwritten_even_after_selection(self):
        self.project_dir.mkdir()
        self.project_file.write_bytes(b"existing workflow")
        self.context_file.write_bytes(b"existing context")
        self.controller.new_collection_from_template_requested_by_gui()
        self.assert_source_restored()
        self.assertEqual(self.project_file.read_bytes(), b"existing workflow")
        self.assertEqual(self.context_file.read_bytes(), b"existing context")
        self.assertEqual(len(list(self.project_dir.iterdir())), 2)

    def test_selection_accepts_existing_folder_but_retries_existing_workflow(self):
        self.project_dir.mkdir()
        selected = str(self.root / "Copy.pysmc")
        with patch("pysm_lib.app_controller.QFileDialog.getSaveFileName", return_value=(selected, "")):
            self.assertEqual(AppController._select_new_collection_target(self.controller, self.root),
                             ("Copy", self.project_dir, self.project_file))
        self.project_file.write_bytes(b"existing")
        with patch(
            "pysm_lib.app_controller.QFileDialog.getSaveFileName",
            side_effect=[(selected, ""), ("", "")],
        ) as dialog:
            self.assertIsNone(AppController._select_new_collection_target(self.controller, self.root))
            self.assertEqual(dialog.call_count, 2)
        self.controller._show_collection_copy_save_error.assert_called_with(
            "dialogs.collection_copy_save.project_exists_error", path=self.project_file,
        )

    def test_failed_publication_removes_new_folder(self):
        real_replace = os.replace

        def fail_collection(source, target):
            if Path(source).name == "collection.json":
                raise OSError("simulated publication failure")
            return real_replace(source, target)

        with patch("pysm_lib.set_manager.os.replace", side_effect=fail_collection):
            self.controller.new_collection_from_template_requested_by_gui()
        self.assert_source_restored()
        self.assertFalse(self.project_dir.exists())

    def test_failed_publication_restores_existing_context_and_other_files(self):
        self.project_dir.mkdir()
        self.context_file.write_bytes(b"original context bytes")
        other = self.project_dir / "keep.txt"
        other.write_bytes(b"keep")
        real_replace = os.replace

        for failing_name in ("context.json", "collection.json"):
            with self.subTest(failing_name=failing_name):
                def fail_publication(source, target):
                    if Path(source).name == failing_name:
                        raise OSError("simulated publication failure")
                    return real_replace(source, target)

                with patch("pysm_lib.set_manager.os.replace", side_effect=fail_publication):
                    self.controller.new_collection_from_template_requested_by_gui()
                self.assert_source_restored()
                self.assertFalse(self.project_file.exists())
                self.assertEqual(self.context_file.read_bytes(), b"original context bytes")
                self.assertEqual(other.read_bytes(), b"keep")
                self.assertEqual(len(list(self.project_dir.iterdir())), 2)

    def test_failed_staging_leaves_existing_files_untouched(self):
        self.project_dir.mkdir()
        self.context_file.write_bytes(b"original context")
        with patch.object(
            self.controller.set_manager, "_atomic_write_json",
            side_effect=OSError("disk full"),
        ):
            self.controller.new_collection_from_template_requested_by_gui()
        self.assert_source_restored()
        self.assertEqual(self.context_file.read_bytes(), b"original context")
        self.assertEqual(list(self.project_dir.iterdir()), [self.context_file])

    def test_path_selection_rejects_outside_root_and_redirected_folder(self):
        with self.assertRaisesRegex(ValueError, "outside_working_root"):
            AppController._copy_project_paths(self.root, self.root.parent / "Copy.pysmc")
        real_resolve = Path.resolve

        def resolve_with_redirect(path, strict=False):
            if path == self.project_dir:
                return self.root.parent / "Redirected"
            return real_resolve(path, strict=strict)

        with patch.object(Path, "resolve", resolve_with_redirect):
            with self.assertRaisesRegex(ValueError, "outside_working_root"):
                AppController._copy_project_paths(self.root, self.root / "Copy.pysmc")
            self.controller.new_collection_from_template_requested_by_gui()
        self.assert_source_restored()
        self.assertFalse(self.project_dir.exists())

    def test_ordinary_save_still_updates_existing_workflow(self):
        self.assertTrue(self.controller.set_manager.save_collection_to_file(self.source_file))
        saved = json.loads(self.source_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["collection_name"], self.source.collection_name)
        self.assertTrue(self.source_file.with_suffix(".context.json").is_file())

    def test_exception_after_successful_save_keeps_saved_copy(self):
        self.controller._log_collection_properties.side_effect = RuntimeError("UI failure")
        self.controller.new_collection_from_template_requested_by_gui()
        self.assertEqual(self.controller.current_collection_file_path, self.project_file)
        self.assertEqual(self.controller.set_manager.current_collection_file_path, self.project_file)
        self.assertFalse(self.controller.set_manager.is_dirty)
        self.assertTrue(self.project_file.is_file())
        self.assertTrue(self.context_file.is_file())

    def test_cancel_preserves_clean_source_state(self):
        self.controller.set_manager._set_dirty(False)
        self.controller._confirm_new_collection_save.return_value = False
        self.controller.new_collection_from_template_requested_by_gui()
        self.assertEqual(self.controller.set_manager.current_collection_model, self.source)
        self.assertEqual(self.controller.current_collection_file_path, self.source_file)
        self.assertFalse(self.controller.set_manager.is_dirty)

    def test_unavailable_rollback_keeps_context_backup_for_recovery(self):
        self.project_dir.mkdir()
        self.context_file.write_bytes(b"original context")
        real_replace = os.replace

        def fail_write_and_restore(source, target):
            if Path(source).name in {"collection.json", "previous-context.json"}:
                raise PermissionError("simulated lock")
            return real_replace(source, target)

        with patch("pysm_lib.set_manager.os.replace", side_effect=fail_write_and_restore):
            self.controller.new_collection_from_template_requested_by_gui()
        self.assert_source_restored()
        self.assertFalse(self.project_file.exists())
        backups = list(self.project_dir.glob(".pysm-copy-*/previous-context.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"original context")


if __name__ == "__main__":
    unittest.main()
