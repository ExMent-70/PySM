"""Регрессии сохранения документа и незавершённого ввода в Qt-таблице."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QMessageBox

import run_list_create as run
from _lib.domain import AppConfig, Student, StudentIdAllocator


class SavingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.managed = patch.object(run, "IS_MANAGED_RUN", False)
        self.managed.start()
        self.addCleanup(self.managed.stop)
        self.window = run.ClassListEditor(AppConfig(wf_autosave_formats=[]))
        self.window.config.wf_dest_dir = self.temp.name
        self.window.id_allocator = StudentIdAllocator("A7K3", 2)
        self.window.table_model.update_data([
            Student(student_id="A7K3-S001", surname="Иванов", name="Иван")
        ])
        self.window._is_dirty = False

    def tearDown(self):
        editor = self.window.processed_table.itemDelegate()._active_editor
        if editor is not None:
            QTest.keyClick(editor, Qt.Key_Escape)
        self.window._is_dirty = False
        self.window.table_model.update_data([])
        self.window.close()
        self.app.processEvents()

    def edit_cell(self, column):
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        index = self.window.proxy_model.index(0, column)
        self.window.processed_table.setCurrentIndex(index)
        self.window.processed_table.edit(index)
        self.app.processEvents()
        return self.window.processed_table.itemDelegate()._active_editor

    def make_file(self, name="custom.list"):
        path = Path(self.temp.name) / name
        run.io_services.save_session(
            path, path.stem, "-", self.window.table_model.get_all_data(), [],
            self.window.id_allocator,
        )
        return path

    def test_ctrl_s_commits_each_kind_of_active_editor(self):
        for col, expected in [(3, "Пётр"), (5, "учитель")]:
            with self.subTest(column=col):
                editor = self.edit_cell(col)
                if isinstance(editor, QComboBox):
                    editor.setCurrentText(expected)
                else:
                    editor.setText(expected)
                QTest.keyClick(editor, Qt.Key_S, Qt.ControlModifier)
                self.app.processEvents()
                data = json.loads(
                    self.window._get_default_filepath("list").read_text(encoding="utf-8")
                )
                field = "name" if col == 3 else "rank"
                self.assertEqual(expected, data["students"][0][field])
                self.assertFalse(self.window._is_dirty)

    def test_close_commits_rank_before_asking_and_cancel_keeps_window(self):
        editor = self.edit_cell(5)
        editor.setCurrentText("учитель")
        with patch.object(
            self.window, "_ask_unsaved_changes_action",
            return_value=QMessageBox.StandardButton.Cancel,
        ) as ask:
            self.assertFalse(self.window.close())
        ask.assert_called_once()
        self.assertEqual("учитель", self.window.table_model.get_all_data()[0].rank)
        self.assertTrue(self.window._is_dirty)

    def test_invalid_active_cell_blocks_save(self):
        editor = self.edit_cell(3)
        editor.setText("123")
        with patch.object(run.QMessageBox, "warning"), patch.object(
            run.io_services, "save_session"
        ) as save:
            self.assertFalse(self.window._save_list())
        save.assert_not_called()
        self.assertEqual("Иван", self.window.table_model.get_all_data()[0].name)

    def test_invalid_input_survives_real_validation_dialog(self):
        editor = self.edit_cell(3)
        editor.setText("123")

        def dismiss_warning():
            dialog = QApplication.activeModalWidget()
            if isinstance(dialog, QMessageBox):
                dialog.accept()

        QTimer.singleShot(0, dismiss_warning)
        self.assertFalse(self.window._save_list())
        self.assertEqual("123", editor.text())
        editor.setText("Пётр")
        self.assertTrue(self.window._save_list())

    def test_opening_unknown_rank_editor_does_not_change_saved_rank(self):
        self.window.table_model.get_all_data()[0].rank = "Особый ранг"
        editor = self.edit_cell(5)
        self.assertEqual("Особый ранг", editor.currentText())
        self.assertTrue(self.window._finish_table_editing())
        self.assertFalse(self.window._is_dirty)

    def test_invalid_input_is_not_lost_when_focus_moves_to_toolbar(self):
        editor = self.edit_cell(3)
        editor.setText("123")
        with patch.object(run.QMessageBox, "warning"):
            self.window.import_panel_button.setFocus()
            self.app.processEvents()
            self.assertIs(editor, self.window.processed_table.itemDelegate()._active_editor)
            self.assertEqual("123", editor.text())
            self.assertFalse(self.window._save_list())

    def test_all_editable_columns_mark_real_changes_only(self):
        model = self.window.table_model
        model.set_info_columns(["Цитата"])
        for col, value in [(0, 1), (2, "Петров"), (3, "Пётр"),
                           (4, "Иванович"), (5, "учитель"), (6, "-"), (9, "Текст")]:
            with self.subTest(column=col):
                self.window._is_dirty = False
                model.setData(model.index(0, col), value)
                self.assertTrue(self.window._is_dirty)
                self.window._is_dirty = False
                model.setData(model.index(0, col), value)
                self.assertFalse(self.window._is_dirty)

    def test_deleting_last_row_marks_document_dirty(self):
        self.window.processed_table.selectRow(0)
        with patch.object(run.QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window._delete_selected_rows()
        self.assertEqual(0, self.window.table_model.rowCount())
        self.assertTrue(self.window._is_dirty)

    def test_loading_cancel_preserves_current_data_and_path(self):
        path = self.make_file()
        student = self.window.table_model.get_all_data()[0]
        student.rank = "учитель"
        self.window._is_dirty = True
        with patch.object(self.window, "_ask_unsaved_changes_action",
                          return_value=QMessageBox.StandardButton.Cancel):
            self.window._load_from_file(path)
        self.assertIs(student, self.window.table_model.get_all_data()[0])
        self.assertEqual("учитель", student.rank)
        self.assertIsNone(self.window._current_list_path)
        self.assertTrue(self.window._is_dirty)

    def test_loading_save_reads_latest_saved_data(self):
        path = self.make_file()
        self.window._load_from_file(path)
        self.window.table_model.setData(self.window.table_model.index(0, 5), "учитель")
        with patch.object(self.window, "_ask_unsaved_changes_action",
                          return_value=QMessageBox.StandardButton.Save):
            self.window._load_from_file(path)
        self.assertEqual("учитель", self.window.table_model.get_all_data()[0].rank)
        self.assertFalse(self.window._is_dirty)

    def test_open_and_save_as_keep_exact_selected_path(self):
        path = self.make_file()
        self.window._load_from_file(path)
        self.assertEqual(path, self.window._get_default_filepath("list"))
        target = Path(self.temp.name) / "renamed.list"
        with patch.object(run.QFileDialog, "getSaveFileName", return_value=(str(target), "")):
            self.assertTrue(self.window._save_list(save_as=True))
        self.assertEqual(target, self.window._get_default_filepath("list"))
        self.assertEqual(target.with_suffix(".csv"), self.window._get_default_filepath("csv"))

    def test_failed_save_as_preserves_document_location(self):
        path = self.make_file()
        self.window._load_from_file(path)
        self.window._is_dirty = True
        target = Path(self.temp.name) / "other" / "renamed.list"
        with patch.object(run.QFileDialog, "getSaveFileName", return_value=(str(target), "")), \
             patch.object(run.io_services, "save_session", side_effect=OSError("failure")), \
             patch.object(run.QMessageBox, "critical"):
            self.assertFalse(self.window._save_list(save_as=True))
        self.assertEqual(path, self.window._current_list_path)
        self.assertEqual(str(path.parent), self.window.config.wf_dest_dir)
        self.assertEqual(path.stem, self.window._class_name)
        self.assertTrue(self.window._is_dirty)

    def test_close_cancel_and_discard_do_not_write_context(self):
        self.window._is_dirty = True
        for reply in [QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Discard]:
            with patch.object(run, "IS_MANAGED_RUN", True), \
                 patch.object(self.window, "_save_student_ids_order") as save, \
                 patch.object(self.window, "_ask_unsaved_changes_action", return_value=reply):
                event = QCloseEvent()
                self.window.closeEvent(event)
                save.assert_not_called()
                self.assertEqual(reply == QMessageBox.StandardButton.Discard, event.isAccepted())

    def test_failed_related_output_keeps_dirty_for_retry(self):
        self.window.config.wf_autosave_formats = ["html", "csv"]
        with patch.object(self.window, "_save_html", return_value=None), \
             patch.object(self.window, "_save_csv", return_value=True) as csv:
            self.assertFalse(self.window._save_list())
        csv.assert_called_once()
        self.assertTrue(self.window._is_dirty)
        self.assertTrue(self.window._get_default_filepath("list").exists())

    def test_context_failure_blocks_close_and_can_be_retried(self):
        self.window._is_dirty = True
        with patch.object(run, "IS_MANAGED_RUN", True), \
             patch.object(self.window, "_save_student_ids_order", return_value=False), \
             patch.object(self.window, "_ask_unsaved_changes_action",
                          return_value=QMessageBox.StandardButton.Save):
            event = QCloseEvent()
            self.window.closeEvent(event)
            self.assertFalse(event.isAccepted())
        self.assertTrue(self.window._is_dirty)
        with patch.object(run, "IS_MANAGED_RUN", True), \
             patch.object(self.window, "_save_student_ids_order", return_value=True):
            self.assertTrue(self.window._save_list())
        self.assertFalse(self.window._is_dirty)

    def test_empty_list_updates_csv_and_context(self):
        self.window.table_model.update_data([])
        self.window.config.wf_autosave_formats = ["csv"]
        with patch.object(run, "IS_MANAGED_RUN", True), \
             patch.object(run.io_services, "save_student_ids_order_to_context",
                          return_value=("test", [])) as save:
            self.assertTrue(self.window._save_list())
        self.assertEqual([], save.call_args.args[2])
        path = self.window._get_default_filepath("csv")
        self.assertEqual(1, len(path.read_text(encoding="utf-16").splitlines()))


if __name__ == "__main__":
    unittest.main()
