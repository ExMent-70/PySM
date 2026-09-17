"""Synthetic contract and real Qt interaction tests for the three stages.

Set PHOTO_SELECTION04_DESKTOP=1 to run the same interactions in visible windows.
No user sessions, workflows, context or photo files are used.
"""

from __future__ import annotations

from argparse import Namespace
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType
import unittest
from unittest.mock import patch

if not os.environ.get("PHOTO_SELECTION04_DESKTOP"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[5]
SUITE = ROOT / "scripts/Workflow2/Tools/photo_selection2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/Workflow2/Tools"))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QPlainTextEdit, QPushButton

from photo_selection2._common import assignment_core as core
from photo_selection2._common import copy_service, workers, window_base
from photo_selection2._common.ai_import import validate_ai_response
from photo_selection2._common.config import make_parser
from photo_selection2._common.csv_import import read_csv_table, import_table, import_personal_file
from photo_selection2._common.domain import SelectionDocument, ImportEntry
from photo_selection2._common.roster import load_roster
from photo_selection2._common.storage import load_document, save_document
from photo_selection2._common.signatures import file_signature
from photo_selection2.selection_import._lib.app import ImportWindow
from photo_selection2.selection_copy._lib import runner as copy_runner
from photo_selection2.selection_assignments._lib.app import AssignmentsWindow

# Only the original photo_selection and the core it actually uses are references.
from scripts.Workflow2.FaceAnalysis._common import photo_selection_core as original_core
from scripts.Workflow2.FaceAnalysis._common import photo_selection_copy as original_copy


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def snapshot(path):
    """Compare file contents without depending on the temporary root."""
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)
        from pysm_lib import theme_api
        theme_api.apply_theme_to_app(cls.app)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.analysis = self.root / "Analysis_SCHOOL"
        self.source = self.root / "Capture"
        self.dest = self.root / "Selects"
        self.list_file = self.root / "Класс.list"
        for folder in (self.analysis, self.source, self.dest):
            folder.mkdir()
        write_json(self.list_file, {"list_id": "A7K3", "students": [
            {"student_id": "A7K3-S001", "surname": "Иванова", "name": "Анна", "rank": "ученик", "alpha_order": 1},
            {"student_id": "A7K3-S002", "surname": "Петров", "name": "Иван", "rank": "ученик", "alpha_order": 2},
        ]})
        self.roster = load_roster(self.list_file)
        self.selection = self.analysis / "photo_selection.json"
        self.assignment = self.analysis / "photo_assignments.json"
        self.document = SelectionDocument("A7K3", "Класс", "SCHOOL")
        self.document.apply("A7K3-S001", ["901256"], source="manual")
        save_document(self.selection, self.document)
        self.info = {
            "IMG_901256.jpg": {"filename": "IMG_901256.jpg", "location_name": "portrait", "faces": [{"student_id": "A7K3-S001"}]},
            "IMG_901270.jpg": {"filename": "IMG_901270.jpg", "location_name": "group_photo_01", "faces": [{"student_id": "A7K3-S001"}, {"student_id": "A7K3-S002"}]},
        }
        write_json(self.analysis / "info_faces.json", self.info)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(window_base, "IS_MANAGED_RUN", False))
        self.stack.enter_context(patch.object(window_base, "pysm_context", None))
        self.messages = []
        for name in ("information", "warning", "critical"):
            self.stack.enter_context(patch.object(QMessageBox, name, side_effect=lambda *args: self.messages.append(args[2])))
        self.windows = []
        self.qt_errors = []
        self.stack.enter_context(patch.object(sys, "excepthook", side_effect=lambda *args: self.qt_errors.append(args)))

    def tearDown(self):
        for window in self.windows:
            if hasattr(window, "_worker"):
                self.wait(lambda: window._worker is None)
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard):
                window.close()
            if hasattr(window, "_image_pipeline"):
                self.wait(lambda: window._image_pipeline.is_closed)
            window.deleteLater()
        self.app.processEvents()
        self.stack.close()
        self.tmp.cleanup()
        self.assertEqual(self.qt_errors, [], "Qt callback raised an unhandled exception")

    def config(self, **overrides):
        values = dict(student_list_file=str(self.list_file), analysis_dir=str(self.analysis),
                      source_dir=str(self.source), dest_dir=str(self.dest), session_name="Класс",
                      photo_session="SCHOOL", exclude_dirs="Masks", on_conflict="skip", mode="copy", title="Проверка Photo Selection 04",
                      message="<p>Инструкция оператора</p>")
        values.update(overrides)
        return Namespace(**values)

    def inputs(self, **overrides):
        values = dict(student_list_file=self.list_file, analysis_dir=self.analysis,
                      source_dir=self.source, dest_dir=self.dest, exclude_dirs=("Masks",))
        values.update(overrides)
        return values

    def copy_config(self):
        """Use the same minimal contract as a fresh copying instance in PySM."""
        return make_parser("selection_copy").parse_args([
            "--analysis_dir", str(self.analysis), "--source_dir", str(self.source),
            "--dest_dir", str(self.dest),
        ])

    def wait(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(10)
        self.assertTrue(predicate(), "Background operation did not complete")

    def open_window(self, cls, config=None):
        window = cls(config or self.config())
        self.windows.append(window)
        window.show()
        QTest.qWait(30)
        if hasattr(window, "_worker"):
            self.wait(lambda: window._worker is None)
            self.assertIsNotNone(window.state.build_result, self.messages)
        return window

    def click(self, window, title):
        button = next(b for b in window.findChildren(QPushButton) if b.text() == title)
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        if hasattr(window, "_worker"):
            self.wait(lambda: window._worker is None)

    def image(self, path, color="steelblue"):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(240, 180, QImage.Format.Format_RGB32)
        image.fill(QColor(color))
        self.assertTrue(image.save(str(path)))

    def file(self, relative, content=b"raw"):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def worker(self, operation, **overrides):
        args = self.inputs(**overrides)
        request = workers.BuildRequest(**args, assignment_path=self.assignment,
            selection_signature=file_signature(self.selection), session_name="Класс", photo_session="SCHOOL")
        worker = workers.PhotoSelectionOperationWorker(request, operation)
        outcomes, errors = [], []
        worker.completed.connect(outcomes.append)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertFalse(errors, errors)
        self.assertEqual(len(outcomes), 1)
        return outcomes[0]

    def copy_now(self, config=None):
        """Run the production noninteractive copy stage and capture its report."""
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = copy_runner.run_copy(config or self.copy_config())
        self.assertEqual(code, 0, output.getvalue() + errors.getvalue())
        return output.getvalue()

    def test_import_manual_click_without_analysis_or_photo_paths(self):
        self.selection.unlink()
        (self.analysis / "info_faces.json").unlink()
        config = self.config()
        del config.source_dir, config.dest_dir, config.exclude_dirs
        with patch.object(core, "build_assignments", side_effect=AssertionError("Importer scanned photographs")):
            window = self.open_window(ImportWindow, config)
            self.assertFalse(hasattr(window, "_image_pipeline"))
            self.assertFalse(hasattr(window, "view_tabs"))
            self.assertIs(window.main_splitter.widget(0), window.table)
            self.assertEqual([window.table.horizontalHeaderItem(i).text()
                              for i in range(window.table.columnCount())],
                             ["student_id", "Фамилия Имя", "Выбранные номера", "Количество", "Источник"])

            def fill_dialog():
                dialog = self.app.activeModalWidget()
                self.assertIsInstance(dialog, QDialog)
                dialog.findChild(QPlainTextEdit).setPlainText("001234; 901256; 001234")
                dialog.accept()

            QTimer.singleShot(20, fill_dialog)
            self.click(window, "Изменить выбранные номера")
            self.assertEqual(json.loads(self.selection.read_text(encoding="utf-8"))["students"]["A7K3-S001"]["selected_numbers"], ["001234", "901256"])
            self.assertFalse(self.assignment.exists())
            self.assertIn("Инструкция оператора", window.import_result.toPlainText())
            self.assertNotIn("Инструкция оператора", window._completion_log_html())

    def test_import_clear_and_explicit_empty_answer(self):
        window = self.open_window(ImportWindow)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.click(window, "Очистить")
        self.assertNotIn("A7K3-S001", json.loads(self.selection.read_text())["students"])
        self.assertIsNone(window.table.cellWidget(0, 4))

        def accept_empty_numbers():
            dialog = self.app.activeModalWidget()
            self.assertIsInstance(dialog, QDialog)
            dialog.findChild(QPlainTextEdit).clear()
            dialog.accept()

        QTimer.singleShot(20, accept_empty_numbers)
        self.click(window, "Изменить выбранные номера")
        self.assertEqual(window.table.item(0, 4).toolTip(), "Данные введены вручную")
        record = json.loads(self.selection.read_text())["students"]["A7K3-S001"]
        self.assertEqual(record["selected_numbers"], [])
        self.assertTrue(record["responded"])

    def test_csv_encodings_separators_and_personal_import(self):
        for encoding, delimiter in [("utf-8-sig", ";"), ("utf-16", "\t"), ("cp1251", ",")]:
            with self.subTest(encoding=encoding):
                path = self.root / "input.csv"
                path.write_text(delimiter.join(["ФИО", "Фото", "Ещё"]) + "\n" + delimiter.join(["Иванова Анна", "901256", "901270"]) + "\n", encoding=encoding)
                table = read_csv_table(path)
                entries, unresolved = import_table(table, self.roster, "ФИО", ["Фото", "Ещё"], min_digits=6, max_digits=6, pad_to_digits=0)
                self.assertFalse(unresolved)
                self.assertEqual(entries[0].selected_numbers, ("901256", "901270"))
        personal = self.root / "Иванова Анна.csv"
        personal.write_text("901256\n901270\n", encoding="utf-8")
        self.assertEqual(import_personal_file(personal, self.roster, min_digits=6, max_digits=6).student_id, "A7K3-S001")

    def test_csv_ui_import_and_repeated_entries(self):
        window = self.open_window(ImportWindow)
        path = self.root / "Иванова Анна.csv"
        path.write_text("901256\n901270\n", encoding="utf-8")
        from PySide6.QtWidgets import QFileDialog
        with patch.object(QFileDialog, "getOpenFileNames", return_value=([str(path)], "")), \
             patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            window.import_personal_csv()
        self.assertEqual(window.document.students["A7K3-S001"].selected_numbers, ["901256", "901270"])
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            window._apply_entries([ImportEntry("A7K3-S001", ("901270",)), ImportEntry("A7K3-S001", ("001234",))], "csv")
        self.assertEqual(window.document.students["A7K3-S001"].selected_numbers, ["901256", "901270", "001234"])

    def test_ai_validation_and_empty_response(self):
        payload = {"matched": [{"student_id": "A7K3-S001", "source_person": "Анна", "selected_numbers": []}], "unresolved": []}
        entries, unresolved = validate_ai_response(payload, self.roster, min_digits=6, max_digits=6)
        self.assertEqual(entries[0].selected_numbers, ())
        self.assertTrue(entries[0].responded)
        payload["matched"][0]["student_id"] = "A7K3-S999"
        with self.assertRaises(ValueError):
            validate_ai_response(payload, self.roster, min_digits=6, max_digits=6)

    def test_foreign_session_is_rejected(self):
        with self.assertRaises(ValueError):
            load_document(self.selection, self.roster, "Другой класс", "SCHOOL")
        with self.assertRaises(ValueError):
            ImportWindow(self.config(photo_session="STREET"))

    def test_parallel_import_does_not_overwrite_external_selection(self):
        first = self.open_window(ImportWindow)
        second = self.open_window(ImportWindow)
        first.document.apply("A7K3-S001", ["901270"], source="manual")
        self.assertTrue(first.save())
        before = self.selection.read_bytes()
        second.document.apply("A7K3-S001", ["001234"], source="manual")
        self.assertFalse(second.save())
        self.assertEqual(self.selection.read_bytes(), before)
        self.assertIn("другим окном", self.messages[-1])

    def test_copy_matches_original_with_raw_sidecars_and_photographer(self):
        for name in ["IMG_901256.CR3", "IMG_901256.xmp", "CaptureOne/Settings/IMG_901256.CR3.cos", "portrait/IMG_901256.jpg", "ph_IMG_901270.psd", "Masks/IMG_901256.png", "nested/Masks/IMG_901270.jpg"]:
            self.file(name)
        new_result = core.build_assignments(**self.inputs())
        old_result = original_core.build_assignments(**self.inputs())
        self.assertEqual(new_result.assignment_payload(), old_result.assignment_payload())
        self.assertEqual([vars(i) for i in new_result.issues], [vars(i) for i in old_result.issues])
        other_dest = self.root / "old-dest"
        old_summary = original_copy.copy_selected_files(old_result, self.source, other_dest)
        copy_result = copy_runner._calculate(self.copy_config())
        self.assertFalse(copy_result.has_errors)
        self.assertEqual(copy_result.assignments, {})
        new_summary = copy_service.copy_selected_files(copy_result, self.source, self.dest)
        self.assertEqual((new_summary.copied, new_summary.skipped), (old_summary.copied, old_summary.skipped))
        self.assertEqual(snapshot(self.dest), snapshot(other_dest))
        self.assertTrue((self.dest / "portrait/CaptureOne/Settings/IMG_901256.CR3.cos").is_file())
        self.assertFalse((self.dest / "portrait/portrait").exists())
        self.assertFalse(any("Masks" in path for path in snapshot(self.dest)))

    def test_copy_worker_does_not_modify_either_json(self):
        self.file("IMG_901256.CR3")
        write_json(self.assignment, {"sentinel": "preserve"})
        before = [(p.read_bytes(), file_signature(p)) for p in (self.selection, self.assignment)]
        outcome = self.worker("copy")
        self.assertEqual(outcome.copy_summary.copied, 1)
        self.assertFalse(outcome.assignment_saved)
        self.assertEqual(before, [(p.read_bytes(), file_signature(p)) for p in (self.selection, self.assignment)])

    def test_copy_idempotence_conflict_and_same_source(self):
        source = self.file("IMG_901256.CR3")
        self.assertEqual(self.worker("copy").copy_summary.copied, 1)
        self.assertEqual(self.worker("copy").copy_summary.skipped, 1)
        source.write_bytes(b"different raw")
        outcome = self.worker("copy")
        self.assertFalse(outcome.result.has_errors)
        self.assertEqual(outcome.copy_summary.skipped, 1)
        self.assertEqual((self.dest / "portrait/IMG_901256.CR3").read_bytes(), b"raw")
        same = self.worker("copy", source_dir=self.dest, dest_dir=self.dest)
        self.assertEqual(same.copy_summary.skipped, 1)

    def test_ph_prefix_already_in_destination_marks_source_frame(self):
        self.file("IMG_901270.psd")
        self.image(self.dest / "group_photo_01/PH_IMG_901270.jpg")
        self.copy_now()
        self.assertTrue((self.dest / "group_photo_01/IMG_901270.psd").is_file())
        outcome = self.worker("copy")
        self.assertTrue((self.dest / "group_photo_01/IMG_901270.psd").is_file())
        self.assertEqual(outcome.result.assignments["A7K3-S002"], ["901270"])

    def test_unknown_location_and_unsafe_location(self):
        self.info["IMG_901256.jpg"].pop("location_name")
        write_json(self.analysis / "info_faces.json", self.info)
        self.file("IMG_901256.CR3")
        self.worker("copy")
        self.assertTrue((self.dest / "unknown/IMG_901256.CR3").exists())
        self.info["IMG_901256.jpg"]["location_name"] = "../outside"
        write_json(self.analysis / "info_faces.json", self.info)
        before = snapshot(self.dest)
        outcome = self.worker("copy")
        self.assertTrue(outcome.result.has_errors)
        self.assertIsNone(outcome.copy_summary)
        self.assertEqual(snapshot(self.dest), before)

    def test_ambiguous_number_blocks_copy(self):
        self.file("IMG_901256_901270.jpg")
        outcome = self.worker("copy")
        self.assertIn("ambiguous_filename", {i.code for i in outcome.result.issues})
        self.assertIsNone(outcome.copy_summary)

    def test_build_requires_destination_image_and_preserves_old_plan(self):
        self.file("IMG_901256.CR3")
        self.worker("copy")
        write_json(self.assignment, {"sentinel": 1})
        before = self.assignment.read_bytes()
        outcome = self.worker("build")
        self.assertFalse(outcome.assignment_saved)
        self.assertEqual(before, self.assignment.read_bytes())
        self.assertIn("assignment_layout_file_missing", {i.code for i in outcome.result.issues})

    def test_build_does_not_copy_or_edit_selection(self):
        self.image(self.dest / "portrait/IMG_901256.jpg")
        before = (self.selection.read_bytes(), file_signature(self.selection))
        with patch.object(workers, "copy_selected_files", side_effect=AssertionError("Build copied files")):
            outcome = self.worker("build")
        self.assertTrue(outcome.assignment_saved)
        self.assertEqual(before, (self.selection.read_bytes(), file_signature(self.selection)))
        self.assertEqual(json.loads(self.assignment.read_text()), original_core.build_assignments(**self.inputs()).assignment_payload())

    def test_worker_drops_selection_changed_during_scan(self):
        self.image(self.dest / "portrait/IMG_901256.jpg")
        original = workers.PhotoSelectionOperationWorker._build

        def mutate(worker):
            result = original(worker)
            self.document.apply("A7K3-S002", ["901270"], source="manual")
            save_document(self.selection, self.document)
            return result

        with patch.object(workers.PhotoSelectionOperationWorker, "_build", mutate):
            outcome = self.worker("build")
        self.assertFalse(outcome.assignment_saved)
        self.assertFalse(self.assignment.exists())

    def test_atomic_assignment_failure_preserves_previous_json(self):
        self.image(self.dest / "portrait/IMG_901256.jpg")
        result = core.build_assignments(**self.inputs())
        self.assignment.write_bytes(b"old json")
        with patch.object(core.os, "replace", side_effect=OSError("test failure")):
            with self.assertRaises(OSError):
                core.save_assignments(self.assignment, result)
        self.assertEqual(self.assignment.read_bytes(), b"old json")
        self.assertFalse(list(self.analysis.glob("*.tmp")))

    def test_gui_raw_copy_then_export_copy_and_plan(self):
        self.file("IMG_901256.CR3")
        before_selection = (self.selection.read_bytes(), file_signature(self.selection))
        report = self.copy_now()
        self.assertIn("Скопировано: 1", report)
        self.assertIn("Копирование завершено", report)
        self.assertNotIn("НЕ ГОТОВО К ВЕРСТКЕ", report)
        self.assertFalse(self.assignment.exists())

        export = self.root / "Output/Альбом/Фото/SCHOOL"
        album = self.root / "Альбом/Фото/SCHOOL"
        self.image(export / "portrait/IMG_901256.jpg")
        self.image(export / "group_photo_01/PH_IMG_901270.jpg", "tan")
        self.image(self.analysis / "JPG/IMG_901256.jpg")
        self.copy_now(self.config(source_dir=str(export), dest_dir=str(album)))
        plan = self.open_window(AssignmentsWindow, self.config(source_dir=str(export), dest_dir=str(album)))
        self.click(plan, "Создать план вёрстки")
        self.assertEqual(json.loads(self.assignment.read_text())["assignments"], {
            "A7K3-S001": ["901256", "901270"], "A7K3-S002": ["901270"]})
        self.assertTrue(plan._album_readiness(plan.state.build_result)["ready"])
        self.assertEqual(before_selection, (self.selection.read_bytes(), file_signature(self.selection)))
        self.assertEqual(plan.view_tabs.count(), 2)
        for window in (plan,):
            self.assertFalse(hasattr(window, "edit_selected_numbers"))
            self.assertFalse(hasattr(window, "_save_document"))
            menu = window._assignment_context_menu(window.student_location_table, window.student_location_table.topLevelItem(0))
            self.assertFalse(any("Изменить" in a.text() or "Очистить" in a.text() for a in menu.actions()))
            menu.deleteLater()
        with self.assertRaises(ValueError):
            plan._start_operation("copy")
        # The existing generator accepts the unchanged schema.
        generator = ROOT / "scripts/Workflow2/InDesign/idsgn_spread_generator"
        sys.path.insert(0, str(generator))
        from common.data import load_children_data
        children = load_children_data(self.list_file, self.assignment)
        self.assertTrue(children)

    def test_gui_jpg_refresh_after_reimport_and_report_export(self):
        self.image(self.source / "IMG_901256.jpg")
        self.image(self.source / "IMG_901270.jpg")
        self.image(self.analysis / "JPG/IMG_901256.jpg")
        self.image(self.analysis / "JPG/IMG_901270.jpg", "tan")
        self.copy_now()
        plan = self.open_window(AssignmentsWindow)
        self.click(plan, "Создать план вёрстки")
        importer = self.open_window(ImportWindow)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            importer._apply_entries([ImportEntry("A7K3-S001", ("901256", "901270"))], "manual")
        self.assertEqual(importer.table.item(0, 2).text(), "901256, 901270")
        self.click(plan, "Обновить список")
        self.assertFalse(plan._album_readiness(plan.state.build_result)["ready"])
        self.copy_now()
        self.click(plan, "Создать план вёрстки")
        self.assertTrue(plan._album_readiness(plan.state.build_result)["ready"])
        from PySide6.QtWidgets import QFileDialog
        for extension, action in [("html", plan._save_student_location_html), ("csv", plan._save_student_location_csv)]:
            output = self.root / f"report.{extension}"
            with patch.object(QFileDialog, "getSaveFileName", return_value=(str(output), "")):
                action()
            self.assertIn("901270", output.read_text(encoding="utf-8-sig"))
        self.assertNotIn("Инструкция оператора", plan._completion_log_html())
        item = plan.photo_table.topLevelItem(0)
        plan.photo_table.setCurrentItem(item)
        self.wait(lambda: plan.preview.pixmap() is not None and not plan.preview.pixmap().isNull())
        artifacts = os.environ.get("PHOTO_SELECTION04_SCREENSHOTS")
        if artifacts:
            folder = Path(artifacts)
            folder.mkdir(parents=True, exist_ok=True)
            for name, window in [("import", importer), ("assignments", plan)]:
                window.raise_()
                QTest.qWait(150)
                self.assertTrue(window.grab().save(str(folder / f"{name}.png")))
        QTest.mouseClick(plan.view_tabs.tabBar(), Qt.MouseButton.LeftButton,
                         pos=plan.view_tabs.tabBar().tabRect(1).center())
        self.assertEqual(plan.view_tabs.currentIndex(), 1)
        menu = plan._assignment_context_menu(plan.student_location_table, None)
        next(action for action in menu.actions() if action.text() == "Развернуть все").trigger()
        student = plan.student_location_table.topLevelItem(0)
        self.assertTrue(student.isExpanded())
        plan.student_location_table.setCurrentItem(student)
        self.wait(lambda: plan.preview.pixmap() is not None and not plan.preview.pixmap().isNull())
        if artifacts:
            self.assertTrue(plan.grab().save(str(folder / "assignments_students.png")))
        menu.deleteLater()

    def test_full_entrypoint_event_loops(self):
        before = (self.selection.read_bytes(), file_signature(self.selection))
        for stage in ("selection_import", "selection_assignments"):
            args = [sys.executable, str(SUITE / "_tests/entrypoint_probe.py"),
                    str(SUITE / stage / f"run_{stage}.py"),
                    "--student_list_file", str(self.list_file),
                    "--analysis_dir", str(self.analysis), "--session_name", "Класс",
                    "--photo_session", "SCHOOL"]
            if stage != "selection_import":
                args += ["--source_dir", str(self.source), "--dest_dir", str(self.dest)]
            result = subprocess.run(args, cwd=self.root, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            outcome = json.loads(result.stdout.splitlines()[-1])
            self.assertTrue(outcome["opened"])
            self.assertEqual(outcome["tabs"], 0 if stage == "selection_import" else 2)
        self.assertEqual(before, (self.selection.read_bytes(), file_signature(self.selection)))

    def test_each_window_persists_its_own_geometry(self):
        windows = [self.open_window(cls) for cls in (ImportWindow, AssignmentsWindow)]
        with patch.object(window_base, "IS_MANAGED_RUN", True), \
             patch.object(window_base, "pysm_context") as context:
            for window in windows:
                window._save_window_state()
        keys = [call.args[0] for call in context.set_structured.call_args_list]
        self.assertEqual(keys, ["win_state.photo_selection04_import", "win_state.photo_selection04_assignments"])
        states = [call.args[1] for call in context.set_structured.call_args_list]
        self.assertNotIn("active_tab", states[0])
        self.assertEqual(states[1]["active_tab"], windows[1].view_tabs.currentIndex())
        with patch.object(window_base, "IS_MANAGED_RUN", True), \
             patch.object(window_base, "pysm_context") as context, \
             patch.object(window_base.logger, "warning") as warning:
            context.get_structured.return_value = {**states[0], "active_tab": 0}
            windows[0]._restore_window_state()
            warning.assert_not_called()

    def test_optional_matches_and_unrecognized_selection(self):
        self.info["IMG_901256.jpg"]["faces"] = [{"student_id": "A7K3-S002"}]
        write_json(self.analysis / "info_faces.json", self.info)
        self.file("IMG_901256.jpg")
        result = core.build_assignments(**self.inputs())
        self.assertEqual(result.assignments["A7K3-S001"], ["901256"])
        self.assertIn("student_not_recognized", {i.code for i in result.issues})
        # Even an empty optional file must preserve the baseline's interpretation.
        write_json(self.analysis / "matches_portrait_to_group.json", {})
        result = core.build_assignments(**self.inputs())
        old = original_core.build_assignments(**self.inputs())
        self.assertEqual(result.assignment_payload(), old.assignment_payload())
        self.assertEqual([vars(i) for i in result.issues], [vars(i) for i in old.issues])

    def test_partial_copy_failure_preserves_successes_and_cleans_temporary_file(self):
        self.file("IMG_901256.CR3")
        self.file("IMG_901256.xmp", b"xmp")
        result = core.build_assignments(**self.inputs())
        original = copy_service.shutil.copy2

        def fail_sidecar(source, target):
            if source.suffix == ".xmp":
                raise OSError("Synthetic sidecar failure")
            return original(source, target)

        with patch.object(copy_service.shutil, "copy2", side_effect=fail_sidecar):
            outcome = copy_service.copy_selected_files(result, self.source, self.dest)
        self.assertEqual(outcome.copied, 1)
        self.assertIn("copy_failed", {i.code for i in outcome.issues})
        self.assertTrue((self.dest / "portrait/IMG_901256.CR3").exists())
        self.assertFalse(list(self.dest.rglob("*.tmp")))

    def test_cli_passports_and_independent_entrypoints(self):
        from pysm_lib.script_scanner import read_script_passport, find_run_file
        for stage in ("selection_import", "selection_copy", "selection_assignments"):
            folder = SUITE / stage
            passport, error = read_script_passport(folder)
            self.assertFalse(error)
            self.assertIsNotNone(passport)
            entry = find_run_file(folder)
            self.assertTrue(entry.is_file())
            result = subprocess.run([sys.executable, str(entry), "--help"], cwd=self.root,
                                    capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            parser = make_parser(stage)
            self.assertEqual(set(passport["command_line_args_meta"]), {a.dest for a in parser._actions if a.dest != "help"})
            self.assertEqual(b"--source_dir" in result.stdout, stage != "selection_import")
            self.assertNotIn(b"--operation", result.stdout)
            self.assertNotIn(b"<b>", result.stdout)

    def test_entrypoints_do_not_share_generic_lib_or_common_modules(self):
        """All stages can coexist with unrelated packages named _lib/_common."""
        with patch.dict(sys.modules, {"_lib": ModuleType("_lib"), "_common": ModuleType("_common")}):
            for stage in ("selection_import", "selection_copy", "selection_assignments"):
                entry = importlib.import_module(f"photo_selection2.{stage}.run_{stage}")
                app_module = importlib.import_module(
                    f"photo_selection2.{stage}._lib.{'runner' if stage == 'selection_copy' else 'app'}"
                )
                function = "run_copy" if stage == "selection_copy" else "run_application"
                with patch.object(entry, "get_config", return_value=Namespace()), \
                     patch.object(app_module, function, return_value=stage):
                    self.assertEqual(entry.main(), stage)

    def test_portable_suite_full_entrypoints_use_the_copied_modules(self):
        portable = self.root / "portable scripts/photo_selection2"
        shutil.copytree(SUITE, portable, ignore=shutil.ignore_patterns("__pycache__", "_tests"))
        self.file("IMG_901256.CR3")
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
        for key in ("PYSM_CONTEXT_FILE", "PYSM_CONTEXT_SHM_NAME", "PYSM_CONTEXT_MODE"):
            env.pop(key, None)
        for stage in ("selection_import", "selection_copy", "selection_assignments"):
            entry = portable / stage / f"run_{stage}.py"
            if stage == "selection_copy":
                args = [sys.executable, str(entry), "--analysis_dir", str(self.analysis),
                        "--source_dir", str(self.source), "--dest_dir", str(self.dest)]
            else:
                args = [sys.executable, str(SUITE / "_tests/entrypoint_probe.py"), str(entry),
                        "--student_list_file", str(self.list_file), "--analysis_dir", str(self.analysis),
                        "--session_name", "Класс", "--photo_session", "SCHOOL"]
                if stage == "selection_assignments":
                    args += ["--source_dir", str(self.source), "--dest_dir", str(self.dest)]
            result = subprocess.run(args, cwd=self.root, env=env, capture_output=True,
                                    encoding="utf-8", timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            if stage != "selection_copy":
                outcome = json.loads(result.stdout.splitlines()[-1])
                self.assertTrue(Path(outcome["module_file"]).is_relative_to(portable))
                self.assertIn("</b>", result.stdout)
            else:
                self.assertIn("Скопировано: 1.", result.stdout)
                self.assertEqual((self.dest / "portrait/IMG_901256.CR3").read_bytes(), b"raw")

    def test_import_resource_report_is_returned_once_and_reports_unsaved_changes(self):
        window = self.open_window(ImportWindow)
        with patch.object(window_base, "pysm_context") as context:
            html = window._completion_log_html()
            self.assertIsInstance(html, str)
            self.assertIn(self.selection.resolve().as_uri(), html)
            self.assertIn("Рабочая<br>папка", html)
            context.log_html.assert_not_called()
            window._emit_final_log_once()
            window._emit_final_log_once()
            context.log_html.assert_called_once_with(html)
        window.document.apply("A7K3-S002", ["901270"], source="manual")
        self.assertIn("Выбор не сохранён", window._completion_log_html())

    def test_copy_report_failure_does_not_fail_completed_move(self):
        source = self.file("IMG_901256.CR3")
        config = self.copy_config()
        config.mode = "move"
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(copy_runner, "StandardTreeBuilder", side_effect=RuntimeError("Report failed")), \
             redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(copy_runner.run_copy(config), 0)
        self.assertFalse(source.exists())
        self.assertEqual((self.dest / "portrait/IMG_901256.CR3").read_bytes(), b"raw")
        self.assertIn("Перемещение завершено", output.getvalue())
        self.assertIn("Предупреждение", errors.getvalue())

    def test_copy_full_entrypoint_without_pysm_or_qt(self):
        self.file("IMG_901256.CR3")
        code = '''
import importlib.abc, runpy, sys
class NoPySM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pysm_lib', 'PySide6'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, NoPySM())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        result = subprocess.run([
            sys.executable, "-c", code, str(SUITE / "selection_copy/run_selection_copy.py"),
            "--analysis_dir", str(self.analysis), "--source_dir", str(self.source),
            "--dest_dir", str(self.dest),
        ], cwd=self.root, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
           capture_output=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Скопировано: 1.", result.stdout)
        self.assertIn(str(self.dest), result.stdout)

    def copy_subprocess(self, *extra_args):
        """Use the real managed progress protocol, without a PySM context file."""
        env = dict(os.environ)
        for key in ("PYSM_CONTEXT_FILE", "PYSM_CONTEXT_SHM_NAME", "PYSM_CONTEXT_MODE"):
            env.pop(key, None)
        env.update(PY_SCRIPT_MANAGER_ACTIVE="1", PYTHONIOENCODING="utf-8")
        args = [sys.executable, str(SUITE / "_tests/copy_entrypoint_probe.py"),
                str(SUITE / "selection_copy/run_selection_copy.py"),
                "--analysis_dir", str(self.analysis), "--source_dir", str(self.source),
                "--dest_dir", str(self.dest), *extra_args]
        result = subprocess.run(args, cwd=self.root, env=env, capture_output=True,
                                encoding="utf-8", timeout=20)
        self.assertIn("COPY_PROBE_NO_GUI", result.stdout, result.stdout + result.stderr)
        progress = []
        for line in result.stderr.splitlines():
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if payload.get("type") == "progress":
                progress.append(payload)
        self.assertTrue(progress)
        self.assertEqual(progress[-1], {"type": "progress", "current": 0, "total": 0, "text": ""})
        return result, progress

    def test_headless_copy_entrypoint_progress_and_json_preservation(self):
        self.file("IMG_901256.CR3")
        self.file("IMG_901256.xmp")
        write_json(self.assignment, {"sentinel": "unchanged"})
        before = [(p.read_bytes(), file_signature(p)) for p in (self.selection, self.assignment)]
        result, progress = self.copy_subprocess()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        frames = [p for p in progress if p["total"] == 2]
        self.assertEqual({p["current"] for p in frames}, {0, 1, 2})
        self.assertIn("Скопировано: 2. Пропущено: 0.", result.stdout)
        self.assertEqual(before, [(p.read_bytes(), file_signature(p)) for p in (self.selection, self.assignment)])
        result, _ = self.copy_subprocess()
        self.assertEqual(result.returncode, 0)
        self.assertIn("Скопировано: 0. Пропущено: 2.", result.stdout)

    def test_headless_default_skip_preserves_existing_file_and_closes_progress(self):
        self.file("IMG_901256.CR3")
        self.file("IMG_901256.xmp", b"sidecar")
        target = self.dest / "portrait/IMG_901256.CR3"
        target.parent.mkdir()
        target.write_bytes(b"RAW")
        result, _ = self.copy_subprocess()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Копирование завершено", result.stdout)
        self.assertEqual(target.read_bytes(), b"RAW")
        self.assertEqual((target.parent / "IMG_901256.xmp").read_bytes(), b"sidecar")
        self.assertIn("Скопировано: 1. Пропущено: 1.", result.stdout)

    def test_headless_overwrite_replaces_existing_bytes(self):
        self.file("IMG_901256.CR3", b"new")
        target = self.dest / "portrait/IMG_901256.CR3"
        target.parent.mkdir()
        target.write_bytes(b"old")
        before = snapshot(self.analysis)
        result, progress = self.copy_subprocess("--on_conflict", "overwrite")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(target.read_bytes(), b"new")
        self.assertIn("Скопировано: 1. Пропущено: 0.", result.stdout)
        self.assertTrue(any(p["current"] == p["total"] == 1 for p in progress))
        result, _ = self.copy_subprocess("--on_conflict", "overwrite")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Скопировано: 1. Пропущено: 0.", result.stdout)
        self.assertEqual(snapshot(self.analysis), before)
        self.assertFalse(list(self.dest.rglob("*.tmp")))

    def test_headless_rename_uses_first_available_suffix_and_repeats(self):
        self.file("portrait/IMG_901256.jpg", b"same")
        target = self.dest / "portrait/IMG_901256.jpg"
        target.parent.mkdir()
        target.write_bytes(b"same")
        target.with_name("IMG_901256 (1).jpg").write_bytes(b"keep one")
        target.with_name("IMG_901256 (3).jpg").write_bytes(b"keep three")
        before = snapshot(self.analysis)
        for suffix in (2, 4):
            result, _ = self.copy_subprocess("--on_conflict", "rename")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(target.with_name(f"IMG_901256 ({suffix}).jpg").read_bytes(), b"same")
            self.assertIn("Скопировано: 1. Пропущено: 0.", result.stdout)
        self.assertEqual(target.read_bytes(), b"same")
        self.assertEqual(target.with_name("IMG_901256 (1).jpg").read_bytes(), b"keep one")
        self.assertEqual(target.with_name("IMG_901256 (3).jpg").read_bytes(), b"keep three")
        self.assertEqual(snapshot(self.analysis), before)
        self.assertFalse(list(self.dest.rglob("*.tmp")))

    def test_overwrite_failure_preserves_old_file_and_other_copies(self):
        self.file("IMG_901256.CR3", b"new raw")
        self.file("IMG_901256.xmp", b"new xmp")
        target = self.dest / "portrait/IMG_901256.CR3"
        target.parent.mkdir()
        target.write_bytes(b"old raw")
        original_copy2 = copy_service.shutil.copy2

        def fail_raw(source, temporary):
            if source.suffix == ".CR3":
                temporary.write_bytes(b"partial")
                raise OSError("Synthetic partial write failure")
            return original_copy2(source, temporary)

        config = self.copy_config()
        config.on_conflict = "overwrite"
        with patch.object(copy_service.shutil, "copy2", side_effect=fail_raw), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(copy_runner.run_copy(config), 1)
        self.assertEqual(target.read_bytes(), b"old raw")
        self.assertEqual(target.with_suffix(".xmp").read_bytes(), b"new xmp")
        self.assertFalse(list(self.dest.rglob("*.tmp")))

    def test_overwrite_publish_failure_preserves_old_file(self):
        self.file("IMG_901256.CR3", b"new")
        target = self.dest / "portrait/IMG_901256.CR3"
        target.parent.mkdir()
        target.write_bytes(b"old")
        config = self.copy_config()
        config.on_conflict = "overwrite"
        with patch.object(copy_service.os, "replace", side_effect=PermissionError("File locked")), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(copy_runner.run_copy(config), 1)
        self.assertEqual(target.read_bytes(), b"old")
        self.assertFalse(list(self.dest.rglob("*.tmp")))

    def test_same_physical_file_is_skipped_in_every_mode(self):
        self.file("portrait/IMG_901256.CR3")
        for mode in ("copy", "move"):
            for conflict in ("skip", "overwrite", "rename"):
                config = self.copy_config()
                config.dest_dir = str(self.source)
                config.on_conflict = conflict
                config.mode = mode
                before = snapshot(self.source)
                report = self.copy_now(config)
                label = "Перемещено" if mode == "move" else "Скопировано"
                self.assertIn(f"{label}: 0. Пропущено: 1.", report)
                self.assertEqual(snapshot(self.source), before)

    def test_skip_and_rename_handle_destination_created_during_publish(self):
        self.file("IMG_901256.CR3", b"new")
        original_rename = copy_service.os.rename
        for mode in ("skip", "rename"):
            config = self.copy_config()
            config.dest_dir = str(self.root / f"race-{mode}")
            config.on_conflict = mode
            target = Path(config.dest_dir) / "portrait/IMG_901256.CR3"
            raced = False

            def publish_after_race(temporary, destination):
                nonlocal raced
                if not raced:
                    raced = True
                    destination.write_bytes(b"other process")
                    raise FileExistsError("Name claimed during publication")
                return original_rename(temporary, destination)

            with patch.object(copy_service.os, "rename", side_effect=publish_after_race):
                report = self.copy_now(config)
            self.assertTrue(raced)
            self.assertEqual(target.read_bytes(), b"other process")
            if mode == "rename":
                self.assertEqual(target.with_name("IMG_901256 (1).CR3").read_bytes(), b"new")
                self.assertIn("Скопировано: 1. Пропущено: 0.", report)
            else:
                self.assertIn("Скопировано: 0. Пропущено: 1.", report)
            self.assertFalse(list(Path(config.dest_dir).rglob("*.tmp")))

    def test_headless_copy_without_roster_matches_or_session_arguments(self):
        self.list_file.unlink()
        for filename in ("matches_portrait_to_group.json", "photo_assignments.json"):
            (self.analysis / filename).write_text("not JSON", encoding="utf-8")
        self.info["IMG_901270.jpg"]["faces"] = [{"student_id": "UNKNOWN"}]
        write_json(self.analysis / "info_faces.json", self.info)
        self.document.apply("A7K3-S002", ["901256", "001234"], source="manual")
        save_document(self.selection, self.document)
        self.file("IMG_901256.CR3")
        self.file("IMG_001234.CR3")
        self.file("nested/PH_IMG_901270.psd")
        before = snapshot(self.analysis)
        result, _ = self.copy_subprocess()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Скопировано: 3. Пропущено: 0.", result.stdout)
        self.assertTrue((self.dest / "portrait/IMG_901256.CR3").is_file())
        self.assertTrue((self.dest / "unknown/IMG_001234.CR3").is_file())
        self.assertTrue((self.dest / "group_photo_01/nested/PH_IMG_901270.psd").is_file())
        self.assertEqual(snapshot(self.analysis), before)

    def test_copy_cli_exposes_only_copy_inputs(self):
        config = self.copy_config()
        self.assertEqual(set(vars(config)), {"analysis_dir", "source_dir", "dest_dir", "exclude_dirs", "on_conflict", "mode"})
        self.assertEqual(config.exclude_dirs, "Masks")
        self.assertEqual(config.on_conflict, "skip")
        self.assertEqual(config.mode, "copy")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            make_parser("selection_copy").parse_args([
                "--analysis_dir", str(self.analysis), "--source_dir", str(self.source),
                "--dest_dir", str(self.dest), "--mode", "invalid",
            ])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            make_parser("selection_copy").parse_args([
                "--analysis_dir", str(self.analysis), "--source_dir", str(self.source),
                "--dest_dir", str(self.dest), "--on_conflict", "invalid",
            ])
        for name in ("student_list_file", "session_name", "photo_session"):
            with self.subTest(name=name), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                make_parser("selection_copy").parse_args([
                    "--analysis_dir", str(self.analysis), "--source_dir", str(self.source),
                    "--dest_dir", str(self.dest), f"--{name}", "unused",
                ])

    def test_headless_move_preserves_structure_json_and_excluded_files(self):
        names = ["portrait/IMG_901256.CR3", "CaptureOne/Settings/IMG_901256.CR3.cos",
                 "IMG_901256.xmp", "exports/PH_IMG_901270.jpg"]
        for name in names:
            self.file(name, name.encode())
        self.file("nested/Masks/IMG_901256.png", b"excluded")
        self.file("IMG_999999.jpg", b"unselected")
        before = snapshot(self.analysis)
        result, progress = self.copy_subprocess("--mode", "move")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Перемещено: 4. Пропущено: 0.", result.stdout)
        self.assertTrue(any(p["current"] == p["total"] == 4 for p in progress))
        for name in names:
            self.assertFalse((self.source / name).exists())
        for relative in ["portrait/IMG_901256.CR3", "portrait/CaptureOne/Settings/IMG_901256.CR3.cos",
                         "portrait/IMG_901256.xmp", "group_photo_01/exports/PH_IMG_901270.jpg"]:
            self.assertTrue((self.dest / relative).is_file())
        self.assertEqual((self.source / "nested/Masks/IMG_901256.png").read_bytes(), b"excluded")
        self.assertTrue((self.source / "IMG_999999.jpg").is_file())
        self.assertEqual(snapshot(self.analysis), before)
        result, _ = self.copy_subprocess("--mode", "move")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Нет исходных файлов для перемещения", result.stdout)

    def test_headless_move_conflict_policies(self):
        target = self.dest / "portrait/IMG_901256.CR3"
        target.parent.mkdir()
        for conflict in ("skip", "overwrite", "rename"):
            with self.subTest(conflict=conflict):
                source = self.file("IMG_901256.CR3", b"new")
                target.write_bytes(b"old")
                result, _ = self.copy_subprocess("--mode", "move", "--on_conflict", conflict)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if conflict == "skip":
                    self.assertEqual(source.read_bytes(), b"new")
                    self.assertEqual(target.read_bytes(), b"old")
                    self.assertIn("Перемещено: 0. Пропущено: 1.", result.stdout)
                else:
                    self.assertFalse(source.exists())
                    self.assertIn("Перемещено: 1. Пропущено: 0.", result.stdout)
                    if conflict == "overwrite":
                        self.assertEqual(target.read_bytes(), b"new")
                    else:
                        self.assertEqual(target.read_bytes(), b"old")
                        self.assertEqual(target.with_name("IMG_901256 (1).CR3").read_bytes(), b"new")

    def test_move_write_and_publish_failures_preserve_original(self):
        source = self.file("IMG_901256.CR3", b"new")
        target = self.dest / "portrait/IMG_901256.CR3"
        target.parent.mkdir()
        target.write_bytes(b"old")
        config = self.copy_config()
        config.mode, config.on_conflict = "move", "overwrite"
        for module, name in [(copy_service.shutil, "copy2"), (copy_service.os, "replace")]:
            with self.subTest(name=name), patch.object(module, name, side_effect=OSError("Synthetic failure")), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(copy_runner.run_copy(config), 1)
            self.assertEqual(source.read_bytes(), b"new")
            self.assertEqual(target.read_bytes(), b"old")
            self.assertFalse(list(self.dest.rglob("*.tmp")))

    def test_move_failed_source_removal_keeps_copy_and_continues(self):
        source = self.file("IMG_901256.CR3", b"raw")
        sidecar = self.file("IMG_901256.xmp", b"xmp")
        original_unlink = Path.unlink

        def refuse_original(path, *args, **kwargs):
            if path == source:
                raise PermissionError("Original is locked")
            return original_unlink(path, *args, **kwargs)

        config = self.copy_config()
        config.mode = "move"
        output = io.StringIO()
        with patch.object(Path, "unlink", refuse_original), redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertEqual(copy_runner.run_copy(config), 1)
        self.assertEqual(source.read_bytes(), b"raw")
        self.assertEqual((self.dest / "portrait/IMG_901256.CR3").read_bytes(), b"raw")
        self.assertEqual((self.dest / "portrait/IMG_901256.xmp").read_bytes(), b"xmp")
        self.assertFalse(sidecar.exists())
        self.assertIn("Перемещено: 1. Пропущено: 0.", output.getvalue())
        self.assertIn("Скопировано без удаления оригинала: 1.", output.getvalue())
        self.assertFalse(list(self.dest.rglob("*.tmp")))

    def test_move_does_not_delete_source_changed_during_copy(self):
        source = self.file("IMG_901256.CR3", b"original")
        original_copy2 = copy_service.shutil.copy2

        def edit_after_copy(path, temporary):
            result = original_copy2(path, temporary)
            path.write_bytes(b"updated original")
            return result

        config = self.copy_config()
        config.mode = "move"
        with patch.object(copy_service.shutil, "copy2", side_effect=edit_after_copy), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(copy_runner.run_copy(config), 1)
        self.assertEqual(source.read_bytes(), b"updated original")
        self.assertEqual((self.dest / "portrait/IMG_901256.CR3").read_bytes(), b"original")

    def test_headless_invalid_selection_blocks_copy(self):
        self.file("IMG_901256.CR3")
        cases = [
            {"schema_version": 99, "students": {}},
            {"schema_version": 1, "students": []},
            {"schema_version": 1, "students": {"A7K3-S001": None}},
            {"schema_version": 1, "students": {"A7K3-S001": {"selected_numbers": "901256"}}},
            {"schema_version": 1, "students": {"A7K3-S001": {"selected_numbers": ["901256", "123"]}}},
        ]
        for selection in cases:
            with self.subTest(selection=selection):
                write_json(self.selection, selection)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(copy_runner.run_copy(self.copy_config()), 1)
                self.assertEqual(snapshot(self.dest), {})

    def test_headless_unsafe_location_and_ambiguous_filename_block_copy(self):
        self.file("IMG_901256.CR3")
        self.info["IMG_901256.jpg"]["location_name"] = "../outside"
        write_json(self.analysis / "info_faces.json", self.info)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(copy_runner.run_copy(self.copy_config()), 1)
        self.assertEqual(snapshot(self.dest), {})
        self.info["IMG_901256.jpg"]["location_name"] = "portrait"
        write_json(self.analysis / "info_faces.json", self.info)
        self.file("IMG_901256_901270.jpg")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(copy_runner.run_copy(self.copy_config()), 1)
        self.assertEqual(snapshot(self.dest), {})

    def test_headless_missing_input_and_empty_source(self):
        result, _ = self.copy_subprocess()
        self.assertEqual(result.returncode, 0)
        self.assertIn("Нет исходных файлов для копирования", result.stdout)
        (self.analysis / "info_faces.json").unlink()
        result, _ = self.copy_subprocess()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Ошибка копирования", result.stderr)

    def test_headless_selection_change_during_scan_prevents_copy(self):
        self.file("IMG_901256.CR3")
        calculate = copy_runner._calculate

        def mutate(config):
            result = calculate(config)
            self.document.apply("A7K3-S002", ["901270"], source="manual")
            save_document(self.selection, self.document)
            return result

        with patch.object(copy_runner, "_calculate", side_effect=mutate), \
             patch.object(copy_runner, "copy_selected_files") as copy, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(copy_runner.run_copy(self.config()), 1)
        copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
