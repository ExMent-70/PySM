"""Standalone importer extracted from the original photo_selection tab."""
from __future__ import annotations
import copy
import sys
from pathlib import Path
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QMenu, QSplitter, QTableWidget, QTableWidgetItem,
    QTextBrowser, QVBoxLayout, QWidget,
)


try:
    from pysm_lib.gui.ai import edit_ai_json_response
    from pysm_lib.pysm_icons import icons as pysm_icons
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    sys.exit(1)

from ..._common.window_base import WindowBase, run_window
from ..._common.import_reports import ImportReportMixin
from ..._common.ai_import import create_ai_import_request
from ..._common.constants import PHOTO_NUMBER_DIGITS
from ..._common.csv_import import import_personal_file, import_table, read_csv_table, read_personal_numbers
from ..._common.domain import ImportEntry, coalesce_import_entries
from ..._common.number_parser import parse_manual_numbers
from ..._common.roster import load_roster
from ..._common.storage import load_document, save_document
from ..._common.signatures import file_signature
from ..._common.ui_widgets import CsvMappingDialog, SelectedNumbersDialog

class ImportWindow(ImportReportMixin, WindowBase):
    """Edit only photo_selection.json; analysis and image folders are unnecessary."""
    WINDOW_STATE_VAR = "win_state.photo_selection04_import"
    HEADERS = ("student_id", "Фамилия Имя", "Выбранные номера", "Количество", "Источник")

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.roster = load_roster(Path(config.student_list_file))
        self.selection_path = Path(config.analysis_dir) / "photo_selection.json"
        self._selection_signature = file_signature(self.selection_path)
        self.document = load_document(self.selection_path, self.roster, config.session_name, config.photo_session)
        self._saved_state = copy.deepcopy(self.document.to_dict())
        self._final_log_emitted = False
        self.user_message_html = str(config.message or "")
        self.setWindowTitle(config.title)
        self.resize(1280, 780)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addWidget(QLabel(f"Список: {self.roster.list_id} | Фотосессия: {config.photo_session}"))
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setIconSize(QSize(20, 20))
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.currentCellChanged.connect(self._on_current_student_changed)
        for column in (1, 2):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.import_result = QTextBrowser()
        self.import_result.setMinimumWidth(320)
        self.import_result.setOpenExternalLinks(False)
        self.import_result.setOpenLinks(False)
        self.import_result.anchorClicked.connect(self._open_report_link)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.table)
        self.main_splitter.addWidget(self.import_result)
        self.main_splitter.setSizes([880, 400])
        layout.addWidget(self.main_splitter, 1)
        buttons = QHBoxLayout()
        import_button = QPushButton("Импорт")
        import_menu = QMenu(import_button)
        import_menu.addAction("Общая CSV", self.import_csv)
        import_menu.addAction("Персональные CSV", self.import_personal_csv)
        import_menu.addAction("AI-промпт / JSON", self.import_ai)
        import_button.setMenu(import_menu)
        buttons.addWidget(import_button)
        for title, handler in (("Изменить выбранные номера", self.edit_selected_numbers), ("Очистить", self.clear_selected), ("Сохранить", self.save)):
            button = QPushButton(title)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self._refresh_table()
        self._restore_window_state()

    def _splitters(self):
        return {"main": self.main_splitter}

    def _on_current_student_changed(self, row, *_args):
        if 0 <= row < len(self.roster.students):
            self._show_student_report(row)

    def _save_document(self, *, show_message):
        """Reject external edits instead of overwriting another import window."""
        try:
            if file_signature(self.selection_path) != self._selection_signature:
                raise ValueError("Файл выбора изменён другим окном. Закройте это окно без сохранения и повторно запустите импорт.")
            save_document(self.selection_path, self.document)
            self._selection_signature = file_signature(self.selection_path)
            self._saved_state = copy.deepcopy(self.document.to_dict())
            if show_message:
                QMessageBox.information(self, "Сохранено", str(self.selection_path))
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            return False

    def _selection_changed(self):
        self._save_document(show_message=False)

    def _completion_log_html(self):
        """Return the resource report for WindowBase to emit exactly once."""
        from html import escape
        saved = self.document.to_dict() == self._saved_state and self.selection_path.is_file()
        if not saved:
            return "<p>Выбор не сохранён: " + escape(str(self.selection_path)) + "</p>"
        tv_builder = StandardTreeBuilder(icon_size=28)
        root_node = ResourceNode(
            "Рабочая<br>папка", self.selection_path.parent, "folder", "Результаты анализа"
        )
        root_node.children.append(ResourceNode("photo_selection.json", self.selection_path, "code"))
        tv_builder.add_section("", [root_node])
        return tv_builder.get_html()

    def closeEvent(self, event):
        if self.document.to_dict() != self._saved_state:
            answer = QMessageBox.question(self, "Несохранённые изменения", "Сохранить изменения перед закрытием?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save and not self._save_document(show_message=False):
                event.ignore()
                return
        self._emit_final_log_once()
        self._save_window_state()
        event.accept()

    def _refresh_table(self):
        """Rebuild visible selection rows while keeping the current student."""
        current_row = self.table.currentRow()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.roster.students))
        for row, student in enumerate(self.roster.students):
            self.table.removeCellWidget(row, 4)
            selection = self.document.students.get(student.student_id)
            numbers = selection.selected_numbers if selection else []
            values = (
                student.student_id,
                student.display_name,
                ", ".join(numbers),
                str(len(numbers)),
                selection.source if selection else "",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column in {0, 3, 4}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 1 and pysm_icons:
                    item.setIcon(pysm_icons.get_qicon("PHOTO_PORTRAIT", 20))
                elif column == 4 and selection and selection.source:
                    source_labels = {
                        "csv": "Данные импортированы из CSV",
                        "ai_json": "Данные импортированы из AI JSON",
                        "manual": "Данные введены вручную",
                    }
                    item.setToolTip(
                        source_labels.get(selection.source, selection.source)
                    )
                    if pysm_icons:
                        icon_names = {
                            "csv": "FILE_CSV",
                            "manual": "FILE_TXT",
                            "ai_json": "FILE_CODE",
                        }
                        icon_name = icon_names.get(selection.source, "FILE_CODE")
                        item.setText("")
                        source_icon = QLabel()
                        source_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        source_icon.setStyleSheet("background: transparent;")
                        source_icon.setPixmap(
                            pysm_icons.get_qicon(icon_name, 20).pixmap(20, 20)
                        )
                        source_icon.setToolTip(item.toolTip())
                        source_icon.setAttribute(
                            Qt.WidgetAttribute.WA_TransparentForMouseEvents
                        )
                        self.table.setCellWidget(row, column, source_icon)
                self.table.setItem(row, column, item)
        if self.roster.students:
            current_row = min(max(current_row, 0), len(self.roster.students) - 1)
            self.table.setCurrentCell(current_row, 1)
            self.table.selectRow(current_row)
        self.table.blockSignals(False)
        if self.roster.students:
            self._show_student_report(current_row)

    def _on_cell_double_clicked(self, row: int, column: int):
        self.edit_selected_numbers(row)

    def edit_selected_numbers(self, row: int | None = None):
        """Validate manual input, confirm removals and persist the new selection."""
        if row is None or isinstance(row, bool):
            row = self.table.currentRow()
        if row is None or row < 0:
            QMessageBox.information(self, "Редактирование", "Сначала выберите ученика.")
            return
        student = self.roster.students[row]
        current = self.document.students.get(student.student_id)
        initial = ", ".join(current.selected_numbers) if current else ""
        dialog = SelectedNumbersDialog(student.display_name, initial, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        text = dialog.text()
        try:
            numbers = parse_manual_numbers(
                text,
                min_digits=PHOTO_NUMBER_DIGITS,
                max_digits=PHOTO_NUMBER_DIGITS,
                pad_to_digits=0,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Некорректные номера", str(exc))
            return
        responded = current.responded if current else True
        if current and numbers == current.selected_numbers:
            return
        removed = [
            number for number in (current.selected_numbers if current else [])
            if number not in numbers
        ]
        if removed:
            answer = QMessageBox.question(
                self,
                "Подтверждение удаления",
                "Будут удалены номера: " + ", ".join(removed) + ". Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.document.apply(
            student.student_id, numbers, source="manual", responded=responded
        )
        self._selection_changed()
        self._refresh_table()

    def clear_selected(self):
        """Remove selected students' saved choices after explicit confirmation."""
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if not rows and self.table.currentRow() >= 0:
            rows = [self.table.currentRow()]
        affected = [
            row for row in rows
            if self.roster.students[row].student_id in self.document.students
        ]
        if not affected:
            QMessageBox.information(
                self, "Очистка", "Выберите ученика с сохранённым выбором."
            )
            return
        names = [self.roster.students[row].display_name for row in affected]
        answer = QMessageBox.question(
            self,
            "Очистить выбор",
            "Будет полностью удалён сохранённый выбор:\n\n"
            + "\n".join(names[:15])
            + (f"\n… и ещё {len(names) - 15}" if len(names) > 15 else "")
            + "\n\nПродолжить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for row in affected:
            student_id = self.roster.students[row].student_id
            self.document.students.pop(student_id, None)
        self._selection_changed()
        self._refresh_table()

    def _apply_entries(self, entries: list[ImportEntry], source: str, unresolved=None):
        unresolved = unresolved or []
        entries = coalesce_import_entries(entries)
        if not entries:
            status = (
                "Все записи требуют ручной проверки"
                if unresolved
                else "Подходящих записей не найдено"
            )
            self._show_import_result(
                entries, unresolved, source=source, status=status
            )
            QMessageBox.information(self, "Импорт", status + ".")
            return
        preview_lines = []
        for entry in entries[:12]:
            student = self.roster.by_id[entry.student_id]
            current = self.document.students.get(entry.student_id)
            old_numbers = current.selected_numbers if current else []
            added = [number for number in entry.selected_numbers if number not in old_numbers]
            removed = [number for number in old_numbers if number not in entry.selected_numbers]
            changes = []
            if added:
                changes.append(f"+ {', '.join(added)}")
            if removed:
                changes.append(f"при замене удалить {', '.join(removed)}")
            if not entry.selected_numbers:
                changes.append("пустой ответ")
            preview_lines.append(f"{student.display_name}: {'; '.join(changes) or 'без изменений'}")
        if len(entries) > 12:
            preview_lines.append(f"… и ещё {len(entries) - 12}")
        preview = "\n".join(preview_lines)
        answer = QMessageBox.question(
            self,
            "Применить импорт",
            f"Будет обновлено учеников: {len(entries)}.\n"
            f"Не разрешено строк: {len(unresolved)}.\n\n{preview}\n\n"
            "Да — заменить выбор затронутых учеников.\nНет — добавить номера.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            self._show_import_result(
                entries, unresolved, source=source, status="Импорт отменён"
            )
            return
        mode = "replace" if answer == QMessageBox.StandardButton.Yes else "merge"
        for entry in entries:
            self.document.apply(
                entry.student_id,
                entry.selected_numbers,
                source=source,
                mode=mode,
                responded=entry.responded,
            )
        self._selection_changed()
        self._refresh_table()
        mode_label = "замена" if mode == "replace" else "добавление"
        self._show_import_result(
            entries,
            unresolved,
            source=source,
            status=f"Импорт применён, режим: {mode_label}",
        )

    def import_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Открыть CSV", "", "CSV (*.csv);;Все файлы (*)")
        if not filename:
            return
        try:
            table = read_csv_table(Path(filename))
            dialog = CsvMappingDialog(table.headers, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            identity, columns = dialog.mapping()
            entries, unresolved = import_table(
                table, self.roster, identity, columns,
                min_digits=PHOTO_NUMBER_DIGITS,
                max_digits=PHOTO_NUMBER_DIGITS,
                pad_to_digits=0,
            )
            self._apply_entries(entries, "csv", unresolved)
        except Exception as exc:
            self._show_import_result([], [], source="csv", status=f"Ошибка: {exc}")
            QMessageBox.critical(self, "Ошибка CSV", str(exc))

    def import_personal_csv(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, "Открыть персональные CSV", "", "CSV (*.csv);;Все файлы (*)")
        entries, unresolved = [], []
        try:
            for filename in filenames:
                entry = import_personal_file(
                    Path(filename), self.roster,
                    min_digits=PHOTO_NUMBER_DIGITS,
                    max_digits=PHOTO_NUMBER_DIGITS,
                    pad_to_digits=0,
                )
                if entry:
                    entries.append(entry)
                else:
                    path = Path(filename)
                    unresolved.append({
                        "source_person": path.stem,
                        "source_file": path.name,
                        "selected_numbers": read_personal_numbers(
                            path,
                            min_digits=PHOTO_NUMBER_DIGITS,
                            max_digits=PHOTO_NUMBER_DIGITS,
                            pad_to_digits=0,
                        ),
                        "reason": "ФИО из имени файла не найдено или неоднозначно",
                    })
            self._apply_entries(entries, "csv", unresolved)
        except Exception as exc:
            self._show_import_result([], [], source="csv", status=f"Ошибка: {exc}")
            QMessageBox.critical(self, "Ошибка CSV", str(exc))

    def import_ai(self):
        result = edit_ai_json_response(create_ai_import_request(self.roster), self)
        if result.accepted:
            entries, unresolved = result.value
            self._apply_entries(entries, "ai_json", unresolved)

    def save(self):
        return self._save_document(show_message=True)


def run_application(config):
    """Print the workflow heading only when actually starting the window."""
    print("<b>ОБРАБОТКА СПИСКА ВЫБРАННЫХ ФОТОГРАФИЙ</b>", flush=True)
    return run_window(ImportWindow, config)
