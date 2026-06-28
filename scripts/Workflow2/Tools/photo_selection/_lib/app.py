"""GUI for collecting per-student selected photo numbers."""

from __future__ import annotations

import argparse
import copy
from html import escape
import logging
from pathlib import Path
import sys

from .assignment_core import normalize_exclude_dirs
from .constants import (
    PHOTO_NUMBER_DIGITS, WINDOW_STATE_VAR, ITEM_NUMBER_ROLE, ITEM_PATHS_ROLE,
    ITEM_STUDENT_ROLE,
)
from .csv_import import (
    import_personal_file, import_table, read_csv_table, read_personal_numbers,
)
from .domain import ImportEntry
from .models import PhotoSelectionSessionState
from .number_parser import parse_manual_numbers
from .roster import load_roster
from .storage import load_document, save_document
from .ui_widgets import AnswerCheckBox, AiDialog, CsvMappingDialog, ImagePreviewLabel, SelectedNumbersDialog
from .workers import BuildRequest, Operation, OperationOutcome, PhotoSelectionOperationWorker
from .assignment_views import AssignmentViewsMixin
from .export_service import ExportMixin
from .preview_service import PreviewMixin
from .report_builder import ReportMixin

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_icons import icons as pysm_icons
    from pysm_lib.pysm_report_api import DashboardBuilder, ResourceNode
    from pysm_lib.window_state_manager import WindowStateManager
    IS_MANAGED_RUN = True
except ImportError:
    ConfigResolver = None
    pysm_context = None
    pysm_icons = None
    theme_api = None
    DashboardBuilder = None
    ResourceNode = None
    WindowStateManager = None
    IS_MANAGED_RUN = False

from PySide6.QtCore import QProcess, QSize, Qt, QTimer, QUrl, QUrlQuery
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QDialog, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton,
    QMenu, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)


logger = logging.getLogger(__name__)
class PhotoSelectionWindow(ReportMixin, AssignmentViewsMixin, PreviewMixin, ExportMixin, QMainWindow):
    HEADERS = ("student_id", "Фамилия Имя", "Выбранные номера", "Количество", "Ответ", "Источник")

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config
        self.roster = load_roster(Path(config.student_list_file))
        self.selection_path = Path(config.analysis_dir) / "photo_selection.json"
        self.assignment_path = Path(config.analysis_dir) / "photo_assignments.json"
        self.document = load_document(
            self.selection_path, self.roster, config.session_name, config.photo_session
        )
        self._saved_state = copy.deepcopy(self.document.to_dict())
        self.state = PhotoSelectionSessionState()
        self._worker: PhotoSelectionOperationWorker | None = None
        self._base_report_html = ""
        self._preview_by_stem: dict[str, Path] = {}
        self._final_log_emitted = False
        self.user_message_html = str(getattr(config, "message", "") or "").strip()
        self.setWindowTitle(
            str(getattr(config, "title", "") or f"Выбор фотографий — {config.photo_session}")
        )
        self.resize(1280, 780)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        session_label = QLabel(
            f"Список: {self.roster.list_id} | Фотосессия: {config.photo_session}"
        )
        session_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(session_label)
        self.view_tabs = QTabWidget()
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setIconSize(QSize(20, 20))
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.currentCellChanged.connect(self._on_current_student_changed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.view_tabs.addTab(self.table, "Импорт")

        self.photo_table = QTreeWidget()
        self._configure_tree(
            self.photo_table,
            ["Номер", "Локация", "Источник", "Распознано", "Назначено", "Файлы"],
            (85, 110, 95, 210, 210, 240),
        )
        self.view_tabs.addTab(self.photo_table, "Общий список выбранных фотографий")

        self.student_location_table = QTreeWidget()
        self._configure_tree(
            self.student_location_table,
            ["ФИО/Номер", "Статус", "Локация", "Источник", "Файлы"],
            (235, 85, 170, 150, 270),
        )
        self.view_tabs.addTab(self.student_location_table, "Персональные списки выбранных фотографий")
        self.view_tabs.currentChanged.connect(self._on_active_tab_changed)

        self.import_result = QTextBrowser()
        self.report = self.import_result
        self.import_result.setObjectName("importResult")
        self.import_result.setOpenExternalLinks(False)
        self.import_result.setOpenLinks(False)
        self.import_result.anchorClicked.connect(self._open_report_link)
        self.import_result.setPlaceholderText(
            "Выберите ученика для просмотра подробной информации."
        )
        self.import_result.setMinimumWidth(320)
        self.preview = ImagePreviewLabel()
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.addWidget(self.import_result)
        self.right_splitter.addWidget(self.preview)
        self.right_splitter.setSizes([390, 300])
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.view_tabs)
        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([880, 400])
        layout.addWidget(self.main_splitter, 1)
        self.import_buttons = QWidget()
        buttons = QHBoxLayout(self.import_buttons)
        buttons.setContentsMargins(0, 0, 0, 0)
        import_button = QPushButton("Импорт")
        import_menu = QMenu(import_button)
        import_menu.addAction("Общая CSV", self.import_csv)
        import_menu.addAction("Персональные CSV", self.import_personal_csv)
        import_menu.addAction("AI-промпт / JSON", self.import_ai)
        import_button.setMenu(import_menu)
        buttons.addWidget(import_button)
        for title, handler in (
            ("Изменить выбранные номера", self.edit_selected_numbers),
            ("Очистить", self.clear_selected),
            ("Сохранить", self.save),
        ):
            button = QPushButton(title)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addWidget(self.import_buttons)

        self.assignment_buttons = QWidget()
        assignment_buttons = QHBoxLayout(self.assignment_buttons)
        assignment_buttons.setContentsMargins(0, 0, 0, 0)
        for title, handler in (
            ("Обновить список", self.refresh_async),
            ("Копировать выбранные файлы", self.copy_files),
            ("Создать список", self.build_assignments_file),
            ("Копировать и создать список", self.copy_and_build),
        ):
            button = QPushButton(title)
            button.clicked.connect(handler)
            assignment_buttons.addWidget(button)
        assignment_buttons.addStretch()
        layout.addWidget(self.assignment_buttons)
        self._operation_buttons = tuple(
            self.assignment_buttons.findChildren(QPushButton)
        )
        self.assignment_buttons.hide()
        self._refresh_table()
        self._restore_window_state()
        if self._can_refresh_assignments():
            QTimer.singleShot(0, self.refresh_async)

    def _configure_tree(
        self,
        table: QTreeWidget,
        headers: list[str],
        widths: tuple[int, ...],
    ) -> None:
        table.setColumnCount(len(headers))
        table.setHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setUniformRowHeights(True)
        table.setRootIsDecorated(True)
        table.setIconSize(QSize(38, 20))
        table.setSortingEnabled(True)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(
            lambda position, source=table: self._show_assignment_context_menu(
                source, position
            )
        )
        table.currentItemChanged.connect(self._on_assignment_item_changed)
        header = table.header()
        header.setSectionsMovable(False)
        header.setSortIndicatorShown(True)
        for column in range(table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        for column, width in enumerate(widths):
            table.setColumnWidth(column, width)

    def _splitters(self) -> dict[str, QSplitter]:
        return {"main": self.main_splitter, "right": self.right_splitter}

    def _on_active_tab_changed(self, index: int) -> None:
        import_mode = index == 0
        self.import_buttons.setVisible(import_mode)
        self.assignment_buttons.setVisible(not import_mode)
        if import_mode and self.table.currentRow() >= 0:
            self._show_student_report(self.table.currentRow())
        elif self.state.build_result is not None:
            self._render_assignment_summary()

    def _restore_window_state(self) -> None:
        if not (IS_MANAGED_RUN and pysm_context and WindowStateManager):
            return
        try:
            saved_state = pysm_context.get_structured(WINDOW_STATE_VAR, {})
            if isinstance(saved_state, dict) and saved_state:
                WindowStateManager.restore_state(
                    window=self,
                    state_data=saved_state,
                    splitters=self._splitters(),
                )
                active_tab = saved_state.get("active_tab")
                if isinstance(active_tab, int) and 0 <= active_tab < self.view_tabs.count():
                    self.view_tabs.setCurrentIndex(active_tab)
        except Exception:
            logger.warning("Не удалось восстановить состояние окна", exc_info=True)

    def _save_window_state(self) -> None:
        if not (IS_MANAGED_RUN and pysm_context and WindowStateManager):
            return
        try:
            state = WindowStateManager.save_state(
                window=self,
                splitters=self._splitters(),
            )
            state["active_tab"] = self.view_tabs.currentIndex()
            pysm_context.set_structured(WINDOW_STATE_VAR, state)
        except Exception:
            logger.warning("Не удалось сохранить состояние окна", exc_info=True)

    def _open_report_link(self, url: QUrl) -> None:
        """Open local report links externally instead of navigating in QTextBrowser."""
        action = ""
        if url.scheme() == "pysm":
            action = url.host()
            path = Path(QUrlQuery(url).queryItemValue("path"))
        else:
            path = Path(url.toLocalFile())
        if not path.exists():
            QMessageBox.information(
                self,
                "Файл ещё не создан",
                f"Объект пока не существует:\n{path}",
            )
            return
        if action == "reveal-file":
            self._reveal_file(path)
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _reveal_file(path: Path) -> None:
        """Open the containing folder and select the file in Windows Explorer."""
        if sys.platform == "win32":
            QProcess.startDetached("explorer.exe", ["/select,", str(path.resolve())])
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    @staticmethod
    def _report_action_url(path: Path, action: str) -> QUrl:
        url = QUrl()
        url.setScheme("pysm")
        url.setHost(action)
        query = QUrlQuery()
        query.addQueryItem("path", str(path.resolve()))
        url.setQuery(query)
        return url

    @classmethod
    def _report_action_link(
        cls,
        path: Path,
        action: str,
        content_html: str,
    ) -> str:
        url = cls._report_action_url(path, action).toString()
        return (
            f'<a href="{escape(url, quote=True)}" '
            f'style="text-decoration:none">{content_html}</a>'
        )

    def _show_assignment_context_menu(self, table: QTreeWidget, position) -> None:
        item = table.itemAt(position)
        if item is None and table is not self.student_location_table:
            return
        menu = QMenu(table)
        edit_numbers = clear_student = None
        collapse_all = expand_all = export_html = export_csv = None
        if table is self.student_location_table:
            collapse_all = menu.addAction("Свернуть все")
            expand_all = menu.addAction("Развернуть все")
            export_html = menu.addAction("Сохранить как HTML")
            export_csv = menu.addAction("Сохранить как CSV")
            if item is not None:
                menu.addSeparator()
                edit_numbers = menu.addAction("Изменить выбранные номера")
                clear_student = menu.addAction("Очистить выбор ученика")

        target = self._target_path_for_item(item) if item is not None else None
        open_folder = open_file = None
        if item is not None:
            open_folder = menu.addAction("Открыть папку с файлом")
            open_file = menu.addAction("Открыть файл")
        available = target is not None and target.is_file()
        if open_folder is not None and open_file is not None:
            open_folder.setEnabled(available)
            open_file.setEnabled(available)
        selected = menu.exec(table.viewport().mapToGlobal(position))
        if selected is collapse_all:
            table.collapseAll()
        elif selected is expand_all:
            table.expandAll()
        elif selected is export_html:
            self._save_student_location_html()
        elif selected is export_csv:
            self._save_student_location_csv()
        elif selected is edit_numbers:
            self._edit_selection_from_assignment_item(item)
        elif selected is clear_student:
            self._clear_selection_from_assignment_item(item)
        elif not available:
            return
        elif selected is open_folder:
            self._reveal_file(target)
        elif selected is open_file:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _refresh_table(self):
        current_row = self.table.currentRow()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.roster.students))
        for row, student in enumerate(self.roster.students):
            selection = self.document.students.get(student.student_id)
            numbers = selection.selected_numbers if selection else []
            values = (
                student.student_id,
                student.display_name,
                ", ".join(numbers),
                str(len(numbers)),
                "Да" if selection and selection.responded else "Нет",
                selection.source if selection else "",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column in {0, 3, 4, 5}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 4:
                    item.setText("")
                elif column == 1 and pysm_icons:
                    item.setIcon(pysm_icons.get_qicon("PHOTO_PORTRAIT", 20))
                elif column == 5 and selection and selection.source:
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
            checkbox = AnswerCheckBox()
            checkbox.setToolTip("Ответ ученика получен")
            checkbox.setChecked(bool(selection and selection.responded))
            checkbox.toggled.connect(
                lambda checked, student_id=student.student_id:
                self._on_answer_toggled(student_id, checked)
            )
            checkbox.rowDoubleClicked.connect(
                lambda row=row: self.edit_selected_numbers(row)
            )
            checkbox_container = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_container)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.addWidget(checkbox)
            self.table.setCellWidget(row, 4, checkbox_container)
        if self.roster.students:
            current_row = min(max(current_row, 0), len(self.roster.students) - 1)
            self.table.setCurrentCell(current_row, 1)
            self.table.selectRow(current_row)
        self.table.blockSignals(False)
        if self.roster.students:
            self._show_student_report(current_row)
            self._show_import_student_preview(current_row)

    def _on_current_student_changed(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if 0 <= current_row < len(self.roster.students):
            self._show_student_report(current_row)
            self._show_import_student_preview(current_row)
        else:
            self.preview.show_message("Выберите ученика для предпросмотра JPG")


    def _on_answer_toggled(self, student_id: str, responded: bool):
        current = self.document.students.get(student_id)
        if current is None and not responded:
            return
        numbers = current.selected_numbers if current else []
        self.document.apply(
            student_id, numbers, source="manual", responded=responded
        )
        self._selection_changed()
        self._refresh_table()

    def _on_cell_double_clicked(self, row: int, column: int):
        self.edit_selected_numbers(row)

    def edit_selected_numbers(self, row: int | None = None):
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
        if not entries and not unresolved:
            self._show_import_result(
                entries, unresolved, source=source, status="Подходящих записей не найдено"
            )
            QMessageBox.information(self, "Импорт", "Подходящих записей не найдено.")
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

    def _exclude_dirs(self) -> list[str]:
        return normalize_exclude_dirs(getattr(self.config, "exclude_dirs", "Masks"))

    def _operation_request(self) -> BuildRequest:
        return BuildRequest(
            student_list_file=Path(self.config.student_list_file),
            analysis_dir=Path(self.config.analysis_dir),
            source_dir=Path(self.config.source_dir),
            dest_dir=Path(self.config.dest_dir),
            exclude_dirs=tuple(self._exclude_dirs()),
            assignment_path=self.assignment_path,
        )

    def _set_busy(self, busy: bool) -> None:
        for button in getattr(self, "_operation_buttons", ()):
            button.setEnabled(not busy)

    def _start_operation(self, operation: Operation) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._set_busy(True)
        worker = PhotoSelectionOperationWorker(
            self._operation_request(), operation, self
        )
        self._worker = worker
        worker.stageChanged.connect(self._show_worker_stage)
        worker.completed.connect(self._operation_completed)
        worker.failed.connect(self._operation_error)
        worker.finished.connect(self._operation_finished)
        worker.start()

    def _show_worker_stage(self, message: str) -> None:
        self.report.setHtml(
            self._message_header_html()
            + f"<h3>Выполнение операции</h3><p>{escape(message)}</p>"
        )

    def _operation_error(self, message: str) -> None:
        self.state.build_result = None
        self.photo_table.clear()
        self.student_location_table.clear()
        self.preview.show_message("Предпросмотр недоступен из-за ошибки проверки")
        self.report.setHtml(
            self._message_header_html()
            + f"<h3 style='color:#b00020'>Ошибка</h3><p>{escape(message)}</p>"
        )
        QMessageBox.critical(self, "Обработка фотографий", message)

    def _operation_completed(self, outcome: OperationOutcome) -> None:
        self.state.build_result = outcome.result
        if outcome.copy_summary is not None:
            self.state.copy_summary = outcome.copy_summary
        if outcome.assignment_saved:
            self.state.assignments_dirty = False
        elif outcome.operation in {"copy", "copy_and_build"}:
            self.state.assignments_dirty = True
        self._render_assignment_views()
        if outcome.operation == "refresh":
            return
        if outcome.result.has_errors:
            QMessageBox.warning(
                self,
                "Обработка фотографий",
                "Исправьте блокирующие ошибки перед выполнением операции.",
            )
            return
        if outcome.copy_summary is not None:
            summary = outcome.copy_summary
            message = f"Скопировано: {summary.copied}. Пропущено: {summary.skipped}."
        else:
            message = "Список назначений сформирован."
        if outcome.assignment_saved:
            message += f"\nФайл создан:\n{self.assignment_path}"
        QMessageBox.information(self, "Обработка фотографий", message)

    def _operation_finished(self) -> None:
        worker = self._worker
        self._worker = None
        self._set_busy(False)
        if worker is not None:
            worker.deleteLater()

    def refresh_async(self) -> None:
        self._start_operation("refresh")

    def copy_files(self) -> None:
        self._start_operation("copy")

    def build_assignments_file(self) -> None:
        self._start_operation("build")

    def copy_and_build(self) -> None:
        self._start_operation("copy_and_build")






















    def _target_path_for_item(self, item: QTreeWidgetItem | None) -> Path | None:
        if item is None:
            return None
        number = str(item.data(0, ITEM_NUMBER_ROLE) or "")
        if number:
            return self._find_preview_jpg(number)
        raw_paths = item.data(0, ITEM_PATHS_ROLE) or []
        paths = [Path(value) for value in raw_paths]
        destination = Path(self.config.dest_dir)
        for path in paths:
            try:
                path.relative_to(destination)
            except ValueError:
                continue
            return path
        return paths[0] if paths else None

    def _student_id_from_assignment_item(self, item: QTreeWidgetItem | None) -> str | None:
        while item is not None:
            student_id = str(item.data(0, ITEM_STUDENT_ROLE) or "")
            if student_id:
                return student_id
            item = item.parent()
        return None

    def _edit_selection_from_assignment_item(self, item: QTreeWidgetItem | None) -> None:
        student_id = self._student_id_from_assignment_item(item)
        if not student_id:
            QMessageBox.information(self, "Редактирование", "Выберите ученика.")
            return
        row = next(
            (
                index for index, student in enumerate(self.roster.students)
                if student.student_id == student_id
            ),
            -1,
        )
        if row >= 0:
            self.view_tabs.setCurrentIndex(0)
            self.table.selectRow(row)
            self.edit_selected_numbers(row)

    def _clear_selection_from_assignment_item(self, item: QTreeWidgetItem | None) -> None:
        student_id = self._student_id_from_assignment_item(item)
        if not student_id or student_id not in self.document.students:
            return
        student = self.roster.by_id[student_id]
        answer = QMessageBox.question(
            self,
            "Очистить выбор",
            f"Удалить выбор ученика {student.display_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.document.students.pop(student_id, None)
        self._selection_changed()
        self._refresh_table()





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
        dialog = AiDialog(self.roster, self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_entries(dialog.entries, "ai_json", dialog.unresolved)

    def _save_document(self, *, show_message: bool) -> bool:
        try:
            save_document(self.selection_path, self.document)
            self._saved_state = copy.deepcopy(self.document.to_dict())
            self.state.selection_dirty = False
            self.state.assignments_dirty = True
            if show_message:
                QMessageBox.information(self, "Сохранено", str(self.selection_path))
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            return False

    def save(self):
        return self._save_document(show_message=True)

    def _selection_changed(self) -> None:
        """Persist changed selections and refresh assignment views."""
        self.state.selection_dirty = True
        self.state.assignments_dirty = True
        if self._save_document(show_message=False) and self._can_refresh_assignments():
            self.refresh_async()

    def _can_refresh_assignments(self) -> bool:
        """Return True when enough analysis inputs exist for assignment refresh."""
        analysis_dir = Path(self.config.analysis_dir)
        return (
            (analysis_dir / "info_faces.json").is_file()
            and self.selection_path.is_file()
        )

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Операция выполняется",
                "Дождитесь завершения сканирования или копирования файлов.",
            )
            event.ignore()
            return
        if self.document.to_dict() == self._saved_state:
            if self.state.assignments_dirty:
                answer = QMessageBox.question(
                    self,
                    "Список назначений неактуален",
                    "Выбор фотографий изменён, но photo_assignments.json не пересоздан.\n"
                    "Перейти к верстке пока нельзя.\n\nЗакрыть окно?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            self._emit_final_log_once()
            self._save_window_state()
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "Несохранённые изменения",
            "Сохранить изменения перед закрытием?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            if self._save_document(show_message=False):
                self._emit_final_log_once()
                self._save_window_state()
                event.accept()
            else:
                event.ignore()
        elif answer == QMessageBox.StandardButton.Discard:
            self._emit_final_log_once()
            self._save_window_state()
            event.accept()
        else:
            event.ignore()


def run_application(config: argparse.Namespace) -> int:
    """Create and run the GUI for an already resolved configuration."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        app = QApplication.instance() or QApplication(sys.argv)
        if IS_MANAGED_RUN and theme_api:
            theme_api.apply_theme_to_app(app)
        window = PhotoSelectionWindow(config)
        window.show()
        return app.exec()
    except Exception as exc:
        logger.exception("Ошибка photo_selection: %s", exc)
        return 1
