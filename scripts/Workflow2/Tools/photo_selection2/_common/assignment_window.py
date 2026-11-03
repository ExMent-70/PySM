"""Two read-only photo views for reviewing and publishing assignments."""
from __future__ import annotations

from html import escape
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QMenu, QSplitter, QTabWidget, QTextBrowser, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from .assignment_core import normalize_exclude_dirs
from .constants import ITEM_NUMBER_ROLE, ITEM_PATHS_ROLE, ITEM_STUDENT_ROLE, ITEM_LOCATION_ROLE
from .image_pipeline import PhotoSelectionImagePipeline
from .models import PhotoSelectionSessionState
from .roster import load_roster
from .ui_widgets import ImagePreviewLabel
from .workers import BuildRequest, OperationOutcome, PhotoSelectionOperationWorker
from .signatures import file_signature
from .destination_inventory import analysis_version
from .assignment_views import AssignmentViewsMixin
from .student_location_table import StudentLocationTable
from pysm_lib.pysm_icons import icons
from .export_service import ExportMixin
from .preview_service import PreviewMixin
from .report_builder import ReportMixin
from .window_base import WindowBase


class AssignmentWindow(ReportMixin, AssignmentViewsMixin, PreviewMixin, ExportMixin, WindowBase):
    """Review logical locations and publish assignments without moving files."""
    OPERATION = "build"
    ACTION_TITLE = "Создать план вёрстки"
    WINDOW_STATE_VAR = "win_state.photo_selection04_assignments"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.roster = load_roster(Path(config.student_list_file))
        self.selection_path = Path(config.analysis_dir) / "photo_selection.json"
        self.assignment_path = Path(config.analysis_dir) / "photo_assignments.json"
        self.state = PhotoSelectionSessionState()
        self._worker = None
        self._refresh_pending = False
        self._base_report_html = ""
        self._preview_by_stem = {}
        self._final_log_emitted = False
        self._close_after_image_shutdown = False
        self.user_message_html = str(config.message or "")
        self._image_pipeline = PhotoSelectionImagePipeline(Path(config.analysis_dir), parent=self)
        self._image_pipeline.shutdownFinished.connect(self._finish_deferred_close)
        self.setWindowTitle(config.title)
        self.resize(1280, 780)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel(f"Список: {self.roster.list_id} | Фотосессия: {config.photo_session}"))
        self.view_tabs = QTabWidget()
        self.photo_table = QTreeWidget()
        self._configure_tree(self.photo_table, ["Номер", "Локация", "Источник", "Распознано", "Назначено", "Файлы"], (85, 110, 95, 210, 210, 240))
        self.view_tabs.addTab(self.photo_table, "Общий список выбранных фотографий")
        self.student_location_table = StudentLocationTable()
        self.student_location_table.photoActivated.connect(self._on_personal_selection)
        self.student_location_table.customContextMenuRequested.connect(
            lambda position: self._show_assignment_context_menu(self.student_location_table, position)
        )
        self.view_tabs.addTab(self.student_location_table, "Персональные списки выбранных фотографий")
        self.view_tabs.currentChanged.connect(self._on_assignment_tab_changed)
        left_layout.addWidget(self.view_tabs, 1)
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(False)
        self.report.setOpenLinks(False)
        self.report.anchorClicked.connect(self._open_report_link)
        self.report.setMinimumWidth(320)
        self.preview = ImagePreviewLabel(self._image_pipeline.loader)
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.addWidget(self.report)
        self.right_splitter.addWidget(self.preview)
        self.right_splitter.setSizes([390, 300])
        right_layout.addWidget(self.right_splitter, 1)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([880, 400])
        layout.addWidget(self.main_splitter, 1)
        self.personal_legend = QLabel(
            f"Выбор: {icons.PHOTO_PORTRAIT(size=16)} клиент &nbsp; "
            f"{icons.CAMERA(size=16)} фотограф &nbsp;&nbsp; "
            f"Файл: {icons.FOLDER(size=16)} исходная папка &nbsp; "
            f"{icons.FOLDER_VIRTUAL(size=16)} целевая папка &nbsp; "
            f"{icons.WARNING(size=16)} файл не найден &nbsp; — выбор отсутствует"
        )
        self.personal_legend.setWordWrap(True)
        self.personal_legend.setVisible(False)
        left_layout.addWidget(self.personal_legend)
        self.assignment_buttons = QWidget()
        buttons = QHBoxLayout(self.assignment_buttons)
        buttons.setContentsMargins(0, 0, 0, 0)
        for title, handler, icon_name in (
            ("Обновить список", self.refresh_async, "REFRESH"),
            (self.ACTION_TITLE, self.perform_operation, "SAVE"),
        ):
            button = QPushButton(title)
            button.setIcon(icons.get_qicon(icon_name, 20))
            button.setIconSize(QSize(20, 20))
            button.clicked.connect(handler)
            buttons.addWidget(button)
        right_layout.addWidget(self.assignment_buttons)
        self._restore_window_state()
        QTimer.singleShot(0, self.refresh_async)

    def _set_busy(self, busy):
        self.view_tabs.setEnabled(not busy)
        self.assignment_buttons.setEnabled(not busy)

    def perform_operation(self):
        self._start_operation(self.OPERATION)

    def _start_operation(self, operation):
        """Reload inputs and prevent calls to another stage's operation."""
        if operation not in {"refresh", self.OPERATION}:
            raise ValueError("Эта операция недоступна в данном скрипте.")
        if self._worker is not None and self._worker.isRunning():
            if operation == "refresh":
                self._refresh_pending = True
            return
        try:
            self.roster = load_roster(Path(self.config.student_list_file))
            request = self._operation_request()
            worker = PhotoSelectionOperationWorker(request, operation, self)
        except Exception as exc:
            self._operation_error(str(exc))
            return
        self.state.copy_summary = None
        self._set_busy(True)
        self._worker = worker
        worker.stageChanged.connect(self._show_worker_stage)
        worker.completed.connect(self._operation_completed)
        worker.failed.connect(self._operation_error)
        worker.finished.connect(self._operation_finished)
        worker.start()

    def _assignment_context_menu(self, table, item):
        """Build a menu containing only navigation and report export."""
        # A table parent can propagate its item-view palette to popup menus.
        menu = QMenu(self)
        if table is self.student_location_table:
            menu.addAction("Сохранить как HTML", self._save_student_location_html)
            menu.addAction("Сохранить как CSV", self._save_student_location_csv)
            if item is not None:
                student = item.data(ITEM_STUDENT_ROLE)
                location = item.data(ITEM_LOCATION_ROLE)
                menu.addAction("Сведения об ученике", lambda: self._show_assignment_student_report(student))
                if location:
                    menu.addAction("Сведения о локации", lambda: self._show_assignment_student_report(student, location=location))
            entry = table.selected_photo() if item is not None else None
            if entry is not None:
                paths = sorted(entry.paths, key=lambda path: (
                    not path.is_relative_to(Path(self.config.dest_dir)), str(path).casefold()))
                target = paths[0] if paths else self._find_preview_jpg(entry.number)
                self._add_file_actions(menu, target)
                if paths:
                    files_menu = menu.addMenu("Все файлы кадра")
                    # Keep Python wrappers alive for the lifetime of the popup.
                    menu._physical_file_menus = [files_menu, files_menu.menuAction()]
                    for path in paths:
                        file_menu = files_menu.addMenu(str(path))
                        menu._physical_file_menus.extend((file_menu, file_menu.menuAction()))
                        self._add_file_actions(file_menu, path)
            return menu
        if item is not None:
            target = self._target_path_for_item(item)
            available = target is not None and target.is_file()
            menu.addAction("Открыть папку с файлом", lambda: self._reveal_file(target)).setEnabled(available)
            menu.addAction("Открыть файл", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))).setEnabled(available)
        return menu

    def _add_file_actions(self, menu, target):
        """Keep every physical variant accessible from a compact matrix cell."""
        available = target is not None and target.is_file()
        menu.addAction("Открыть папку с файлом", lambda: self._reveal_file(target)).setEnabled(available)
        menu.addAction("Открыть файл", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))).setEnabled(available)

    def _show_assignment_context_menu(self, table, position):
        if table is self.student_location_table:
            table.select_photo_at(position)
        menu = self._assignment_context_menu(table, table.itemAt(position))
        if menu.actions():
            menu.exec(table.viewport().mapToGlobal(position))
        menu.deleteLater()

    def closeEvent(self, event):
        if self._close_after_image_shutdown:
            event.accept() if self._image_pipeline.is_closed else event.ignore()
            return
        if self._worker is not None:
            QMessageBox.information(self, "Операция выполняется", "Дождитесь завершения операции.")
            event.ignore()
            return
        self._accept_close(event)

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
            selection_signature=file_signature(self.selection_path),
            session_name=self.config.session_name,
            photo_session=self.config.photo_session,
        )

    def _show_worker_stage(self, message: str) -> None:
        self.report.setHtml(
            self._message_header_html()
            + f"<h3>Выполнение операции</h3><p>{escape(message)}</p>"
        )

    def _operation_error(self, message: str) -> None:
        self.state.build_result = None
        self.photo_table.clear()
        self.student_location_table.setRowCount(0)
        self.preview.show_message("Предпросмотр недоступен из-за ошибки проверки")
        self.report.setHtml(
            self._message_header_html()
            + f"<h3 style='color:#b00020'>Ошибка</h3><p>{escape(message)}</p>"
        )
        QMessageBox.critical(self, "Обработка фотографий", message)

    def _operation_completed(self, outcome: OperationOutcome) -> None:
        try:
            analysis_current = outcome.analysis_signature == analysis_version(
                Path(self.config.analysis_dir) / "info_faces.json")
        except OSError:
            analysis_current = False
        if outcome.selection_signature != file_signature(self.selection_path) or not analysis_current:
            self.state.assignments_dirty = True
            self._refresh_pending = True
            self.report.setHtml(
                self._message_header_html()
                + "<h3>Исходные данные изменены</h3>"
                + "<p>Устаревший результат фоновой операции отброшен. "
                + "После завершения будет выполнен новый пересчёт.</p>"
            )
            return
        selected = (self._selection_key(self.photo_table.currentItem()),
                    self.student_location_table.selection_key())
        self.state.build_result = outcome.result
        self._preview_by_stem = dict(outcome.preview_by_stem or {})
        if outcome.copy_summary is not None:
            self.state.copy_summary = outcome.copy_summary
        if outcome.assignment_saved:
            self.state.assignments_dirty = False
        elif outcome.operation == "refresh" and self.OPERATION == "build":
            self.state.assignments_dirty = self.state.assignments_dirty or not self._assignment_file_matches_result(
                outcome.result
            )
        elif outcome.operation == "copy":
            self.state.assignments_dirty = True
        self.preview.show_message("Выберите фотографию для предпросмотра")
        self._render_assignment_views()
        self._restore_selection(self.photo_table, selected[0])
        self.student_location_table.restore_selection(selected[1])
        self._on_assignment_tab_changed()
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

    def _on_assignment_tab_changed(self, *_):
        """Refresh the selected cell's details when switching between views."""
        personal = self.view_tabs.currentWidget() is self.student_location_table
        self.personal_legend.setVisible(personal)
        if personal:
            self._on_personal_selection()
        elif self.photo_table.currentItem() is not None:
            self._on_assignment_item_changed(self.photo_table.currentItem(), None)
        else:
            self._render_assignment_summary()

    def _on_personal_selection(self):
        """Use existing report and preview services for a matrix cell or photo line."""
        table = self.student_location_table
        item = table.currentItem()
        result = self.state.build_result
        if item is None or result is None:
            self._render_assignment_summary()
            return
        number = table.selected_number()
        if number and number in result.records:
            self._show_photo_report(result.records[number])
            self._show_preview_for_number(number)
            return
        student_id = item.data(ITEM_STUDENT_ROLE)
        location = item.data(ITEM_LOCATION_ROLE)
        self._show_assignment_student_report(student_id, location=location)
        number = self._preferred_preview_number_for_student(student_id, location)
        if number:
            self._show_preview_for_number(number)
        else:
            self.preview.show_message("Для выбранной ячейки нет фотографии")

    @staticmethod
    def _selection_key(item):
        """Identify a selected row independently of row order and renamed locations."""
        if item is None:
            return None
        paths = tuple(str(path) for path in (item.data(0, ITEM_PATHS_ROLE) or []))
        number = str(item.data(0, ITEM_NUMBER_ROLE) or "")
        student = ""
        ancestor = item
        while ancestor is not None:
            student = student or str(ancestor.data(0, ITEM_STUDENT_ROLE) or "")
            number = number or str(ancestor.data(0, ITEM_NUMBER_ROLE) or "")
            ancestor = ancestor.parent()
        location = str(item.data(0, ITEM_LOCATION_ROLE) or "") if not number else ""
        return student, number, location, paths

    def _restore_selection(self, table, key):
        """Restore a surviving row without refreshing the inactive tab's report."""
        if key is None:
            return
        pending = [table.topLevelItem(i) for i in range(table.topLevelItemCount())]
        while pending:
            item = pending.pop()
            if self._selection_key(item) == key:
                previous = table.blockSignals(True)
                try:
                    table.setCurrentItem(item)
                    parent = item.parent()
                    while parent is not None:
                        parent.setExpanded(True)
                        parent = parent.parent()
                finally:
                    table.blockSignals(previous)
                return
            pending.extend(item.child(i) for i in range(item.childCount()))

    def _operation_finished(self) -> None:
        worker = self._worker
        self._worker = None
        self._set_busy(False)
        if worker is not None:
            worker.deleteLater()
        if self._refresh_pending:
            self._refresh_pending = False
            QTimer.singleShot(0, self.refresh_async)

    def refresh_async(self) -> None:
        self._start_operation("refresh")

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

    def _accept_close(self, event) -> None:
        """Finalize UI state and close after image workers have retired."""

        self._emit_final_log_once()
        self._save_window_state()
        self.preview.cancel_requests()
        if self._image_pipeline.shutdown():
            event.accept()
            return
        self._close_after_image_shutdown = True
        event.ignore()

    def _finish_deferred_close(self) -> None:
        if self._close_after_image_shutdown:
            self.close()
