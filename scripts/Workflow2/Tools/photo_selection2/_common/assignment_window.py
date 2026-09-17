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
from .constants import ITEM_NUMBER_ROLE, ITEM_PATHS_ROLE
from .image_pipeline import PhotoSelectionImagePipeline
from .models import PhotoSelectionSessionState
from .roster import load_roster
from .ui_widgets import ImagePreviewLabel
from .workers import BuildRequest, OperationOutcome, PhotoSelectionOperationWorker
from .signatures import file_signature
from .assignment_views import AssignmentViewsMixin
from .export_service import ExportMixin
from .preview_service import PreviewMixin
from .report_builder import ReportMixin
from .window_base import WindowBase


class AssignmentWindow(ReportMixin, AssignmentViewsMixin, PreviewMixin, ExportMixin, WindowBase):
    """Read-only views with exactly one permitted mutating operation."""
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
        layout.addWidget(QLabel(f"Список: {self.roster.list_id} | Фотосессия: {config.photo_session}"))
        self.view_tabs = QTabWidget()
        self.photo_table = QTreeWidget()
        self._configure_tree(self.photo_table, ["Номер", "Локация", "Источник", "Распознано", "Назначено", "Файлы"], (85, 110, 95, 210, 210, 240))
        self.view_tabs.addTab(self.photo_table, "Общий список выбранных фотографий")
        self.student_location_table = QTreeWidget()
        self._configure_tree(self.student_location_table, ["ФИО/Номер", "Статус", "Локация", "Источник", "Файлы"], (235, 85, 170, 150, 270))
        self.view_tabs.addTab(self.student_location_table, "Персональные списки выбранных фотографий")
        self.view_tabs.currentChanged.connect(lambda _: self._render_assignment_summary())
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
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.view_tabs)
        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setSizes([880, 400])
        layout.addWidget(self.main_splitter, 1)
        self.assignment_buttons = QWidget()
        buttons = QHBoxLayout(self.assignment_buttons)
        for title, handler in (("Обновить список", self.refresh_async), (self.ACTION_TITLE, self.perform_operation)):
            button = QPushButton(title)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addWidget(self.assignment_buttons)
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
        except Exception as exc:
            self._operation_error(str(exc))
            return
        self.state.copy_summary = None
        self._set_busy(True)
        worker = PhotoSelectionOperationWorker(request, operation, self)
        self._worker = worker
        worker.stageChanged.connect(self._show_worker_stage)
        worker.completed.connect(self._operation_completed)
        worker.failed.connect(self._operation_error)
        worker.finished.connect(self._operation_finished)
        worker.start()

    def _assignment_context_menu(self, table, item):
        """Build a menu containing only navigation and report export."""
        menu = QMenu(table)
        if table is self.student_location_table:
            menu.addAction("Свернуть все", table.collapseAll)
            menu.addAction("Развернуть все", table.expandAll)
            menu.addAction("Сохранить как HTML", self._save_student_location_html)
            menu.addAction("Сохранить как CSV", self._save_student_location_csv)
        if item is not None:
            target = self._target_path_for_item(item)
            available = target is not None and target.is_file()
            menu.addAction("Открыть папку с файлом", lambda: self._reveal_file(target)).setEnabled(available)
            menu.addAction("Открыть файл", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))).setEnabled(available)
        return menu

    def _show_assignment_context_menu(self, table, position):
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
        self.student_location_table.clear()
        self.preview.show_message("Предпросмотр недоступен из-за ошибки проверки")
        self.report.setHtml(
            self._message_header_html()
            + f"<h3 style='color:#b00020'>Ошибка</h3><p>{escape(message)}</p>"
        )
        QMessageBox.critical(self, "Обработка фотографий", message)

    def _operation_completed(self, outcome: OperationOutcome) -> None:
        if outcome.selection_signature != file_signature(self.selection_path):
            self.state.assignments_dirty = True
            self._refresh_pending = True
            self.report.setHtml(
                self._message_header_html()
                + "<h3>Выбор изменён</h3>"
                + "<p>Устаревший результат фоновой операции отброшен. "
                + "После завершения будет выполнен новый пересчёт.</p>"
            )
            return
        self.state.build_result = outcome.result
        self._preview_by_stem = dict(outcome.preview_by_stem or {})
        if outcome.copy_summary is not None:
            self.state.copy_summary = outcome.copy_summary
        if outcome.assignment_saved:
            self.state.assignments_dirty = False
        elif outcome.operation == "refresh" and self.OPERATION == "build":
            self.state.assignments_dirty = not self._assignment_file_matches_result(
                outcome.result
            )
        elif outcome.operation == "copy":
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
