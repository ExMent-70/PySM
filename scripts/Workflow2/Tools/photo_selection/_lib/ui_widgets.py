"""Reusable Qt widgets and dialogs for the photo selection workflow."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QTextOption
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from .ai_import import build_prompt, extract_json_object, load_prompt_template, validate_ai_response
from .constants import PHOTO_NUMBER_DIGITS
from .csv_import import suggest_columns
from .domain import ImportEntry
from .roster import StudentRoster


class AnswerCheckBox(QCheckBox):
    """Theme-aware answer checkbox that also forwards row double-clicks."""

    rowDoubleClicked = Signal()

    def mouseDoubleClickEvent(self, event):
        self.rowDoubleClicked.emit()
        event.accept()


class ImagePreviewLabel(QLabel):
    """Theme-friendly image preview that keeps the original aspect ratio."""

    CACHE_LIMIT = 32

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path: Path | None = None
        self._source_pixmap = QPixmap()
        self._pixmap_cache: OrderedDict[tuple[str, int, int], QPixmap] = OrderedDict()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(220)
        self.setWordWrap(True)
        self.setText("Выберите строку фотографии для предпросмотра JPG")

    def show_image(self, path: Path) -> None:
        pixmap = self._load_preview_pixmap(path)
        if pixmap.isNull():
            self.show_message(f"Не удалось открыть JPG:\n{path.name}")
            return
        self.image_path = path
        self._source_pixmap = pixmap
        self._update_pixmap()
        self.setToolTip(str(path))

    def show_message(self, text: str) -> None:
        self.image_path = None
        self._source_pixmap = QPixmap()
        self.clear()
        self.setText(text)
        self.setToolTip("")

    def clear_cache(self) -> None:
        """Drop cached preview pixmaps when analysis inputs are rebuilt."""
        self._pixmap_cache.clear()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            return
        available = self.contentsRect().size()
        if available.width() < 1 or available.height() < 1:
            return
        self.setPixmap(self._source_pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _load_preview_pixmap(self, path: Path) -> QPixmap:
        """Load JPG/JPEG preview lazily and keep a small LRU cache."""
        if path.suffix.casefold() not in {".jpg", ".jpeg"}:
            return QPixmap(str(path))
        try:
            stat = path.stat()
        except OSError:
            return QPixmap()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        cached = self._pixmap_cache.get(key)
        if cached is not None:
            self._pixmap_cache.move_to_end(key)
            return cached
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return pixmap
        self._pixmap_cache[key] = pixmap
        self._pixmap_cache.move_to_end(key)
        while len(self._pixmap_cache) > self.CACHE_LIMIT:
            self._pixmap_cache.popitem(last=False)
        return pixmap


class SelectedNumbersDialog(QDialog):
    """Multiline number editor that wraps long lists to the available width."""

    def __init__(self, display_name: str, initial_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбранные номера")
        self.resize(430, 260)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"{display_name}\n"
            "Введите номера через пробел, запятую или точку с запятой:"
        ))

        self.editor = QPlainTextEdit(initial_text)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.editor.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.editor.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        layout.addWidget(self.editor, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def text(self) -> str:
        return self.editor.toPlainText()


class CsvMappingDialog(QDialog):
    def __init__(self, headers: tuple[str, ...], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сопоставление столбцов CSV")
        layout = QVBoxLayout(self)
        identity, numbers = suggest_columns(headers)
        layout.addWidget(QLabel("Столбец с student_id или ФИО:"))
        self.identity_combo = QComboBox()
        self.identity_combo.addItems(headers)
        if identity:
            self.identity_combo.setCurrentText(identity)
        layout.addWidget(self.identity_combo)
        layout.addWidget(QLabel("Столбцы с номерами фотографий:"))
        self.number_list = QListWidget()
        for header in headers:
            item = QListWidgetItem(header)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if header in numbers else Qt.CheckState.Unchecked
            )
            self.number_list.addItem(item)
        layout.addWidget(self.number_list)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def mapping(self) -> tuple[str, list[str]]:
        columns = [
            self.number_list.item(index).text()
            for index in range(self.number_list.count())
            if self.number_list.item(index).checkState() == Qt.CheckState.Checked
        ]
        return self.identity_combo.currentText(), columns


class AiDialog(QDialog):
    def __init__(self, roster: StudentRoster, config: argparse.Namespace, parent=None):
        super().__init__(parent)
        self.roster = roster
        self.config = config
        self.entries: list[ImportEntry] = []
        self.unresolved: list[dict] = []
        self.setWindowTitle("AI-импорт выбора")
        self.resize(760, 600)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        generate = QWidget()
        generate_layout = QVBoxLayout(generate)
        generate_layout.addWidget(QLabel("Неструктурированный текст:"))
        self.raw_text = QPlainTextEdit()
        generate_layout.addWidget(self.raw_text)
        copy_button = QPushButton("Сформировать промпт и скопировать")
        copy_button.clicked.connect(self._copy_prompt)
        generate_layout.addWidget(copy_button)
        self.prompt_preview = QPlainTextEdit()
        self.prompt_preview.setReadOnly(True)
        generate_layout.addWidget(self.prompt_preview)
        tabs.addTab(generate, "1. Промпт")

        response = QWidget()
        response_layout = QVBoxLayout(response)
        response_layout.addWidget(QLabel("JSON-ответ AI:"))
        self.response_text = QPlainTextEdit()
        response_layout.addWidget(self.response_text)
        open_button = QPushButton("Открыть JSON-файл")
        open_button.clicked.connect(self._open_response)
        response_layout.addWidget(open_button)
        apply_button = QPushButton("Проверить ответ")
        apply_button.clicked.connect(self._validate_response)
        response_layout.addWidget(apply_button)
        tabs.addTab(response, "2. Ответ")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _copy_prompt(self):
        text = self.raw_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Нет текста", "Вставьте исходный текст.")
            return
        template = load_prompt_template(
            Path(__file__).parent / "resources" / "ai_prompt_template.txt"
        )
        prompt = build_prompt(template, self.roster, text)
        QApplication.clipboard().setText(prompt)
        self.prompt_preview.setPlainText(prompt)

    def _validate_response(self):
        try:
            self.entries, self.unresolved = validate_ai_response(
                extract_json_object(self.response_text.toPlainText()),
                self.roster,
                min_digits=PHOTO_NUMBER_DIGITS,
                max_digits=PHOTO_NUMBER_DIGITS,
                pad_to_digits=0,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка AI JSON", str(exc))
            return
        QMessageBox.information(
            self,
            "Ответ проверен",
            f"Однозначных записей: {len(self.entries)}\n"
            f"Не разрешено: {len(self.unresolved)}",
        )
        self.accept()

    def _open_response(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Открыть JSON-ответ AI", "", "JSON (*.json);;Все файлы (*)"
        )
        if not filename:
            return
        try:
            self.response_text.setPlainText(Path(filename).read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError) as exc:
            QMessageBox.critical(self, "Ошибка JSON", str(exc))
