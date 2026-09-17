"""Reusable Qt widgets and dialogs for the photo selection workflow."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPixmap, QTextOption
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFrame, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from pysm_lib.pysm_image_cache import (
    AsyncImageLoader,
    AsyncImageResult,
    ImageRequest,
)

from .csv_import import suggest_columns


class AnswerCheckBox(QCheckBox):
    """Theme-aware answer checkbox that also forwards row double-clicks."""

    rowDoubleClicked = Signal()

    def mouseDoubleClickEvent(self, event):
        self.rowDoubleClicked.emit()
        event.accept()


class ImagePreviewLabel(QLabel):
    """Theme-friendly preview backed by the shared asynchronous image API."""

    RESIZE_DEBOUNCE_MS = 80

    def __init__(self, image_loader: AsyncImageLoader, parent=None):
        super().__init__(parent)
        self._image_loader = image_loader
        self._channel = ("photo-selection-preview", id(self))
        self._request_id: int | None = None
        self.image_path: Path | None = None
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._request_current_image)
        self._image_loader.imageReady.connect(self._on_image_ready)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(220)
        self.setWordWrap(True)
        self.setText("Выберите строку фотографии для предпросмотра JPG")

    def show_image(self, path: Path) -> None:
        """Schedule a non-blocking preview request for ``path``."""

        self.image_path = Path(path)
        self.clear()
        self.setText("Загрузка…")
        self.setToolTip(str(self.image_path))
        self._resize_timer.stop()
        self._request_current_image()

    def show_message(self, text: str) -> None:
        self.cancel_requests()
        self.image_path = None
        self.clear()
        self.setText(text)
        self.setToolTip("")

    def cancel_requests(self) -> None:
        """Prevent pending resize callbacks from submitting new work."""

        self._resize_timer.stop()
        self._image_loader.cancel(self._channel)
        self._request_id = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.image_path is not None:
            self._resize_timer.start(self.RESIZE_DEBOUNCE_MS)

    @Slot()
    def _request_current_image(self) -> None:
        path = self.image_path
        if path is None:
            return
        available = self.contentsRect().size()
        if available.width() < 1 or available.height() < 1:
            return
        request = ImageRequest(
            path,
            (available.width(), available.height()),
            mode="fit",
            allow_upscale=False,
            variant="photo_selection.preview.v1",
        )
        self._request_id = self._image_loader.request(
            request,
            channel=self._channel,
            persist=True,
            disk_format="JPG",
            quality=90,
        )

    @Slot(object)
    def _on_image_ready(self, result: AsyncImageResult) -> None:
        if result.channel != self._channel or result.request_id != self._request_id:
            return
        self._request_id = None
        if result.image.isNull():
            path = self.image_path
            name = path.name if path is not None else ""
            self.show_message(f"Не удалось открыть JPG:\n{name}")
            return
        self.clear()
        self.setPixmap(QPixmap.fromImage(result.image))


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
