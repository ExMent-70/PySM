"""Qt dialog for the provider-neutral manual AI JSON workflow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...ai.manual_json import AiJsonRequest


T = TypeVar("T")


class AiJsonDialogStatus(str, Enum):
    """How a manual AI JSON dialog session ended."""

    VALIDATED = "validated"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class AiJsonDialogResult(Generic[T]):
    """Validated domain value and user-entered source material."""

    status: AiJsonDialogStatus
    value: T | None
    prompt: str
    raw_text: str
    raw_response: str

    @property
    def accepted(self) -> bool:
        """Return whether the response passed the request validator."""

        return self.status is AiJsonDialogStatus.VALIDATED


class AiJsonDialog(QDialog, Generic[T]):
    """Reusable two-step dialog: prepare prompt, then validate AI JSON."""

    def __init__(self, request: AiJsonRequest[T], parent: QWidget | None = None):
        super().__init__(parent)
        self.request = request
        self._last_prompt = ""
        self._accepted_result: AiJsonDialogResult[T] | None = None

        self.setWindowTitle(request.title)
        self.resize(760, 600)
        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        layout.addWidget(tabs)

        prompt_tab = QWidget(tabs)
        prompt_layout = QVBoxLayout(prompt_tab)
        self.raw_text_edit: QPlainTextEdit | None = None
        if request.raw_text_token is not None:
            prompt_layout.addWidget(QLabel(request.raw_text_label, prompt_tab))
            self.raw_text_edit = QPlainTextEdit(prompt_tab)
            self.raw_text_edit.setPlaceholderText(request.raw_text_placeholder)
            prompt_layout.addWidget(self.raw_text_edit)

        self.copy_prompt_button = QPushButton("Сформировать промпт и скопировать", prompt_tab)
        self.copy_prompt_button.clicked.connect(self._copy_prompt)
        prompt_layout.addWidget(self.copy_prompt_button)
        prompt_layout.addWidget(QLabel("Предпросмотр промпта:", prompt_tab))
        self.prompt_preview = QPlainTextEdit(prompt_tab)
        self.prompt_preview.setReadOnly(True)
        prompt_layout.addWidget(self.prompt_preview)
        tabs.addTab(prompt_tab, "1. Промпт")

        response_tab = QWidget(tabs)
        response_layout = QVBoxLayout(response_tab)
        response_layout.addWidget(QLabel(request.response_label, response_tab))
        self.response_text_edit = QPlainTextEdit(response_tab)
        response_layout.addWidget(self.response_text_edit)
        self.open_response_button = QPushButton("Открыть JSON-файл", response_tab)
        self.open_response_button.clicked.connect(self._open_response)
        response_layout.addWidget(self.open_response_button)
        self.validate_response_button = QPushButton("Проверить и продолжить", response_tab)
        self.validate_response_button.clicked.connect(self._validate_response)
        response_layout.addWidget(self.validate_response_button)
        tabs.addTab(response_tab, "2. Ответ")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def result(self) -> AiJsonDialogResult[T]:
        """Return a validated result or a cancellation snapshot."""

        if self._accepted_result is not None:
            return self._accepted_result
        return AiJsonDialogResult(
            status=AiJsonDialogStatus.CANCELLED,
            value=None,
            prompt=self._last_prompt,
            raw_text=self._raw_text(),
            raw_response=self.response_text_edit.toPlainText(),
        )

    def _raw_text(self) -> str:
        return self.raw_text_edit.toPlainText() if self.raw_text_edit is not None else ""

    def _copy_prompt(self) -> None:
        try:
            prompt = self.request.build_prompt(self._raw_text())
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка AI-промпта", str(exc))
            return

        self._last_prompt = prompt
        QApplication.clipboard().setText(prompt)
        self.prompt_preview.setPlainText(prompt)
        QMessageBox.information(
            self,
            "Промпт скопирован",
            "Промпт скопирован в буфер обмена. Передайте его выбранному AI-сервису.",
        )

    def _open_response(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть JSON-ответ AI",
            "",
            self.request.response_file_filter,
        )
        if not filename:
            return
        try:
            self.response_text_edit.setPlainText(
                Path(filename).read_text(encoding="utf-8-sig")
            )
        except (OSError, UnicodeError) as exc:
            QMessageBox.critical(self, "Ошибка AI JSON", str(exc))

    def _validate_response(self) -> None:
        raw_response = self.response_text_edit.toPlainText()
        try:
            value = self.request.validate_response(raw_response)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка AI JSON", str(exc))
            return

        self._accepted_result = AiJsonDialogResult(
            status=AiJsonDialogStatus.VALIDATED,
            value=value,
            prompt=self._last_prompt,
            raw_text=self._raw_text(),
            raw_response=raw_response,
        )
        if self.request.show_success_message:
            message = (
                self.request.success_message(value)
                if self.request.success_message is not None
                else "JSON-ответ AI успешно проверен."
            )
            QMessageBox.information(self, "Ответ проверен", message)
        self.accept()


def create_ai_json_dialog(
    request: AiJsonRequest[T],
    parent: QWidget | None = None,
) -> AiJsonDialog[T]:
    """Create a non-blocking reusable manual AI JSON dialog."""

    return AiJsonDialog(request, parent)


def edit_ai_json_response(
    request: AiJsonRequest[T],
    parent: QWidget | None = None,
) -> AiJsonDialogResult[T]:
    """Open the dialog modally and return its structured result.

    The helper creates a temporary ``QApplication`` only for standalone callers
    and never closes an existing application owned by the calling script.
    """

    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    dialog = create_ai_json_dialog(request, parent)
    dialog.exec()
    return dialog.result


__all__ = [
    "AiJsonDialog",
    "AiJsonDialogResult",
    "AiJsonDialogStatus",
    "create_ai_json_dialog",
    "edit_ai_json_response",
]
