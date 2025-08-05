# pysm_lib/gui/widgets/editor_factory.py

import json
import pathlib
from datetime import date, datetime
from typing import Optional, Any, Dict, List, Tuple

from PySide6.QtCore import Qt, Signal, QLocale
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QComboBox,
    QCheckBox,
    QInputDialog,
    QPlainTextEdit,
    QFileDialog,
    QDateEdit,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QGridLayout,
    QSizePolicy,
)

from ...locale_manager import LocaleManager


class BaseEditor(QWidget):
    """Базовый класс для всех редакторов, чтобы иметь общий тип и стиль."""

    valueChanged = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAutoFillBackground(False)


class CheckBoxEditor(BaseEditor):
    def __init__(self, value: Optional[bool], parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.checkbox = QCheckBox()
        if value is not None:
            self.checkbox.setChecked(bool(value))
        layout.addWidget(self.checkbox)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.checkbox.toggled.connect(self.valueChanged.emit)


class ChoicesEditorWidget(BaseEditor):
    choicesChanged = Signal(list)

    def __init__(
        self,
        value: Optional[str],
        choices: Optional[List[str]],
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self._choices = choices or []
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.combo = QComboBox()
        self.combo.setProperty("isEditor", True)
        self.combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.combo.addItems(self._choices)
        if value is not None:
            self.combo.setCurrentText(str(value))
        self.edit_btn = QPushButton("...")
        self.edit_btn.setFixedWidth(30)
        self.edit_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        self.edit_btn.setToolTip(
            self.locale_manager.get("dialogs.context_editor.edit_choices_tooltip")
        )
        layout.addWidget(self.combo, 0, 0)
        layout.addWidget(self.edit_btn, 0, 1)
        layout.setColumnStretch(0, 1)
        self.combo.currentTextChanged.connect(self.valueChanged.emit)
        self.edit_btn.clicked.connect(self._edit_choices)

    def _edit_choices(self):
        text, ok = QInputDialog.getMultiLineText(
            self,
            self.locale_manager.get("dialogs.context_editor.edit_choices_title"),
            self.locale_manager.get("dialogs.context_editor.edit_choices_label"),
            "\n".join(self._choices),
        )
        if ok:
            new_choices = [line.strip() for line in text.splitlines() if line.strip()]
            self._choices = new_choices
            self.choicesChanged.emit(self._choices)
            current_text = self.combo.currentText()
            self.combo.blockSignals(True)
            self.combo.clear()
            self.combo.addItems(self._choices)
            self.combo.setCurrentText(current_text)
            self.combo.blockSignals(False)


class DialogEditorWidget(BaseEditor):
    def __init__(
        self,
        value: Any,
        var_type: str,
        options: Dict[str, Any],
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.var_type = var_type
        self.options = options
        self.locale_manager = locale_manager
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.line_edit = QLineEdit(str(value) if value is not None else "")
        self.line_edit.setToolTip(self.line_edit.text())
        self.line_edit.setCursorPosition(0)
        self.line_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.line_edit.setStyleSheet(
            """
            QLineEdit { border: 1px solid transparent; background-color: transparent; }
            QLineEdit:focus { border: 1px solid #5693c2; background-color: white; }
        """
        )
        if self.var_type in ("string_multiline", "json"):
            self.line_edit.setReadOnly(True)
        button = QPushButton("...")
        button.setFixedWidth(30)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.line_edit, 0, 0)
        layout.addWidget(button, 0, 1)
        layout.setColumnStretch(0, 1)
        button.clicked.connect(self.on_button_click)
        if not self.line_edit.isReadOnly():
            self.line_edit.editingFinished.connect(
                lambda: self.valueChanged.emit(self.line_edit.text())
            )

    def on_button_click(self):
        current_value = (
            self.line_edit.text()
            if self.var_type in ("file_path", "dir_path")
            else self.options.get("value")
        )
        new_value, changed = EditorFactory._handle_button_dialogs(
            self, self.var_type, current_value, self.options, self.locale_manager
        )
        if changed:
            self.line_edit.setText(str(new_value))
            self.line_edit.setToolTip(str(new_value))
            self.line_edit.setCursorPosition(0)
            self.valueChanged.emit(new_value)


class DateEditor(BaseEditor):
    def __init__(self, value: Optional[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.editor = QDateEdit(self)
        self.editor.setCalendarPopup(True)
        self.editor.setDisplayFormat("yyyy-MM-dd")
        if value:
            self.editor.setDate(date.fromisoformat(value))
        else:
            self.editor.setDate(date.today())
        layout.addWidget(self.editor)
        self.editor.dateChanged.connect(
            lambda d: self.valueChanged.emit(d.toString(Qt.DateFormat.ISODate))
        )


class DateTimeEditor(BaseEditor):
    def __init__(self, value: Optional[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.editor = QDateTimeEdit(self)
        self.editor.setCalendarPopup(True)
        self.editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        if value:
            self.editor.setDateTime(datetime.fromisoformat(value))
        else:
            self.editor.setDateTime(datetime.now())
        layout.addWidget(self.editor)
        self.editor.dateTimeChanged.connect(
            lambda dt: self.valueChanged.emit(dt.toString(Qt.DateFormat.ISODateWithMs))
        )


class ListEditorWidget(BaseEditor):
    """Редактор для типа 'list', использующий диалог для многострочного ввода."""

    def __init__(
        self,
        value: Optional[List[str]],
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self.current_value = value or []
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.line_edit = QLineEdit()
        self.line_edit.setReadOnly(True)
        self.line_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.line_edit.setStyleSheet(
            "QLineEdit { border: 1px solid transparent; background-color: transparent; }"
        )
        self._update_display_text()
        button = QPushButton("...")
        button.setFixedWidth(30)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.line_edit, 0, 0)
        layout.addWidget(button, 0, 1)
        layout.setColumnStretch(0, 1)
        button.clicked.connect(self.on_button_click)

    def on_button_click(self):
        text_value = "\n".join(self.current_value)
        text, ok = QInputDialog.getMultiLineText(
            self, "List Editor", "Enter one value per line:", text_value
        )
        if ok:
            new_list = [line.strip() for line in text.splitlines() if line.strip()]
            self.current_value = new_list
            self._update_display_text()
            self.valueChanged.emit(self.current_value)

    def _update_display_text(self):
        display_text = ", ".join(self.current_value)
        self.line_edit.setText(display_text)
        self.line_edit.setToolTip(display_text)
        self.line_edit.setCursorPosition(0)


class LineEditEditor(BaseEditor):
    def __init__(
        self,
        value: Optional[str],
        validator: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.line_edit = QLineEdit(str(value) if value is not None else "")
        self.line_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        if validator:
            self.line_edit.setValidator(validator)
        layout.addWidget(self.line_edit)
        layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit.setStyleSheet(
            """
            QLineEdit { border: 1px solid transparent; background-color: transparent; }
            QLineEdit:focus { border: 1px solid #5693c2; background-color: white; }
        """
        )
        self.line_edit.editingFinished.connect(
            lambda: self.valueChanged.emit(self.line_edit.text())
        )


class EditorFactory:
    @staticmethod
    def create_editor(
        var_type: str,
        options: Dict[str, Any],
        locale_manager: LocaleManager,
        is_passport_mode: bool = False,
    ) -> Optional[QWidget]:
        value = options.get("value")
        if var_type == "bool":
            return CheckBoxEditor(value)
        elif var_type == "int":
            return LineEditEditor(value, QIntValidator())
        elif var_type == "float":
            validator = QDoubleValidator()
            validator.setLocale(QLocale(QLocale.Language.C))
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            return LineEditEditor(value, validator)
        elif var_type == "choice":
            return ChoicesEditorWidget(value, options.get("choices"), locale_manager)
        elif var_type == "date":
            return DateEditor(value)
        elif var_type == "datetime":
            return DateTimeEditor(value)
        elif var_type in (
            "file_path",
            "dir_path",
            "password",
            "string_multiline",
            "json",
        ):
            return DialogEditorWidget(value, var_type, options, locale_manager)
        elif var_type == "list":
            return ListEditorWidget(value, locale_manager)
        elif var_type == "string":
            return LineEditEditor(value)
        else:
            return None

    @staticmethod
    def _handle_button_dialogs(
        parent: QWidget,
        var_type: str,
        current_value: Any,
        options: Dict[str, Any],
        locale_manager: LocaleManager,
    ) -> Tuple[Any, bool]:
        collection_dir = options.get("collection_dir", pathlib.Path.home())
        start_path = str(current_value) if current_value else str(collection_dir)
        if var_type == "file_path":
            new_path, _ = QFileDialog.getOpenFileName(parent, "Select File", start_path)
            return (new_path, True) if new_path else (current_value, False)
        if var_type == "dir_path":
            new_path = QFileDialog.getExistingDirectory(
                parent, "Select Directory", start_path
            )
            return (new_path, True) if new_path else (current_value, False)
        if var_type == "password":
            new_pass, ok = QInputDialog.getText(
                parent,
                "Enter Password",
                "Password:",
                QLineEdit.EchoMode.Password,
                current_value,
            )
            return (new_pass, ok)
        if var_type == "string_multiline":
            text, ok = QInputDialog.getMultiLineText(
                parent, "Edit Text", "Value:", current_value
            )
            return (text, ok)
        if var_type == "json":
            dialog = QDialog(parent)
            dialog.setWindowTitle(
                locale_manager.get(
                    "dialogs.context_editor.json_editor_title", name="Value"
                )
            )
            layout = QVBoxLayout(dialog)
            editor = QPlainTextEdit()
            try:
                text_to_edit = (
                    json.dumps(current_value, indent=2, ensure_ascii=False)
                    if current_value
                    else ""
                )
            except (json.JSONDecodeError, TypeError):
                text_to_edit = str(current_value or "")
            editor.setPlainText(text_to_edit)
            layout.addWidget(editor)
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            )
            layout.addWidget(button_box)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            if dialog.exec():
                text = editor.toPlainText()
                try:
                    return json.loads(text), True
                except json.JSONDecodeError:
                    QMessageBox.warning(
                        parent,
                        locale_manager.get("general.error_title"),
                        locale_manager.get("dialogs.context_editor.json_invalid_error"),
                    )
                    return current_value, False
        return current_value, False
