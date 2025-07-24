# pysm_lib/gui/dialogs/collection_passport_dialog.py

from typing import Optional, Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QPlainTextEdit,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QLabel,
    QToolBox,
    QGroupBox,
)

from ...models import ScriptRootModel, ScriptSetsCollectionModel
from ...locale_manager import LocaleManager
from ..widgets.path_list_editor import PathListEditor
from ..widgets.context_editor_widget import ContextEditorWidget
from ...app_constants import APPLICATION_ROOT_DIR


class CollectionPassportDialog(QDialog):
    def __init__(
        self,
        controller: Any,
        collection_model: "ScriptSetsCollectionModel",
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self.setWindowTitle(
            self.locale_manager.get("dialogs.collection_passport.title")
        )
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)

        collection_file_path = controller.current_collection_file_path
        self.collection_model = collection_model

        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # --- 1. БЛОК: Создаем и стилизуем QToolBox ---
        self.properties_toolbox = QToolBox()
        self.properties_toolbox.setStyleSheet(
            """
            QToolBox::tab {
                background-color: #ffc300;
                color: black;
                padding: 4px;
                border-radius: 5px;                                
            }
            QToolBox::tab:!selected:hover {
                color: #ffffff;
                font: bold;
                
            }             
            QToolBox::tab:selected {
                font: bold;
                background-color: white;
            }
        """
        )
        left_layout.addWidget(self.properties_toolbox)

        # --- 2. БЛОК: Создаем виджет для первой вкладки (Свойства) ---
        props_widget = QWidget()
        props_layout = QVBoxLayout(
            props_widget
        )  # Используем QVBoxLayout для корректной работы QTextEdit
        self.name_label = QLabel(
            self.locale_manager.get("dialogs.collection_passport.name_label")
        )
        self.name_edit = QLineEdit(self.collection_model.collection_name)
        props_layout.addWidget(self.name_label)
        props_layout.addWidget(self.name_edit)
        self.description_label = QLabel(
            self.locale_manager.get("dialogs.collection_passport.description_label")
        )
        
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        # Заменяем QTextEdit на QPlainTextEdit для удобного редактирования
        # простого текста, включая HTML-теги.
        self.description_edit = QPlainTextEdit(self.collection_model.description or "")
        # Строка .setAcceptRichText(False) больше не нужна.
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

        props_layout.addWidget(self.description_label)
        props_layout.addWidget(self.description_edit, 1)
        self.properties_toolbox.addItem(
            props_widget,
            self.locale_manager.get("dialogs.collection_passport.properties_group"),
        )        


        # --- 3. БЛОК: Создаем виджет для второй вкладки (Корневые директории) ---
        roots_widget = QWidget()
        roots_layout = QVBoxLayout(roots_widget)
        start_dir_for_roots = str(
            collection_file_path.parent
            if collection_file_path and collection_file_path.is_file()
            else APPLICATION_ROOT_DIR / "scripts"
        )
        self.roots_editor = PathListEditor(
            locale_manager=self.locale_manager,
            dialog_title=self.locale_manager.get(
                "dialogs.collection_passport.select_dir_dialog_title"
            ),
            allow_editing=False,
            default_dialog_path=start_dir_for_roots,
        )
        self.roots_editor.set_paths(
            [root.path for root in self.collection_model.script_roots]
        )
        roots_layout.addWidget(self.roots_editor)
        self.properties_toolbox.addItem(
            roots_widget,
            self.locale_manager.get("dialogs.collection_passport.roots_group"),
        )

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        context_group = QGroupBox(
            self.locale_manager.get("dialogs.collection_passport.context_group")
        )
        context_layout = QVBoxLayout(context_group)
        all_args_details = controller.get_known_args_with_details()
        self.context_editor = ContextEditorWidget(
            collection_file_path, all_args_details, self.locale_manager
        )
        self.context_editor.set_data(self.collection_model.context_data)
        context_layout.addWidget(self.context_editor)
        right_layout.addWidget(context_group)
        splitter.addWidget(right_panel)

        total_width = self.width()
        left_width = int(total_width * 0.3)
        right_width = total_width - left_width
        splitter.setSizes([left_width, right_width])

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        main_layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setEnabled(False)
        self.name_edit.textChanged.connect(lambda: self.ok_button.setEnabled(True))
        self.description_edit.textChanged.connect(
            lambda: self.ok_button.setEnabled(True)
        )
        self.context_editor.data_changed.connect(
            lambda: self.ok_button.setEnabled(True)
        )
        self.roots_editor.data_changed.connect(lambda: self.ok_button.setEnabled(True))

    def get_data(self) -> Optional[Dict[str, Any]]:
        if self.result() == QDialog.DialogCode.Accepted:
            new_root_paths = self.roots_editor.get_paths()
            new_script_roots = [ScriptRootModel(path=p) for p in new_root_paths]
            return {
                "name": self.name_edit.text(),
                "description": self.description_edit.toPlainText(),
                "script_roots": new_script_roots,
                "context_data": self.context_editor.get_data(),
            }
        return None
