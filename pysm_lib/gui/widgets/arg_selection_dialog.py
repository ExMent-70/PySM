# pysm_lib/gui/widgets/arg_selection_dialog.py

from typing import Dict, Optional, Tuple, List

from PySide6.QtCore import Qt, QSignalBlocker, Slot
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QLineEdit,
    QListWidgetItem,
    QFormLayout,
    QWidget,
)

from ...models import ScriptArgMetaDetailModel
from ...locale_manager import LocaleManager


class ArgSelectionDialog(QDialog):
    """
    Диалог для выбора аргумента из списка известных в коллекции.
    Показывает детали выбранного аргумента.
    """

    def __init__(
        self,
        title: str,
        label: str,
        args_meta_map: Dict[str, List[Tuple[str, ScriptArgMetaDetailModel]]],
        locale_manager: LocaleManager,
        parent: Optional[QWidget] = None,
        read_only: bool = False,
    ):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self.args_meta_map = args_meta_map
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        main_layout = QVBoxLayout(self)
        filter_layout = QFormLayout()
        self.name_edit = QLineEdit()

        if read_only:
            self.name_edit.setReadOnly(True)

        filter_layout.addRow(label, self.name_edit)
        main_layout.addLayout(filter_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)
        self.arg_list = QListWidget()
        splitter.addWidget(self.arg_list)
        self.details_view = QTextEdit()
        self.details_view.setReadOnly(True)
        splitter.addWidget(self.details_view)
        splitter.setSizes([200, 400])

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        main_layout.addWidget(button_box)

        self.arg_list.addItems(sorted(self.args_meta_map.keys()))

        if read_only:
            ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
            ok_button.setEnabled(False)
            button_box.accepted.connect(self.reject)
        else:
            button_box.accepted.connect(self.accept)

        button_box.rejected.connect(self.reject)

        # --- БЛОК 1: ИЗМЕНЕННЫЕ ПОДКЛЮЧЕНИЯ СИГНАЛОВ ---
        self.name_edit.textChanged.connect(self._filter_list)
        # Сигнал selectionChanged теперь отвечает ТОЛЬКО за обновление правой панели
        self.arg_list.currentItemChanged.connect(self._update_details_view)
        # Новый сигнал itemClicked отвечает за установку текста в QLineEdit
        self.arg_list.itemClicked.connect(self._on_item_clicked)

        if not read_only:
            # Двойной клик по-прежнему подтверждает выбор
            self.arg_list.itemDoubleClicked.connect(self.accept)

        if self.arg_list.count() > 0:
            self.arg_list.setCurrentRow(0)

    def _filter_list(self, text: str):
        # Блокируем сигналы, чтобы избежать мерцания при обновлении
        self.arg_list.blockSignals(True)
        self.arg_list.clear()

        # Заполняем список отфильтрованными элементами
        for arg_name in sorted(self.args_meta_map.keys()):
            # if text.lower() in arg_name.lower():
            # Меняем проверку с 'in' на 'startswith' для более интуитивного поведения
            if arg_name.lower().startswith(text.lower()):
                self.arg_list.addItem(arg_name)

        # Разблокируем сигналы
        self.arg_list.blockSignals(False)

        # Если в списке есть элементы, выбираем первый
        if self.arg_list.count() > 0:
            self.arg_list.setCurrentRow(0)
        else:
            # Иначе очищаем панель деталей
            self.details_view.clear()

    # 2. БЛОК: Новый метод _on_item_clicked
    @Slot(QListWidgetItem)
    def _on_item_clicked(self, item: QListWidgetItem):
        """Вызывается при клике на элемент списка. Устанавливает текст в поле ввода."""
        if not item:
            return
        # Блокируем сигналы на QLineEdit, чтобы его изменение не вызвало фильтрацию
        with QSignalBlocker(self.name_edit):
            self.name_edit.setText(item.text())

    # 3. БЛОК: Старый метод _on_selection_changed переименован и упрощен
    @Slot(QListWidgetItem, QListWidgetItem)
    def _update_details_view(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Обновляет правую панель с детальной информацией о выбранном элементе."""
        if not current:
            self.details_view.clear()
            return

        arg_name = current.text()
        usage_list = self.args_meta_map.get(arg_name)

        if not usage_list:
            self.details_view.clear()
            return

        header_style = "background-color: #e8f0fe; padding: 5px; border-radius: 3px; font-weight: bold;"
        var_style = "color: #FFFFFF; background-color: #000080; padding: 5px; border-radius: 3px; font-size: 14px;"

        html_parts = []
        html_parts.append(f"""<div style="{var_style}">Переменная: {arg_name}</div>""")
        html_parts.append("""<p> </p>""")

        total_items = len(usage_list)

        for i, (script_name, meta) in enumerate(usage_list):
            help_text = (
                meta.description
                or "<i>"
                + self.locale_manager.get(
                    "dialogs.script_properties.no_description_text"
                )
                + "</i>"
            )
            type_text = meta.type
            default_text = (
                str(meta.default) if meta.default is not None else "<i>None</i>"
            )

            block_content = f"""
            <div style="{header_style}">Используется в скрипте: {script_name}.py</div>
            <div style="padding-left: 10px; padding-top: 5px;">
                <b>Тип данных:</b> {type_text}<br>
                <b>Значение по умолчанию:</b> {default_text}<br>
                <b>Описание:</b>
                <div style='padding-left: 15px;'>{help_text}</div>
            </div>
            """
            html_parts.append(block_content)

            if i < total_items - 1:
                html_parts.append(
                    '<hr style="border: 0; border-top: 1px solid #ccc; margin: 10px 0;">'
                )

        self.details_view.setHtml("".join(html_parts))

    # 4. БЛОК: Метод get_selected_arg_meta (без изменений)
    def get_selected_arg_meta(
        self,
    ) -> Tuple[Optional[ScriptArgMetaDetailModel], Optional[str]]:
        if self.exec() == QDialog.DialogCode.Accepted:
            arg_name = self.name_edit.text()
            usage_list = self.args_meta_map.get(arg_name)
            if usage_list:
                first_meta = usage_list[0][1]
                return first_meta, arg_name
        return None, None
