# pysm_lib/gui/widgets/parameter_editor_widget.py

"""
Модуль содержит универсальный виджет `ParameterEditorWidget`, 
предназначенный для редактирования параметров скриптов в виде таблицы.
Поддерживает три режима работы: редактирование паспорта скрипта (EditMode.PASSPORT), 
экземпляра скрипта (EditMode.INSTANCE) и контекстных переменных (EditMode.CONTEXT_VARS).
"""

from typing import Dict, Optional, Any, List, Union

from PySide6.QtCore import Qt, Slot, QSignalBlocker, QTimer, QEvent, QObject, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QPushButton,
)

from ...models import (
    ScriptArgMetaDetailModel,
    ScriptSetEntryValueEnabled,
    ContextVariableType,
    ContextVariableModel,
    ScriptSetEntryModel,
)
from ...locale_manager import LocaleManager
from ...theme_manager import ThemeManager
from ...app_enums import EditMode
from .editor_factory import EditorFactory
from .editor_context import EditorContext


class ParamTableColumn:
    """
    Константы индексов колонок таблицы для различных режимов редактирования.
    Обеспечивают строгую привязку данных к колонкам в зависимости от EditMode.
    """
    # Колонки для режима экземпляра (EditMode.INSTANCE)
    INSTANCE_ENABLE = 0
    INSTANCE_NAME = 1
    INSTANCE_VALUE = 2
    INSTANCE_ACTIONS = 3

    # Колонки для режима паспорта (EditMode.PASSPORT)
    PASSPORT_REQUIRED = 0
    PASSPORT_NAME = 1
    PASSPORT_TYPE = 2
    PASSPORT_DEFAULT = 3
    PASSPORT_DESCRIPTION = 4

    # Колонки для режима контекстных переменных (EditMode.CONTEXT_VARS)
    CONTEXT_NAME = 0
    CONTEXT_TYPE = 1
    CONTEXT_VALUE = 2
    CONTEXT_READONLY = 3
    CONTEXT_DESCRIPTION = 4


class ParameterEditorWidget(QWidget):
    """
    Универсальный табличный редактор параметров.

    Динамически генерирует строки и колонки таблицы, встраивая в ячейки
    соответствующие редакторы (через EditorFactory) на основе типа данных переменной.
    """
    
    # Сигнал испускается при любом изменении данных в таблице (значений, типов, флагов)
    data_changed = Signal()

    def __init__(
        self,
        mode: EditMode,
        locale_manager: LocaleManager,
        theme_manager: ThemeManager,
        script_entries: Optional[List[ScriptSetEntryModel]] = None,
        get_script_name_func: Optional[callable] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Инициализирует виджет редактора параметров.

        :param mode: Режим работы (PASSPORT, INSTANCE, CONTEXT_VARS).
        :param locale_manager: Менеджер локализации для перевода интерфейса.
        :param theme_manager: Менеджер тем оформления.
        :param script_entries: Список известных экземпляров скриптов (нужен для редакторов экземпляров).
        :param get_script_name_func: Функция получения информации о скрипте.
        :param parent: Родительский виджет.
        """
        super().__init__(parent)
        self.mode = mode
        
        # Контекст передается внутрь каждого индивидуального редактора значения
        self.editor_context = EditorContext(
            theme_manager=theme_manager,
            locale_manager=locale_manager,
            get_script_info_func=get_script_name_func or (lambda x: None),
            script_entries=script_entries or[],
        )

        # Хранилища данных в зависимости от режима
        self._args_meta: Dict[str, ScriptArgMetaDetailModel] = {}
        self._args_values: Dict[str, ScriptSetEntryValueEnabled] = {}
        self._context_vars: Dict[str, ContextVariableModel] = {}

        # Настройки палитры для визуального выделения префиксов контекстных переменных
        self.prefix_color_palette =[
            QColor("#e8f0fe"), QColor("#eaf5ea"), QColor("#fff5e6"),
            QColor("#fdecf0"), QColor("#f0eefc"),
        ]
        self._prefix_to_color_map: Dict[str, QColor] = {}

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Создает и настраивает базовый UI (компоновку и саму таблицу)."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        # Отключаем стандартное редактирование текста, так как используются кастомные виджеты
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        
        layout.addWidget(self.table, 1)

    def _connect_signals(self):
        """Подключает базовые сигналы таблицы."""
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)

    def on_cell_double_clicked(self, row: int, column: int):
        """
        Обрабатывает двойной клик по ячейке.
        Вызывает кастомные диалоги редакторов (если они привязаны к ячейке) 
        или позволяет редактировать текст, если колонка разрешена для этого.

        :param row: Индекс строки.
        :param column: Индекс колонки.
        """
        allowed_columns =[]
        
        if self.mode == EditMode.PASSPORT:
            allowed_columns = [ParamTableColumn.PASSPORT_DEFAULT, ParamTableColumn.PASSPORT_DESCRIPTION]
        elif self.mode == EditMode.INSTANCE:
            allowed_columns = [ParamTableColumn.INSTANCE_VALUE]
        elif self.mode == EditMode.CONTEXT_VARS:
            allowed_columns =[ParamTableColumn.CONTEXT_VALUE, ParamTableColumn.CONTEXT_DESCRIPTION]

        if column in allowed_columns:
            widget = self.table.cellWidget(row, column)
            # Если в ячейке установлен кастомный редактор с кнопкой '...' (например, ListEditor)
            if hasattr(widget, "on_button_click"):
                widget.on_button_click()
            else:
                # Иначе открываем стандартное редактирование ячейки
                item = self.table.item(row, column)
                if item and item.flags() & Qt.ItemFlag.ItemIsEditable:
                    self.table.editItem(item)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Перехватывает события прокрутки колесика мыши для QComboBox.
        Предотвращает случайное изменение типа переменной при прокрутке таблицы.
        """
        if event.type() == QEvent.Type.Wheel and isinstance(watched, QComboBox):
            return True
        return super().eventFilter(watched, event)

    def set_data(
        self,
        data: Dict[str, Any],
        instance_values: Optional[Dict[str, ScriptSetEntryValueEnabled]] = None,
    ):
        """
        Загружает данные в виджет и инициирует перерисовку таблицы.

        :param data: Основной словарь с метаданными или переменными контекста.
        :param instance_values: Словарь со значениями конкретного экземпляра (только для EditMode.INSTANCE).
        """
        # Глубокое копирование используется для предотвращения случайного изменения оригинальных моделей
        if self.mode == EditMode.PASSPORT:
            self._args_meta = {k: v.model_copy(deep=True) for k, v in data.items()}
        elif self.mode == EditMode.INSTANCE:
            self._args_meta = {k: v.model_copy(deep=True) for k, v in data.items()}
            self._args_values = {k: v.model_copy(deep=True) for k, v in (instance_values or {}).items()}
        elif self.mode == EditMode.CONTEXT_VARS:
            self._context_vars = {k: v.model_copy(deep=True) for k, v in data.items()}
            
        self._populate_table()

    def get_updated_meta(self) -> Dict[str, ScriptArgMetaDetailModel]:
        """Возвращает измененные метаданные (для паспорта)."""
        return self._args_meta

    def get_updated_values(self) -> Dict[str, ScriptSetEntryValueEnabled]:
        """Возвращает измененные значения экземпляра (для настроек скрипта)."""
        return self._args_values

    def get_updated_context_vars(self) -> Dict[str, ContextVariableModel]:
        """Возвращает измененные контекстные переменные."""
        return self._context_vars

    def _populate_table(self):
        """
        Полностью перестраивает структуру и содержимое таблицы на основе текущих данных.
        Временно блокирует сигналы таблицы во избежание рекурсивных вызовов во время генерации.
        """
        self._prefix_to_color_map.clear()
        
        with QSignalBlocker(self.table):
            self.table.clear()
            self.table.setRowCount(0)
            
            # Настраиваем заголовки и количество колонок
            if self.mode == EditMode.PASSPORT:
                self._setup_passport_mode()
            elif self.mode == EditMode.INSTANCE:
                self._setup_instance_mode()
            else:
                self._setup_context_mode()
                
            # Определяем источник данных для итерации
            data_source = self._context_vars if self.mode == EditMode.CONTEXT_VARS else self._args_meta
            
            # Построчно заполняем таблицу отсортированными ключами
            for row, (name, model) in enumerate(sorted(data_source.items())):
                self.table.insertRow(row)
                if self.mode == EditMode.PASSPORT:
                    self._create_passport_row(row, name, model)
                elif self.mode == EditMode.INSTANCE:
                    self._create_instance_row(row, name, model)
                else:
                    self._create_context_row(row, name, model)
                    
        # Отложенный вызов для корректной подгонки ширины колонок после отрисовки
        QTimer.singleShot(0, self._adjust_table_columns)

    def _setup_passport_mode(self):
        """Настраивает заголовки таблицы для режима редактирования паспорта."""
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_required"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_name"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_type"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_default"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_description"),
        ])

    def _setup_instance_mode(self):
        """Настраивает заголовки таблицы для режима редактирования экземпляра."""
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_enable"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_name"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_value"),
            self.editor_context.locale_manager.get("dialogs.script_properties.args_tab.header_actions"),
        ])

    def _setup_context_mode(self):
        """Настраивает заголовки таблицы для режима редактирования контекстных переменных."""
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            self.editor_context.locale_manager.get("dialogs.context_editor.header_name"),
            self.editor_context.locale_manager.get("dialogs.context_editor.header_type"),
            self.editor_context.locale_manager.get("dialogs.context_editor.header_value"),
            self.editor_context.locale_manager.get("dialogs.context_editor.header_readonly"),
            self.editor_context.locale_manager.get("dialogs.context_editor.header_description"),
        ])

    def _create_context_row(self, row: int, name: str, var: ContextVariableModel):
        """
        Создает и заполняет одну строку таблицы для контекстной переменной.
        """
        # 1. Цветовое выделение по префиксу (например 'SYS_path' будет одного цвета, 'USER_name' другого)
        prefix = name.split("_", 1)[0] if "_" in name else None
        bg_color = self._get_color_for_prefix(prefix)
        
        def create_item(text=""):
            item = QTableWidgetItem(text)
            if bg_color:
                item.setBackground(bg_color)
            return item

        # Имя (только для чтения)
        name_item = create_item(name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, ParamTableColumn.CONTEXT_NAME, name_item)
        
        # Тип (выпадающий список)
        self.table.setItem(row, ParamTableColumn.CONTEXT_TYPE, create_item())
        type_combo = QComboBox()
        type_combo.addItems(list(ContextVariableType.__args__))
        type_combo.installEventFilter(self)
        type_combo.setCurrentText(var.type)
        type_combo.currentTextChanged.connect(lambda t, r=row: self._on_type_changed(r, t))
        self.table.setCellWidget(row, ParamTableColumn.CONTEXT_TYPE, type_combo)
        
        # Значение (динамический редактор)
        self.table.setItem(row, ParamTableColumn.CONTEXT_VALUE, create_item())
        self._create_value_editor(row, name, var)
        
        # Read Only (Чекбокс)
        self.table.setItem(row, ParamTableColumn.CONTEXT_READONLY, create_item())
        ro_check = QCheckBox()
        ro_check.setChecked(var.read_only)
        ro_check.toggled.connect(lambda s, r=row: self._on_readonly_changed(r, s))
        self.table.setCellWidget(row, ParamTableColumn.CONTEXT_READONLY, self._center_widget(ro_check))
        
        # Описание
        desc_item = create_item(var.description or "")
        desc_item.setToolTip(var.description)
        self.table.setItem(row, ParamTableColumn.CONTEXT_DESCRIPTION, desc_item)

    def _create_instance_row(self, row: int, name: str, meta: ScriptArgMetaDetailModel):
        """
        Создает и заполняет одну строку таблицы для экземпляра скрипта.
        Позволяет переопределять значения по умолчанию.
        """
        entry = self._args_values.get(name)
        
        # Enable (Чекбокс переопределения)
        chk_enable = QCheckBox()
        chk_enable.setChecked(entry.enabled if entry else False)
        chk_enable.setEnabled(not meta.required) # Обязательные параметры всегда включены
        chk_enable.toggled.connect(lambda state, r=row: self._on_enable_toggled(r, state))
        self.table.setCellWidget(row, ParamTableColumn.INSTANCE_ENABLE, self._center_widget(chk_enable))
        
        # Имя
        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        name_item.setToolTip(meta.description or name)
        self.table.setItem(row, ParamTableColumn.INSTANCE_NAME, name_item)
        
        # Значение (динамический редактор)
        self.table.setItem(row, ParamTableColumn.INSTANCE_VALUE, QTableWidgetItem())
        self._create_value_editor(row, name, meta)
        
        # Actions (Кнопка сброса к значению из паспорта)
        reset_btn = QPushButton(
            self.editor_context.locale_manager.get("dialogs.script_properties.reset_button")
        )        
        reset_btn.clicked.connect(lambda checked=False, r=row, n=name: self._on_reset_to_default(r, n))
        self.table.setCellWidget(row, ParamTableColumn.INSTANCE_ACTIONS, self._center_widget(reset_btn))

    def _create_passport_row(self, row: int, name: str, meta: ScriptArgMetaDetailModel):
        """
        Создает и заполняет одну строку таблицы для паспорта скрипта.
        Позволяет задавать типы и значения по умолчанию.
        """
        # Required (Чекбокс)
        chk_required = QCheckBox()
        chk_required.setChecked(meta.required)
        chk_required.toggled.connect(lambda state, r=row: self._on_required_toggled(r, state))
        self.table.setCellWidget(row, ParamTableColumn.PASSPORT_REQUIRED, self._center_widget(chk_required))
        
        # Имя
        name_item = QTableWidgetItem(name)
        self.table.setItem(row, ParamTableColumn.PASSPORT_NAME, name_item)
        
        # Тип (Выпадающий список)
        type_combo = QComboBox()
        type_combo.addItems(list(ContextVariableType.__args__))
        type_combo.installEventFilter(self)
        type_combo.setCurrentText(meta.type)
        type_combo.currentTextChanged.connect(lambda text, r=row: self._on_type_changed(r, text))
        self.table.setCellWidget(row, ParamTableColumn.PASSPORT_TYPE, type_combo)
        
        # Дефолтное значение (динамический редактор)
        self.table.setItem(row, ParamTableColumn.PASSPORT_DEFAULT, QTableWidgetItem())
        self._create_value_editor(row, name, meta)
        
        # Описание
        desc_item = QTableWidgetItem(meta.description or "")
        self.table.setItem(row, ParamTableColumn.PASSPORT_DESCRIPTION, desc_item)

    def _create_value_editor(self, row: int, name: str, model: Union[ScriptArgMetaDetailModel, ContextVariableModel]):
        """
        Универсальный метод для создания конкретного редактора значения.
        Использует `EditorFactory` для получения нужного виджета (например, `DateEditor`, `ListEditor` и т.д.)
        в зависимости от `model.type`. Подключает сигналы изменения к нужным слотам в зависимости от `mode`.
        """
        is_passport = self.mode == EditMode.PASSPORT
        is_context = self.mode == EditMode.CONTEXT_VARS
        
        target_col = -1
        value = None
        var_type = "string"
        choices = None
        
        # 1. Извлекаем данные для редактора в зависимости от режима
        if is_context:
            target_col = ParamTableColumn.CONTEXT_VALUE
            value = model.value
            var_type = model.type
            choices = getattr(model, "choices", None)
            
        elif is_passport:
            target_col = ParamTableColumn.PASSPORT_DEFAULT
            value = getattr(model, "default", None)
            var_type = model.type
            choices = getattr(model, "choices", None)
            
        else: # EditMode.INSTANCE
            target_col = ParamTableColumn.INSTANCE_VALUE
            entry = self._args_values.get(name)
            value = entry.value if entry else None
            var_type = model.type
            choices = getattr(model, "choices", None)
        
        # Очищаем ячейку перед вставкой нового редактора
        if self.table.cellWidget(row, target_col):
            self.table.removeCellWidget(row, target_col)
        
        # 2. Создаем редактор через фабрику
        options = {"choices": choices}
        editor = EditorFactory.create_editor(var_type, value, self.editor_context, options)
        
        if editor:
            # 3. Подключаем сигнал изменения значения к правильному обработчику
            if is_passport:
                editor.valueChanged.connect(lambda v, r=row: self._on_default_value_changed(r, v))
            elif is_context:
                editor.valueChanged.connect(lambda v, r=row: self._on_context_value_changed(r, v))
            else:
                editor.valueChanged.connect(lambda v, r=row: self._on_instance_value_changed(r, v))
            
            # Если редактор поддерживает список выбора (например, ChoicesEditor), обрабатываем его
            if (is_passport or is_context) and hasattr(editor, "choicesChanged"):
                editor.choicesChanged.connect(lambda c, r=row: self._on_choices_changed(r, c))

            # 4. Проверяем, нужно ли делать редактор активным
            entry = self._args_values.get(name) if not (is_passport or is_context) else None
            is_enabled = is_passport or is_context or (entry and entry.enabled)

            # Вставляем редактор в таблицу или ставим заглушку "N/A", если он отключен
            if is_enabled:
                self.table.setCellWidget(row, target_col, editor)
            else:
                item = QTableWidgetItem("N/A")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, target_col, item)

    def _get_color_for_prefix(self, prefix: Optional[str]) -> Optional[QColor]:
        """Возвращает цвет фона для заданной группы контекстных переменных на основе префикса."""
        if not prefix:
            return None
            
        if prefix not in self._prefix_to_color_map:
            color_index = len(self._prefix_to_color_map) % len(self.prefix_color_palette)
            self._prefix_to_color_map[prefix] = self.prefix_color_palette[color_index]
            
        return self._prefix_to_color_map[prefix]

    def _get_name_from_row(self, row: int) -> Optional[str]:
        """Утилита для получения имени переменной/аргумента из указанной строки таблицы."""
        col_map = {
            EditMode.CONTEXT_VARS: ParamTableColumn.CONTEXT_NAME, 
            EditMode.PASSPORT: ParamTableColumn.PASSPORT_NAME, 
            EditMode.INSTANCE: ParamTableColumn.INSTANCE_NAME
        }
        item = self.table.item(row, col_map[self.mode])
        return item.text() if item else None

    # --- Слоты для обработки изменений от UI элементов ---

    @Slot(int, bool)
    def _on_required_toggled(self, row: int, state: bool):
        """Слот: изменение флага обязательности (Passport)."""
        name = self._get_name_from_row(row)
        if name:
            self._args_meta[name].required = state
            self.data_changed.emit()

    @Slot(int, str)
    def _on_type_changed(self, row: int, new_type: str):
        """Слот: изменение типа данных переменной (Passport / Context)."""
        name = self._get_name_from_row(row)
        if not name:
            return
            
        model = self._context_vars[name] if self.mode == EditMode.CONTEXT_VARS else self._args_meta[name]
        
        # Если тип действительно изменился
        if model.type != new_type:
            # Сбрасываем старое значение
            if self.mode == EditMode.CONTEXT_VARS:
                model.value = None
            else:
                setattr(model, 'default', None)
                
            model.type = new_type
            model.choices =[] if new_type == "choice" else None
            
            # Пересоздаем виджет редактора для нового типа
            self._create_value_editor(row, name, model)
            self.data_changed.emit()

    @Slot(int, object)
    def _on_default_value_changed(self, row: int, value: Any):
        """Слот: изменение значения по умолчанию (Passport)."""
        name = self._get_name_from_row(row)
        if name:
            self._args_meta[name].default = value
            self.data_changed.emit()

    @Slot(int, object)
    def _on_context_value_changed(self, row: int, value: Any):
        """Слот: изменение значения контекстной переменной (Context Vars)."""
        name = self._get_name_from_row(row)
        if name:
            self._context_vars[name].value = value
            self.data_changed.emit()

    @Slot(int, list)
    def _on_choices_changed(self, row: int, choices: List[str]):
        """Слот: обновление списка доступных вариантов для типа 'choice'."""
        name = self._get_name_from_row(row)
        if name:
            model = self._context_vars[name] if self.mode == EditMode.CONTEXT_VARS else self._args_meta[name]
            model.choices = choices
            self.data_changed.emit()

    @Slot(int, bool)
    def _on_enable_toggled(self, row: int, state: bool):
        """Слот: включение/отключение переопределения значения (Instance)."""
        name = self._get_name_from_row(row)
        if name:
            self._args_values[name].enabled = state
            self._create_value_editor(row, name, self._args_meta[name])
            self.data_changed.emit()

    @Slot(int, object)
    def _on_instance_value_changed(self, row: int, value: Any):
        """Слот: изменение переопределенного значения (Instance)."""
        name = self._get_name_from_row(row)
        if name:
            self._args_values[name].value = value
            self.data_changed.emit()

    @Slot(int, str)
    def _on_reset_to_default(self, row: int, name: str):
        """Слот: сброс значения экземпляра до дефолтного из паспорта (Instance)."""
        meta = self._args_meta.get(name)
        if meta:
            self._args_values[name].value = meta.default
            self._create_value_editor(row, name, meta)
            self.data_changed.emit()

    @Slot(int, bool)
    def _on_readonly_changed(self, row: int, checked: bool):
        """Слот: переключение флага Read Only (Context Vars)."""
        name = self._get_name_from_row(row)
        if name:
            self._context_vars[name].read_only = checked
            self.data_changed.emit()

    # --- Вспомогательные UI методы ---

    def _center_widget(self, widget: QWidget) -> QWidget:
        """Оборачивает виджет (например, QCheckBox) в пустой контейнер для центрирования в ячейке таблицы."""
        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return cell

    def _adjust_table_columns(self):
        """Подгоняет ширину колонок в зависимости от текущего режима отображения."""
        self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader()
        
        if self.mode == EditMode.PASSPORT:
            header.setSectionResizeMode(ParamTableColumn.PASSPORT_DESCRIPTION, QHeaderView.ResizeMode.Stretch)
        elif self.mode == EditMode.INSTANCE:
            header.setSectionResizeMode(ParamTableColumn.INSTANCE_VALUE, QHeaderView.ResizeMode.Stretch)
        elif self.mode == EditMode.CONTEXT_VARS:
            header.setSectionResizeMode(ParamTableColumn.CONTEXT_NAME, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(ParamTableColumn.CONTEXT_VALUE, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(ParamTableColumn.CONTEXT_DESCRIPTION, QHeaderView.ResizeMode.Stretch)