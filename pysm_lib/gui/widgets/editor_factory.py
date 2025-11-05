# pysm_lib/gui/widgets/editor_factory.py

from typing import Optional, Any, Dict
from PySide6.QtCore import QLocale
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QWidget

from .editor_context import EditorContext
from ._registry import EDITOR_REGISTRY
from ._editors.line_edit_editor import LineEditEditor


class EditorFactory:
    """
    Фабрика для создания виджетов-редакторов на основе типа данных.
    Использует предварительно заполненный реестр EDITOR_REGISTRY.
    """
    @staticmethod
    def create_editor(
        var_type: str,
        value: Any,
        context: EditorContext,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[QWidget]:
        
        # Передаем var_type в options на случай, если он понадобится редактору
        if options is None:
            options = {}
        options["var_type"] = var_type

        # 1. Ищем специализированный редактор в реестре
        editor_class = EDITOR_REGISTRY.get(var_type)
        
        # 2. Если не найден, определяем редактор по умолчанию
        if not editor_class:
            if var_type == "int":
                options["validator"] = QIntValidator()
                editor_class = LineEditEditor
            elif var_type == "float":
                validator = QDoubleValidator()
                validator.setLocale(QLocale(QLocale.Language.C))
                validator.setNotation(QDoubleValidator.Notation.StandardNotation)
                options["validator"] = validator
                editor_class = LineEditEditor
            else: # Для всех остальных неизвестных типов
                editor_class = LineEditEditor

        # 3. Создаем экземпляр найденного или дефолтного класса
        if editor_class:
            return editor_class(value=value, context=context, options=options)
        
        return None