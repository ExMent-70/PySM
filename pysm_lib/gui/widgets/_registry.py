# pysm_lib/gui/widgets/_registry.py

from typing import Dict, Type, Any, Optional
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal
from .editor_context import EditorContext


class BaseEditor(QWidget):
    """
    Базовый класс для всех виджетов-редакторов.
    Определяет общий конструктор и сигнал valueChanged.
    """
    valueChanged = Signal(object)

    def __init__(
        self,
        value: Any,
        context: EditorContext,
        options: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.value = value
        self.context = context
        self.options = options or {}
        self.setAutoFillBackground(False)

# Глобальный реестр для хранения пар "имя_типа -> класс_редактора"
EDITOR_REGISTRY: Dict[str, Type[BaseEditor]] = {}

def register_editor(type_name: str):
    """
    Декоратор для регистрации класса редактора для определенного типа данных.
    
    Пример использования:
    @register_editor("bool")
    class BoolEditor(BaseEditor):
        ...
    """
    def decorator(editor_class: Type[BaseEditor]) -> Type[BaseEditor]:
        if type_name in EDITOR_REGISTRY:
            # Предупреждение, если тип уже зарегистрирован
            print(f"Warning: Editor for type '{type_name}' is being overridden.")
        EDITOR_REGISTRY[type_name] = editor_class
        return editor_class
    return decorator