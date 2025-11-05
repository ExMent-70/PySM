# pysm_lib/gui/widgets/__init__.py

# 1. Сначала импортируем публичные классы, которые нужны "снаружи"
from .parameter_editor_widget import ParameterEditorWidget
from .context_editor_widget import ContextEditorWidget
from .path_list_editor import PathListEditor
from .editor_factory import EditorFactory
from .editor_context import EditorContext
from ._registry import BaseEditor, register_editor

# 2. Затем импортируем наш пакет с редакторами.
#    Этот импорт выполнит код в _editors/__init__.py, который, в свою очередь,
#    импортирует все модули редакторов и запустит их регистрацию в реестре.
from . import _editors

# 3. Определяем, что будет доступно при "from .widgets import *"
__all__ = [
    "ParameterEditorWidget",
    "ContextEditorWidget",
    "PathListEditor",
    "EditorFactory",
    "EditorContext",
    "BaseEditor",
    "register_editor",
]