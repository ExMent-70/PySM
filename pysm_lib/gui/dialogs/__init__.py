# pysm_lib/gui/dialogs/__init__.py

# Экспортируем классы для удобного импорта из других модулей
from .script_properties_dialog import ScriptPropertiesDialog, EditMode
from .collection_passport_dialog import CollectionPassportDialog
from .settings_dialog import SettingsDialog
from ..widgets.parameter_editor_widget import ParameterEditorWidget

# --- НАЧАЛО ИЗМЕНЕНИЙ ---
# КОММЕНТАРИЙ: Явно указываем, какие имена экспортируются из этого пакета,
# чтобы линтер не считал импорты неиспользуемыми.
__all__ = [
    "ScriptPropertiesDialog",
    "EditMode",
    "CollectionPassportDialog",
    "SettingsDialog",
    "ParameterEditorWidget",
]
# --- КОНЕЦ ИЗМЕНЕНИЙ ---
