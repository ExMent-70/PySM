# pysm_lib/gui/widgets/editor_context.py

from dataclasses import dataclass
from typing import List, Optional, Callable

from ...models import ScriptSetEntryModel, ScriptInfoModel
from ...theme_manager import ThemeManager
from ...locale_manager import LocaleManager


@dataclass
class EditorContext:
    """
    Контейнер для зависимостей, передаваемых в редакторы параметров.
    """
    theme_manager: ThemeManager
    locale_manager: LocaleManager
    get_script_info_func: Callable[[str], Optional[ScriptInfoModel]]
    script_entries: List[ScriptSetEntryModel]