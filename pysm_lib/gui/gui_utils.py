# pysm_lib/gui/gui_utils.py

import re
from typing import Dict
from ..config_manager import ConfigManager

# --- 1. БЛОК: Функция переименована в приватную ---
def _resolve_placeholders_in_text(text: str, styles: Dict[str, str]) -> str:
    """
    Внутренняя функция. Заменяет плейсхолдеры {theme.style_name}
    на значения из переданного словаря.
    """
    if not text or "{theme." not in text:
        return text

    def replacer(match):
        style_name = match.group(1)
        return styles.get(style_name, f"/* unknown style: {style_name} */")

    return re.sub(r"{theme\.([a-zA-Z0-9_]+)}", replacer, text)

# --- 2. БЛОК: Это теперь ЕДИНСТВЕННАЯ публичная функция ---
def resolve_themed_text(text: str, config_manager: ConfigManager) -> str:
    """
    Получает активную тему из ConfigManager и заменяет в тексте
    плейсхолдеры {theme.style_name} на реальные CSS-стили.
    """
    active_theme = config_manager.get_active_theme()
    styles = active_theme.get_styles_as_dict()
    return _resolve_placeholders_in_text(text, styles)