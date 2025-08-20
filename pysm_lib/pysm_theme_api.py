# pysm_lib/pysm_theme_api.py

import sys
from typing import Optional, Dict

# 1. Блок: Безопасный импорт QApplication
# ==============================================================================
# Это позволяет импортировать модуль даже в окружениях без GUI,
# не вызывая ошибок.
try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    class QApplication:  # type: ignore
        """Пустышка, если PySide6 не установлен."""
        pass

from .theme_manager import ThemeManager
from . import pysm_context

# 2. Блок: Класс ThemeAPI
# ==============================================================================
class ThemeAPI:
    """
    Предоставляет простой API для пользовательских скриптов для доступа и применения
    информации о темах из основного приложения PyScriptManager.
    """
    def __init__(self):
        # 1. Читаем имя активной темы из файла контекста.
        active_theme_name = pysm_context.get("pysm_active_theme_name", "default")
        
        # 2. Создаем чистый экземпляр ThemeManager внутри дочернего процесса.
        self._manager = ThemeManager()
        
        # 3. Указываем менеджеру, какую тему нужно загрузить.
        self._manager.set_active_theme(active_theme_name)

    def get_dynamic_style(self, style_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Получает значение одного динамического стиля (например, цвет, размер шрифта)
        из файла theme.toml активной темы.

        Args:
            style_name (str): Ключ стиля (например, "script_stdout").
            default (Optional[str]): Значение по умолчанию, если ключ не найден.

        Returns:
            Optional[str]: Строка со стилем CSS или значение по умолчанию.
        """
        return self._manager.get_active_theme_dynamic_styles().get(style_name, default)

    def get_all_dynamic_styles(self) -> Dict[str, str]:
        """
        Получает все динамические стили из активной темы в виде словаря.

        Returns:
            Dict[str, str]: Словарь ключей стилей и их значений CSS.
        """
        return self._manager.get_active_theme_dynamic_styles()

    def apply_theme_to_app(self, app: QApplication):
        """
        Применяет QSS-стили активной темы к экземпляру QApplication.
        Предназначено для пользовательских скриптов с собственным GUI на PySide6.

        Args:
            app (QApplication): Экземпляр QApplication пользовательского скрипта.
        """
        if "PySide6.QtWidgets" in sys.modules and isinstance(app, QApplication):
            qss_content = self._manager.get_active_theme_qss()
            if qss_content:
                try:
                    app.setStyleSheet(qss_content)
                except Exception as e:
                    print(f"[PySM ThemeAPI] Ошибка применения стилей: {e}", file=sys.stderr)
        else:
             print(f"[PySM ThemeAPI] Предупреждение: 'app' не является валидным экземпляром QApplication.", file=sys.stderr)

# 3. Блок: Глобальный экземпляр-синглтон
# ==============================================================================
# Создается один раз при импорте и обеспечивает простой доступ.
theme_api = ThemeAPI()