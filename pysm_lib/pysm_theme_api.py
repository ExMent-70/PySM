# pysm_lib/pysm_theme_api.py

import sys
from typing import Dict, List, Optional, Union

# 1. Блок: Безопасные импорты
# ==============================================================================
try:
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication, QWidget
except ImportError:

    class QApplication:  # type: ignore
        """Пустышка, если PySide6 не установлен."""
        pass

    class QColor:  # type: ignore
        """Пустышка, если PySide6 не установлен."""
        def __init__(self, *args, **kwargs):
            pass

    class QWidget:  # type: ignore
        """Пустышка, если PySide6 не установлен."""
        pass

try:
    from ipymarkup.palette import Color, Palette, Rgb
except ImportError:
    Color, Palette, Rgb = None, None, None


from .theme_manager import ThemeManager
from . import pysm_context

# 2. Блок: Класс ThemeAPI
# ==============================================================================
class ThemeAPI:
    """
    Предоставляет API для доступа и применения информации о темах
    из основного приложения PyScriptManager.
    """

    def __init__(self):
        active_theme_name = pysm_context.get("pysm_active_theme_name", "default")
        self._manager = ThemeManager()
        self._manager.set_active_theme(active_theme_name)

    def _parse_css_string(self, style_str: str) -> Dict[str, str]:
        """Парсит строку CSS-свойств в словарь."""
        styles = {}
        for part in style_str.split(";"):
            if ":" in part:
                key, value = part.split(":", 1)
                styles[key.strip()] = value.strip()
        return styles

    def get_dynamic_style(self, style_name: str, default: Optional[str] = None) -> Optional[str]:
        """Получает значение одного динамического стиля из theme.toml."""
        return self._manager.get_active_theme_dynamic_styles().get(style_name, default)

    def get_all_dynamic_styles(self) -> Dict[str, str]:
        """Получает все динамические стили из активной темы в виде словаря."""
        return self._manager.get_active_theme_dynamic_styles()

    def get_parsed_style(
        self, style_name: str, default: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Получает стиль и возвращает его в виде распарсенного словаря.
        """
        style_str = self.get_dynamic_style(style_name, default)
        if style_str:
            return self._parse_css_string(style_str)
        return {}

    def get_qcolor(
        self, style_name: str, css_property: str, default: str
    ) -> "QColor":
        """Получает стиль, извлекает HEX-код и возвращает готовый объект QColor."""
        styles = self.get_parsed_style(style_name)
        hex_code = styles.get(css_property, default)
        return QColor(hex_code)

    def get_ipymarkup_color(
        self, style_name: str, defaults: Dict[str, str]
    ) -> Optional["Color"]:
        """Получает стиль и возвращает готовый объект ipymarkup.palette.Color."""
        if not all([Color, Rgb, Palette]):
            return None

        styles = self.get_parsed_style(style_name)
        bg = styles.get("background-color", defaults.get("background-color"))
        border = styles.get("border-color", defaults.get("border-color"))
        text = styles.get("color", defaults.get("color"))

        try:
            return Color("Style", background=Rgb(bg), border=Rgb(border), text=Rgb(text))
        except (ValueError, TypeError) as e:
            print(f"[PySM ThemeAPI] Ошибка создания цвета ipymarkup: {e}", file=sys.stderr)
            return Color(
                "Style",
                background=Rgb(defaults.get("background-color")),
                border=Rgb(defaults.get("border-color")),
                text=Rgb(defaults.get("color")),
            )

    def apply_theme_to_app(self, app: QApplication):
        """Применяет QSS-стили активной темы к экземпляру QApplication."""
        if "PySide6.QtWidgets" in sys.modules and isinstance(app, QApplication):
            qss_content = self._manager.get_active_theme_qss()
            if qss_content:
                try:
                    app.setStyleSheet(qss_content)
                except Exception as e:
                    print(f"[PySM ThemeAPI] Ошибка применения стилей: {e}", file=sys.stderr)
        else:
            print(
                "[PySM ThemeAPI] Предупреждение: 'app' не является валидным экземпляром QApplication.",
                file=sys.stderr,
            )


# 3. Блок: Глобальный экземпляр-синглтон
# ==============================================================================
theme_api = ThemeAPI()


# 4. Блок: Публичные функции-помощники
# ==============================================================================
def format_ipymarkup_box(text, spans, palette) -> str:
    """
    Кастомная версия format_span_box_markup из ipymarkup, исправляющая цвет текста.
    """
    from html import escape

    def order_spans(spans):
        return sorted(spans, key=lambda s: s[0])

    def span_text_sections(text, spans):
        previous = 0
        for span in spans:
            start, stop, _ = span
            yield text[previous:start], None
            yield text[start:stop], span
            previous = stop
        yield text[previous:], None

    lines = []
    spans = order_spans(spans)
    lines.append('<div class="tex2jax_ignore" style="white-space: pre-wrap">')
    for text_part, span in span_text_sections(text, spans):
        text_part = escape(text_part)
        if not span:
            lines.append(text_part)
            continue

        color = palette.get(span[2])
        lines.append(
            '<span style="'
            "padding: 2px; "
            "border-radius: 4px; "
            "border: 1px solid {border}; "
            "background: {background};"
            "color: {text_color};"
            '">'.format(
                background=color.background.value,
                border=color.border.value,
                text_color=color.text.value,
            )
        )
        lines.append(text_part)
        if span[2]:
            lines.append(
                '<span style="'
                "vertical-align: middle; "
                "margin-left: 2px; "
                "font-size: 0.7em; "
                "color: {color};"
                '">'.format(color=color.text.value)
            )
            lines.append(span[2])
            lines.append("</span>")
        lines.append("</span>")
    lines.append("</div>")
    return "".join(lines)


def set_widget_class(widget: QWidget, class_names: Union[str, List[str]]):
    """
    Устанавливает динамическое свойство 'class' для виджета и немедленно
    применяет соответствующий QSS-стиль из таблицы стилей приложения.

    Args:
        widget (QWidget): Виджет, к которому применяется класс.
        class_names (Union[str, List[str]]): Имя или список имен классов.
    """
    class_string = " ".join(class_names) if isinstance(class_names, list) else class_names
    widget.setProperty("class", class_string)

    widget.style().unpolish(widget)
    widget.style().polish(widget)