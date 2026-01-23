# pysm_lib/pysm_icons.py

import base64
from typing import Optional, Dict

# Импортируем API тем
from .pysm_theme_api import theme_api

# Безопасный импорт PySide6 для UI
try:
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtCore import QByteArray
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QIcon: pass
    class QPixmap: pass

# --- Блок 1: Хранилище путей SVG ---
SVG_PATHS: Dict[str, str] = {
    # --- БАЗОВЫЕ ---
    "FOLDER": '<path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>',
    "FOLDER_OPEN": '<path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/>',
    "FILE": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/>',
    
    # --- ТИПЫ ФАЙЛОВ (ДЕТАЛИЗИРОВАННЫЕ) ---
    
    # CAPTURE ONE
    "FILE_C1": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><text x="12" y="17" font-family="Arial" font-size="10" fill="#FFF" text-anchor="middle" font-weight="bold">C1</text>',  
    
    
    # INDD (Буквы Id)
    "FILE_INDD": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><text x="12" y="17" font-family="Arial" font-size="10" fill="#FFF" text-anchor="middle" font-weight="bold">Id</text>',     

    # HTML (Крупный тег </>)
    "FILE_HTML": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><text x="12" y="16" font-family="Consolas, monospace" font-size="9" fill="#FFF" text-anchor="middle" font-weight="bold">&lt;/&gt;</text>', 
    
    # JSON / CODE (Крупные скобки { })
    "FILE_CODE": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><text x="12" y="16" font-family="Consolas, monospace" font-size="11" fill="#FFF" text-anchor="middle" font-weight="bold">{ }</text>',
    
    # PSD (Текст Ps)
    "FILE_PSD": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><text x="12" y="17" font-family="Arial" font-size="9" fill="#FFF" text-anchor="middle" font-weight="bold">Ps</text>',
    
    # TXT (Текст TXT)
    "FILE_TXT": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><text x="12" y="17" font-family="Arial" font-size="8" fill="#FFF" text-anchor="middle" font-weight="bold">TXT</text>',

    # JPG / IMAGE (Картинка)
    "FILE_IMAGE": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><path fill="#FFF" fill-opacity="0.9" d="M14 13l-3-4-2.25 3-1.75-2.25L5 13h14l-2.25-3z"/>',
    
    # DATABASE
    "FILE_DB": '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/><ellipse cx="12" cy="10" rx="5" ry="2" fill="currentColor"/><path d="M7 10v4c0 1.1 2.24 2 5 2s5-.9 5-2v-4" fill="none" stroke="currentColor" stroke-width="1.5"/>',

    # ARCHIVE (ZIP)
    "FILE_ARCHIVE": '<path d="M20 6h-8l-2-2H4c-1.1.0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-10 8H8v-2h2v2zm0-4H8V8h2v2zm2 4h-2v-2h2v2zm0-4h-2V8h2v2zm2 4h-2v-2h2v2zm0-4h-2V8h2v2z"/>',

    # --- ДЕЙСТВИЯ (КНОПКИ) ---
    "NEW": '<path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 14h-3v3h-2v-3H8v-2h3v-3h2v3h3v2zm-3-7V3.5L18.5 9H13z"/>', 
    "OPEN": '<path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/><path d="M12 10l-4 4h3v4h2v-4h3z" fill="#FFF" fill-opacity="0.7"/>',
    "SAVE": '<path d="M17 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm-5 16c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3zm3-10H5V5h10v4z"/>',
    "SLIDERS": '<path d="M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z"/>', 
    "REFRESH": '<path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>',
    "SETTINGS": '<path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c.59-.24 1.13-.57 1.62-.94l-2.39-.96c-.22-.08-.47 0-.59.22L5.09 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.04.24.24.41.48.41h3.84c.24 0 .43-.17.47-.41l.36-2.54c.59-.24 1.13-.57 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>',
    "CONSOLE": '<path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zM4 18V6h16v12H4z"/><path d="M7.5 13l-1.41-1.41L8.67 9l-2.59-2.59L7.5 5l4 4-4 4zm5.5-1h5v2h-5z"/>',
    
    # --- ОКНА / ПРИЛОЖЕНИЕ ---
    "EXIT": '<path d="M10.09 15.59L11.5 17l5-5-5-5-1.41 1.41L12.67 11H3v2h9.67l-2.58 2.59zM19 3H5c-1.11 0-2 .9-2 2v4h2V5h14v14H5v-4H3v4c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"/>',
    "CLOSE": '<path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>',    
    
    # --- PYTHON / ВИРТУАЛЬНЫЕ ---
    "FOLDER_PY": '<path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/><path d="M16.5 12l-1.5 1.5-2.5-2.5 2.5-2.5 1.5 1.5-1 1 1 1zm-9 0l1.5-1.5-1-1 1-1.5-2.5 2.5 2.5 2.5-1.5-1.5 1-1-1-1z" fill-opacity="0.8"/>', 
    "FILE_PY": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><path d="M11.5 11c1.1 0 2 .9 2 2h-4c0-1.1.9-2 2-2zm0 6c-1.1 0-2-.9-2-2h4c0 1.1-.9 2-2 2z" fill-opacity="0.8"/>', 
    "FOLDER_VIRTUAL": '<path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-2.06 11L15 15.28 12.06 17l.78-3.33-2.59-2.24 3.41-.29L15 8l1.34 3.14 3.41.29-2.59 2.24.78 3.33z"/>',
    "INSTANCE_SET": '<path d="M2 6H0v5h.01L0 20c0 1.1.9 2 2 2h18v-2H2V6zm20-2h-8l-2-2H6c-1.1 0-1.99.9-1.99 2L4 16c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 12H6V6h14v10z"/>',
    "INSTANCE_ITEM": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><path d="M12 12c-1.7 0-3 1.3-3 3s1.3 3 3 3 3-1.3 3-3-1.3-3-3-3zm0 4.5c-.8 0-1.5-.7-1.5-1.5s.7-1.5 1.5-1.5 1.5.7 1.5 1.5-.7 1.5-1.5 1.5z" opacity="0.8"/><circle cx="12" cy="15" r="3.5" stroke="currentColor" stroke-width="1.5" fill="none" stroke-dasharray="2 1"/>',

    # --- УПРАВЛЕНИЕ ВЫПОЛНЕНИЕМ ---
    "PLAY": '<path d="M8 5v14l11-7z"/>',
    "STOP": '<path d="M6 6h12v12H6z"/>',
    "NEXT": '<path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/>',
    
    # --- СТАТУСЫ ---
    "OK": '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>',
    "ERROR": '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>',
    "WARNING": '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>',
    "INFO": '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>',
    "LOCK": '<path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z"/>',
    "ADD": '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11h-4v4h-2v-4H7v-2h4V7h2v4h4v2z"/>',
    "DELETE": '<path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>',
    "ARROW_SUB": '<path d="M19 15l-6 6-1.42-1.42L15.17 16H4V4h2v10h9.17l-3.59-3.58L13 9l6 6z"/>',
}

# Маппинг имен методов на ключи цветов в theme.toml
THEME_KEYS = {
    # Actions
    "NEW": "icon_primary", "OPEN": "icon_folder", "SAVE": "icon_primary",
    "SLIDERS": "icon_primary", "REFRESH": "icon_info", "SETTINGS": "icon_primary", "CONSOLE": "icon_archive",

    # Window Actions
    "EXIT": "icon_danger",
    "CLOSE": "icon_danger",    
    
    # Python / Virtual
    "FOLDER_PY": "icon_folder", 
    "FILE_PY": "icon_file",
    "FOLDER_VIRTUAL": "icon_info", "INSTANCE_SET": "icon_folder", "INSTANCE_ITEM": "icon_file",

    # Execution
    "PLAY": "icon_success", "STOP": "icon_danger", "NEXT": "icon_warning",

    # Standard
    "FOLDER": "icon_folder", "FOLDER_OPEN": "icon_folder",
    "FILE": "icon_file", 
    "FILE_CODE": "icon_code", 
    "FILE_ARCHIVE": "icon_archive",
    "FILE_IMAGE": "icon_info", 
    "FILE_HTML": "icon_folder", 
    
    # --- НОВЫЕ ТИПЫ ФАЙЛОВ ---
    "FILE_INDD": "icon_adobe", 
    "FILE_PSD": "icon_info",   
    "FILE_TXT": "icon_file",  
    "FILE_C1": "icon_folder",    
    # -------------------------
    
    "FILE_DB": "icon_primary",
    "OK": "icon_success", "ADD": "icon_success",
    "ERROR": "icon_danger", "DELETE": "icon_danger",
    "WARNING": "icon_warning",
    "INFO": "icon_info",
    "LOCK": "icon_primary", "ARROW_SUB": "icon_primary",
}

# Цвета по умолчанию (Fallback)
DEFAULT_PALETTE = {
    "icon_primary": "#7F8C8D", "icon_folder": "#F39C12", "icon_file": "#3498DB",
    "icon_success": "#27AE60", "icon_warning": "#F1C40F", "icon_danger": "#E74C3C",
    "icon_info": "#2980B9", "icon_adobe": "#D9005F", "icon_code": "#2ECC71", "icon_archive": "#34495E",
}


# --- Блок 2: Основной класс API ---
class PysmIcons:
    """
    Универсальный провайдер иконок для PySM.
    Поддерживает вывод в HTML (для логов) и QIcon (для UI).
    """

    def _get_theme_color(self, theme_key: str) -> str:
        """Получает HEX-цвет иконки из активной темы PySM."""
        style_dict = theme_api.get_parsed_style(theme_key)
        color_hex = style_dict.get("color")
        return color_hex if color_hex else DEFAULT_PALETTE.get(theme_key, "#000000")

    def _generate_svg_xml(self, name: str, size: int, color: Optional[str] = None) -> str:
        """Генерирует чистую XML строку SVG."""
        svg_path = SVG_PATHS.get(name, "")
        if not svg_path:
            return ""

        theme_key = THEME_KEYS.get(name, "icon_primary")
        fill_color = color if color else self._get_theme_color(theme_key)

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'width="{size}" height="{size}" fill="{fill_color}">{svg_path}</svg>'
        )

    # --- Публичный API для HTML (Log Report) ---
    def _render_html(self, name: str, size: int = 20, **kwargs) -> str:
        """Возвращает тег <img> с Base64."""
        color = kwargs.get('color')
        raw_svg = self._generate_svg_xml(name, size, color)
        
        if not raw_svg:
            return ""

        b64 = base64.b64encode(raw_svg.encode('utf-8')).decode('utf-8')
        return (
            f'<img src="data:image/svg+xml;base64,{b64}" '
            f'width="{size}" height="{size}" '
            f'style="vertical-align: middle; border: none;">'
        )

    # --- Публичный API для UI (PySide6) ---
    def get_qicon(self, name: str, size: int = 24, color: Optional[str] = None) -> "QIcon":
        """
        Возвращает объект QIcon для использования в интерфейсе.
        """
        if not HAS_QT:
            raise ImportError("PySide6 не установлен. Невозможно создать QIcon.")

        raw_svg = self._generate_svg_xml(name, size, color)
        if not raw_svg:
            return QIcon()

        data = QByteArray(raw_svg.encode('utf-8'))
        pixmap = QPixmap()
        pixmap.loadFromData(data, "SVG")
        return QIcon(pixmap)

    # --- Методы-хелперы (Backward Compatibility) ---
    def __getattr__(self, name: str):
        if name in SVG_PATHS:
            def wrapper(size: int = 20, **kwargs):
                return self._render_html(name, size, **kwargs)
            return wrapper
        raise AttributeError(f"'PysmIcons' object has no attribute '{name}'")


icons = PysmIcons()