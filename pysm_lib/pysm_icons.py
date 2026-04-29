# pysm_lib/pysm_icons.py

import base64
from typing import Optional, Dict, List

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


    # --- ИЗБРАННОЕ / QUICK ACCESS ---
    "STAR": '<path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/>',
    "ROCKET": '<path d="M13.13 22.19L11.5 18.36C13.07 17.78 14.54 17 15.9 16.09L13.13 22.19zM5.64 12.5l-3.83-1.63 6.1-2.77C7 9.46 7.78 10.93 8.36 12.5L5.64 12.5zM21.03 2.97a9.38 9.38 0 0 0-6.19-2.91c-4.47 0-8.64 2.39-10.97 6.34-.1.17-.18.35-.25.53L1.5 8.13c-.4.17-.65.57-.6 1.01.05.44.38.79.82.88l3.65.73c.53 2.19 1.5 4.22 2.82 5.96l-1.92 4.13c-.19.42-.08.92.27 1.22.21.18.47.28.74.28.16 0 .32-.04.47-.11l4.58-2.16c1.86 1.25 3.99 2.05 6.22 2.31.13.02.26.02.39.02 4.54 0 8.61-2.82 10.23-7.09.18-.46.3-1.02.3-1.6V2.97zM20 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2z"/>',
    "LIGHTNING": '<path d="M7 2v11h3v9l7-12h-4l4-8z"/>',
    "WAND": '<path d="M7.5 5.6L10 7 8.6 4.5 10 2 7.5 3.4 5 2l1.4 2.5L5 7zm12 9.8L17 14l1.4 2.5L17 19l2.5-1.4L22 19l-1.4-2.5L22 14zM22 2l-2.5 1.4L17 2l1.4 2.5L17 7l2.5-1.4L22 7l-1.4-2.5zm-7.63 5.29c-.39-.39-1.02-.39-1.41 0L1.29 18.96c-.39.39-.39 1.02 0 1.41l2.83 2.83c.39.39 1.02.39 1.41 0l11.66-11.66c.39-.39.39-1.02 0-1.41l-2.82-2.84z"/>',
    "WRENCH": '<path d="M22.7 19l-9.1-9.1c.9-2.3.4-5-1.5-6.9-2-2-5-2.4-7.4-1.3L9 6 6 9 1.6 4.7C.4 7.1.9 10.1 2.9 12.1c1.9 1.9 4.6 2.4 6.9 1.5l9.1 9.1c.4.4 1 .4 1.4 0l2.3-2.3c.5-.4.5-1.1.1-1.4z"/>',
    "TARGET": '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm0 10c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4zm0-6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>',
    "BUG": '<path d="M20 8h-2.81c-.45-.78-1.07-1.45-1.82-1.96L17 4.41 15.59 3l-2.17 2.17C12.96 5.06 12.49 5 12 5c-.49 0-.96.06-1.41.17L8.41 3 7 4.41l1.62 1.63C7.88 6.55 7.26 7.22 6.81 8H4v2h2.09c-.05.33-.09.66-.09 1v1H4v2h2v1c0 .34.04.67.09 1H4v2h2.81c1.04 1.79 2.97 3 5.19 3s4.15-1.21 5.19-3H20v-2h-2.09c.05-.33.09-.66.09-1v-1h2v-2h-2v-1c0-.34-.04-.67-.09-1H20V8zm-6 8h-4v-2h4v2zm0-4h-4v-2h4v2z"/>',
    
    # --- НОВЫЕ ИКОНКИ (ПОЛЬЗОВАТЕЛЬСКИЕ) ---
    "REPORT": '<path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm2 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>',
    "TABLE": '<path d="M3 3h18v18H3V3zm2 2v4h4V5H5zm6 0v4h4V5h-4zm6 0v4h4V5h-4zM5 11v4h4v-4H5zm6 0v4h4v-4h-4zm6 0v4h4v-4h-4zM5 17v4h4v-4H5zm6 0v4h4v-4h-4zm6 0v4h4v-4h-4z"/>',
    "LIST": '<path d="M4 6h2v2H4zm0 5h2v2H4zm0 5h2v2H4zm4-10h12v2H8zm0 5h12v2H8zm0 5h12v2H8z"/>',
    "CAMERA": '<path d="M12 12c1.65 0 3-1.35 3-3s-1.35-3-3-3-3 1.35-3 3 1.35 3 3 3zm9-6h-3.17L16 4h-8l-1.83 2H3c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-9 13c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"/>',
    "FILE_JPG": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><text x="12" y="17" font-family="Arial" font-size="8" fill="#FFF" text-anchor="middle" font-weight="bold">JPG</text>',
    "COPY_JPG": '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1z"/><path d="M19 5H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2z"/><text x="13.5" y="17" font-family="Arial" font-size="6" fill="#FFF" text-anchor="middle" font-weight="bold">JPG</text>',
    "COPY_PSD": '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1z"/><path d="M19 5H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2z"/><text x="13.5" y="17" font-family="Arial" font-size="6" fill="#FFF" text-anchor="middle" font-weight="bold">PSD</text>',
    "SELECT_FILES": '<path d="M13 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><path fill="#FFF" d="M10.9 16.1l-2.8-2.8 1.4-1.4 1.4 1.4 4.6-4.6 1.4 1.4-6 6z"/>',
    "APP_PHOTOSHOP": '<rect x="2" y="2" width="20" height="20" rx="4" ry="4"/><text x="12" y="16" font-family="Arial" font-size="11" fill="#FFF" text-anchor="middle" font-weight="bold">Ps</text>',    

    # --- ПРОИЗВОДНЫЕ ОТ C1, INDD И НОВЫЙ ФОРМАТ XMP ---
    "APP_C1": '<rect x="2" y="2" width="20" height="20" rx="4" ry="4"/><text x="12" y="16" font-family="Arial" font-size="11" fill="#FFF" text-anchor="middle" font-weight="bold">C1</text>',
    "COPY_C1": '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1z"/><path d="M19 5H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2z"/><text x="13.5" y="16" font-family="Arial" font-size="9" fill="#FFF" text-anchor="middle" font-weight="bold">C1</text>',
    
    "APP_INDD": '<rect x="2" y="2" width="20" height="20" rx="4" ry="4"/><text x="12" y="16" font-family="Arial" font-size="11" fill="#FFF" text-anchor="middle" font-weight="bold">Id</text>',
    "COPY_INDD": '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1z"/><path d="M19 5H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2z"/><text x="13.5" y="16" font-family="Arial" font-size="9" fill="#FFF" text-anchor="middle" font-weight="bold">Id</text>',

    
    "FILE_XMP": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><text x="12" y="17" font-family="Arial" font-size="8" fill="#FFF" text-anchor="middle" font-weight="bold">XMP</text>',
    "APP_XMP": '<rect x="2" y="2" width="20" height="20" rx="4" ry="4"/><text x="12" y="16" font-family="Arial" font-size="9" fill="#FFF" text-anchor="middle" font-weight="bold">XMP</text>',
    "COPY_XMP": '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1z"/><path d="M19 5H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2z"/><text x="13.5" y="17" font-family="Arial" font-size="6" fill="#FFF" text-anchor="middle" font-weight="bold">XMP</text>',

    # --- ТИПЫ ФОТОГРАФИЙ ---
    "PHOTO_PORTRAIT": '<path d="M19 5v14H5V5h14m0-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 9c1.65 0 3-1.35 3-3s-1.35-3-3-3-3 1.35-3 3 1.35 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-1.5c0-2.33-4.67-3.5-7-3.5z"/>',
    "PHOTO_GROUP": '<path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zm-9.5-9.5c.83 0 1.5-.67 1.5-1.5S10.33 6.5 9.5 6.5 8 7.17 8 8s.67 1.5 1.5 1.5zm5 0c.83 0 1.5-.67 1.5-1.5S15.33 6.5 14.5 6.5 13 7.17 13 8s.67 1.5 1.5 1.5zm-5 1.5c-1.83 0-5.5.92-5.5 2.75V16h11v-2.25c0-1.83-3.67-2.75-5.5-2.75zm5 0c-.26 0-.54.02-.82.07 1.01.7 1.7 1.64 1.7 2.85V16h3v-2.25c0-1.83-3.67-2.75-3.88-2.75z"/>',
    "PHOTO_NATURE": '<path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>',

    # --- РЕДАКТИРОВАНИЕ ФОТОГРАФИЙ (С КАРАНДАШОМ В УГЛУ) ---
    "PHOTO_PORTRAIT_EDIT": '<path d="M3 3h4v2H3z M10 3h4v2h-4z M17 3h4v2h-4z M3 19h4v2H3z M10 19h4v2h-4z M17 19h4v2h-4z M3 7h2v4H3z M3 13h2v4H3z M19 7h2v4h-2z M19 13h2v4h-2z M12 12c1.65 0 3-1.35 3-3s-1.35-3-3-3-3 1.35-3 3 1.35 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-1.5c0-2.33-4.67-3.5-7-3.5z"/>',
    "PHOTO_GROUP_EDIT": '<path d="M3 3h4v2H3z M10 3h4v2h-4z M17 3h4v2h-4z M3 19h4v2H3z M10 19h4v2h-4z M17 19h4v2h-4z M3 7h2v4H3z M3 13h2v4H3z M19 7h2v4h-2z M19 13h2v4h-2z M9.5 7c.83 0 1.5.67 1.5 1.5S10.33 10 9.5 10 8 9.33 8 8.5 8.67 7 9.5 7zm5 0c.83 0 1.5.67 1.5 1.5S15.33 10 14.5 10 13 9.33 13 8.5 13.67 7 14.5 7zm-5 4.5c-1.83 0-5.5.92-5.5 2.75V17h11v-2.25c0-1.83-3.67-2.75-5.5-2.75zm5 0c-.26 0-.54.02-.82.07 1.01.7 1.7 1.64 1.7 2.85V17h3v-2.25c0-1.83-3.67-2.75-3.88-2.75z"/>',
    "PHOTO_NATURE_EDIT": '<path d="M3 3h4v2H3z M10 3h4v2h-4z M17 3h4v2h-4z M3 19h4v2H3z M10 19h4v2h-4z M17 19h4v2h-4z M3 7h2v4H3z M3 13h2v4H3z M19 7h2v4h-2z M19 13h2v4h-2z M8.5 14.5l2.5 3.01L14.5 13l4.5 6H5l3.5-4.5z"/>',

    # --- РАБОТА С ПЕРЕМЕННЫМИ (Индикация {x} внутри блока) ---
    "VAR_COPY": '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1z"/><path d="M19 5H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2z"/><text x="13.5" y="16" font-family="Consolas, monospace" font-size="9" fill="#FFF" text-anchor="middle" font-weight="bold">{x}</text>',
    "VAR_SET": '<text x="12" y="16.5" font-family="Consolas, monospace" font-size="14" text-anchor="middle" font-weight="bold">{x}</text>',
    "VAR_REMOVE": '<circle cx="12" cy="12" r="10"/><text x="12" y="16.5" font-family="Consolas, monospace" font-size="12" fill="#FFF" text-anchor="middle" font-weight="bold">{x}</text>',

    # --- ЛОГИКА / УСЛОВИЯ (Ромб IF) ---
    "LOGIC_IF": '<path d="M12 2L2 12l10 10 10-10L12 2z"/><text x="12" y="15.5" font-family="Arial" font-size="10" fill="#FFF" text-anchor="middle" font-weight="bold">IF</text>',
    
    # --- ФОРМАТ RAW И КОНВЕРТАЦИЯ ---
    "FILE_RAW": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7zm2 5V3.5L18.5 9H15z"/><text x="12" y="17" font-family="Arial" font-size="7" fill="#FFF" text-anchor="middle" font-weight="bold">RAW</text>',
    "COPY_RAW": '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1z"/><path d="M19 5H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2z"/><text x="13.5" y="16" font-family="Arial" font-size="6.5" fill="#FFF" text-anchor="middle" font-weight="bold">RAW</text>',
    "CONVER_RAW": '<rect x="1" y="5" width="9" height="14" rx="2"/><text x="5.5" y="15.5" font-family="Arial" font-size="10" fill="#FFF" text-anchor="middle" font-weight="bold">R</text><polygon points="11,10 13.5,12 11,14"/><rect x="14" y="5" width="9" height="14" rx="2"/><text x="18.5" y="15.5" font-family="Arial" font-size="10" fill="#FFF" text-anchor="middle" font-weight="bold">J</text>',    

    # --- КОМАНДА МАСКИ (Классический значок маски) ---
    "FILE_MASK": '<path fill-rule="evenodd" d="M3 5h18v14H3V5zm9 12c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5z"/>',   

    # --- СТАТУСЫ И АТРИБУТЫ ЛИЦ ---
    "GENDER_MALE": '<path d="M15 2v2h2.59l-4.59 4.59C11.83 7.55 10.46 7 9 7 5.13 7 2 10.13 2 14s3.13 7 7 7 7-3.13 7-7c0-1.46-.55-2.83-1.59-3.99L19 6.41V9h2V2h-7zM9 19c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"/>',
    "GENDER_FEMALE": '<path d="M12 2C8.69 2 6 4.69 6 8c0 2.97 2.16 5.44 5 5.92V16H8v2h3v4h2v-4h3v-2h-3v-2.08c2.84-.48 5-2.95 5-5.92 0-3.31-2.69-6-6-6zm0 10c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4z"/>',    
    "EYE_CLOSED": '<path d="M12 7c2.76 0 5 2.24 5 5 0 .65-.13 1.26-.36 1.83l2.92 2.92c1.51-1.26 2.7-2.89 3.43-4.75-1.73-4.39-6-7.5-11-7.5-1.4 0-2.74.25-3.98.7l2.16 2.16C10.74 7.13 11.35 7 12 7zM2 4.27l2.28 2.28.46.46C3.08 8.3 1.78 10.02 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l.42.42L19.73 22 21 20.73 3.27 3 2 4.27zM7.53 9.8l1.55 1.55c-.05.21-.08.43-.08.65 0 1.66 1.34 3 3 3 .22 0 .44-.03.65-.08l1.55 1.55c-.67.33-1.41.53-2.2.53-2.76 0-5-2.24-5-5 0-.79.2-1.53.53-2.2zm4.31-.78l3.15 3.15.02-.16c0-1.66-1.34-3-3-3l-.17.01z"/>',
    "MOUTH_OPEN": '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-2.5-8.5c.83 0 1.5-.67 1.5-1.5S10.33 8.5 9.5 8.5 8 9.17 8 10s.67 1.5 1.5 1.5zm5 0c.83 0 1.5-.67 1.5-1.5S15.33 8.5 14.5 8.5 13 9.17 13 10s.67 1.5 1.5 1.5zm-2.5 2.5c-1.5 0-2.5 1-2.5 2.5S10.5 19 12 19s2.5-1 2.5-2.5S13.5 14 12 14z"/>',    
    
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
    
    # Favorites
    "STAR": "icon_primary", "ROCKET": "icon_primary", "LIGHTNING": "icon_primary",
    "WAND": "icon_primary", "WRENCH": "icon_primary", "TARGET": "icon_primary", "BUG": "icon_primary",   

    # --- НОВЫЕ ИКОНКИ ---
    "REPORT": "icon_primary",
    "TABLE": "icon_primary",
    "LIST": "icon_primary",
    "CAMERA": "icon_info",
    "FILE_JPG": "icon_info",
    "COPY_JPG": "icon_file",
    "COPY_PSD": "icon_info",
    "SELECT_FILES": "icon_success",
    "APP_PHOTOSHOP": "icon_info", # Используем синий цвет (как для PSD)
    
    # App & Copies Derivatives
    "APP_C1": "icon_folder", "COPY_C1": "icon_folder",
    "APP_INDD": "icon_adobe", "COPY_INDD": "icon_adobe",
    "FILE_XMP": "icon_code", "APP_XMP": "icon_code", "COPY_XMP": "icon_code",

    # Photos
    "PHOTO_PORTRAIT": "icon_info", "PHOTO_GROUP": "icon_info", "PHOTO_NATURE": "icon_info",
    "PHOTO_PORTRAIT_EDIT": "icon_primary", "PHOTO_GROUP_EDIT": "icon_primary", "PHOTO_NATURE_EDIT": "icon_primary",

    # Variables Actions
    "VAR_COPY": "icon_primary", 
    "VAR_SET": "icon_success", 
    "VAR_REMOVE": "icon_danger",

    # Logic
    "LOGIC_IF": "icon_warning",

    # RAW & Mask
    "FILE_RAW": "icon_info", 
    "COPY_RAW": "icon_info", 
    "CONVER_RAW": "icon_success", # Зеленый цвет подчеркивает "действие/успех" конвертации
    "FILE_MASK": "icon_primary",  # Синий/основной цвет как для стандартных команд  

    # Attributes
    "GENDER_MALE": "icon_info",    # Синий оттенок
    "GENDER_FEMALE": "icon_adobe", # Розово-красный оттенок    
    "EYE_CLOSED": "icon_danger",
    "MOUTH_OPEN": "icon_warning",    
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

# Категоризация иконок для отображения в диалоге выбора
ICON_CATEGORIES: Dict[str, List[str]] = {
    "system":["FOLDER", "FOLDER_OPEN", "FILE", "NEW", "OPEN", "SAVE", "SLIDERS", "REFRESH", "SETTINGS", "CONSOLE", "EXIT", "CLOSE"],
    "files":[
        "FILE_PY", "FILE_CODE", "FILE_TXT", "FILE_HTML", "FILE_IMAGE", "FILE_DB", "FILE_ARCHIVE", 
        "FILE_JPG", "FILE_PSD", "FILE_INDD", "FILE_C1", "REPORT", "TABLE", "LIST",
        "FILE_XMP", "PHOTO_PORTRAIT", "PHOTO_GROUP", "PHOTO_NATURE",
        "FILE_RAW", "FILE_MASK"  # <--- Добавлены сюда
    ],
    "actions":[
        "PLAY", "STOP", "NEXT", "SELECT_FILES", "COPY_JPG", "COPY_PSD", "APP_PHOTOSHOP", "CAMERA", "ADD", "DELETE",
        "APP_C1", "COPY_C1", "APP_INDD", "COPY_INDD", "APP_XMP", "COPY_XMP",
        "PHOTO_PORTRAIT_EDIT", "PHOTO_GROUP_EDIT", "PHOTO_NATURE_EDIT",
        "VAR_COPY", "VAR_SET", "VAR_REMOVE", "LOGIC_IF",
        "COPY_RAW", "CONVER_RAW"
    ],
    "statuses":["OK", "ERROR", "WARNING", "INFO", "LOCK", "ARROW_SUB", "TARGET", "BUG"],
    "pysm":["FOLDER_PY", "FOLDER_VIRTUAL", "INSTANCE_SET", "INSTANCE_ITEM", "STAR", "ROCKET", "LIGHTNING", "WAND", "WRENCH"],
    "attributes":["GENDER_MALE", "GENDER_FEMALE", "EYE_CLOSED", "MOUTH_OPEN"]    
}