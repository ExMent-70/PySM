# debug_show_icons.py

import sys
from pathlib import Path

# --- Стандартный блок настройки путей PySM ---
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pysm_lib.pysm_context import pysm_context
    from pysm_lib.pysm_icons import icons, SVG_PATHS, THEME_KEYS
    
    IS_MANAGED_RUN = True
except ImportError as e:
    print(f"Ошибка импорта библиотек PySM: {e}")
    sys.exit(1)

def main():
    if not IS_MANAGED_RUN or not pysm_context:
        print("Запустите этот скрипт внутри PySM.")
        return

    # Заголовок
    html_parts = [
        '<div style="font-family: sans-serif; color: #2c3e50;">',
        '<h2>🎨 Галерея иконок PySM</h2>',
        '<p>Ниже представлены все иконки, доступные через <code>pysm_lib.pysm_icons</code>.</p>',
        '<table style="width: 100%; border-collapse: collapse; margin-top: 15px;">',
        '<tr style="background-color: #ecf0f1; text-align: left;">',
        '<th style="padding: 10px; border: 1px solid #bdc3c7;">Иконка (32px)</th>',
        '<th style="padding: 10px; border: 1px solid #bdc3c7;">Код вызова</th>',
        '<th style="padding: 10px; border: 1px solid #bdc3c7;">Ключ темы (Цвет)</th>',
        '<th style="padding: 10px; border: 1px solid #bdc3c7;">Категория</th>',
        '</tr>'
    ]

    # Группировка для наглядности (логическая)
    categories = {
        "Действия и UI": ["NEW", "OPEN", "SAVE", "SETTINGS", "SLIDERS", "REFRESH", "CONSOLE", "ADD", "DELETE", "ARROW_SUB"],
        "Выполнение": ["PLAY", "STOP", "NEXT"],
        "PySM Сущности": ["FOLDER_PY", "FILE_PY", "FOLDER_VIRTUAL", "INSTANCE_SET", "INSTANCE_ITEM"],
        "Файловая система": ["FOLDER", "FOLDER_OPEN", "FOLDER_ADD", "FILE", "FILE_CODE", "FILE_IMAGE", "FILE_HTML", "FILE_DB", "FILE_INDD", "FILE_ARCHIVE"],
        "Статусы": ["OK", "ERROR", "WARNING", "INFO", "LOCK"]
    }

    # Собираем все ключи, чтобы найти те, что не попали в категории
    all_keys = set(SVG_PATHS.keys())
    categorized_keys = set()

    for cat_name, keys in categories.items():
        html_parts.append(f'<tr><td colspan="4" style="background-color: #d6eaf8; padding: 5px 10px; font-weight: bold; color: #2980b9;">{cat_name}</td></tr>')
        
        for key in keys:
            if key in SVG_PATHS:
                categorized_keys.add(key)
                _add_row(html_parts, key)

    # Остальные (если забыли добавить в категории выше)
    remaining = all_keys - categorized_keys
    if remaining:
        html_parts.append('<tr><td colspan="4" style="background-color: #fae5d3; padding: 5px 10px; font-weight: bold; color: #d35400;">Остальные</td></tr>')
        for key in sorted(remaining):
            _add_row(html_parts, key)

    html_parts.append('</table></div>')
    
    # Вывод
    pysm_context.log_html("".join(html_parts))

def _add_row(html_parts, key):
    # Получаем метод динамически
    icon_method = getattr(icons, key)
    # Генерируем HTML иконки размером 32px
    icon_img = icon_method(size=32)
    theme_key = THEME_KEYS.get(key, "icon_primary")
    
    row = (
        f'<tr style="border-bottom: 1px solid #eee;">'
        f'<td style="padding: 8px; text-align: center;">{icon_img}</td>'
        f'<td style="padding: 8px; font-family: monospace; color: #e74c3c;">icons.{key}()</td>'
        f'<td style="padding: 8px; color: #7f8c8d;">{theme_key}</td>'
        f'<td style="padding: 8px; font-size: 0.9em; color: #95a5a6;">SVG</td>'
        f'</tr>'
    )
    html_parts.append(row)

if __name__ == "__main__":
    main()