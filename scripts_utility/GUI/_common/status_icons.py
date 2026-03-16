"""
Модуль содержит инициализированные иконки статусов для использования в отчетах.
Включает безопасную инициализацию на случай отсутствия библиотеки pysm_lib.
"""
try:
    from pysm_lib import theme_api
    from pysm_lib.pysm_icons import icons
except ImportError:
    theme_api = None
    icons = None

icon_size = 18

if theme_api and icons:
    # Получение стилей из темы
    style_warning = theme_api.get_parsed_style("icon_warning", default="color: #F1C40F;")
    style_error = theme_api.get_parsed_style("icon_error", default="color: #E74C3C;")

    # Инициализация иконок
    icon_warning = icons.WARNING(size=icon_size)
    icon_ok = icons.OK(size=icon_size)
    icon_error = icons.ERROR(size=icon_size)
    icon_info = icons.INFO(size=icon_size)
    icon_delete = icons.DELETE(size=icon_size)
    icon_play = icons.PLAY(size=icon_size)
    icon_save = icons.SAVE(size=icon_size)
    # Цветные версии иконки сохранения
    icon_save_warning = icons.SAVE(size=icon_size, color=style_warning.get("color"))
    icon_save_error = icons.SAVE(size=icon_size, color=style_error.get("color"))
else:
    # Заглушки (пустые строки) если библиотека недоступна
    icon_warning = ""
    icon_ok = ""
    icon_error = ""
    icon_info = ""
    icon_delete = ""
    icon_play = ""
    icon_save = ""
    icon_save_warning = ""
    icon_save_error = ""    