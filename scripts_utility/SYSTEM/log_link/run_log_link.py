# run_log_link_demo.py

# --- Блок 1: Импорты ---
# ==============================================================================
import argparse
import sys
from argparse import Namespace

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    # Заглушки для автономного запуска
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg)
    tqdm = TqdmWriteMock


# --- Блок 2: Получение конфигурации ---
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы для вывода гиперссылки.
    """
    parser = argparse.ArgumentParser(
        description="Демонстрирует вывод кликабельной гиперссылки в консоль PyScriptManager.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Определяем аргументы для управления выводом ссылки
    parser.add_argument("--link_href", type=str, help="URL или локальный путь, на который будет указывать ссылка.")
    parser.add_argument("--link_text", type=str, help="Видимый текст ссылки. Если не указан, используется сам URL/путь.")
    parser.add_argument("--link_align", type=str, default="left", choices=["left", "center", "right"], help="Выравнивание ссылки.")
    parser.add_argument("--link_margin", type=int, default=5, help="Вертикальный отступ (сверху и снизу) в пикселях.")

    if IS_MANAGED_RUN:
        # Принудительно указываем ConfigResolver, что аргумент 'link_href'
        # всегда должен обрабатываться как путь.
        resolver = ConfigResolver(parser, force_path_args=["link_href"])
        return resolver.resolve_all()
    
    # Для автономного запуска обработка путей не так критична
    return parser.parse_args()


# --- Блок 3: Основная логика скрипта ---
# ==============================================================================
def main():
    """
    Основной рабочий процесс скрипта.
    """
    if not IS_MANAGED_RUN:
        print("ОШИБКА: Этот скрипт предназначен для запуска в среде PyScriptManager.", file=sys.stderr)
        sys.exit(1)

    config = get_config()

    # Валидация обязательного параметра
    if not config.link_href:
        tqdm.write("ОШИБКА: Параметр 'link_href' является обязательным.")
        sys.exit(1)

    # Вызов API-функции для отображения ссылки в консоли
    pysm_context.log_link(
        url_or_path=config.link_href,
        text=config.link_text,
        align=config.link_align,
        margin=config.link_margin,
    )
    sys.exit(0)


# --- Блок 4: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    main()