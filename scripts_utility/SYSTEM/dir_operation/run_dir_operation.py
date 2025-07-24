# run_dir_operation.py

# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import os
import sys
from argparse import Namespace

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib import pysm_operations
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_operations = None
    ConfigResolver = None


# 2. БЛОК: Получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы и возвращает полностью обработанную конфигурацию.
    """
    parser = argparse.ArgumentParser(
        description="Копирует или перемещает файлы/папки, поддерживая многопоточность и шаблоны путей.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Аргументы остаются без изменений
    parser.add_argument("-s", "--do_source_dir", type=str, help="Исходная директория. Может содержать шаблоны вида {var}.")
    parser.add_argument("-d", "--do_dest_dir", type=str, help="Директория назначения. Может содержать шаблоны вида {var}.")
    parser.add_argument("-m", "--do_mode", type=str, choices=["copy", "move"], default="copy", help="Режим работы: 'copy' или 'move'.")
    parser.add_argument("-c", "--do_on_conflict", type=str, choices=["skip", "overwrite", "rename"], default="skip", help="Действие при конфликте имен.")
    parser.add_argument("-t", "--all_threads", type=int, default=os.cpu_count() or 4, help="Количество потоков для выполнения.")
    parser.add_argument("--do_copy_base_folder", action=argparse.BooleanOptionalAction, default=False, help="Копировать корневую папку источника в папку назначения.")
    parser.add_argument("-i", "--do_include", type=str, nargs='+', default=["*"], help="Паттерны для включения файлов (glob-синтаксис).")

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        # Вся магия разрешения путей и шаблонов теперь здесь:
        return resolver.resolve_all()
    else:
        # Для автономного режима также используем resolver, который обработает пути
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()


# 3. БЛОК: Основная логика скрипта
# ==============================================================================
def main():
    """
    Главная функция: получает конфигурацию, валидирует ее и вызывает API.
    """
    if not IS_MANAGED_RUN or not pysm_operations:
        print("ERROR: This script requires the PySM environment.", file=sys.stderr)
        sys.exit(1)

    config = get_config()

    # Валидация, специфичная для логики этого скрипта
    if not config.do_source_dir or not config.do_dest_dir:
        print(
            "ERROR: Source and Destination directories are required. Check your arguments or context variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Вызываем основную функцию API
    exit_code = pysm_operations.perform_directory_operation(
        source_dir_str=config.do_source_dir,
        dest_dir_str=config.do_dest_dir,
        mode=config.do_mode,
        on_conflict=config.do_on_conflict,
        threads=config.all_threads,
        copy_base_folder=config.do_copy_base_folder,
        include_patterns=config.do_include,
    )
    sys.exit(exit_code)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()