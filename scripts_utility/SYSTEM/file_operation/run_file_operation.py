# run_file_operation.py

# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import sys
from argparse import Namespace

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib import pysm_operations
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_operations = None
    ConfigResolver = None
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg)
    tqdm = TqdmWriteMock


# 2. БЛОК: Получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы и возвращает полностью обработанную конфигурацию.
    """
    parser = argparse.ArgumentParser(
        description="Выполняет одиночные операции с файлами (копирование, перемещение, удаление, переименование).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Аргументы остаются без изменений
    parser.add_argument("--fo_operation", type=str, default=None, choices=['copy', 'move', 'rename', 'delete'], help="Тип выполняемой операции с файлом.")
    parser.add_argument("--fo_source_path", type=str, help="Исходный файл. Может содержать шаблоны {var}.")
    parser.add_argument("--fo_destination_path", type=str, help="Путь назначения. Может содержать шаблоны {var}.")
    parser.add_argument("--fo_overwrite", action='store_true', help="Разрешить перезапись, если файл назначения уже существует.")
    parser.add_argument("--fo_create_parents", action='store_true', help="Создавать родительские папки для пути назначения.")

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
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
    op = config.fo_operation

    # Программная валидация, специфичная для этого скрипта
    if not op:
        tqdm.write("ERROR: Operation type (--fo_operation) must be specified.")
        sys.exit(1)
    if op in ['copy', 'move', 'rename'] and (not config.fo_source_path or not config.fo_destination_path):
        tqdm.write(f"ERROR: Для операции '{op}' требуются и fo_source_path, и fo_destination_path.")
        sys.exit(1)
    if op == 'delete' and not config.fo_source_path:
        tqdm.write(f"ERROR: Для операции '{op}' требуется fo_source_path.")
        sys.exit(1)

    # Вызов функции API
    exit_code = pysm_operations.perform_file_operation(
        operation=config.fo_operation,
        source_path_str=config.fo_source_path,
        destination_path_str=config.fo_destination_path,
        overwrite=config.fo_overwrite,
        create_parents=config.fo_create_parents,
    )
    sys.exit(exit_code)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()