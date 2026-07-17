# run_batch_rename.py

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
def str2bool(value) -> bool:
    """Безопасно преобразует булевые значения из PySM UI и CLI."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("yes", "true", "t", "1", "on")


def get_config() -> Namespace:
    """
    Определяет аргументы и возвращает полностью обработанную конфигурацию.
    """
    parser = argparse.ArgumentParser(
        description="Пакетно переименовывает файлы после сортировки по времени изменения.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--br_source_dir", type=str, help="Папка с файлами. Может содержать шаблоны {var}.")
    parser.add_argument("--br_include", type=str, nargs="+", default=["*.CR3"], help="Паттерны для включения файлов.")
    parser.add_argument("--br_template", type=str, default="%prefix%%index%%suffix%%ext%", help="Шаблон нового имени. Токены скрипта: %%index%%, %%ext%%, %%stem%%, %%name%%, %%prefix%%, %%suffix%%.")
    parser.add_argument("--br_prefix", type=str, default="", help="Префикс для токена %%prefix%%.")
    parser.add_argument("--br_suffix", type=str, default="", help="Суффикс для токена %%suffix%%.")
    parser.add_argument("--br_start_index", type=int, default=1, help="Первый порядковый номер.")
    parser.add_argument("--br_index_digits", type=int, default=4, help="Количество цифр в %%index%%.")
    parser.add_argument("--br_on_conflict", type=str, choices=["error", "skip", "rename"], default="error", help="Действие при конфликте имён.")
    parser.add_argument("--br_sort_method", type=str, choices=["created_time", "modified_time", "name", "none"], default="modified_time", help="Метод сортировки файлов перед переименованием.")
    parser.add_argument("--br_mode", type=str, choices=["rename", "dry_run"], default="rename", help="Режим работы: rename - переименовать, dry_run - только показать план.")
    parser.add_argument("--br_recursive", nargs="?", const="True", default="False", help="Искать файлы во вложенных папках.")

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
    br_recursive = str2bool(config.br_recursive)
    br_dry_run = config.br_mode == "dry_run"

    if not config.br_source_dir:
        tqdm.write("ERROR: Source directory (--br_source_dir) must be specified.")
        sys.exit(1)

    if config.br_start_index < 0:
        tqdm.write("ERROR: br_start_index must be greater than or equal to 0.")
        sys.exit(1)

    if config.br_index_digits < 0:
        tqdm.write("ERROR: br_index_digits must be greater than or equal to 0.")
        sys.exit(1)

    exit_code = pysm_operations.perform_batch_rename_operation(
        source_dir_str=config.br_source_dir,
        include_patterns=config.br_include,
        rename_template=config.br_template,
        start_index=config.br_start_index,
        index_digits=config.br_index_digits,
        prefix=config.br_prefix,
        suffix=config.br_suffix,
        recursive=br_recursive,
        on_conflict=config.br_on_conflict,
        dry_run=br_dry_run,
        sort_method=config.br_sort_method,
        lowercase_extension=True,
        sanitize_filename=True,
    )
    sys.exit(exit_code)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()
