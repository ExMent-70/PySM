# run_wf_session_open.py

# --- Блок 1: Импорты ---
# ==============================================================================
import argparse
import os
import platform
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

# Попытка импорта библиотек из экосистемы PySM
try:
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver = None


# --- Блок 2: Вспомогательные функции ---
# ==============================================================================
def open_file_os_agnostic(filepath: str) -> bool:
    """
    Открывает файл с помощью стандартного для ОС приложения.
    Эта функция остается без изменений.
    """
    # Проверяем, что путь не пустой и файл существует
    if not filepath or not Path(filepath).is_file():
        print(f"ОШИБКА: Файл не найден или путь не указан: '{filepath}'", file=sys.stderr)
        return False
        
    try:
        print(f"Запуск программы <b>Capture One</b>...\n")
        if platform.system() == 'Darwin':
            subprocess.run(['open', filepath], check=True)
        elif platform.system() == 'Windows':
            os.startfile(filepath)
        else:
            subprocess.run(['xdg-open', filepath], check=True)
        print(f"Открытие файла сессии <i>{filepath}</i>...<br>")
        return True
    except Exception as e:
        print(f"Ошибка при попытке открыть файл сессии: {e}", file=sys.stderr)
        return False


# --- Блок 3: Получение конфигурации (ПОЛНОСТЬЮ ПЕРЕРАБОТАН) ---
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver.
    """
    parser = argparse.ArgumentParser(
        description="Открывает указанный файл сессии Capture One.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Определяем один-единственный аргумент для пути к файлу
    parser.add_argument(
        "--__wf_session_file_path",
        type=str,
        help="Полный путь к файлу сессии (.cosessiondb), может содержать шаблоны."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# --- Блок 4: Основная логика (ПОЛНОСТЬЮ ПЕРЕРАБОТАН) ---
# ==============================================================================
def main():
    """
    Основная логика: получает путь к файлу и открывает его.
    """
    config = get_config()
    
    # Получаем уже полностью готовый путь из config
    session_file_to_open = config.__wf_session_file_path
      
    if open_file_os_agnostic(session_file_to_open):
        sys.exit(0)
    else:
        sys.exit(1)


# --- Блок 5: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    main()