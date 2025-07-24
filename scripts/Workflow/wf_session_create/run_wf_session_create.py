# run_wf_raw_import.py

# --- Блок 1: Импорты ---
# ==============================================================================
import argparse
import os
import pathlib
import re
import shutil
import sys
from argparse import Namespace

# Попытка импорта библиотек из экосистемы PySM.
# Это обеспечивает универсальность скрипта.
try:
    from pysm_lib import pysm_context, pysm_operations
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    # Заглушки для автономного запуска
    IS_MANAGED_RUN = False
    pysm_context = None
    pysm_operations = None
    ConfigResolver = None
    # Если tqdm недоступен, будет использована базовая реализация
    from tqdm import tqdm


# --- Блок 2: Константы ---
# ==============================================================================
# Эта логика уникальна для данного скрипта и остается без изменений.
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
TEMPLATE_BASE_DIR_NAME = "_TEMPLATE_"
TEMPLATE_SESSION_DIR_NAME = "_TEMPLATE_SESSION_"
TEMPLATE_SESSION_FILE_NAME_BASE = "_TEMPLATE_SESSION_"
OUTPUT_SUBDIR_NAME = "Capture"
COSESSIONDB_EXT = ".cosessiondb"

INVALID_FOLDER_NAME_CHARS = r'[\<\>\:\"\/\\\|\?\*]'
RESERVED_FOLDER_NAMES = {".", "..", "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"}


# --- Блок 3: Вспомогательные функции ---
# ==============================================================================
def is_valid_foldername(name: str) -> tuple[bool, str]:
    """
    Проверяет корректность имени для создания папки.
    Функция сохранена, так как содержит специфическую бизнес-логику.
    """
    if not isinstance(name, str) or not name.strip():
        return False, "Имя не может быть пустым."
    if re.search(INVALID_FOLDER_NAME_CHARS, name):
        return False, f"Имя содержит недопустимые символы: <>:\"/\\|?*"
    if name.strip().lower() in RESERVED_FOLDER_NAMES:
        return False, f"Имя '{name}' зарезервировано системой."
    return True, ""


# --- Блок 4: Получение конфигурации ---
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver.
    Это заменяет ручной парсинг и функцию get_parameter.
    """
    parser = argparse.ArgumentParser(
        description="Создает сессию Capture One и импортирует в нее RAW-файлы.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--wf_session_name", type=str, default="{wf_session_name}", help="Имя для новой сессии (например, имя класса).")
    parser.add_argument("--wf_raw_path", type=str, default="{wf_raw_path}", help="Папка с исходными RAW-файлами для импорта.")
    parser.add_argument("--wf_session_path", type=str, default="{wf_session_path}", help="Папка, в которой будет создана новая сессия.")
    parser.add_argument("--wf_copy_threads", type=int, default=os.cpu_count() or 4, help="Количество потоков для копирования файлов.")

    # ConfigResolver сам обработает приоритеты (CLI > Context > Default)
    # и разрешит пути и шаблоны.
    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    return parser.parse_args()


# --- Блок 5: Основная логика скрипта ---
# ==============================================================================
def main():
    """
    Основной рабочий процесс скрипта.
    """
    if not IS_MANAGED_RUN:
        print("ОШИБКА: Этот скрипт предназначен для запуска в среде PyScriptManager.", file=sys.stderr)
        sys.exit(1)

    # 5.1. Получение и валидация параметров
    # --------------------------------------------------------------------------
    print("<b>Этап 1: Получение и валидация параметров</b>")
    config = get_config()

    if not all([config.wf_session_name, config.wf_raw_path, config.wf_session_path]):
        tqdm.write("ОШИБКА: Параметры 'wf_session_name', 'wf_raw_path' и 'wf_session_path' обязательны.")
        sys.exit(1)

    is_valid, reason = is_valid_foldername(config.wf_session_name)
    if not is_valid:
        tqdm.write(f"ОШИБКА: Некорректное имя сессии: {reason}")
        sys.exit(1)

    source_path = pathlib.Path(config.wf_raw_path)
    target_path = pathlib.Path(config.wf_session_path)

    print(f"Источник RAW-файлов: {source_path}")
    print(f"Корневая папка сессий Capture One: {target_path}")
    print(f"Имя новой сессии Capture One: {config.wf_session_name}")

    # 5.2. Создание структуры сессии из шаблона
    # --------------------------------------------------------------------------
    print("\n<b>Этап 2: Создание структуры сессии</b>")
    template_source_path = SCRIPT_DIR / TEMPLATE_BASE_DIR_NAME / TEMPLATE_SESSION_DIR_NAME
    if not template_source_path.is_dir():
        tqdm.write(f"ОШИБКА: Папка шаблона сессии не найдена: {template_source_path}")
        sys.exit(1)

    final_session_path = target_path / config.wf_session_name
    if final_session_path.exists():
        tqdm.write(f"ОШИБКА: Целевая папка '{final_session_path}' уже существует.")
        sys.exit(1)

    try:
        shutil.copytree(template_source_path, final_session_path)
        print(f"1. Шаблон скопирован в: {final_session_path}")

        original_db_file = final_session_path / (TEMPLATE_SESSION_FILE_NAME_BASE + COSESSIONDB_EXT)
        target_db_file = final_session_path / (config.wf_session_name + COSESSIONDB_EXT)
        original_db_file.rename(target_db_file)
        print(f"2. Файл сессии переименован в: {target_db_file.name}")

    except (OSError, IOError) as e:
        tqdm.write(f"ОШИБКА при создании структуры сессии: {e}")
        if final_session_path.exists():
            shutil.rmtree(final_session_path)
        sys.exit(1)

    print(" ", file=sys.stderr)
    pysm_context.log_link(
        url_or_path=str(final_session_path), # Передаем строку, а не объект Path
        text=f"Открыть папку сессии Capture One",
    )


    """

    # 5.3. Копирование RAW-файлов с использованием API
    # --------------------------------------------------------------------------
    print("\n<b>Этап 3: Копирование RAW-файлов</b>")
    capture_folder_path = final_session_path / OUTPUT_SUBDIR_NAME

    # Делегируем всю логику копирования функции из API.
    # Это заменяет ручную реализацию на ThreadPoolExecutor.
    exit_code = pysm_operations.perform_directory_operation(
        source_dir_str=str(source_path),
        dest_dir_str=str(capture_folder_path),
        mode="copy",
        on_conflict="rename",  # Безопасный режим по умолчанию
        threads=config.wf_copy_threads,
        copy_base_folder=False, # Копируем только содержимое
        include_patterns=["*"], # Все файлы, как в оригинальной логике
    )

    if exit_code != 0:
        tqdm.write("ОШИБКА: Произошла ошибка на этапе копирования файлов. Прерывание.")
        sys.exit(1)

    # 5.4. Сохранение результата в контекст
    # --------------------------------------------------------------------------
    #print("\n--- Этап 4: Завершение и сохранение в контекст ---")
    session_path_str = str(final_session_path)
    context_data = {
        "session_path": session_path_str,
        "capture_path": str(capture_folder_path),
        # Пример создания других путей на будущее
        "output_path": f"{session_path_str}/Output",
        "selects_path": f"{session_path_str}/Selects",
    }
    # Используем метод update для атомарной записи всех данных
    pysm_context.update(context_data)
    tqdm.write("Пути созданной сессии успешно сохранены в контекст:")
    for key, value in context_data.items():
        tqdm.write(f"- {key}: {value}")

    if IS_MANAGED_RUN:
        print(" ", file=sys.stderr)
        pysm_context.log_link(
            url_or_path=str(final_session_path),
            text="Открыть папку сессии</i>",
        )      
        pysm_context.log_link(
            url_or_path=str(capture_folder_path),
            text="Открыть папку Capture сессии</i>",
        )      
    """
    print(" ", file=sys.stderr)

    sys.exit(0)


# --- Блок 6: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    main()