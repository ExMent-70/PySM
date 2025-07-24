# run_wf_raw_analysis.py

# --- Блок 1: Импорты и настройка окружения ---
# ==============================================================================
print(f"<b>ЗАПУСК СКРИПТА. ИНИЦИАЛИЗАЦИЯ БИБИЛИОТЕК...</b>")


import argparse
import logging
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, Any

# Попытка импорта библиотек из экосистемы PySM
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver = None
    pysm_context = None
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable=None, *args, **kwargs):
            return iterable if iterable is not None else None

# Импорт основной логики анализатора
try:
    from main import run_full_processing, setup_logging
    from fc_lib.fc_config import ConfigManager
except ImportError as e:
    msg = f"ОШИБКА: Не удалось импортировать модули (main, fc_lib). Убедитесь, что они доступны. Детали: {e}"
    print(msg, file=sys.stderr)
    sys.exit(1)

# Глобальные переменные
logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "face_config.toml"


# --- Блок 2: Получение конфигурации ---
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver.
    """
    parser = argparse.ArgumentParser(
        description="Запуск полного цикла обработки изображений лиц.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    # Убираем default. Если аргумент не задан, он будет None.
    parser.add_argument("--__wf_config", type=str, help="Путь к базовому файлу .toml.")
    # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    parser.add_argument("--__wf_folder_path", type=str, help="Переопределить путь к папке с исходниками.")
    parser.add_argument("--__wf_output_path", type=str, help="Переопределить путь к папке для результатов.")
    parser.add_argument("--__wf_children_file", type=str, help="Переопределить путь к файлу со списком детей.")
    parser.add_argument("--__wf_log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Переопределить уровень логирования.")

    resolver = ConfigResolver(parser)
    return resolver.resolve_all()

# --- Блок 3: Основная логика скрипта ---
# ==============================================================================
def main():
    """
    Основная функция: настраивает окружение и запускает обработку.
    """
    # 3.1. Получение конфигурации через ConfigResolver
    # --------------------------------------------------------------------------
    launch_config = get_config()
    
    # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    # 3.2. Определение пути к файлу конфигурации
    # --------------------------------------------------------------------------
    config_file_path = launch_config.__wf_config
    # Если путь не был передан ни через CLI, ни через контекст, используем наш локальный default
    if config_file_path is None:
        config_file_path = str(DEFAULT_CONFIG_PATH)
        logger.info(f"Путь к конфигурации не указан, используется значение по умолчанию: {config_file_path}")
    # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    
    # 3.3. Настройка логирования
    # --------------------------------------------------------------------------
    log_level_override = launch_config.__wf_log_level
    setup_logging(log_level_str=log_level_override, log_file="face_processing.log", gui_log_handler=None)
    #logger.info("Логирование настроено.")

    # 3.4. Инициализация ConfigManager и применение переопределений
    # --------------------------------------------------------------------------
    try:
        # Используем определенный на предыдущем шаге путь
        config_manager = ConfigManager(config_file_path)
    except FileNotFoundError:
        logger.critical(f"Базовый файл конфигурации не найден: {config_file_path}")
        sys.exit(1)

    # Собираем словарь переопределений из полученных аргументов
    overrides: Dict[str, Any] = {}
    if launch_config.__wf_folder_path:
        overrides["paths.folder_path"] = launch_config.__wf_folder_path
    if launch_config.__wf_output_path:
        overrides["paths.output_path"] = launch_config.__wf_output_path
    if launch_config.__wf_children_file:
        overrides["paths.children_file"] = launch_config.__wf_children_file
    if log_level_override:
        overrides["logging_level"] = log_level_override.upper()
    
    # Если переопределения есть, применяем их
    if overrides:
        #logger.info(f"Применение переопределений: {list(overrides.keys())}")
        config_manager.override_config(overrides)

    # 3.4. Адаптеры для колбэков
    # --------------------------------------------------------------------------
    progress_bar = tqdm(total=100, desc="Инициализация", unit="%", dynamic_ncols=True)
    last_progress = {"current": 0}

    def log_adapter(message: str):
        getattr(logging, message.split(":", 1)[0].lower(), logging.info)(message)

    def progress_adapter(current: int, total: int):
        if total > 0 and progress_bar.total != total:
            progress_bar.total = total
        delta = current - last_progress["current"]
        if delta > 0:
            progress_bar.update(delta)
            last_progress["current"] = current

    # 3.5. Запуск основного процесса
    # --------------------------------------------------------------------------
    start_time = time.time()
    success = False
    try:
        success = run_full_processing(
            config_path_or_obj=config_manager,
            log_callback=log_adapter,
            progress_callback=progress_adapter,
        )
        progress_bar.set_description("Завершено" if success else "Завершено с ошибками", refresh=True)

    except Exception as e:
        logger.critical(f"Неперехваченное исключение в `main`: {e}", exc_info=True)
        success = False
    finally:
        # Корректное завершение прогресс-бара и логирование
        if progress_bar.n < progress_bar.total:
            progress_bar.update(progress_bar.total - progress_bar.n)
        progress_bar.close()
        duration = time.time() - start_time
        print("")
        print(f"<b>ОБРАБОТКА УСПЕШНО ЗАВЕРШЕНА</b>")


        print(" ", file=sys.stderr)
        pysm_context.log_link(
            url_or_path=str(launch_config.__wf_output_path)+"\\face_clustering_report.html", # Передаем строку, а не объект Path
            text=f"Открыть HTML-отчета с результатами работы скрипта",
        )                  
        print(" ", file=sys.stderr)
        pysm_context.log_link(
            url_or_path=str(launch_config.__wf_output_path)+"\\_JPG", # Передаем строку, а не объект Path
            text=f"Открыть папку с файлами JPG</i>",
        )                  

        print(" ", file=sys.stderr)
        pysm_context.log_link(
            url_or_path=str(launch_config.__wf_folder_path), # Передаем строку, а не объект Path
            text=f"Открыть папку c RAW файлами",
        )                  

        print(" ", file=sys.stderr)
        pysm_context.log_link(
            url_or_path=str(launch_config.__wf_output_path), # Передаем строку, а не объект Path
            text=f"Открыть папку с файлами JSON",
        )                  
        print(" ", file=sys.stderr)


        logger.info(f"Итоговый статус: {'УСПЕХ' if success else 'НЕУДАЧА'}")
        logger.info(f"Время выполнения: {duration:.2f} секунд.<br>")

        sys.exit(0 if success else 1)


# --- Блок 4: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    if not IS_MANAGED_RUN:
        print("Скрипт запущен автономно. Логи будут в консоли и файле 'face_processing.log'.")
    main()