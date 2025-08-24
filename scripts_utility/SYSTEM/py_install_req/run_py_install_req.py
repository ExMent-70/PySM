# run_py_install_req.py

import argparse
import logging
import pathlib
import sys
import platform
import os
import ctypes
from argparse import Namespace

try:
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context = None, None

# Предполагаем, что installer_lib находится рядом или в sys.path
from installer_lib import (
    SystemAnalyzer,
    RequirementsParser,
    InstallationManager
)
from installer_lib.utils import find_requirements_file

def get_config() -> Namespace:
    """Определяет и разрешает CLI-аргументы."""
    parser = argparse.ArgumentParser(
        description="Интеллектуальная установка Python-зависимостей для Windows.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--inst_search_path", type=str, default="",
        help="Путь к файлу (requirements.txt/pyproject.toml) или к директории для поиска."
    )
    # По умолчанию пустая строка, чтобы мы могли отловить, был ли параметр передан
    parser.add_argument("--inst_python_interpreter", type=str, default="",
        help="Путь к исполняемому файлу Python (python.exe)."
    )
    parser.add_argument("--inst_upgrade", action="store_true")
    parser.add_argument("--inst_verbose", "-v", action="store_true")
    parser.add_argument("--inst_analyze_only", action="store_true")

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    else:
        return parser.parse_args()

def main():
    """
    Главная функция для запуска процесса установки зависимостей.
    """
    config = get_config()
    
    log_level = "INFO"
    if IS_MANAGED_RUN and pysm_context:
        #log_level = pysm_context.get("sys_log_level", "INFO")
        log_level = pysm_context.get_structured("pysm_sys_info.log_level", default="INFO")
    
    if config.inst_verbose:
        log_level = "DEBUG"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(message)s',
        stream=sys.stdout,
        force=True
    )

    logging.info("<b>УСТАНОВКА ЗАВИСИМОСТЕЙ PYTHON</b>")

    try:
        if platform.system() != "Windows":
            logging.error("ОШИБКА: Этот скрипт предназначен для работы только под Windows.")
            sys.exit(1)

        # Этап 1: Анализ системы и подробный вывод
        analyzer = SystemAnalyzer()
        system_info = analyzer.analyze()
        
        logging.info("\n<b>Конфигурация системы:</b>")
        if system_info.gpu:
            gpu = system_info.gpu
            logging.info(f"  - <b>GPU:</b> {gpu.name}")
            logging.info(f"  - <b>Поколение:</b> {gpu.generation.capitalize() if gpu.generation else 'N/A'}")
            logging.info(f"  - <b>Память:</b> {gpu.memory_mb / 1024:.1f} GB")
            logging.info(f"  - <b>Бэкенд:</b> {gpu.backend.upper()}")
            logging.info(f"  - <b>Compute Capability:</b> {gpu.compute_capability}")
            logging.info(f"  - <b>Поддержка TensorRT (через ONNX):</b> {'Да' if gpu.tensorrt_support else 'Нет'}")
        else:
            logging.info("  - <b>GPU:</b> Не обнаружен")
        
        if system_info.cuda and system_info.cuda.is_available:
            cuda = system_info.cuda
            logging.info(f"  - <b>CUDA (драйвер):</b> {cuda.driver_version or 'N/A'}")
            logging.info(f"  - <b>CUDA (рекомендовано):</b> {cuda.recommended_version or 'N/A'}")
        else:
            logging.info("  - <b>CUDA:</b> Недоступна")
        
        if config.inst_analyze_only:
            logging.info("\n--- Работа завершена в режиме 'только анализ' ---")
            sys.exit(0)



        # 1. Блок определения путей с правильными приоритетами (без изменений).
        #    Этот блок ГАРАНТИРУЕТ, что 'search_target' будет валидным объектом Path.
        # --- Определение пути к Python интерпретатору ---
        target_python_exe = None
        if config.inst_python_interpreter:
            candidate = pathlib.Path(config.inst_python_interpreter)
            if candidate.is_file(): target_python_exe = candidate

        if not target_python_exe and IS_MANAGED_RUN and pysm_context:
            path_from_context = pysm_context.get_structured("pysm_sys_info.python_interpreter")
            if path_from_context and pathlib.Path(path_from_context).is_file():
                target_python_exe = pathlib.Path(path_from_context)

        if not target_python_exe:
            target_python_exe = pathlib.Path(sys.executable)

        # --- Определение пути для поиска зависимостей ---
        search_target = None
        if config.inst_search_path:
            candidate = config.inst_search_path.resolve()
            if candidate.exists(): search_target = candidate

        if not search_target and IS_MANAGED_RUN and pysm_context:
            path_from_context = pysm_context.get_structured("pysm_sys_info.app_root_dir")
            if path_from_context and pathlib.Path(path_from_context).exists():
                search_target = pathlib.Path(path_from_context).resolve()

        if not search_target:
            search_target = pathlib.Path('.').resolve()

        logging.info(f"\n<b>Определение параметров установки зависимостей</b>")
        if search_target.is_file():
            logging.info(f"Используется явно указанный файл: <i>{search_target}</i>")
            requirements_file = search_target
        elif search_target.is_dir():
            logging.info(f"Выполняется поиск в директории: <i>{search_target}</i>")
            requirements_file = find_requirements_file(search_target)
        
        if not requirements_file:
            logging.error(f"ОШИБКА: Не удалось найти requirements.txt или pyproject.toml в '{search_target}'.")
            sys.exit(1)



        # Этап 3: Парсинг
        logging.info(f"  - <b>Целевой Python:</b> <i>{target_python_exe}</i>")
        logging.info(f"  - <b>Файл зависимостей:</b> <i>{requirements_file.name}</i>")
        logging.info(f"  - <b>Режим обновления:</b> <i>{'Включен' if config.inst_upgrade else 'Выключен'}</i><br>")

        parser = RequirementsParser(system_info)
        installation_plan = parser.parse(requirements_file)
        
        # Этап 4: Выполнение
        manager = InstallationManager(
            plan=installation_plan,
            system_info=system_info,
            python_executable=target_python_exe,
            force_upgrade=config.inst_upgrade
        )
        manager.execute_plan()

    except Exception as e:
        logging.error(f"\nПроизошла критическая ошибка: {e}", exc_info=config.inst_verbose)
        sys.exit(1)

if __name__ == "__main__":
    main()