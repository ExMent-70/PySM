"""CLI-точка входа интеллектуального установщика Python-зависимостей PySM.

Скрипт сначала определяет целевой Python и файл зависимостей, затем строит
план установки с учетом GPU/CUDA и только после этого запускает установку.
Дефолты важны для безопасного поведения в PySM: без явного Python используется
текущий интерпретатор, без явного пути зависимостей - корень PySM.
"""

import argparse
import json
import logging
import pathlib
import sys
import platform
from argparse import Namespace

try:
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context = None, None

from installer_lib import (
    SystemAnalyzer,
    RequirementsParser,
    InstallationManager
)
from installer_lib.utils import find_requirements_file, run_command


def resolve_target_python(config: Namespace) -> pathlib.Path:
    """Определяет целевой интерпретатор Python для анализа и установки."""
    if config.inst_python_interpreter:
        candidate = pathlib.Path(config.inst_python_interpreter).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Целевой Python не найден: {candidate}")
        return candidate.resolve()

    return pathlib.Path(sys.executable).resolve()


def resolve_search_target(config: Namespace) -> pathlib.Path:
    """Определяет, где искать requirements.txt / pyproject.toml."""
    if config.inst_search_path:
        candidate = pathlib.Path(config.inst_search_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Путь поиска зависимостей не найден: {candidate}")
        return candidate

    return resolve_pysm_root()


def resolve_pysm_root() -> pathlib.Path:
    """Возвращает корневой каталог PySM."""
    if IS_MANAGED_RUN and pysm_context:
        path_from_context = pysm_context.get_structured("pysm_sys_info.app_root_dir")
        if path_from_context and pathlib.Path(path_from_context).exists():
            return pathlib.Path(path_from_context).resolve()

    current = pathlib.Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "main.py").is_file() and (parent / "config.toml").is_file():
            return parent

    return pathlib.Path(".").resolve()


def log_torch_runtime_status(python_exe: pathlib.Path) -> None:
    """Показывает фактическое состояние PyTorch в целевом Python."""
    cmd = [
        str(python_exe),
        "-c",
        (
            "import json\n"
            "try:\n"
            " import torch\n"
            " print(json.dumps({"
            "'installed': True, "
            "'version': torch.__version__, "
            "'cuda_version': getattr(torch.version, 'cuda', None), "
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0"
            "}))\n"
            "except Exception as e:\n"
            " print(json.dumps({'installed': False, 'error': str(e)}))\n"
        ),
    ]
    success, stdout, stderr = run_command(cmd)
    if not success:
        logging.warning(f"  - <b>PyTorch:</b> не удалось проверить ({stderr})")
        return

    try:
        import json
        info = json.loads(stdout)
    except Exception:
        logging.warning(f"  - <b>PyTorch:</b> не удалось разобрать ответ проверки ({stdout})")
        return

    if not info.get("installed"):
        logging.info(f"  - <b>PyTorch:</b> не установлен или не импортируется ({info.get('error')})")
        return

    logging.info(f"  - <b>PyTorch version:</b> {info.get('version')}")
    logging.info(f"  - <b>PyTorch CUDA build:</b> {info.get('cuda_version') or 'CPU-only'}")
    logging.info(f"  - <b>torch.cuda.is_available():</b> {info.get('cuda_available')}")
    logging.info(f"  - <b>CUDA devices в PyTorch:</b> {info.get('device_count')}")


def log_onnx_runtime_status(python_exe: pathlib.Path) -> None:
    """Показывает фактические ONNX Runtime providers в целевом Python."""
    cmd = [
        str(python_exe),
        "-c",
        (
            "import json\n"
            "try:\n"
            " import onnxruntime as ort\n"
            " print(json.dumps({"
            "'installed': True, "
            "'version': getattr(ort, '__version__', None), "
            "'providers': ort.get_available_providers()"
            "}))\n"
            "except Exception as e:\n"
            " print(json.dumps({'installed': False, 'error': str(e)}))\n"
        ),
    ]
    success, stdout, stderr = run_command(cmd)
    if not success:
        logging.warning(f"  - <b>ONNX Runtime:</b> не удалось проверить ({stderr})")
        return

    try:
        import json
        info = json.loads(stdout)
    except Exception:
        logging.warning(f"  - <b>ONNX Runtime:</b> не удалось разобрать ответ проверки ({stdout})")
        return

    if not info.get("installed"):
        logging.info(f"  - <b>ONNX Runtime:</b> не установлен или не импортируется ({info.get('error')})")
        return

    logging.info(f"  - <b>ONNX Runtime version:</b> {info.get('version')}")
    logging.info(f"  - <b>ONNX Runtime providers:</b> {info.get('providers')}")


def get_target_marker_environment(python_exe: pathlib.Path) -> dict | None:
    """Возвращает PEP 508 marker environment именно для целевого Python.

    Requirements markers вроде ``python_version`` должны оцениваться не по
    Python, которым запущен PySM, а по интерпретатору, куда будут ставиться
    пакеты. Если запрос к целевому Python не удался, используется environment
    текущего процесса как безопасный fallback с предупреждением в логе.
    """
    try:
        from packaging.markers import default_environment
    except Exception:
        return None

    environment = default_environment()
    cmd = [
        str(python_exe),
        "-c",
        (
            "import json, os, platform, sys\n"
            "impl_version = sys.implementation.version\n"
            "implementation_version = '.'.join(str(part) for part in impl_version[:3])\n"
            "if impl_version.releaselevel != 'final':\n"
            " implementation_version += impl_version.releaselevel[0] + str(impl_version.serial)\n"
            "print(json.dumps({"
            "'python_version': f'{sys.version_info.major}.{sys.version_info.minor}', "
            "'python_full_version': platform.python_version(), "
            "'implementation_name': sys.implementation.name, "
            "'implementation_version': implementation_version, "
            "'platform_machine': platform.machine(), "
            "'platform_release': platform.release(), "
            "'platform_system': platform.system(), "
            "'platform_version': platform.version(), "
            "'os_name': os.name, "
            "'sys_platform': sys.platform"
            "}))\n"
        ),
    ]
    success, stdout, stderr = run_command(cmd)
    if not success:
        logging.warning(f"Не удалось получить marker environment целевого Python: {stderr}")
        return environment
    try:
        environment.update(json.loads(stdout))
    except Exception:
        logging.warning(f"Не удалось разобрать marker environment целевого Python: {stdout}")
    return environment

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
    parser.add_argument("--inst_plan_only", action="store_true")

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

        target_python_exe = resolve_target_python(config)

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
            logging.info(f"  - <b>CUDA (portable PySM):</b> {cuda.portable_version or 'N/A'}")
            if cuda.portable_path:
                logging.info(f"  - <b>Путь portable CUDA:</b> <i>{cuda.portable_path}</i>")
            logging.info(f"  - <b>CUDA (рекомендовано):</b> {cuda.recommended_version or 'N/A'}")
            logging.info(f"  - <b>CUDA для wheel:</b> {cuda.selected_version or 'CPU'} ({cuda.selected_source or 'fallback'})")
            for warning in cuda.warnings:
                logging.warning(f"  - <b>Предупреждение CUDA:</b> {warning}")
        else:
            logging.info("  - <b>CUDA:</b> Недоступна")
            if system_info.cuda and system_info.cuda.portable_version:
                logging.info(f"  - <b>CUDA (portable PySM):</b> {system_info.cuda.portable_version}")
                if system_info.cuda.portable_path:
                    logging.info(f"  - <b>Путь portable CUDA:</b> <i>{system_info.cuda.portable_path}</i>")

        logging.info(f"  - <b>Целевой Python:</b> <i>{target_python_exe}</i>")
        log_torch_runtime_status(target_python_exe)
        log_onnx_runtime_status(target_python_exe)
        
        if config.inst_analyze_only:
            logging.info("\n--- Работа завершена в режиме 'только анализ' ---")
            sys.exit(0)



        search_target = resolve_search_target(config)
        requirements_file = None

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

        parser = RequirementsParser(system_info, marker_environment=get_target_marker_environment(target_python_exe))
        installation_plan = parser.parse(requirements_file)
        
        # Этап 4: Выполнение
        manager = InstallationManager(
            plan=installation_plan,
            system_info=system_info,
            python_executable=target_python_exe,
            force_upgrade=config.inst_upgrade,
            plan_only=config.inst_plan_only,
        )
        manager.execute_plan()

    except Exception as e:
        logging.error(f"\nПроизошла критическая ошибка: {e}", exc_info=config.inst_verbose)
        sys.exit(1)

if __name__ == "__main__":
    main()
