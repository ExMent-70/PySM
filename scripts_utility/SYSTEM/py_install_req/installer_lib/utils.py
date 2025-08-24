# installer_lib/utils.py

import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Tuple

def run_command(command: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str, str]:
    """
    Выполняет внешнюю команду и возвращает (успех, stdout, stderr).
    """
    # 1. Запуск внешней команды с захватом вывода.
    #    Добавлен флаг CREATE_NO_WINDOW, чтобы утилиты вроде nvidia-smi
    #    не вызывали мигание консольного окна.
    logging.debug(f"Executing command: {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        # 2. Обработка результата. Возвращаем stdout и stderr даже при ошибке
        #    для дальнейшего анализа и логирования.
        if process.returncode != 0:
            logging.debug(f"Command failed with code {process.returncode}")
            logging.debug(f"Stderr: {process.stderr.strip()}")
            return False, process.stdout.strip(), process.stderr.strip()
        
        return True, process.stdout.strip(), process.stderr.strip()
    except FileNotFoundError:
        logging.error(f"Command not found: {command[0]}")
        return False, "", f"Command not found: {command[0]}"
    except Exception as e:
        logging.error(f"Exception during command execution: {e}")
        return False, "", str(e)

def find_executable(name: str) -> Optional[Path]:
    """
    Ищет исполняемый файл (например, nvidia-smi.exe) в стандартных директориях Windows.
    Это делает вызов более надежным, чем просто полагаться на системную переменную PATH.
    """
    # 3. Список стандартных путей для поиска утилиты NVIDIA.
    search_paths = [
        Path(r"C:\Program Files\NVIDIA Corporation\NVSMI"),
        Path(r"C:\Windows\System32")
    ]
    # 4. Итерация по путям и проверка наличия файла.
    for path in search_paths:
        executable_path = path / f"{name}.exe"
        if executable_path.is_file():
            logging.debug(f"Найден {name}.exe в {path}")
            return executable_path
            
    logging.warning(f"{name}.exe не найден в стандартных директориях. Полагаемся на системный PATH.")
    # 5. Если файл не найден, возвращаем None. run_command попробует запустить его по имени.
    return None

def find_requirements_file(search_path: Path) -> Optional[Path]:
    """
    Ищет файл зависимостей (pyproject.toml, requirements.txt) 
    в указанной директории, повторяя логику поиска из Rust-версии.
    """
    # 6. Приоритет №1: pyproject.toml в корневой директории.
    pyproject_path = search_path / "pyproject.toml"
    if pyproject_path.is_file():
        logging.info(f"Найден приоритетный файл: {pyproject_path}")
        return pyproject_path

    # 7. Приоритет №2: Поиск requirements.txt по известным точным путям.
    #    Этот список в точности соответствует логике из Rust-кода.
    specific_candidates = [
        search_path / "requirements.txt",
        search_path / "requirements_pyp.txt", # для зависимостей из pyproject
        search_path / "requirements" / "requirements_nvidia.txt",
        search_path / "requirements" / "requirements.txt",
        search_path / "install" / "requirements.txt",
    ]
    for candidate in specific_candidates:
        if candidate.is_file():
            logging.info(f"Найден файл зависимостей по точному пути: {candidate}")
            return candidate
            
    # 8. Приоритет №3: Поиск файлов по шаблону 'requirements_*.txt' в корне.
    #    Это позволяет находить файлы вроде requirements_windows.txt.
    try:
        for item in search_path.glob("requirements_*.txt"):
            if item.is_file():
                logging.info(f"Найден файл по шаблону 'requirements_*.txt': {item}")
                return item
    except Exception as e:
        logging.warning(f"Ошибка при поиске файлов по шаблону: {e}")

    # 9. Если ничего не найдено, возвращаем None.
    logging.warning(f"Файл зависимостей в директории {search_path} не найден.")
    return None