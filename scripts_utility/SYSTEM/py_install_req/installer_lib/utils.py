# installer_lib/utils.py

import subprocess
import logging
import queue
import re
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from pysm_lib.pysm_progress_reporter import IS_RUNNING_UNDER_PYSM, JsonProgressReporter
except ImportError:
    IS_RUNNING_UNDER_PYSM = False
    JsonProgressReporter = None

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

def run_command(command: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str, str]:
    """
    Выполняет внешнюю команду и возвращает (успех, stdout, stderr).
    """
    logging.debug(f"Executing command: {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False,
            creationflags=CREATE_NO_WINDOW
        )
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


def run_command_streaming(
    command: List[str],
    cwd: Optional[Path] = None,
    progress_title: str = "Установка пакетов",
    log_output: bool = False,
) -> Tuple[bool, str, str]:
    """
    Выполняет внешнюю команду с живым progress-bar PySM.

    pip/uv могут долго скачивать и устанавливать wheel-файлы. Захват вывода через
    subprocess.run() делает этот этап "немым", поэтому здесь вывод читается в
    отдельном потоке, а progress-bar получает heartbeat даже во время тихих фаз.
    """
    logging.debug(f"Executing streaming command: {' '.join(command)}")
    output_lines: List[str] = []
    output_queue: "queue.Queue[object]" = queue.Queue()
    stream_eof = object()
    progress = JsonProgressReporter(total=0, desc=progress_title) if IS_RUNNING_UNDER_PYSM and JsonProgressReporter else None

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError:
        if progress is not None:
            progress.close()
        logging.error(f"Command not found: {command[0]}")
        return False, "", f"Command not found: {command[0]}"
    except Exception as e:
        if progress is not None:
            progress.close()
        logging.error(f"Exception during command execution: {e}")
        return False, "", str(e)

    def reader() -> None:
        assert process.stdout is not None
        buffer = ""
        while True:
            char = process.stdout.read(1)
            if char == "":
                if buffer.strip():
                    output_queue.put(buffer.strip())
                break
            if char in ("\r", "\n"):
                if buffer.strip():
                    output_queue.put(buffer.strip())
                buffer = ""
            else:
                buffer += char
        output_queue.put(stream_eof)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    start_time = time.monotonic()
    last_heartbeat = start_time
    last_progress_percent = -1
    last_text = progress_title
    reader_done = False

    try:
        while True:
            try:
                item = output_queue.get(timeout=0.25)
            except queue.Empty:
                now = time.monotonic()
                if progress is not None and now - last_heartbeat >= 1.0:
                    elapsed = int(now - start_time)
                    progress.set_description(f"{last_text} ({elapsed} сек.)")
                    last_heartbeat = now
                if process.poll() is not None and reader_done:
                    break
                continue

            if item is stream_eof:
                reader_done = True
                if process.poll() is not None:
                    break
                continue

            line = str(item)
            output_lines.append(line)
            if log_output and _should_log_stream_line(line):
                logging.info(f"  {line}")

            progress_text = _format_progress_text(progress_title, line)
            last_text = progress_text
            percent = _extract_percent(line)
            if progress is not None:
                if percent is not None:
                    if progress.total != 100:
                        progress.reset(total=100)
                        last_progress_percent = 0
                    progress.set_description(progress_text, refresh=False)
                    if percent > last_progress_percent:
                        progress.update(percent - last_progress_percent)
                    else:
                        progress.set_description(progress_text)
                    last_progress_percent = max(last_progress_percent, percent)
                else:
                    progress.set_description(progress_text)

        return_code = process.wait()
        reader_thread.join(timeout=1)
    finally:
        if progress is not None:
            progress.close()

    combined_output = "\n".join(output_lines).strip()
    if return_code != 0:
        logging.debug(f"Command failed with code {return_code}")
        logging.debug(f"Output: {combined_output}")
        return False, combined_output, combined_output
    return True, combined_output, ""


def _extract_percent(line: str) -> Optional[int]:
    matches = re.findall(r"(\d{1,3})%", line)
    if not matches:
        return None
    value = int(matches[-1])
    if 0 <= value <= 100:
        return value
    return None


def _format_progress_text(title: str, line: str) -> str:
    compact_line = _compact_progress_line(line, "")
    if not compact_line:
        return title

    if compact_line == title or compact_line.startswith(f"{title}:"):
        return compact_line

    return _compact_progress_line(f"{title}: {compact_line}", title)


def _compact_progress_line(line: str, fallback: str) -> str:
    cleaned = re.sub(r"\s+", " ", line).strip()
    if not cleaned:
        return fallback
    if len(cleaned) > 140:
        cleaned = cleaned[:137] + "..."
    return cleaned


def _should_log_stream_line(line: str) -> bool:
    lower = line.lower()
    return any(marker in lower for marker in ("error", "failed", "warning", "installed", "resolved", "prepared"))

def find_executable(name: str) -> Optional[Path]:
    """
    Ищет исполняемый файл (например, nvidia-smi.exe) в стандартных директориях Windows.
    Это делает вызов более надежным, чем просто полагаться на системную переменную PATH.
    """
    search_paths = [
        Path(r"C:\Program Files\NVIDIA Corporation\NVSMI"),
        Path(r"C:\Windows\System32")
    ]
    for path in search_paths:
        executable_path = path / f"{name}.exe"
        if executable_path.is_file():
            logging.debug(f"Найден {name}.exe в {path}")
            return executable_path
            
    logging.warning(f"{name}.exe не найден в стандартных директориях. Полагаемся на системный PATH.")
    return None

def find_requirements_file(search_path: Path) -> Optional[Path]:
    """
    Ищет файл зависимостей (pyproject.toml, requirements.txt) в указанной директории.
    """
    pyproject_path = search_path / "pyproject.toml"
    if pyproject_path.is_file():
        logging.info(f"Найден приоритетный файл: {pyproject_path}")
        return pyproject_path

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
            
    try:
        for item in search_path.glob("requirements_*.txt"):
            if item.is_file():
                logging.info(f"Найден файл по шаблону 'requirements_*.txt': {item}")
                return item
    except Exception as e:
        logging.warning(f"Ошибка при поиске файлов по шаблону: {e}")

    logging.warning(f"Файл зависимостей в директории {search_path} не найден.")
    return None
