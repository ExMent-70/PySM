# run_install_requirements.py

import argparse
import subprocess
import sys
import pathlib
import re
from typing import Dict, List
import threading

try:
    from tqdm import tqdm
    from pysm_lib import pysm_context
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        class tqdm:
            def __init__(self, iterable=None, *args, **kwargs):
                self.iterable = iterable if iterable is not None else []
            def __iter__(self):
                return iter(self.iterable)
            def update(self, n=1): pass
            def set_description(self, desc=None): pass
            def close(self): pass
            def write(self, s, *args, **kwargs): print(s)
    class DummyContext:
        def get_data(self, *args, **kwargs): return None
        def set_data(self, *args, **kwargs): pass
        def resolve_path(self, p): return pathlib.Path(p).resolve()
    pysm_context = DummyContext()


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")


def stream_reader(stream, progress_bar: tqdm):
    """
    Читает строки из потока и выводит их.
    Ищет начало загрузки для отображения неопределенного прогресс-бара.
    """
    DOWNLOAD_START_RE = re.compile(r"Downloading\s+(?P<url>\S+)\s+\((?P<size>[\d.]+\s[a-zA-Z]+)\)")
    is_downloading = False

    for line in iter(stream.readline, ''):
        line_strip = line.strip()
        if not line_strip:
            continue
        
        progress_bar.write(f"| {line_strip}")
        
        match = DOWNLOAD_START_RE.search(line_strip)
        if match:
            if is_downloading:
                progress_bar.close()

            is_downloading = True
            filename = match.group('url').split('/')[-1]
            size = match.group('size')
            
            progress_bar.reset(total=0) 
            progress_bar.set_description(f"Загрузка {filename} ({size})")
        
        if "successfully installed" in line_strip.lower():
            if is_downloading:
                is_downloading = False
                progress_bar.close()

    stream.close()


def get_installed_packages(python_exe: pathlib.Path) -> Dict[str, str]:
    print("\n[1/4] Проверка установленных пакетов...")
    try:
        result = subprocess.run(
            [str(python_exe), "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        import json
        packages_list = json.loads(result.stdout)
        installed = {_normalize_name(pkg["name"]): pkg["version"] for pkg in packages_list}
        print(f"    - Найдено {len(installed)} установленных пакетов.")
        return installed
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"    - ПРЕДУПРЕЖДЕНИЕ: Не удалось получить список установленных пакетов: {e}", file=sys.stderr)
        return {}


def parse_requirements(file_path: pathlib.Path) -> List[str]:
    print(f"\n[2/4] Чтение файла зависимостей: {file_path}")
    if not file_path.is_file():
        print("    - ОШИБКА: Файл не найден.", file=sys.stderr)
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    print(f"    - Найдено {len(lines)} требований.")
    return lines


def compare_and_get_to_install(
    requirements: List[str], installed: Dict[str, str], upgrade: bool
) -> List[str]:
    print("\n[3/4] Анализ и сравнение зависимостей...")
    to_install = []
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet
    except ImportError:
        Requirement = Version = SpecifierSet = None
        print("    - ПРЕДУПРЕЖДЕНИЕ: Библиотека 'packaging' не найдена. Сравнение версий может быть неточным.")

    if not requirements:
        print("    - Список требований пуст. Пропуск установки.")
        return []

    for req_str in requirements:
        normalized_req_name = ""
        specifier = None
        if Requirement:
            try:
                req = Requirement(req_str)
                normalized_req_name = _normalize_name(req.name)
                specifier = req.specifier
            except Exception:
                normalized_req_name = _normalize_name(req_str)
                specifier = SpecifierSet("")
        else:
            match = re.match(r"([a-zA-Z0-9\-_]+)", req_str)
            if match:
                normalized_req_name = _normalize_name(match.group(1))
        
        if not normalized_req_name:
            print(f"    - [!] Не удалось распознать имя пакета в строке: '{req_str}'. Пропуск.")
            continue
        
        if normalized_req_name not in installed:
            print(f"    - [+] Будет установлен: {req_str}")
            to_install.append(req_str)
        else:
            current_version_str = installed[normalized_req_name]
            if upgrade:
                print(f"    - [U] Будет обновлен (режим --upgrade): {req_str}")
                to_install.append(req_str)
            elif specifier and Version and specifier:
                if not specifier.contains(Version(current_version_str)):
                     print(f"    - [!] Конфликт версий для '{normalized_req_name}': требуется '{specifier}', установлен '{current_version_str}'. Пропуск.")
                else:
                    print(f"    - [=] Уже установлен и соответствует: {req_str}")
            else:
                 print(f"    - [=] Уже установлен (версия не проверялась): {normalized_req_name}=={current_version_str}")
    return to_install


def main():
    parser = argparse.ArgumentParser(
        description="Интеллектуально устанавливает пакеты из файла requirements.txt, проверяя уже установленные.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--req_requirements_file", required=True, help="Путь к файлу requirements.txt")
    parser.add_argument("--req_python_interpreter", help="Путь к целевому python.exe. Если не указан, используется текущий.")
    parser.add_argument("--req_upgrade", action="store_true", help="Принудительно обновить все пакеты до последних версий, удовлетворяющих требованиям.")
    args = parser.parse_args()
    
    if args.req_python_interpreter:
        target_python_exe = pathlib.Path(args.req_python_interpreter)
    else:
        target_python_exe = pathlib.Path(sys.executable)
    if pysm_context:
        req_file = pysm_context.resolve_path(args.req_requirements_file)
    else:
        req_file = pathlib.Path(args.req_requirements_file).resolve()
    
    print("--- Скрипт установки зависимостей ---")
    print(f"Целевой интерпретатор: {target_python_exe}")
    print(f"Файл зависимостей:    {req_file}")
    print(f"Режим обновления:      {'Включен' if args.req_upgrade else 'Выключен'}")
    print("-" * 40)
    
    installed_packages = get_installed_packages(target_python_exe)
    requirements_list = parse_requirements(req_file)
    if not requirements_list:
        print("\nФайл зависимостей пуст или не найден. Завершение работы.")
        sys.exit(0)
    packages_to_install = compare_and_get_to_install(requirements_list, installed_packages, args.req_upgrade)
    if not packages_to_install:
        print("\n[4/4] Все зависимости уже установлены. Ничего не требуется.")
        print("-" * 40)
        print("Установка успешно завершена (без изменений).")
        sys.exit(0)
    
    print("\n[4/4] Запуск установки пакетов...")
    
    command = [
        str(target_python_exe), "-u", "-m", "pip", "install",
        "-v",
        "--progress-bar", "off",
        "--no-cache-dir",
    ]
    if args.req_upgrade:
        command.append("--upgrade")
    command.extend(packages_to_install)

    print(f"    - Команда: {' '.join(command)}")
    print("    - Вывод pip:")

    try:
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, startupinfo=startupinfo
        )
        
        progress_bar = tqdm(desc="Установка пакетов...", leave=True, bar_format='{l_bar}{bar}|')
        
        stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, progress_bar))
        stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, progress_bar))
        stdout_thread.start()
        stderr_thread.start()

        process.wait()

        stdout_thread.join()
        stderr_thread.join()
        
        progress_bar.close()

        print("-" * 40)
        if process.returncode == 0:
            print("Установка успешно завершена.")
        else:
            print(f"Установка завершена с ошибкой (код: {process.returncode}).")
        sys.exit(process.returncode)

    except (FileNotFoundError, Exception) as e:
        print(f"\nКритическая ошибка при выполнении установки: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()