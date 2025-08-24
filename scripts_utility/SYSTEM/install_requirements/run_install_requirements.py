# run_install_requirements.py

import argparse
import subprocess
import sys
import pathlib
import re
from typing import Dict, List, Tuple
from argparse import Namespace

try:
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context = None, None

def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")

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
    except Exception as e:
        print(f"    - ПРЕДУПРЕЖДЕНИЕ: Не удалось получить список пакетов: {e}", file=sys.stderr)
        return {}

def parse_requirements(file_path: pathlib.Path) -> Tuple[List[str], List[str]]:
    print(f"\n[2/4] Чтение файла зависимостей: {file_path}")
    if not file_path.is_file():
        print("    - ОШИБКА: Файл не найден.", file=sys.stderr)
        return [], []
    
    packages, options = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.startswith("--"):
                    options.extend(line.split(' ', 1))
                else:
                    packages.append(line)
    print(f"    - Найдено {len(packages)} требований.")
    print(f"    - Найдено {len(options) // 2} опций для pip.")
    return packages, options

def compare_and_get_to_install(
    requirements: List[str], installed: Dict[str, str], upgrade: bool
) -> List[str]:
    print("\n[3/4] Анализ и сравнение зависимостей...")
    to_install = []
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        Requirement = Version = None
        print("    - ПРЕДУПРЕЖДЕНИЕ: 'packaging' не найдена. Сравнение версий неточное.")

    if not requirements:
        print("    - Список требований пуст.")
        return []

    for req_str in requirements:
        clean_req_str = re.sub(r'\+.+$', '', req_str)
        try:
            req = Requirement(clean_req_str) if Requirement else None
            req_name = _normalize_name(req.name if req else re.split(r'[=<>!~]', clean_req_str)[0])
            specifier = req.specifier if req else None
        except Exception as e:
            print(f"    - [!] Не удалось распознать '{req_str}': {e}. Пропуск.")
            continue
        
        if req_name not in installed:
            print(f"    - [+] Будет установлен: {req_str}")
            to_install.append(req_str)
        else:
            current_version = installed[req_name]
            if upgrade:
                print(f"    - [U] Будет обновлен: {req_str}")
                to_install.append(req_str)
            elif specifier and Version and not specifier.contains(Version(current_version)):
                print(f"    - [!] Конфликт версий для '{req_name}': требуется '{specifier}', установлен '{current_version}'. Обновление.")
                to_install.append(req_str)
            else:
                print(f"    - [=] Уже установлен: {req_str}")
    return to_install

def get_config() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Устанавливает пакеты из файла requirements.txt.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--req_requirements_file", help="Путь к файлу requirements.txt.")
    parser.add_argument("--req_python_interpreter", help="Путь к python.exe.")
    parser.add_argument("--req_upgrade", action="store_true", help="Принудительно обновить все пакеты.")
    
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()

def main():
    config = get_config()

    target_python_exe = None
    if config.req_python_interpreter:
        target_python_exe = pathlib.Path(config.req_python_interpreter)
    elif IS_MANAGED_RUN and pysm_context:
        pysm_config_path = pysm_context.get_structured("pysm_config.paths.python_interpreter")
        if pysm_config_path:
            target_python_exe = pysm_context.resolve_path(pysm_config_path)
    
    if not target_python_exe or not target_python_exe.is_file():
        target_python_exe = pathlib.Path(sys.executable)

    if not config.req_requirements_file:
        print("ОШИБКА: Путь к файлу requirements.txt не указан (--req_requirements_file).", file=sys.stderr)
        sys.exit(1)
    req_file = pathlib.Path(config.req_requirements_file).resolve()

    print("--- Скрипт установки зависимостей ---")
    print(f"Целевой интерпретатор: {target_python_exe}")
    print(f"Файл зависимостей:    {req_file}")
    print(f"Режим обновления:      {'Включен' if config.req_upgrade else 'Выключен'}")
    print("-" * 40)
    
    installed_packages = get_installed_packages(target_python_exe)
    requirements_list, pip_options = parse_requirements(req_file)

    if not requirements_list:
        print("\nФайл зависимостей пуст или не найден. Завершение работы.")
        sys.exit(0)
        
    packages_to_install = compare_and_get_to_install(requirements_list, installed_packages, config.req_upgrade)
    
    if not packages_to_install:
        print("\n[4/4] Все зависимости уже установлены.")
        print("Установка успешно завершена (без изменений).")
        sys.exit(0)
    
    print("\n[4/4] Запуск установки/обновления пакетов...")
    
    command = [str(target_python_exe), "-m", "pip", "install"]
    command.extend(pip_options)
    if config.req_upgrade:
        command.append("--upgrade")
    command.extend(packages_to_install)

    print(f"    - Итоговая команда: {' '.join(command)}")
    print("--- Вывод pip ---")

    try:
        # --- НАЧАЛО ИЗМЕНЕНИЙ: Упрощенный вызов subprocess ---
        # Просто запускаем процесс и позволяем ему писать напрямую в stdout/stderr
        # PySM перехватит этот вывод.
        process = subprocess.run(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        print("--- Конец вывода pip ---")
        print("-" * 40)
        
        if process.returncode == 0:
            print("Установка успешно завершена.")
            sys.exit(0)
        else:
            print(f"Установка завершена с ошибкой (код возврата: {process.returncode}).")
            sys.exit(process.returncode)

    except Exception as e:
        print(f"\nКритическая ошибка при запуске pip: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()