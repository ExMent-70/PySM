# analyze_imports.py

# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import ast
import os
import pathlib
import sys
from argparse import Namespace
from typing import Set, Dict

# Попытка импорта библиотек PySM
try:
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context = None, None
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs): return iterable
        tqdm.write = print

# 2. БЛОК: Константы и вспомогательные данные
# ==============================================================================
try:
    STD_LIB_MODULES = sys.stdlib_module_names
except AttributeError:
    STD_LIB_MODULES = {'os', 'sys', 're', 'json', 'argparse', 'pathlib', 'datetime', 'collections', 'typing', 'shutil', 'subprocess', 'logging', 'time', 'io', 'csv', 'ast'}
    tqdm.write("[ПРЕДУПРЕЖДЕНИЕ] используется неполный список стандартных библиотек. Результат может быть неточным. Рекомендуется Python 3.10+.")

PIP_NAME_MAP = {
    "PIL": "Pillow", "PySide6": "PySide6", "bs4": "beautifulsoup4",
    "cv2": "opencv-python", "sklearn": "scikit-learn", "jinja2": "Jinja2",
    "psd_tools": "psd-tools", "pywintypes": "pypiwin32"
}

# 3. БЛОК: Функции анализа
# ==============================================================================
def get_imports_from_file(file_path: pathlib.Path) -> Set[str]:
    """Анализирует один Python-файл и возвращает множество импортированных модулей."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
    except Exception as e:
        tqdm.write(f"\n[ПРЕДУПРЕЖДЕНИЕ] Не удалось обработать файл: {file_path}\n -> {e}")
    return imports

def find_project_modules(scan_path: pathlib.Path) -> Set[str]:
    """Находит имена всех локальных модулей и пакетов в проекте."""
    project_modules = set()
    for path in scan_path.glob('*.py'):
        project_modules.add(path.stem)
    for path in scan_path.glob('*/__init__.py'):
        project_modules.add(path.parent.name)
    return project_modules

# 4. БЛОК: Получение конфигурации (ИЗМЕНЕН)
# ==============================================================================
def get_config() -> Namespace:
    """Определяет и получает конфигурацию с помощью ConfigResolver."""
    parser = argparse.ArgumentParser(
        description="Анализирует зависимости Python в указанной директории."
    )
    parser.add_argument(
        "--ai_scan_directory", type=str,
        help="Директория для сканирования."
    )
    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    parser.add_argument(
        "--ai_output_directory", type=str,
        help="Папка для сохранения файлов отчета. Если не указана, используется текущая директория."
    )
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
    parser.add_argument(
        "--ai_generate_requirements_only", action="store_true",
        help="Если флаг установлен, будет создан только файл requirements.txt."
    )
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()

# 5. БЛОК: Основная логика (ИЗМЕНЕН)
# ==============================================================================
def main():
    config = get_config()

    if not config.ai_scan_directory or not pathlib.Path(config.ai_scan_directory).is_dir():
        tqdm.write(f"[ОШИБКА] Директория для сканирования не найдена: '{config.ai_scan_directory}'")
        sys.exit(1)

    scan_path = pathlib.Path(config.ai_scan_directory)
    dir_name = scan_path.name
    
    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # Определяем папку для сохранения отчетов
    if config.ai_output_directory:
        output_dir = pathlib.Path(config.ai_output_directory)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Отчеты будут сохранены в: <i>{output_dir.resolve()}</i>")
        except OSError as e:
            tqdm.write(f"[ОШИБКА] Не удалось создать директорию для отчетов '{output_dir}': {e}")
            sys.exit(1)
    else:
        output_dir = pathlib.Path('.') # Текущая директория
        tqdm.write("Папка для отчетов не указана, используется текущая директория.")
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    print(f"Сканирование директории: <i>{scan_path}</i>\n\n")

    project_modules = find_project_modules(scan_path)

    py_files = list(scan_path.rglob('*.py'))
    if not py_files:
        tqdm.write("В указанной директории .py файлы не найдены.")
        sys.exit(0)

    all_imports = set()
    for py_file in tqdm(py_files, desc="Анализ файлов"):
        all_imports.update(get_imports_from_file(py_file))

    std_lib_imports = sorted([imp for imp in all_imports if imp in STD_LIB_MODULES])
    project_imports = sorted([imp for imp in all_imports if imp in project_modules])
    third_party_imports = sorted([imp for imp in all_imports if imp not in STD_LIB_MODULES and imp not in project_modules])
    
    requirements_content = [PIP_NAME_MAP.get(imp, imp) for imp in third_party_imports]
    # --- ИЗМЕНЕНИЕ: Используем output_dir ---
    requirements_path = output_dir / f"{dir_name}_requirements.txt"
    try:
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write("# Файл сгенерирован автоматически скриптом analyze_imports.py\n")
            f.writelines(f"{req}\n" for req in sorted(requirements_content))
        if pysm_context: pysm_context.log_link(str(output_dir), f"Открыть папку {output_dir}")
        if pysm_context: pysm_context.log_link(str(requirements_path), f"Файл {requirements_path.name} успешно создан")
    except IOError as e:
        tqdm.write(f"[ОШИБКА] Не удалось сохранить файл '{requirements_path.name}': {e}")
    
    if not config.ai_generate_requirements_only:
        # --- ИЗМЕНЕНИЕ: Используем output_dir ---
        report_path = output_dir / f"{dir_name}_all_imports.txt"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# Полный отчет по импортам для: {scan_path}\n\n")
                f.write(f"--- СТОРОННИЕ ПАКЕТЫ ({len(third_party_imports)}) ---\n")
                f.writelines(f"{imp}\n" for imp in third_party_imports)
                f.write(f"\n--- ВНУТРЕННИЕ МОДУЛИ ПРОЕКТА ({len(project_imports)}) ---\n")
                f.writelines(f"{imp}\n" for imp in project_imports)
                f.write(f"\n--- МОДУЛИ СТАНДАРТНОЙ БИБЛИОТЕКИ ({len(std_lib_imports)}) ---\n")
                f.writelines(f"{imp}\n" for imp in std_lib_imports)
            if pysm_context: pysm_context.log_link(str(report_path), f"Файл {report_path.name} успешно создан\n")
        except IOError as e:
            tqdm.write(f"[ОШИБКА] Не удалось сохранить файл '{report_path.name}': {e}")
            
    # Вывод в консоль остается без изменений
    print(f"\n<b>Сторонние пакеты ({len(third_party_imports)}):</b>")
    print("<i>" + ", ".join(third_party_imports) + "</i>")
    print(f"\n<b>Внутренние модули проекта ({len(project_imports)}):</b>")
    print("<i>" + ", ".join(project_imports) + "</i>")
    print(f"\n<b>Модули стандартной библиотеки ({len(std_lib_imports)}):</b>")
    print("<i>" + ", ".join(std_lib_imports) + "</i>")
    print("\n")

# 6. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()