# run_wf_csv_generate.py

"""
Скрипт для автоматической генерации CSV-файла на основе структуры папок с фотографиями.

Назначение:
- Рекурсивно находит папки с портретами по заданным именам.
- Создает в CSV-файле уникальный столбец для каждого найденного пути к папке.
- Извлекает из имен файлов информацию о персонах (имя, фамилия).
- Агрегирует все найденные фотографии по персонам в единую таблицу.
- Сохраняет результат в CSV-файл, совместимый с различными программами.
"""

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import csv
import io
import os
import pathlib
import sys
import shutil
from argparse import Namespace
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# Попытка импорта библиотек из экосистемы PyScriptManager
try:
    from pysm_lib.pysm_context import pysm_context, ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver = None
    pysm_context = None


# 2. БЛОК: Определение параметров и получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы командной строки и получает их значения, используя
    ConfigResolver для интеграции с PyScriptManager.

    Returns:
        argparse.Namespace: Объект с обработанными параметрами запуска.
    """
    parser = argparse.ArgumentParser(
        description="Генерирует CSV-файл из структуры папок с фотографиями."
    )
    # Аргумент для указания корневой папки, в которой будет производиться поиск.
    parser.add_argument(
        "--__wf_psd_path",
        type=str,
        help="Путь к корневой папке для анализа (например, .../Альбом/Фото)."
    )
    # Аргумент для указания пути сохранения итогового CSV-файла.
    parser.add_argument(
        "--wf_csv_file_name",
        type=str,
        default="",
        help="Полный путь для сохранения итогового CSV файла."
    )
    # Аргумент для списка базовых имен папок, которые нужно искать.
    parser.add_argument(
        "--wf_portrait_folders",
        type=str,
        nargs='+',
        help="Список имен папок с портретами для рекурсивного поиска."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        # В автономном режиме используются стандартные аргументы.
        return parser.parse_args()


# 3. БЛОК: Основная логика обработки данных
# ==============================================================================
# run_wf_csv_generate.py

# 3. БЛОК: ОСНОВНАЯ ЛОГИКА (ОКОНЧАТЕЛЬНО ИСПРАВЛЕНО)
# ==============================================================================
def process_folders_and_generate_data(config: Namespace) -> tuple[list, list[dict]]:
    """
    Рекурсивно сканирует папки, формирует динамические столбцы и корректно
    агрегирует данные для CSV.
    """
    base_path = pathlib.Path(config.__wf_psd_path)
    portrait_folder_names_base = config.wf_portrait_folders
    if isinstance(portrait_folder_names_base, str):
        portrait_folder_names_base = [folder.strip() for folder in portrait_folder_names_base.splitlines() if folder.strip()]

    print(f"<b>Базовый путь для поиска фотографий:</b>\n<i>{base_path}</i>\n")
    print(f"<b>Целевые имена папок для поиска:</b>\n<i>{', '.join(portrait_folder_names_base)}</i>\n")

    found_portrait_paths: List[pathlib.Path] = []
    for folder_name in portrait_folder_names_base:
        for path in base_path.rglob(folder_name):
            if path.is_dir():
                found_portrait_paths.append(path)
    
    if not found_portrait_paths:
        print("Целевые папки с портретами не найдены. Работа завершена.")
        return [], []
        
    found_portrait_paths.sort(key=lambda p: (len(p.relative_to(base_path).parts), str(p)))

    path_to_header_map: Dict[pathlib.Path, str] = {}
    dynamic_img_headers: List[str] = []

    print("<b>Найденные папки и сгенерированные для них столбцы CSV:</b>")
    for path in found_portrait_paths:
        relative_path_parts = path.relative_to(base_path).parts
        base_name_index = portrait_folder_names_base.index(path.name)
        intermediate_parts = relative_path_parts[:-1]
        unique_suffix = "_".join(list(intermediate_parts) + [str(base_name_index)])
        header_name = f"@img_{unique_suffix}"
        path_to_header_map[path] = header_name
        dynamic_img_headers.append(header_name)
        print(f" - <i>{'/'.join(relative_path_parts)}</i> -> <b>{header_name}</b>")
    print()

    fieldnames = ["Name", "Surname"] + dynamic_img_headers + ["Text"]
    data = defaultdict(lambda: {"Name": "", "Surname": "", "Text": "", **{h: [] for h in dynamic_img_headers}})
    processing_stats = defaultdict(lambda: {"processed": 0, "skipped": 0})
    
    print("<b>Сканирование папок...</b>")
    for folder_path, header_name in path_to_header_map.items():
        folder_display_name = str(folder_path.relative_to(base_path))

        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith((".psd", ".jpg")): continue
            try:
                name_without_ext = pathlib.Path(filename).stem
                parts = name_without_ext.split("-")
                
                # --- НАЧАЛО КЛЮЧЕВОГО ИСПРАВЛЕНИЯ ---
                key = ""
                name = ""
                surname = ""

                if len(parts) == 3:
                    number, name_surname, photo_number = parts
                    if not (number.isdigit() and len(number) == 2 and photo_number.isdigit() and len(photo_number) == 4): raise ValueError("Некорректный формат")
                    surname, name = name_surname.split()
                    # Ключ включает порядковый номер, но не номер фото
                    key = f"{number}-{name} {surname}"
                elif len(parts) == 2:
                    name_surname, photo_number = parts
                    if not (photo_number.isdigit() and len(photo_number) == 4): raise ValueError("Некорректный формат")
                    surname, name = name_surname.split()
                    # Ключ НЕ включает номер фото, только имя и фамилию
                    key = f"{name} {surname}"
                else: 
                    raise ValueError("Не соответствует шаблону")
                # --- КОНЕЦ КЛЮЧЕВОГО ИСПРАВЛЕНИЯ ---

                data[key]["Name"] = name
                data[key]["Surname"] = surname
                
                album_path = base_path.parent
                relative_path_for_csv = (folder_path / filename).relative_to(album_path)
                file_path_for_csv = "/" + relative_path_for_csv.as_posix()
                
                data[key][header_name].append(file_path_for_csv)
                processing_stats[folder_display_name]["processed"] += 1
            except (ValueError, IndexError) as e:
                processing_stats[folder_display_name]["skipped"] += 1
                continue
    
    print("\n<b>Отчет по обработке:</b>")
    total_processed = 0
    for folder_display_name, stats in sorted(processing_stats.items()):
        total_processed += stats['processed']
        print(f"Папка <i>{folder_display_name}</i>: обработано файлов - <b>{stats['processed']}</b>, пропущено - <b>{stats['skipped']}</b>.")

    print(f"\nВсего обработано файлов: <b>{total_processed}</b>")
    print(f"Всего найдено уникальных персон: <b>{len(data)}</b>")
    
    output_rows = []
    for key, person_data in sorted(data.items()):
        max_rows = max(len(person_data.get(h, [])) for h in dynamic_img_headers) if dynamic_img_headers else 0
        if max_rows == 0:
            max_rows = 1

        for row_index in range(max_rows):
            new_row = {"Name": person_data["Name"], "Surname": person_data["Surname"], "Text": ""}
            has_photo_in_row = False
            for header in dynamic_img_headers:
                photo_list = person_data.get(header, [])
                if row_index < len(photo_list):
                    new_row[header] = photo_list[row_index]
                    has_photo_in_row = True
                else:
                    new_row[header] = ""
            if row_index == 0 or has_photo_in_row:
                 output_rows.append(new_row)

    return fieldnames, output_rows


# 4. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    """
    Главная функция: управляет запуском, выполняет проверку параметров,
    вызывает основную логику и сохраняет результат.
    """
    # Флаг для локального тестирования без реальных данных
    TEST_MODE = False
    
    if TEST_MODE:
        print("--- ЗАПУСК В ТЕСТОВОМ РЕЖИМЕ ---")
        test_dir = pathlib.Path("./temp_test_environment").resolve()
        
        mock_config = Namespace(
            __wf_psd_path=str(test_dir / "Альбом" / "Фото"),
            wf_portrait_folders=["Портрет0", "Портрет1"],
            wf_csv_file_name=""
        )
        
        test_structure = {
            "Портрет0": ["01-Иванов Иван-0001.jpg"],
            "Портрет1": ["02-Петров Петр-0002.jpg"],
            "School/Портрет0": ["01-Иванов Иван-0003.jpg", "01-Иванов Иван-0004.jpg"],
            "School/Портрет1": ["01-Иванов Иван-0005.jpg"]
        }
        try:
            base_photo_path = pathlib.Path(mock_config.__wf_psd_path)
            for rel_path, filenames in test_structure.items():
                full_path = base_photo_path / rel_path
                full_path.mkdir(parents=True, exist_ok=True)
                for fname in filenames:
                    (full_path / fname).touch()
            print(f"Временная структура папок создана в: {base_photo_path}")
            
            fieldnames, rows = process_folders_and_generate_data(mock_config)
            
            print("\n--- РЕЗУЛЬТАТ ГЕНЕРАЦИИ CSV (вывод в консоль) ---\n")
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
            print(output.getvalue())

        finally:
            if test_dir.exists():
                shutil.rmtree(test_dir)
                print(f"\nВременная папка {test_dir} удалена.")
    else:
        # Основной (рабочий) режим выполнения
        config = get_config()

        if not config.__wf_psd_path or not os.path.isdir(config.__wf_psd_path):
            print(f"Путь для анализа (--__wf_psd_path) не указан или не существует:", file=sys.stderr)
            print(f"<i>{config.__wf_psd_path}</i>", file=sys.stderr)
            sys.exit(1)

        fieldnames, rows = process_folders_and_generate_data(config)
        
        output_path: pathlib.Path
        # Если путь к CSV-файлу был явно указан, используем его.
        if config.wf_csv_file_name and config.wf_csv_file_name != "":
            output_path = pathlib.Path(config.wf_csv_file_name)
        else:
            # Иначе формируем путь по умолчанию: на уровень выше папки "Фото",
            # с именем "Фото.csv".
            psd_path = pathlib.Path(config.__wf_psd_path)
            output_filename = f"{psd_path.name}.csv"
            output_path = psd_path.parent / output_filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, mode="w", encoding="utf-16", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)
            
            # Если скрипт запущен в среде PySM, генерируем удобные ссылки в логе.
            if IS_MANAGED_RUN and pysm_context:
                pysm_context.log_link(
                    url_or_path=str(output_path),
                    text=f"Файл <i>{output_path.name}</i> успешно создан<br>",
                )
                pysm_context.log_link(
                    url_or_path=str(output_path.parent),
                    text=f"Открыть родительскую папку файла <i>{output_path.name}</i><br>",
                )
                pysm_context.log_link(
                    url_or_path=str(config.__wf_psd_path),
                    text=f"Открыть папку с портретными фотографиями<br>",
                )
        except IOError as e:
            print(f"ОШИБКА: Не удалось сохранить CSV-файл '{output_path}': {e}", file=sys.stderr)
            sys.exit(1)


# 5. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()