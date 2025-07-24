# run_wf_csv_generate.py

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
from typing import Dict, List, Any

try:
    from pysm_lib.pysm_context import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver = None


# 2. БЛОК: Определение параметров и получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver.
    """
    parser = argparse.ArgumentParser(
        description="Генерирует CSV-файл из структуры папок с фотографиями."
    )
    parser.add_argument("--__wf_psd_path", type=str, help="Путь к корневой папке сессии для анализа.")
    parser.add_argument("--wf_csv_file_name", type=str, default = "", help="Полный путь для сохранения итогового CSV файла.")
    parser.add_argument("--wf_portrait_folders", type=str, nargs='+', help="Список папок с портретами для обработки.")

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: ОСНОВНАЯ ЛОГИКА
# ==============================================================================
def process_folders_and_generate_data(config: Namespace) -> tuple[list, list[dict]]:
    """
    Сканирует папки и формирует данные для CSV.
    """
    portrait_folders_list = config.wf_portrait_folders
    if isinstance(portrait_folders_list, str):
        portrait_folders_list = [folder.strip() for folder in portrait_folders_list.splitlines() if folder.strip()]
    
    num_folders = len(portrait_folders_list)
    img_headers = [f"@img{i+1}" for i in range(num_folders)]
    fieldnames = ["Name", "Surname"] + img_headers + ["Text"]

    def data_factory() -> Dict[str, Any]:
        row = {"Name": "", "Surname": "", "Text": ""}
        for header in img_headers: row[header] = []
        return row
    
    data = defaultdict(data_factory)
    base_path = pathlib.Path(config.__wf_psd_path)
    print(f"<b>Базовый путь для поиска фотографий:</b>")
    print(f"<i>{base_path}</i>")
    print(f" ")
    print(f"<b>Сканирование папок...</b>")
  
    processing_stats = defaultdict(lambda: {"processed": 0, "skipped": 0})
    base_folder_name = base_path.name
    for i, folder_name in enumerate(portrait_folders_list):
        folder_path = base_path / folder_name
        if not folder_path.is_dir():
            print(f"Папка '{folder_path}' не найдена, пропущена.")
            continue
        
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith((".psd", ".jpg")): continue
            try:
                name_without_ext = pathlib.Path(filename).stem
                parts = name_without_ext.split("-")
                if len(parts) == 3:
                    number, name_surname, photo_number = parts
                    if not (number.isdigit() and len(number) == 2 and photo_number.isdigit() and len(photo_number) == 4): raise ValueError("Некорректный формат")
                    surname, name = name_surname.split()
                    key = f"{number}-{name} {surname}"
                elif len(parts) == 2:
                    name_surname, photo_number = parts
                    if not (photo_number.isdigit() and len(photo_number) == 4): raise ValueError("Некорректный формат")
                    surname, name = name_surname.split()
                    key = f"{photo_number}-{name} {surname}"
                else: raise ValueError("Не соответствует шаблону")

                data[key]["Name"] = name
                data[key]["Surname"] = surname
                # ИСПРАВЛЕНИЕ: Используем pathlib.Path для строки "Фото"
                relative_path = pathlib.Path("Фото") / base_folder_name / folder_name / filename
                file_path_for_csv = "/" + relative_path.as_posix()
                img_key = f"@img{i+1}"
                data[key][img_key].append(file_path_for_csv)
                processing_stats[folder_name]["processed"] += 1
            except (ValueError, IndexError) as e:
                print(f"  [WARN] Файл '{filename}' пропущен (причина: {e}).")
                processing_stats[folder_name]["skipped"] += 1
                continue
    
    total_processed = 0
    total_skipped = 0
    for folder_name in portrait_folders_list:
        stats = processing_stats[folder_name]
        total_processed += stats['processed']
        total_skipped += stats['skipped']
        if (base_path / folder_name).is_dir():
            print(f"Папка <i>{folder_name}</i>: обработано файлов - <b>{stats['processed']}</b>, пропущено - <b>{stats['skipped']}</b>.")
    print(" ")
    print(f"Всего обработано файлов: <b>{total_processed}</b>")
    print(f"Всего найдено уникальных персон: <b>{len(data)}</b>")
    
    output_rows = []
    for key, person_data in sorted(data.items()):
        max_rows = max(len(person_data.get(h, [])) for h in img_headers) if img_headers else 0
        if max_rows == 0: max_rows = 1

        for row_index in range(max_rows):
            new_row = {"Name": person_data["Name"], "Surname": person_data["Surname"], "Text": ""}
            has_photo_in_row = False
            for i, header in enumerate(img_headers):
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
    TEST_MODE = False
    
    if TEST_MODE:
        print("--- ЗАПУСК В ТЕСТОВОМ РЕЖИМЕ ---")
        test_dir = pathlib.Path("./temp_test_environment").resolve()
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Обновляем mock_config, чтобы он соответствовал новым именам аргументов
        mock_config = Namespace(
            __wf_psd_path=str(test_dir / "Альбом" / "Фото"), # Используем новое имя и полную структуру пути
            wf_portrait_folders=["Портрет0", "Портрет1", "Портрет2"], # Используем новое имя
            wf_csv_file_name=""  # Имитируем пустой ввод для проверки логики по умолчанию
        )
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        
        test_files = [
            ("Портрет0", "01-Савченко Михаил-0004.psd"), ("Портрет0", "01-Савченко Михаил-0014.psd"),
            ("Портрет1", "01-Савченко Михаил-0054.psd"), ("Портрет0", "02-Тимофеев Илья-0027.psd"),
            ("Портрет0", "03-Петров Иван-0100.jpg"), ("Портрет1", "03-Петров Иван-0101.jpg"),
            ("Портрет1", "03-Петров Иван-0102.jpg"), ("Портрет0", "Сидорова Анна-0030.jpg"),
            ("Портрет0", "04-Кузнецов Олег-0200.psd"), ("Портрет2", "04-Кузнецов Олег-0201.psd"),
            ("Портрет0", "Просто картинка.jpg"), ("Портрет0", "05-Иванов-00.psd"),
            ("Портрет0", "06-Петров-Сидоров-0040.jpg"),
        ]
        try:
            # Создаем папки внутри тестовой директории
            base_photo_path = pathlib.Path(mock_config.__wf_psd_path)
            base_photo_path.mkdir(parents=True, exist_ok=True)
            for folder, filename in test_files:
                (base_photo_path / folder).mkdir(exist_ok=True)
                (base_photo_path / folder / filename).touch()
            print(f"Временная структура папок создана в: {base_photo_path}")
            
            fieldnames, rows = process_folders_and_generate_data(mock_config)
            print("\n--- РЕЗУЛЬТАТ ГЕНЕРАЦИИ CSV (вывод в консоль) ---\n")
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
            print(output.getvalue())

            # Проверяем логику сохранения файла
            source_path = pathlib.Path(mock_config.__wf_psd_path)
            output_filename = f"{source_path.name}.csv"
            expected_output_path = source_path.parent / output_filename
            print(f"Ожидаемый путь для сохранения файла: {expected_output_path}")

        finally:
            # Очищаем всю тестовую директорию, а не только base_photo_path
            if test_dir.exists():
                shutil.rmtree(test_dir)
                print(f"\nВременная папка {test_dir} удалена.")
    else:
        config = get_config()

        if not config.__wf_psd_path or not os.path.isdir(config.__wf_psd_path):
            print(f"ОШИБКА: Путь для анализа (--__wf_psd_path) не указан или не существует: '{config.__wf_psd_path}'", file=sys.stderr)
            sys.exit(1)

        fieldnames, rows = process_folders_and_generate_data(config)
        
        # ИСПРАВЛЕНИЕ: Возвращаем логику определения output_path
        output_path: pathlib.Path
        source_path = pathlib.Path(config.__wf_psd_path)
        if config.wf_csv_file_name and config.wf_csv_file_name !="":
            # Если путь к CSV указан, используем его
            output_path = pathlib.Path(config.wf_csv_file_name)
        else:
            # Если не указан, формируем его по логике по умолчанию
            output_filename = "output.csv"
            output_path = source_path.parent.parent / output_filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        #print(f"Сохранение данных в файл: {output_path}")
        with open(output_path, mode="w", encoding="utf-16", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        pysm_context.log_link(
            url_or_path=str(output_path), # Передаем строку, а не объект Path
            text=f"Файл <i>{output_path.name}</i> успешно создан<br>",
    )         
        
        pysm_context.log_link(
            url_or_path=str(output_path.parent), # Передаем строку, а не объект Path
            text=f"Открыть родительскую папку файла <i>{output_path.name}</i><br>",
    )         

        pysm_context.log_link(
            url_or_path=str(config.__wf_psd_path), # Передаем строку, а не объект Path
            text=f"Открыть папку с портретными фотографиями<br>",
    )         


# 5. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()