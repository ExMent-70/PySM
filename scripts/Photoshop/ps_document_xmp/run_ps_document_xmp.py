# dump_xmp_metadata.py
import argparse
import os
import sys
import pathlib

# Используем только необходимые импорты
from photoshop import api
from photoshop.api.enumerations import SaveOptions

def main():
    """
    Открывает PSD-файл и выгружает все его XMP-метаданные в сыром XML-формате
    в файл metadata_dump.xml для анализа.
    """
    parser = argparse.ArgumentParser(description="Выгрузка сырых XMP-метаданных из PSD-файла.")
    parser.add_argument("--path", type=str, required=True, help="Полный путь к PSD-файлу для анализа.")
    args = parser.parse_args()

    file_path = args.path
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл не найден по пути '{file_path}'", file=sys.stderr)
        sys.exit(1)
            
    doc = None
    try:
        print("Подключение к Adobe Photoshop...")
        app = api.Application()
        print(f"Открытие документа для анализа: {os.path.basename(file_path)}")
        doc = app.open(file_path)

        # --- КЛЮЧЕВОЙ МОМЕНТ: ПОЛУЧЕНИЕ ВСЕХ СЫРЫХ ДАННЫХ ---
        print("Извлечение сырых XMP метаданных...")
        raw_xmp_data = doc.xmpMetadata.rawData
        
        # Сохраняем результат в файл для удобного анализа
        output_file_path = pathlib.Path(__file__).parent / "metadata_dump.xml"
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(raw_xmp_data)
            
        print("\n--- ДАННЫЕ УСПЕШНО ВЫГРУЖЕНЫ ---")
        print(f"Все XMP-метаданные сохранены в файл: {output_file_path}")
        print("Пожалуйста, откройте этот файл и найдите в нем ваши данные.")
        # Также выводим в консоль для быстрой проверки
        # print("\n--- Содержимое XMP ---")
        # print(raw_xmp_data)
        # print("---------------------\n")


    except Exception as e:
        print(f"\nПроизошла ошибка во время выполнения: {e}", file=sys.stderr)
    finally:
        if doc:
            print("Закрытие документа...")
            doc.close(SaveOptions.DoNotSaveChanges)
        print("Диагностика завершена.")


if __name__ == "__main__":
    main()