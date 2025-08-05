# run_ps_document_meta.py

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Union

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context, ConfigResolver = None, None
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg, file=sys.stderr)
    tqdm = TqdmWriteMock()


# 2. БЛОК: Получение конфигурации (ОБНОВЛЕН)
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver,
    который автоматически обрабатывает пути и шаблоны.
    """
    if not IS_MANAGED_RUN:
        tqdm.write("ОШИБКА: Этот скрипт предназначен только для запуска из среды PyScriptManager.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Извлекает метаданные из файла и записывает их в контекст.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ps_file_path", type=str, 
        help="Полный путь к файлу (PSD, JPG, etc.). Поддерживает шаблоны и авто-разрешение путей."
    )
    parser.add_argument(
        "--ps_meta_fields", type=str, nargs='*',
        help="Имена полей для извлечения. Используйте '__all__' или оставьте пустым для всех полей."
    )
    parser.add_argument(
        "--ps_clear_context", action="store_true",
        help="Очистить связанные переменные контекста (`psd_meta_*`) перед записью новых."
    )
    
    # ConfigResolver сам позаботится о разрешении путей и шаблонов.
    resolver = ConfigResolver(parser)
    return resolver.resolve_all()


# 3. БЛОК: Главная функция-оркестратор (ИЗМЕНЕН)
# ==============================================================================
def main():
    # Получаем конфигурацию одним вызовом.
    config = get_config()

    # Логика обработки и валидации полей `ps_meta_fields`
    fields_to_extract = config.ps_meta_fields if config.ps_meta_fields else ["__all__"]
    
    if len(fields_to_extract) == 1 and fields_to_extract[0].lower() == "__all__":
        final_fields = "__all__"
    else:
        available_fields = set(pysm_context.get_available_metadata_fields())
        invalid_fields = [f for f in fields_to_extract if f not in available_fields]
        
        if invalid_fields:
            tqdm.write(f"Обнаружены недопустимые имена полей: {', '.join(invalid_fields)}")
            tqdm.write(f"Доступные поля: {', '.join(sorted(list(available_fields)))}")
            sys.exit(1)
        final_fields = fields_to_extract

    try:
        # Извлекаем метаданные. В контекст они записываются ПОЛНОСТЬЮ.
        extracted_data = pysm_context.get_document_metadata(
            doc_path=config.ps_file_path,
            fields=final_fields,
            clear_before_write=config.ps_clear_context
        )

        if not extracted_data:
            print("Метаданные не были извлечены или файл не содержит запрошенных полей.")
        else:
            print("\nМетаданные успешно извлечены и сохранены в контекст.")
            
            # --- НАЧАЛО ПОЛНОГО БЛОКА ДЕМОНСТРАЦИИ ---
            print("\n<b>1. Демонстрация доступа к данным через API</b>")
            headline = pysm_context.get("psd_meta_Headline", "Заголовок не найден")
            print(f"<b>Пример 1:</b> Получение простого строкового значения 'Headline':<br><i>{headline}</i>")
            
            keywords = pysm_context.get("psd_meta_Keywords", [])
            if keywords:
                print(f"<b>Пример 2:</b> Ключевые слова (получен список):<br><i>{', '.join(keywords)}</i>")
            
            person_name = pysm_context.get_structured("psd_meta_SubjectCode.F0_person", "Имя не найдено")
            print(f"<b>Пример 3:</b> Имя персоны (вложенный доступ):<br><i>{person_name}</i>")
            
            subject_code_dict = pysm_context.get("psd_meta_SubjectCode", {})
            if subject_code_dict:
                gender = subject_code_dict.get("F0_gender_faceonnx", "не определен")
                emotion = subject_code_dict.get("F0_emotion_faceonnx", "не определена")
                print(f"<b>Пример 4:</b> Пол/Эмоция (из словаря):<br><i>{gender} / {emotion}</i>")
            # --- КОНЕЦ ПОЛНОГО БЛОКА ДЕМОНСТРАЦИИ ---

            # КОММЕНТАРИЙ: Полностью переписан блок вывода для корректной обработки
            # вложенных структур и усечения только длинных СТРОК.
            print("\n<b>2. Полный список извлеченных переменных (с усечением для лога)</b>")

            def format_and_truncate(value: any, max_len: int = 120) -> str:
                """Усекает только строки, оставляя другие типы как есть."""
                if isinstance(value, str) and len(value) > max_len:
                    # Усекаем и добавляем теги для курсива
                    return f"<i>{value[:max_len]}...</i>"
                # Добавляем теги для курсива для всех остальных значений
                return f"<i>{value if value is not None else '(пусто)'}</i>"

            for key, value in extracted_data.items():
                print(f"<b>{key}:</b>")
                
                if value is None:
                    print("  <i>(пусто)</i>")
                elif isinstance(value, list):
                    if not value:
                        print("  <i>(пустой список)</i>")
                    else:
                        for item in value:
                            # Применяем форматирование и усечение к каждому элементу списка
                            print(f"  - {format_and_truncate(item)}")
                elif isinstance(value, dict):
                    if not value:
                        print("  <i>(пустой объект)</i>")
                    else:
                        for k, v in value.items():
                            # Применяем форматирование и усечение к каждому значению в словаре
                            print(f"  - <b>{k}:</b> {format_and_truncate(v)}")
                else:
                    # Это для простых типов (строка, число, bool), не вложенных.
                    # Добавляем отступ для единообразия.
                    print(f"  {format_and_truncate(value)}")

    except FileNotFoundError as e:
        tqdm.write(f"ОШИБКА: {e}")
        sys.exit(1)
    except Exception as e:
        tqdm.write(f"Произошла непредвиденная ошибка: {e}")
        sys.exit(1)

    print("\n ")
    sys.exit(0)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()