# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import sys
from argparse import Namespace

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib import pysm_context # pysm_context не используется, но оставляем для унификации
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs): print(msg)
    tqdm = TqdmWriteMock()

# Импорт ключевых зависимостей
try:
    from photoshop import api
    from photoshop.api.enumerations import SaveOptions
    from photoshop.api.errors import PhotoshopPythonAPIError
    from comtypes import COMError
except ImportError:
    tqdm.write("ОШИБКА: Библиотеки 'photoshop-python-api' и 'comtypes' не найдены.")
    sys.exit(1)


# 2. БЛОК: Получение конфигурации (ОБНОВЛЕННЫЙ ПАТТЕРН)
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver.
    """
    parser = argparse.ArgumentParser(
        description="Закрывает активный или указанный по имени документ."
    )
    parser.add_argument(
        "--ps_document_name", type=str, required=False,
        help="Имя документа для закрытия. По умолчанию - активный. Поддерживает шаблоны."
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--ps_force_close", action='store_true',
        help="Принудительно закрыть без сохранения."
    )
    save_group.add_argument(
        "--ps_save_on_close", action='store_true',
        help="Сохранить изменения перед закрытием."
    )
    
    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    # Получаем готовую конфигурацию одним вызовом
    config = get_config()

    print("<b>Закрыть документ Adobe Photoshop</b>")

    try:
        app = api.Application()
        if not app.documents:
            tqdm.write("Нет открытых документов для закрытия.")
            sys.exit(0)
    except COMError:
        tqdm.write("ОШИБКА: Не удалось подключиться к Photoshop.")
        sys.exit(1)

    doc_to_close = None
    try:
        # ConfigResolver уже обработал шаблоны в имени документа
        if config.ps_document_name:
            print(f"Закрытие документа <i>{config.ps_document_name}</i>...")
            doc_to_close = app.documents.getByName(config.ps_document_name)
        else:
            print("Закрытие активного документа.")
            doc_to_close = app.activeDocument
    except PhotoshopPythonAPIError as e:
        tqdm.write(f"ОШИБКА: Не удалось найти документ с именем '{config.ps_document_name}'. {e}")
        sys.exit(1)
    except Exception as e:
        tqdm.write(f"Произошла ошибка при выборе документа: {e}")
        sys.exit(1)

    save_option = SaveOptions.PromptToSaveChanges
    if config.ps_force_close:
        save_option = SaveOptions.DoNotSaveChanges
        print("Режим: принудительное закрытие без сохранения.")
    elif config.ps_save_on_close:
        save_option = SaveOptions.SaveChanges
        print("Режим: сохранить перед закрытием.")
    else:
        print("Режим: показать диалог сохранения (если есть изменения).")

    try:
        doc_name = doc_to_close.name
        #tqdm.write(f"Закрытие документа '{doc_name}'...")
        doc_to_close.close(save_option)
        print(f"\n<b>Документ '{doc_name}' успешно закрыт.</b>\n")
    except Exception as e:
        tqdm.write(f"\n<b>Процесс закрытия документа был прерван:</b>")
        tqdm.write(f"<i>{e}</i>\n")        
        #tqdm.write("Это может произойти, если пользователь нажал 'Отмена' в диалоге, или если при сохранении новый документ требует указания пути.\n")
        sys.exit(1)

    #tqdm.write("--- Скрипт закрытия документа: Успешное завершение ---")
    sys.exit(0)

# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()