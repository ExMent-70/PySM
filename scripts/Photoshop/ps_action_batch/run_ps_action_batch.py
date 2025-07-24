# 1. БЛОК: Импорты
# ==============================================================================
"""
Выполняет пакетную обработку файлов с помощью указанного экшена Photoshop.

Скрипт поддерживает несколько режимов работы, определяемых параметром --ps_mode:
- 'active_document': Обрабатывает только текущий активный документ.
- 'active_document_folder': Обрабатывает все файлы в папке, где находится активный документ.
- 'selected_file': Обрабатывает один конкретный файл, указанный в --ps_file_path.
- 'selected_file_folder': Обрабатывает все файлы в папке, где находится указанный файл.
"""
import argparse
import sys
from argparse import Namespace
from pathlib import Path
from typing import List

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context = None, None
    try: from tqdm import tqdm
    except ImportError:
        class TqdmMock:
            def __init__(self, i, *a, **kw): self.i = i
            def __iter__(self): return iter(self.i)
            @staticmethod
            def write(m, *a, **kw): print(m)
            def set_description(self, *a, **kw): pass
        tqdm = TqdmMock

# Импорт ключевых зависимостей
try:
    from photoshop import api
    from photoshop.api.enumerations import SaveOptions
    from comtypes import COMError
except ImportError:
    tqdm.write("ОШИБКА: Библиотеки 'photoshop-python-api' и 'comtypes' не найдены.")
    sys.exit(1)


# 2. БЛОК: Получение конфигурации (ОБНОВЛЕННЫЙ ПАТТЕРН)
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver,
    который автоматически обрабатывает пути и шаблоны.
    """
    parser = argparse.ArgumentParser(
        description="Пакетная обработка файлов с помощью экшена Photoshop в разных режимах."
    )
    parser.add_argument(
        "--ps_mode", type=str, required=True,
        choices=['active_document', 'active_document_folder', 'selected_file', 'selected_file_folder'],
        help="Режим работы скрипта."
    )
    parser.add_argument(
        "--ps_file_path", type=str, required=False,
        help="Путь к файлу (используется в режимах 'selected_file' и 'selected_file_folder'). Поддерживает шаблоны и авто-разрешение."
    )
    parser.add_argument(
        "--ps_action_set", type=str, required=True, help="Имя набора экшенов. Поддерживает шаблоны."
    )
    parser.add_argument(
        "--ps_action_name", type=str, required=True, help="Имя экшена для выполнения. Поддерживает шаблоны."
    )
    parser.add_argument(
        "--ps_recursive", action="store_true",
        help="Рекурсивный поиск (для режимов работы с папками)."
    )
    
    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()

# 3. БЛОК: Определение списка файлов для обработки
# ==============================================================================
def get_files_to_process(config: Namespace, app: api.Application) -> List[str]:
    """
    На основе режима работы и параметров определяет и возвращает
    список путей к файлам для обработки.

    Args:
        config (Namespace): Объект конфигурации, содержащий параметры запуска.
        app (api.Application): Экземпляр приложения Photoshop.

    Returns:
        List[str]: Отсортированный список абсолютных путей к файлам.
    """
    mode = config.ps_mode
    print(f"Определение списка файлов для режима: '{mode}'")
    
    target_folder = None

    if mode == 'active_document':
        if len(app.documents) == 0:
            tqdm.write("ОШИБКА: Режим <i>active_document</i> требует наличия открытого документа.")
            return []
        try:
            # Для обработки одного документа возвращаем список с одним элементом
            return [str(app.activeDocument.fullName)]
        except COMError:
            tqdm.write("ОШИБКА: Активный документ должен быть сохранен хотя бы один раз, чтобы получить его путь.")
            return []

    elif mode == 'active_document_folder':
        if len(app.documents) == 0:
            tqdm.write("ОШИБКА: Режим <i>active_document_folder</i> требует наличия открытого документа.")
            return []
        try:
            target_folder = Path(app.activeDocument.path)
            print(f"Целевая папка (получена из активного документа): {target_folder}")
        except COMError:
            tqdm.write("ОШИБКА: Активный документ должен быть сохранен, чтобы определить его папку.")
            return []

    elif mode == 'selected_file':
        if not config.ps_file_path:
            tqdm.write("ОШИБКА: Для режима <i>selected_file</i> необходимо указать путь в --ps_file_path.")
            return []
        target_file = Path(config.ps_file_path)
        if not target_file.is_file():
            tqdm.write(f"ОШИБКА: Файл не найден по пути: {target_file}")
            return []
        return [str(target_file)]

    elif mode == 'selected_file_folder':
        if not config.ps_file_path:
            tqdm.write("ОШИБКА: Для режима <i>selected_file_folder</i> необходимо указать путь в --ps_file_path.")
            return []
        target_file = Path(config.ps_file_path)
        if not target_file.is_file():
            tqdm.write(f"ОШИБКА: Файл не найден по пути: {target_file}")
            return []
        target_folder = target_file.parent
        print(f"Целевая папка (получена из имени указанного файла): {target_folder}")

    else:
        # Этот блок на практике не должен вызываться, так как argparse choices ограничивает значения
        tqdm.write(f"ОШИБКА: Неизвестный режим работы '{mode}'.")
        return []

    # Общая логика сбора файлов для режимов, работающих с папками
    if target_folder:
        print(f"Поиск изображений в: {target_folder} (Рекурсия: {'Вкл' if config.ps_recursive else 'Выкл'})")
        # Примечание: можно расширить, добавив другие расширения.
        valid_extensions = {".psd"}
        
        if config.ps_recursive:
            image_files_paths = [f for ext in valid_extensions for f in target_folder.rglob(f"*{ext}")]
        else:
            image_files_paths = [f for ext in valid_extensions for f in target_folder.glob(f"*{ext}")]
        
        return sorted([str(f) for f in image_files_paths if f.is_file()])

    return []

# 4. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    """
    Главная функция-оркестратор: подключается к Photoshop, определяет
    список файлов, итерирует по ним и выполняет экшен.
    """
    config = get_config()

    print("<b>Пакетная обработка файлов</b>")
    
    try:
        app = api.Application()
    except COMError:
        tqdm.write("ОШИБКА: Не удалось подключиться к Photoshop.")
        sys.exit(1)

    # Запоминаем имя исходного документа, если он был, чтобы случайно не закрыть его в конце.
    initial_active_doc_name = None
    if len(app.documents) > 0 and config.ps_mode in ['active_document', 'active_document_folder']:
        try:
            initial_active_doc_name = app.activeDocument.name
        except COMError:
            # Активный документ не сохранен, имени нет, ничего страшного.
            pass 

    image_files = get_files_to_process(config, app)

    if not image_files:
        tqdm.write("Не найдено файлов для обработки. Завершение работы.")
        sys.exit(0)
        
    print(f"Найдено {len(image_files)} файлов для обработки.")

    progress_bar = tqdm(image_files, desc="Обработка", unit="file", dynamic_ncols=True)
    for full_path in progress_bar:
        doc = None
        file_name = Path(full_path).name
        try:
            progress_bar.set_description(f"Обработка: {file_name}")
            
            # Открываем очередной файл
            doc = app.open(full_path)
            
            # Выполняем экшен
            app.doAction(config.ps_action_name, config.ps_action_set)
            
            # Сохраняем изменения
            doc.save()
            
        except Exception as e:
            tqdm.write(f"\nОШИБКА при обработке '{file_name}': {e}")
        finally:
            # Гарантированно закрываем документ, если он был открыт
            if doc:
                # Проверяем, не является ли этот документ тем, с которого мы начали.
                # Это предотвращает закрытие исходного документа пользователя.
                if doc.name == initial_active_doc_name:
                    continue
                doc.close(SaveOptions.DoNotSaveChanges)

    print("\n<b>Пакетная обработка завершена.</b>\n")
    sys.exit(0)

# 5. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()