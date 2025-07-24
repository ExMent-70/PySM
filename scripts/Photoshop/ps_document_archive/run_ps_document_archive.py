# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import sys
from argparse import Namespace
from pathlib import Path

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context = None, None
    class TqdmMock:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        @staticmethod
        def write(m, *a, **kw): print(m)
        def set_description(self, *a, **kw): pass
        def update(self, n=1): pass
    tqdm = TqdmMock

# Импорт ключевых зависимостей
try:
    from photoshop import api
    from photoshop.api import save_options
    from photoshop.api.enumerations import SaveOptions
    from comtypes import COMError
except ImportError:
    tqdm.write("ОШИБКА: Библиотеки 'photoshop-python-api' и 'comtypes' не найдены.")
    sys.exit(1)

# Импорт GUI-библиотеки
try:
    from PySide6.QtWidgets import QApplication, QFileDialog
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


# 2. БЛОК: Получение конфигурации (ОБНОВЛЕННЫЙ ПАТТЕРН)
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы, получает значения и при необходимости запрашивает
    недостающие пути в интерактивном режиме.
    """
    parser = argparse.ArgumentParser(
        description="Архивирует PSD файлы: запускает экшен, сохраняет JPG и пересохраняет PSD."
    )
    parser.add_argument(
        "--ps_source_folder", type=str,
        help="Папка с исходными PSD. Если не указан, появится диалог выбора."
    )
    parser.add_argument(
        "--ps_output_folder", type=str,
        help="Папка для сохранения JPG. Если не указан, появится диалог выбора."
    )
    parser.add_argument(
        "--ps_action_set", type=str, default="PySM_Actions",
        help="Имя набора экшенов для выполнения."
    )
    parser.add_argument(
        "--ps_action_name", type=str, default="PySM_RETUCHE_ARCHIVE",
        help="Имя самого экшена для выполнения."
    )
    parser.add_argument(
        "--ps_recursive", action="store_true",
        help="Включить рекурсивный поиск файлов в исходной папке."
    )
    
    # ConfigResolver автоматически обработает пути и шаблоны, если они заданы.
    if IS_MANAGED_RUN:
        config = ConfigResolver(parser).resolve_all()
    else:
        config = parser.parse_args()

    # Интерактивный запрос недостающих путей
    if not config.ps_source_folder or not config.ps_output_folder:
        if not PYSIDE_AVAILABLE:
            tqdm.write("ОШИБКА: PySide6 не найдена. Укажите пути через параметры --ps_source_folder и --ps_output_folder.")
            sys.exit(1)
        
        # Гарантируем, что QApplication существует
        q_app = QApplication.instance() or QApplication(sys.argv)
        
        if not config.ps_source_folder:
            tqdm.write("\nИсходная папка не указана. Открытие диалога...")
            source_path = QFileDialog.getExistingDirectory(None, "Выберите исходную папку с PSD", "")
            if not source_path:
                tqdm.write("Операция отменена пользователем."); sys.exit(0)
            config.ps_source_folder = source_path

        if not config.ps_output_folder:
            tqdm.write("\nПапка назначения не указана. Открытие диалога...")
            output_path = QFileDialog.getExistingDirectory(None, "Выберите папку для сохранения JPG", "")
            if not output_path:
                tqdm.write("Операция отменена пользователем."); sys.exit(0)
            config.ps_output_folder = output_path
            
    return config


# 3. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    # Вся логика конфигурации, включая интерактивный запрос, инкапсулирована в get_config()
    config = get_config()

    print("<b>Скрипт архивации документов</b>")

    source_folder = Path(config.ps_source_folder)
    output_folder = Path(config.ps_output_folder)

    if not source_folder.is_dir():
        tqdm.write(f"ОШИБКА: Исходная папка не найдена: '{source_folder}'"); sys.exit(1)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nИсходная папка: <i>{source_folder}</i>")
    print(f"Папка для JPG: <i>{output_folder}</i>")

    # Поиск файлов
    search_pattern = "*.psd"
    print(f"\n{'Рекурсивный поиск' if config.ps_recursive else 'Поиск'}  {search_pattern} файлов в исходной папке")
    if config.ps_recursive:
        image_files = sorted(list(source_folder.rglob(search_pattern)))
    else:
        image_files = sorted(list(source_folder.glob(search_pattern)))
    
    image_files = [f for f in image_files if f.is_file()]

    if not image_files:
        tqdm.write("В исходной папке не найдено .psd файлов."); sys.exit(0)
        
    print(f"Найдено <b>{len(image_files)}</b> файлов для обработки.")

    try:
        app = api.Application()
    except COMError:
        tqdm.write("ОШИБКА: Не удалось подключиться к Photoshop."); sys.exit(1)

    # Корректная работа с tqdm для ручного обновления прогресса
    with tqdm(total=len(image_files), desc="Архивация", unit="file", dynamic_ncols=True) as progress_bar:
        for full_path in image_files:
            doc = None
            file_name = full_path.name
            try:
                progress_bar.set_description(f"Обработка: {file_name}")
                
                doc = app.open(str(full_path))
                
                # 1. Выполнить экшен
                app.doAction(config.ps_action_name, config.ps_action_set)
                
                # 2. Сохранить как JPG
                jpg_output_path = output_folder / f"{full_path.stem}.jpg"
                jpg_options = save_options.JPEGSaveOptions(quality=12)
                doc.saveAs(str(jpg_output_path), jpg_options)
                
                # 3. Пересохранить исходный PSD
                doc.save()
                
            except Exception as e:
                tqdm.write(f"\nОШИБКА при обработке '{file_name}': {e}")
            finally:
                if doc:
                    doc.close(SaveOptions.DoNotSaveChanges)
                # Обновляем прогресс-бар после всех операций с файлом
                progress_bar.update(1)
                
    tqdm.write("\n<b>Пакетная обработка завершена.</b>")
    if IS_MANAGED_RUN:
        pysm_context.log_link(str(source_folder), "Открыть папку с PSD файлами")
    print("\n")    
    if IS_MANAGED_RUN:
        pysm_context.log_link(str(output_folder), "Открыть папку с JPG файлами")

    sys.exit(0)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()