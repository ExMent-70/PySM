# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import sys
from argparse import Namespace
from pathlib import Path

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    # Заглушка для tqdm
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg)
    tqdm = TqdmWriteMock()

# Импорт ключевых зависимостей
try:
    from photoshop import api
    from photoshop.api import save_options
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
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver,
    который автоматически обрабатывает пути и шаблоны.
    """
    parser = argparse.ArgumentParser(
        description="Сохраняет активный документ Photoshop, имитируя поведение Ctrl+S."
    )
    parser.add_argument(
        "--ps_save_path", type=str, default=None,
        help="Принудительно использовать 'Сохранить как...' с указанным путем. Поддерживает шаблоны и авто-разрешение путей."
    )
    parser.add_argument(
        "--ps_save_as_copy", action=argparse.BooleanOptionalAction, default=False,
        help="Принудительно сохранить как копию (актуально только с --ps_save_path)."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Вспомогательная функция "Сохранить как"
# ==============================================================================
def _save_document_as(doc, file_path, as_copy):
    """Инкапсулирует логику 'Save As' для переиспользования."""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.psd':
        options = save_options.PhotoshopSaveOptions()
        options.layers = True
    elif file_ext == '.png':
        options = save_options.PNGSaveOptions()
        options.compression = 6
    elif file_ext in ['.jpg', '.jpeg']:
        options = save_options.JPEGSaveOptions()
        options.quality = 12
    elif file_ext == '.tif':
        options = save_options.TiffSaveOptions()
    else:
        # Для неизвестных расширений используем опции по умолчанию
        options = save_options.PhotoshopSaveOptions()
    
    try:
        print(f"\nВыполняется команда <b>Сохранить как...</b>:")
        print(f"<i>{file_path}</i>")
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        doc.saveAs(file_path, options, asCopy=as_copy)
        print("\n<b>Документ успешно сохранен.</b>\n")
    except Exception as e:
        tqdm.write(f"Ошибка при выполнении <b>Сохранить как...</b>: {e}")
        sys.exit(1)


# 4. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    """Интеллектуально сохраняет активный документ."""
    # Получаем готовую конфигурацию одним вызовом
    config = get_config()
    
    print("<b>Сохранение документа Adobe Photoshop</p>")

    try:
        app = api.Application()
        if not app.documents:
            tqdm.write("Нет открытых документов для сохранения.")
            sys.exit(0)
        doc = app.activeDocument
    except Exception as e:
        tqdm.write(f"Ошибка подключения к Photoshop: {e}")
        sys.exit(1)

    if doc.saved and not config.ps_save_path:
        # Если есть --ps_save_path, то нужно сохранить в любом случае (Save As)
        tqdm.write(f"\nДокумент <i>{doc.name}</i> не требует сохранения.\n")
        sys.exit(0)

    if not doc.saved:
        print(f"\nОбнаружены несохраненные изменения в документе '{doc.name}'.")

    if config.ps_save_path:
        # ConfigResolver уже преобразовал путь в абсолютный.
        _save_document_as(doc, config.ps_save_path, config.ps_save_as_copy)
    else:
        try:
            _ = doc.fullName
            print(f"\nВыполняется сохранение документа:'")
            print(f"<i>{doc.name}</i>")
            doc.save()
            print("\n<b>Документ успешно сохранен.</b>\n")
        
        except COMError:
            tqdm.write(f"Документ '{doc.name}' новый. Открываю диалоговое окно сохранения...")
            
            if not PYSIDE_AVAILABLE:
                tqdm.write("ОШИБКА: PySide6 не найдена, интерактивный режим невозможен.")
                sys.exit(1)

            q_app = QApplication.instance() or QApplication(sys.argv)

            chosen_path, _ = QFileDialog.getSaveFileName(
                None, "Сохранить как...", str(Path.home() / doc.name),
                "Photoshop Document (*.psd);;PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tif)"
            )
            
            if not chosen_path:
                tqdm.write("\n<b>Сохранение отменено пользователем.</b>\n")
                sys.exit(0)
            
            _save_document_as(doc, chosen_path, as_copy=False)

        except Exception as e:
            tqdm.write(f"Непредвиденная ошибка при сохранении: {e}")
            sys.exit(1)

# 5. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()