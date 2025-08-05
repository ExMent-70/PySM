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
    # Заглушка для tqdm, если он не установлен
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg)
    tqdm = TqdmWriteMock()

# Импорт ключевых зависимостей
try:
    from photoshop import api
    from pywintypes import com_error
except ImportError:
    tqdm.write("ОШИБКА: Библиотека 'photoshop-python-api' не найдена.")
    tqdm.write("Пожалуйста, выполните: pip install photoshop-python-api pywin32")
    sys.exit(1)

# Импорт GUI-библиотеки (опционально, нужна только для диалога)
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
        description="Открывает документ в Adobe Photoshop."
    )
    parser.add_argument(
        "--ps_file_path",
        type=str,
        default=None,
        help="Путь к файлу для открытия. Поддерживает шаблоны и авто-разрешение путей. Если не указан, появится диалог выбора файла."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Основные логические функции
# ==============================================================================
def open_document(file_path: str) -> bool:
    """
    Открывает указанный документ в Photoshop.
    Возвращает True в случае успеха, False в случае ошибки.
    """
    path_obj = Path(file_path)
    if not path_obj.is_file():
        tqdm.write(f"ОШИБКА: Файл не найден по пути '{file_path}'.")
        return False
        
    try:
        print("Подключение к Adobe Photoshop...")
        app = api.Application()
        print(f"Открытие документа '{path_obj.name}'...")
        app.open(file_path)
        print(f"\n<b>Документ '{path_obj.name}' успешно открыт.</b>\n")
        return True
    except com_error as e:
        tqdm.write(f"ОШИБКА COM: Не удалось открыть файл: {e}")
        tqdm.write("Убедитесь, что Photoshop запущен и не заблокирован диалоговым окном.")
        return False
    except Exception as e:
        tqdm.write(f"НЕИЗВЕСТНАЯ ОШИБКА: {e}")
        tqdm.write("Убедитесь, что файл не поврежден и поддерживается Photoshop.")
        return False

def open_document_via_dialog() -> bool:
    """
    Открывает диалог выбора файла и пытается открыть выбранный файл.
    Возвращает True, если файл был выбран и успешно открыт.
    """
    print("\n<i>Путь к файлу не указан. Запуск интерактивного режима...</i>\n")
    
    if not PYSIDE_AVAILABLE:
        tqdm.write("ОШИБКА: Библиотека PySide6 не найдена, интерактивный режим невозможен.")
        tqdm.write("Пожалуйста, укажите путь к файлу через параметр --ps_file_path или установите PySide6.")
        return False

    # Гарантируем наличие экземпляра QApplication для работы диалога
    q_app = QApplication.instance() or QApplication(sys.argv)
    
    chosen_path, _ = QFileDialog.getOpenFileName(
        None,
        "Выберите файл для открытия в Photoshop",
        "",
        "Изображения (*.psd *.psb *.png *.jpg *.jpeg *.tif *.tiff);;Все файлы (*.*)"
    )
    
    if not chosen_path:
        tqdm.write("<b>Операция отменена пользователем.</b>\n")
        # Отмена пользователем не является ошибкой, поэтому завершаемся с успехом
        sys.exit(0)
    
    return open_document(chosen_path)


# 4. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    """
    Главная функция: получает конфигурацию и решает, как открыть файл —
    автоматически по пути или интерактивно через диалог.
    """
    # Получаем готовую конфигурацию одним вызовом.
    config = get_config()

    print("<b>Открытие документа Adobe Photoshop</b>")

    success = False
    if config.ps_file_path:
        # Сценарий 1: Путь задан, работаем в автоматическом режиме.
        # ConfigResolver уже преобразовал его в абсолютный путь.
        success = open_document(config.ps_file_path)
    else:
        # Сценарий 2: Путь не задан, работаем в интерактивном режиме
        success = open_document_via_dialog()

    if success:
        #tqdm.write("--- Скрипт открытия документа: Успешное завершение ---")
        sys.exit(0)
    else:
        tqdm.write("--- Скрипт открытия документа: Завершено с ошибкой ---")
        sys.exit(1)


# 5. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()