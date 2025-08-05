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
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs): print(msg)
    tqdm = TqdmWriteMock()

# Импорт ключевых зависимостей
try:
    from photoshop import api
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
    parser = argparse.ArgumentParser(description="Загружает .atn файл в палитру экшенов.")
    parser.add_argument(
        "--ps_atn_file_path", type=str, required=False,
        help="Путь к .atn файлу. Поддерживает шаблоны и авто-разрешение. Если не указан, появится диалог выбора."
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
    atn_path_str = config.ps_atn_file_path

    print("<b>Загрузка набора экшенов в палитру Photoshop</b>\n")

    if not atn_path_str:
        print("Путь к набору экшенов не указан. Запуск интерактивного режима...")
        if not PYSIDE_AVAILABLE:
            tqdm.write("ОШИБКА: PySide6 не найдена, интерактивный режим невозможен.")
            sys.exit(1)
        
        q_app = QApplication.instance() or QApplication(sys.argv)
        chosen_path, _ = QFileDialog.getOpenFileName(
            None, "Загрузить набор экшенов", "", "Наборы экшенов (*.atn);;Все файлы (*.*)"
        )
        
        if not chosen_path:
            tqdm.write("\n<b>Операция отменена пользователем.</b>\n")
            sys.exit(0)
        
        atn_path_str = chosen_path
    
    atn_path = Path(atn_path_str)
    if not atn_path.is_file() or atn_path.suffix.lower() != '.atn':
        tqdm.write(f"\nФайл не найден или не является .atn файлом: '{atn_path}'")
        sys.exit(1)

    try:
        print("Подключение к Adobe Photoshop...")
        app = api.Application()
        print(f"\nЗагрузка набора экшенов из файла:")
        print(f"<i>{atn_path.name}</i>\n")
        app.load(str(atn_path))
        print("<b>Набор экшенов успешно загружен</b>")
    except COMError:
        tqdm.write(f"ОШИБКА COM: Не удалось подключиться к Photoshop или загрузить файл.")
        sys.exit(1)
    except Exception as e:
        tqdm.write(f"ОШИБКА при загрузке набора экшенов: {e}")
        tqdm.write("Убедитесь, что файл не поврежден и является корректным .atn файлом.")
        sys.exit(1)

    #tqdm.write("--- Скрипт загрузки набора экшенов: Успешное завершение ---")
    sys.exit(0)

# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()