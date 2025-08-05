# pysm_lib/app_constants.py
import pathlib
import sys


# === Блок 1: Путь к корню приложения (НОВЫЙ) ===
def get_application_root() -> pathlib.Path:
    """Определяет корневую директорию приложения."""
    if getattr(sys, "frozen", False):
        # Если приложение "заморожено" (скомпилировано в .exe)
        return pathlib.Path(sys.executable).parent.resolve()
    else:
        # Если запускается как .py скрипт
        return pathlib.Path(__file__).parent.parent.resolve()


APPLICATION_ROOT_DIR = get_application_root()
# === Блок 1.1: Константы для файлов (НОВЫЙ) ===
COLLECTION_EXTENSION = ".pysmc"
COLLECTION_FILE_TYPE_NAME = "PySM Collection"


# === Блок 1.1: Пути к иконкам ===
ICON_DIR = APPLICATION_ROOT_DIR / "pysm_lib" / "gui" / "icons"
ICON_OPEN_DIR = str(ICON_DIR / "folder-open.png")
ICON_REFRESH = str(ICON_DIR / "refresh.png")
ICON_CONSOLE = str(ICON_DIR / "terminal.png")
ICON_SETTINGS = str(ICON_DIR / "settings.png")
ICON_EXIT = str(ICON_DIR / "exit.png")
