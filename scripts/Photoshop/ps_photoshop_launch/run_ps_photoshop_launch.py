# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import os
import subprocess
import sys
import time
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
    # Заглушка для tqdm, если он не установлен в автономной среде
    # Используем простую print-функцию для вывода сообщений
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg)
    tqdm = TqdmWriteMock()


# Импорт сторонних библиотек
try:
    import psutil
    from photoshop import api
    from pywintypes import com_error
except ImportError:
    tqdm.write("ОШИБКА: Необходимые библиотеки (psutil, photoshop-python-api, pywin32) не установлены.")
    tqdm.write("Пожалуйста, выполните: pip install psutil photoshop-python-api pywin32")
    sys.exit(1)


# 2. БЛОК: Константы
# ==============================================================================
PHOTOSHOP_PROCESS_NAME_WIN = "Photoshop.exe"
PHOTOSHOP_PROCESS_NAME_MAC = "Adobe Photoshop"
PHOTOSHOP_EXECUTABLE_REL_PATH_MAC = Path("Contents/MacOS/Adobe Photoshop")
WAIT_TIMEOUT = 160  # Таймаут ожидания запуска в секундах


# 3. БЛОК: Получение конфигурации (ОБНОВЛЕННЫЙ ПАТТЕРН)
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver,
    который автоматически обрабатывает пути и шаблоны.
    """
    parser = argparse.ArgumentParser(
        description="Запускает Adobe Photoshop и проверяет подключение к нему через API."
    )
    parser.add_argument(
        "--ps_app_dir",
        type=str,
        default=None,
        help="Путь к ПАПКЕ, где находится приложение Photoshop. Поддерживает шаблоны и авто-разрешение путей."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        # ConfigResolver сам позаботится о разрешении путей и шаблонов.
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        # В автономном режиме используем стандартный парсинг.
        return parser.parse_args()

# 4. БЛОК: Вспомогательные функции для работы с Photoshop
# ==============================================================================
def is_photoshop_running() -> bool:
    """Проверяет, запущен ли процесс Photoshop."""
    process_name = (PHOTOSHOP_PROCESS_NAME_WIN if sys.platform == "win32" else PHOTOSHOP_PROCESS_NAME_MAC)
    for proc in psutil.process_iter(['name']):
        if process_name.lower() in proc.info['name'].lower():
            print(f"Обнаружен запущенный процесс Photoshop: {proc.info['name']}")
            return True
    return False


def find_photoshop_executable_auto() -> str | None:
    """Ищет путь к исполняемому файлу Photoshop в стандартных местах."""
    print("Запуск автоматического поиска программы Adobe Photoshop...")
    if sys.platform == "win32":
        import winreg
        photoshop_keys = [
            r"SOFTWARE\Adobe\Photoshop\260.0", # v2025
            r"SOFTWARE\Adobe\Photoshop\210.0", # v2024
            r"SOFTWARE\Adobe\Photoshop\190.0", # CC 2020+
            r"SOFTWARE\Adobe\Photoshop\180.0", # CC 2019
        ]
        for key in photoshop_keys:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as reg_key:
                    path, _ = winreg.QueryValueEx(reg_key, "ApplicationPath")
                    exe_path = Path(path) / PHOTOSHOP_PROCESS_NAME_WIN
                    if exe_path.exists():
                        print(f"Путь к программе Adobe Photoshop найден в реестре:")
                        print(f"<i>{exe_path}</i>")

                        return str(exe_path)
            except FileNotFoundError:
                continue
    elif sys.platform == "darwin":
        base_dir = Path("/Applications")
        for version_dir in base_dir.iterdir():
            if "Adobe Photoshop" in version_dir.name and version_dir.is_dir():
                app_bundle = next(version_dir.glob("*.app"), None)
                if app_bundle:
                    binary_path = app_bundle / PHOTOSHOP_EXECUTABLE_REL_PATH_MAC
                    if binary_path.exists():
                        print(f"Photoshop найден: {binary_path}")
                        return str(binary_path)
    print("Автоматический поиск программы Adobe Photoshop не дал результатов.")
    return None


def get_executable_from_path(search_path_str: str) -> str | None:
    """Ищет исполняемый файл Photoshop в указанной директории."""
    if not search_path_str: return None
        
    print(f"Поиск программы Adobe Photoshop в указанной папке: {search_path_str}")
    search_path = Path(search_path_str)
    if not search_path.is_dir():
        print(f"ОШИБКА: Указанный путь '{search_path}' не является папкой или не существует.")
        return None

    if sys.platform == "win32":
        potential_path = search_path / PHOTOSHOP_PROCESS_NAME_WIN
        if potential_path.is_file():
            return str(potential_path)
    elif sys.platform == "darwin":
        # На macOS ищем бандл приложения *.app внутри указанной папки
        app_bundle = next(search_path.glob("Adobe Photoshop*.app"), None)
        if app_bundle:
            potential_path = app_bundle / PHOTOSHOP_EXECUTABLE_REL_PATH_MAC
            if potential_path.is_file():
                return str(potential_path)

    print(f"В папке '{search_path}' исполняемый файл программы Adobe Photoshop не найден.")
    return None


# 5. БЛОК: Основные логические функции
# ==============================================================================
def find_and_launch_photoshop(config: Namespace) -> bool:
    """Определяет путь к Photoshop, запускает его и ожидает готовности."""
    photoshop_exe_path = None
    # Приоритет 1: Путь, указанный пользователем.
    # ConfigResolver уже преобразовал его в абсолютный путь.
    if config.ps_app_dir:
        photoshop_exe_path = get_executable_from_path(config.ps_app_dir)

    # Приоритет 2: Автоматический поиск, если в указанном пути ничего не найдено.
    if not photoshop_exe_path:
        photoshop_exe_path = find_photoshop_executable_auto()

    if not photoshop_exe_path:
        print("КРИТИЧЕСКАЯ ОШИБКА: Не удалось найти исполняемый файл программы Adobe Photoshop.")
        print("Проверьте, что программ Adobe Photoshop установлена на ваш компьютер.")
        return False

    print(f"\nИспользуется исполняемый файл:")
    print(f"<i>{photoshop_exe_path}</i>\n")    
    try:
        subprocess.Popen([photoshop_exe_path])
    except OSError as e:
        tqdm.write(f"ОШИБКА: Не удалось запустить процесс Photoshop: {e}")
        return False

    print(f"Ожидание появления процесса Photoshop (таймаут: {WAIT_TIMEOUT} сек)...")
    start_time = time.time()
    while time.time() - start_time < WAIT_TIMEOUT:
        if is_photoshop_running():
            print("Процесс Photoshop обнаружен. Ожидание полной загрузки (10 сек)...")
            time.sleep(10)
            return True
        time.sleep(1)

    tqdm.write("ОШИБКА: Таймаут ожидания истек. Photoshop не запустился.")
    return False


def connect_to_photoshop_api() -> bool:
    """Пытается подключиться к Photoshop через COM/API."""
    try:
        print("\nПодключение к Photoshop через API...")
        app = api.Application()
        print(f"Успешное подключение к: <b>{app.name} {app.version}</b>")
        print(f"\nОткрыто документов: <b>{len(app.documents)}</b>")
        return True
    except com_error as e:
        tqdm.write(f"ОШИБКА COM: Не удалось подключиться к Photoshop: {e}")
        tqdm.write("Убедитесь, что Photoshop полностью загрузился и не показывает диалоговых окон.")
        return False
    except Exception as e:
        tqdm.write(f"НЕИЗВЕСТНАЯ ОШИБКА: Произошла ошибка при подключении: {e}")
        return False


# 6. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    """Главная функция скрипта."""
    # Получаем конфигурацию одним вызовом.
    config = get_config()
    
    print("<b>Запуск программы Adobe Photoshop</b>")
    
    photoshop_ready = False
    if is_photoshop_running():
        photoshop_ready = True
    else:
        print("Photoshop не запущен. Попытка запуска...")
        photoshop_ready = find_and_launch_photoshop(config)

    if not photoshop_ready:
        sys.exit(1)

    if not connect_to_photoshop_api():
        sys.exit(1)

    #tqdm.write("--- Скрипт запуска Photoshop: Успешное завершение ---")
    sys.exit(0)


# 7. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()