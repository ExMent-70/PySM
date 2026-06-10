# ==============================================================================
# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import signal
import sys
import time
from argparse import Namespace

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
    tqdm.write(
        "ОШИБКА: Необходимые библиотеки "
        "(psutil, photoshop-python-api, pywin32) не установлены."
    )

    tqdm.write(
        "Пожалуйста, выполните:\n"
        "pip install psutil photoshop-python-api pywin32"
    )

    sys.exit(1)


# ==============================================================================
# 2. БЛОК: Константы
# ==============================================================================
PHOTOSHOP_PROCESS_NAME_WIN = "Photoshop.exe"
PHOTOSHOP_PROCESS_NAME_MAC = "Adobe Photoshop"

WAIT_TIMEOUT = 60


# ==============================================================================
# 3. БЛОК: Получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения.
    """

    parser = argparse.ArgumentParser(
        description="Закрывает Adobe Photoshop."
    )

    parser.add_argument(
        "--force_kill",
        action="store_true",
        help=(
            "Принудительно завершить процесс Photoshop, "
            "если штатное закрытие не удалось."
        )
    )

    if IS_MANAGED_RUN and ConfigResolver:

        resolver = ConfigResolver(parser)

        return resolver.resolve_all()

    return parser.parse_args()


# ==============================================================================
# 4. БЛОК: Вспомогательные функции
# ==============================================================================
def get_process_name() -> str:
    """
    Возвращает имя процесса Photoshop для текущей ОС.
    """

    return (
        PHOTOSHOP_PROCESS_NAME_WIN
        if sys.platform == "win32"
        else PHOTOSHOP_PROCESS_NAME_MAC
    )


def find_photoshop_processes() -> list[psutil.Process]:
    """
    Возвращает список процессов Photoshop.
    """

    process_name = get_process_name().lower()

    result = []

    for proc in psutil.process_iter(["pid", "name"]):

        try:
            proc_name = proc.info["name"]

            if proc_name and process_name in proc_name.lower():
                result.append(proc)

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            continue

    return result


def is_photoshop_running() -> bool:
    """
    Проверяет, запущен ли Photoshop.
    """

    return len(find_photoshop_processes()) > 0


def wait_photoshop_closed(
    timeout: int = WAIT_TIMEOUT
) -> bool:
    """
    Ожидает полного завершения Photoshop.
    """

    start_time = time.time()

    while time.time() - start_time < timeout:

        if not is_photoshop_running():
            return True

        time.sleep(1)

    return False


# ==============================================================================
# 5. БЛОК: Штатное закрытие через API
# ==============================================================================
def close_photoshop_via_api() -> bool:
    """
    Пытается корректно закрыть Photoshop через API.

    Используется ExtendScript-команда закрытия Photoshop,
    так как app.quit() в photoshop-python-api
    работает некорректно.
    """

    try:
        print("Подключение к Photoshop через API...")

        app = api.Application()

        print(
            f"Подключено к: "
            f"<b>{app.name} {app.version}</b>"
        )

        print(
            "Отправка команды закрытия "
            "Photoshop через JavaScript..."
        )

        app.doJavaScript(
            """
            var idquit = charIDToTypeID("quit");
            executeAction(idquit, undefined, DialogModes.ALL);
            """
        )

        return True

    except com_error as e:

        tqdm.write(
            f"ОШИБКА COM: Не удалось подключиться "
            f"к Photoshop: {e}"
        )

        return False

    except Exception as e:

        tqdm.write(
            f"НЕИЗВЕСТНАЯ ОШИБКА: "
            f"{e}"
        )

        return False


# ==============================================================================
# 6. БЛОК: Принудительное завершение процесса
# ==============================================================================
def force_kill_photoshop() -> bool:
    """
    Принудительно завершает процессы Photoshop.
    """

    processes = find_photoshop_processes()

    if not processes:

        print("Процессы Photoshop не найдены.")

        return True

    print(
        f"Найдено процессов Photoshop: "
        f"{len(processes)}"
    )

    success = True

    for proc in processes:

        try:
            print(
                f"Завершение PID={proc.pid} "
                f"NAME={proc.name()}"
            )

            if sys.platform == "win32":
                proc.kill()

            else:
                proc.send_signal(signal.SIGKILL)

        except Exception as e:

            tqdm.write(
                f"Ошибка завершения процесса "
                f"PID={proc.pid}: {e}"
            )

            success = False

    return success


# ==============================================================================
# 7. БЛОК: Главная функция
# ==============================================================================
def main():
    """
    Главная функция скрипта.
    """

    config = get_config()

    print("<b>Закрытие Adobe Photoshop</b>")

    if not is_photoshop_running():

        print("Photoshop не запущен.")

        sys.exit(0)

    print("Обнаружен запущенный Photoshop.")

    closed_via_api = close_photoshop_via_api()

    if closed_via_api:

        print(
            f"Ожидание завершения Photoshop "
            f"(таймаут: {WAIT_TIMEOUT} сек)..."
        )

        if wait_photoshop_closed():

            print("Photoshop успешно закрыт.")

            sys.exit(0)

        print(
            "Photoshop не завершился "
            "после штатной команды закрытия."
        )

    else:

        print(
            "Штатное закрытие через API "
            "не удалось."
        )

    if config.force_kill:

        print(
            "Запущено принудительное "
            "завершение процессов..."
        )

        if not force_kill_photoshop():
            sys.exit(1)

        if wait_photoshop_closed():

            print(
                "Photoshop успешно "
                "завершен принудительно."
            )

            sys.exit(0)

        tqdm.write(
            "ОШИБКА: Не удалось завершить Photoshop."
        )

        sys.exit(1)

    tqdm.write(
        "ОШИБКА: Photoshop не был закрыт."
    )

    tqdm.write(
        "Попробуйте использовать параметр "
        "--force_kill"
    )

    sys.exit(1)


# ==============================================================================
# 8. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()