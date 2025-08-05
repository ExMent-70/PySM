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
    # Заглушка для tqdm
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg)
    tqdm = TqdmWriteMock()

# Импорт ключевых зависимостей
try:
    from photoshop import api
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
        description="Запускает экшен из указанного набора для активного документа."
    )
    parser.add_argument(
        "--ps_action_set", type=str, required=True,
        help="Имя набора экшенов (папки), в котором находится экшен. Поддерживает шаблоны."
    )
    parser.add_argument(
        "--ps_action_name", type=str, required=True,
        help="Имя самого экшена, который нужно запустить. Поддерживает шаблоны."
    )
    
    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    """Запускает указанный экшен для активного документа Photoshop."""
    # Получаем готовую конфигурацию одним вызовом
    config = get_config()

    print("<b>Запуск экшена Photoshop</b>\n")

    try:
        print("Подключение к Adobe Photoshop...")
        app = api.Application()
        print("Подключение выполнено успешно.\n")
        if not app.documents:
            tqdm.write("ОШИБКА: Для запуска экшена должен быть открыт хотя бы один документ.")
            sys.exit(1)
    except COMError:
        tqdm.write("ОШИБКА: Не удалось подключиться к Photoshop. Убедитесь, что он запущен.")
        sys.exit(1)
    except Exception as e:
        tqdm.write(f"НЕИЗВЕСТНАЯ ОШИБКА при подключении: {e}")
        sys.exit(1)

    try:
        # ConfigResolver уже обработал шаблоны в именах
        print(f"Запуск экшена <i>{config.ps_action_name}</i> из набора <i>{config.ps_action_set}</i>...")
        app.doAction(config.ps_action_name, config.ps_action_set)
        print(f"\n<b>Экшен <i>{config.ps_action_name}</i> успешно выполнен для документа <i>{app.activeDocument.name}</i></b>.\n")
    except Exception as e:
        tqdm.write(f"ОШИБКА при выполнении экшена: {e}")
        tqdm.write("Убедитесь, что имена набора и экшена указаны правильно (с учетом регистра и пробелов).")
        tqdm.write("Также убедитесь, что условия для выполнения экшена соблюдены (например, выбран нужный слой).")
        sys.exit(1)
        
    #tqdm.write("--- Скрипт запуска экшена: Успешное завершение ---")
    sys.exit(0)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()