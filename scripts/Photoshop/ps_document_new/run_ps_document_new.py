# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import sys
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
    # Заглушка для tqdm
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg)
    tqdm = TqdmWriteMock()

# Импорт ключевых зависимостей
try:
    from photoshop import api
    from photoshop.api.enumerations import NewDocumentMode
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
        description="Создает новый документ в Adobe Photoshop."
    )
    parser.add_argument(
        "--ps_width", type=int, default=1920,
        help="Ширина документа в пикселях."
    )
    parser.add_argument(
        "--ps_height", type=int, default=1080,
        help="Высота документа в пикселях."
    )
    parser.add_argument(
        "--ps_resolution", type=int, default=300,
        help="Разрешение документа в dpi."
    )
    parser.add_argument(
        "--ps_name", type=str, default="Новый документ",
        help="Имя нового документа. Поддерживает шаблоны."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        # ConfigResolver сам получит значения из контекста или командной строки.
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    """
    Главная функция: подключается к Photoshop и создает новый документ
    с заданными параметрами.
    """
    # Получаем готовую конфигурацию одним вызовом.
    config = get_config()

    print("<b>Создание нового документа Adobe Photoshop</b>")

    try:
        print("Подключение к Adobe Photoshop...")
        app = api.Application()
        print("Подключение выполнено успешно.")
    except COMError:
        tqdm.write("ОШИБКА: Не удалось подключиться к Photoshop.")
        tqdm.write("Пожалуйста, убедитесь, что Photoshop запущен и готов к работе.")
        sys.exit(1)
    except Exception as e:
        tqdm.write(f"НЕИЗВЕСТНАЯ ОШИБКА при подключении: {e}")
        sys.exit(1)

    try:
        print(
            f"Создание документа '{config.ps_name}' с размерами "
            f"{config.ps_width}x{config.ps_height}px, {config.ps_resolution}dpi..."
        )
        app.documents.add(
            width=config.ps_width,
            height=config.ps_height,
            resolution=config.ps_resolution,
            name=config.ps_name,
            mode=NewDocumentMode.NewRGB
        )
        print(f"<b>Документ '{config.ps_name}' успешно создан.</b>\n")
        
    except COMError as e:
        tqdm.write(f"ОШИБКА COM: Не удалось создать документ. {e}")
        tqdm.write("Возможно, введены некорректные параметры (например, слишком большой размер).")
        sys.exit(1)
    except Exception as e:
        tqdm.write(f"НЕИЗВЕСТНАЯ ОШИБКА при создании документа: {e}")
        sys.exit(1)
    
    #tqdm.write("--- Скрипт создания документа: Успешное завершение ---")
    sys.exit(0)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()