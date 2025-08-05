# run_img_to_log.py

# --- Блок 1: Импорты (без изменений) ---
import argparse
import sys
from argparse import Namespace
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
        def write(msg, *args, **kwargs): print(msg)
    tqdm = TqdmWriteMock


# --- Блок 2: Получение конфигурации (изменен) ---
def get_config() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Демонстрирует вывод изображения в консоль PyScriptManager.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--img_source_path", type=str, help="Путь к файлу изображения для вывода.")
    parser.add_argument("--img_width", type=int, default=300, help="Ширина изображения в пикселях.")
    parser.add_argument("--img_align", type=str, default="left", choices=["left", "center", "right"], help="Выравнивание изображения.")
    parser.add_argument("--img_margin", type=int, default=5, help="Вертикальный отступ (сверху и снизу) в пикселях.")
    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    parser.add_argument("--img_desc", type=str, help="Текстовая подпись под изображением.")
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    
    return parser.parse_args()


# --- Блок 3: Основная логика скрипта (изменен) ---
def main():
    if not IS_MANAGED_RUN:
        print("ОШИБКА: Этот скрипт предназначен для запуска в среде PyScriptManager.", file=sys.stderr)
        sys.exit(1)

    config = get_config()

    if not config.img_source_path:
        tqdm.write("ОШИБКА: Параметр 'img_source_path' является обязательным.")
        sys.exit(1)
  

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # Вызов API-функции с новым параметром
    pysm_context.log_image(
        image_path=config.img_source_path,
        width=config.img_width,
        align=config.img_align,
        margin=config.img_margin,
        img_desc=config.img_desc,
    )
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
    sys.exit(0)


# --- Блок 4: Точка входа (без изменений) ---
if __name__ == "__main__":
    main()