# run_log_html.py

import argparse
import sys
import pathlib
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
        def write(msg, *args, **kwargs):
            print(msg)
    tqdm = TqdmWriteMock


def get_config() -> Namespace:
    """
    Определяет аргументы для вывода HTML-контента.
    """
    parser = argparse.ArgumentParser(
        description="Выводит произвольный HTML-текст или содержимое HTML-файла в консоль PyScriptManager.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    parser.add_argument("--html_content", type=str, help="Строка с HTML-разметкой для вывода.")
    parser.add_argument("--html_file", type=str, help="Путь к файлу, содержимое которого нужно вывести.")
    parser.add_argument("--html_align", type=str, default="left", choices=["left", "center", "right"], help="Выравнивание контейнера.")
    parser.add_argument("--html_margin", type=int, default=5, help="Вертикальный отступ (сверху и снизу) в пикселях.")

    if IS_MANAGED_RUN:
        # Принудительно обрабатываем html_file как путь для разрешения относительных путей
        resolver = ConfigResolver(parser, force_path_args=["html_file"])
        return resolver.resolve_all()
    
    return parser.parse_args()


def main():
    if not IS_MANAGED_RUN:
        print("ОШИБКА: Этот скрипт предназначен для запуска в среде PyScriptManager.", file=sys.stderr)
        sys.exit(1)

    config = get_config()

    if not config.html_content and not config.html_file:
        tqdm.write("ОШИБКА: Необходимо указать хотя бы один источник данных: '--html_content' или '--html_file'.")
        sys.exit(1)

    # ИЗМЕНЕНИЕ: Вместо склеивания строк в одну, мы выводим их последовательно.
    # Это гарантирует, что даже если в html_content есть незакрытый тег,
    # html_file будет выведен отдельным блоком, а не вложенным.

    pysm_context.log_html(
        html_content="<br>",
        align=config.html_align,
        margin=config.html_margin,
    )

    # 1. Вывод текста (если есть)
    if config.html_content:
        pysm_context.log_html(
            html_content=str(config.html_content),
            align=config.html_align,
            margin=config.html_margin,
        )

    # 2. Вывод содержимого файла (если есть)
    if config.html_file:
        file_path = pathlib.Path(config.html_file)
        if not file_path.is_file():
            tqdm.write(f"ОШИБКА: Файл не найден: {file_path}")
            # Не прерываем выполнение, если текст уже был выведен, но можно выйти с ошибкой, если критично
            sys.exit(1)
        
        try:
            file_content = file_path.read_text(encoding="utf-8")
            if file_content.strip():
                pysm_context.log_html(
                    html_content=file_content,
                    align=config.html_align,
                    margin=config.html_margin,
                )
        except Exception as e:
            tqdm.write(f"ОШИБКА: Не удалось прочитать файл '{file_path}': {e}")
            sys.exit(1)
        finally:
            pysm_context.log_html(
                html_content="<br>",
                align=config.html_align,
                margin=config.html_margin,
            )

    sys.exit(0)


if __name__ == "__main__":
    main()