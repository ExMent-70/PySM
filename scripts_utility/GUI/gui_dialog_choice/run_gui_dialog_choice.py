# run_gui_dialog_choice.py

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
import logging
from argparse import Namespace

IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
    from pysm_lib.context_variable_ops import format_error, format_success
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.input_processor import InputProcessor
    from pysm_lib import theme_api
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
    InputProcessor = None
    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"
    def format_success(var_name: str, value) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"

try:
    from PySide6.QtWidgets import QApplication, QInputDialog
except ImportError:
    print("Ошибка: для работы этого скрипта требуется PySide6. "
          "Установите его командой: pip install pyside6", file=sys.stderr)
    sys.exit(1)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# 2. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Показывает диалог для выбора значения из списка."
    )
    parser.add_argument(
        "--dlg_choice_var", type=str,
        help="Имя переменной для сохранения результата в контекст. Поддерживается точечная нотация, например project.choice.",
        required=True
    )
    parser.add_argument(
        "--dlg_choice_title", type=str,
        default="Выбор опции",
        help="Заголовок, который будет отображаться в верхней части диалогового окна."
    )
    parser.add_argument(
        "--dlg_choice_message", type=str,
        default="Выберите один из следующих вариантов:",
        help="Основной текст сообщения, который будет показан пользователю."
    )
    parser.add_argument(
        "--dlg_choice_list", type=str, nargs='+',
        help="Список вариантов для выбора.",
        required=True
    )
    parser.add_argument(
        "--dlg_choice_dvalue", type=str,
        help="Значение, которое будет выбрано в списке по умолчанию."
    )

    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Основная логика
# ==============================================================================
def main():
    """Основная функция-оркестратор."""
    config = get_config()

    if not IS_MANAGED_RUN or not pysm_context:
        logger.critical(format_error("Этот скрипт может быть запущен только в среде PySM"))
        sys.exit(1)

    choices = config.dlg_choice_list
    if isinstance(choices, str):
        choices = [item.strip() for item in choices.splitlines() if item.strip()]

    if not choices:
        logger.critical(format_error("Список для выбора (--dlg_choice_list) пуст."))
        sys.exit(1)

    processor = InputProcessor(config, pysm_context, IS_MANAGED_RUN)

    # Определяем значение по умолчанию с учетом приоритетов
    default_value = None
    
    # Приоритет 1: Значение из контекста
    context_value = processor.get_initial_value(
        config.dlg_choice_var,
        config.dlg_choice_dvalue or "",
    )
    if context_value is not None and context_value in choices:
        default_value = context_value
        logger.debug(f"Текущее значение переменной <i>{config.dlg_choice_var}</i> = <b>{context_value}</b>\n")
  
    # Приоритет 2: Значение из аргумента --dlg_choice_dvalue
    elif config.dlg_choice_dvalue and config.dlg_choice_dvalue in choices:
        default_value = config.dlg_choice_dvalue
        logger.debug(f"Используется значение по умолчанию из параметра: <b>'{default_value}'</b>")

    # Определяем индекс для QInputDialog
    current_index = 0
    if default_value:
        try:
            current_index = choices.index(default_value)
        except ValueError:
            # На случай, если значение есть, но его нет в списке
            logger.warning(f"Предупреждение: значение по умолчанию '{default_value}' не найдено в списке вариантов.")
    q_app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(q_app)
    
    selected_item, ok = QInputDialog.getItem(
        None,
        config.dlg_choice_title,
        config.dlg_choice_message,
        choices,
        current_index, # <--- Используем вычисленный индекс
        False
    )

    if ok and selected_item:
        try:
            processor.process(
                raw_value=selected_item,
                var_name=config.dlg_choice_var,
                value_type="string",
            )
            logger.debug(f"Переменная <i>{config.dlg_choice_var}</i> = <b>{selected_item}</b> успешно сохранена\n\n")
            logger.info(f"<b>{config.dlg_choice_title}</b>")
            logger.info(format_success(config.dlg_choice_var, selected_item) + "\n")
            sys.exit(0)
        except Exception as e:
            logger.critical(format_error(f"Ошибка при сохранении данных в контекст: {e}"))
            sys.exit(1)
    else:
        logger.critical(format_error("Операция отменена пользователем<br>"))
        sys.exit(1)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()
