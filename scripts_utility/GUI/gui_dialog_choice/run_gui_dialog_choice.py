# run_gui_dialog_choice.py

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
from argparse import Namespace

IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib import theme_api
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
    # Создаем заглушку для tqdm.write, чтобы избежать ошибок NameError
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs): print(msg)
    tqdm = TqdmWriteMock

try:
    from PySide6.QtWidgets import QApplication, QInputDialog
except ImportError:
    print("Ошибка: для работы этого скрипта требуется PySide6. "
          "Установите его командой: pip install pyside6", file=sys.stderr)
    sys.exit(1)


# 2. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Показывает диалог для выбора значения из списка."
    )
    parser.add_argument(
        "--dlg_choice_var", type=str,
        help="Имя переменной для сохранения результата в контекст.",
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
    
    if not IS_MANAGED_RUN or not pysm_context:
        tqdm.write("ОШИБКА: Этот скрипт может быть запущен только в среде PyScriptManager.")
        sys.exit(1)

    config = get_config()

    choices = config.dlg_choice_list
    if isinstance(choices, str):
        choices = [item.strip() for item in choices.splitlines() if item.strip()]

    if not choices:
        tqdm.write("ОШИБКА: Список для выбора (--dlg_choice_list) пуст.")
        sys.exit(1)

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # Определяем значение по умолчанию с учетом приоритетов
    default_value = None
    
    # Приоритет 1: Значение из контекста
    context_value = pysm_context.get(config.dlg_choice_var)
    if context_value is not None and context_value in choices:
        default_value = context_value
        print(f"Текущее значение переменной <i>{config.dlg_choice_var}</i> = <b>{context_value}</b>\n")
  
    # Приоритет 2: Значение из аргумента --dlg_choice_dvalue
    elif config.dlg_choice_dvalue and config.dlg_choice_dvalue in choices:
        default_value = config.dlg_choice_dvalue
        print(f"Используется значение по умолчанию из параметра: <b>'{default_value}'</b>")

    # Определяем индекс для QInputDialog
    current_index = 0
    if default_value:
        try:
            current_index = choices.index(default_value)
        except ValueError:
            # На случай, если значение есть, но его нет в списке
            tqdm.write(f"Предупреждение: значение по умолчанию '{default_value}' не найдено в списке вариантов.")
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
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
            pysm_context.set(config.dlg_choice_var, selected_item)
            print(f"Переменная <i>{config.dlg_choice_var}</i> = <b>{selected_item}</b> успешно сохранена\n\n")
            sys.exit(0)
        except Exception as e:
            tqdm.write(f"Критическая ошибка при сохранении в контекст: {e}")
            sys.exit(1)
    else:
        tqdm.write("Операция отменена пользователем. Выполнение набора скриптов будет остановлено.\n")
        sys.exit(1)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()