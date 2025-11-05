# run_gui_dialog_goto.py

"""
Тестовый и диагностический скрипт для PyScriptManager.

Назначение:
1. Продемонстрировать использование условных переходов.
2. Показать, как из скрипта можно получить список всех доступных для перехода
   экземпляров скриптов в текущем наборе.
3. Показать, как программно установить следующий скрипт для выполнения.
4. Поддерживает два режима: интерактивный (с GUI-диалогом) и
   программный (прямой переход по ID через аргумент командной строки).

Принцип работы:
- **Интерактивный режим (по умолчанию):**
  - Скрипт получает из контекста PySM список всех экземпляров (ID и имя).
  - С помощью GUI-диалога (QInputDialog) он предлагает пользователю выбрать,
    какой скрипт должен выполниться следующим.
  - Выбор пользователя сопоставляется с ID, который записывается в контекст.
- **Программный режим (если задан --dlg_goto_script_id):**
  - Диалоговое окно не отображается.
  - Значение, переданное в аргументе, напрямую записывается в контекст
    как ID следующего скрипта для выполнения.
"""

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
from argparse import Namespace

IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
    from pysm_lib import theme_api
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
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
        description="Показывает диалог выбора следующего скрипта для выполнения."
    )
    # 1. --- НАЧАЛО ИЗМЕНЕНИЙ В БЛОКЕ ---
    parser.add_argument(
        "--dlg_goto_title",
        type=str,
        default="Выбор следующего действия",
        help="Текст, который будет отображаться в заголовке диалогового окна."
    )
    parser.add_argument(
        "--dlg_goto_message",
        type=str,
        default="Выберите скрипт, к которому нужно перейти:",
        help="Основной текст сообщения, который будет показан пользователю."
    )
    parser.add_argument(
        "--dlg_goto_script_id",
        type=str,
        default=None,
        help="ID скрипта для прямого перехода без отображения диалога."
    )
    # 1. --- КОНЕЦ ИЗМЕНЕНИЙ В БЛОКЕ ---

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
        tqdm.write("ОШИБКА: Этот скрипт может быть запущен только в среде PySM.")
        sys.exit(1)

    config = get_config()

    # 2. --- НАЧАЛО ИЗМЕНЕНИЙ В БЛОКЕ ---
    # Проверяем, был ли передан ID скрипта напрямую для программного перехода
    if config.dlg_goto_script_id:
        # Режим прямого перехода без GUI
        selected_id = config.dlg_goto_script_id
        tqdm.write(f"Выполняется прямой переход к скрипту с ID: {selected_id}")
        try:
            pysm_context.set_next_script(selected_id)
            print(f"Очередь выполнения скриптов изменена. Следующий скрипт <i>(id: {selected_id})</i><br>")
            sys.exit(0)
        except Exception as e:
            tqdm.write(f"Критическая ошибка при установке следующего скрипта: {e}")
            sys.exit(1)
    else:
        # Интерактивный режим с GUI-диалогом (предыдущая логика)
        print("Получение списка экземпляров скриптов из текущего набора...")
        all_instances = pysm_context.list_instances()

        if not all_instances:
            tqdm.write("В текущем наборе не найдено экземпляров скриптов. Переход невозможен.")
            sys.exit(0)

        instance_names_for_dialog = [instance['name'] for instance in all_instances]
        name_to_id_map = {instance['name']: instance['id'] for instance in all_instances}

        q_app = QApplication.instance() or QApplication(sys.argv)
        theme_api.apply_theme_to_app(q_app)

        selected_name, ok = QInputDialog.getItem(
            None,
            config.dlg_goto_title,
            config.dlg_goto_message,
            instance_names_for_dialog,
            0,
            False
        )

        if ok and selected_name:
            selected_id = name_to_id_map.get(selected_name)
            if not selected_id:
                tqdm.write(f"Критическая ошибка: не удалось найти ID для имени '{selected_name}'")
                sys.exit(1)

            try:
                pysm_context.set_next_script(selected_id)
                print(f"Очередь выполнения скриптов изменена. Следующий скрипт <b>{selected_name}</b> <i>(id: {selected_id})</i><br>")
                sys.exit(0)
            except Exception as e:
                tqdm.write(f"Критическая ошибка при установке следующего скрипта: {e}")
                sys.exit(1)
        else:
            tqdm.write("Операция отменена пользователем. Выполнение будет остановлено.")
            sys.exit(1)
    # 2. --- КОНЕЦ ИЗМЕНЕНИЙ В БЛОКЕ ---


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()