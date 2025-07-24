# run_gui_dialog_goto.py

"""
Тестовый и диагностический скрипт для PyScriptManager.

Назначение:
1. Продемонстрировать использование условных переходов.
2. Показать, как из скрипта можно получить список всех доступных для перехода
   экземпляров скриптов в текущем наборе.
3. Показать, как программно установить следующий скрипт для выполнения.

Принцип работы:
- Скрипт получает из контекста PySM список всех экземпляров (ID и имя).
- С помощью GUI-диалога (QInputDialog) он предлагает пользователю выбрать,
  какой скрипт должен выполниться следующим, показывая ему понятные имена.
- Выбор пользователя (имя) сопоставляется с ID, и этот ID записывается
  в контекст с помощью `pysm_context.set_next_script()`.
- Исполнитель PySM (AppController) на следующем шаге прочтет это значение и
  передаст управление выбранному скрипту.
"""

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
from argparse import Namespace

IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
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


# 2. БЛОК: Определение и получение конфигурации (без изменений)
# ==============================================================================
def get_config() -> Namespace:
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Показывает диалог выбора следующего скрипта для выполнения."
    )
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
    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Основная логика (ИЗМЕНЕН)
# ==============================================================================
def main():
    """Основная функция-оркестратор."""
    
    if not IS_MANAGED_RUN or not pysm_context:
        tqdm.write("ОШИБКА: Этот скрипт может быть запущен только в среде PySM.")
        sys.exit(1)

    config = get_config()

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    print("Получение списка экземпляров скриптов из текущего набора...")
    # Используем новый метод, возвращающий список словарей
    all_instances = pysm_context.list_instances()

    if not all_instances:
        tqdm.write("В текущем наборе не найдено экземпляров скриптов. Переход невозможен.")
        sys.exit(0)

    # Готовим данные для диалога:
    # 1. Список имен для отображения пользователю.
    # 2. Словарь для поиска ID по имени после выбора.
    instance_names_for_dialog = [instance['name'] for instance in all_instances]
    name_to_id_map = {instance['name']: instance['id'] for instance in all_instances}

    #tqdm.write(f"Найдено {len(all_instances)} экземпляров. Показ диалога выбора.")

    q_app = QApplication.instance() or QApplication(sys.argv)
    
    # Показываем пользователю понятные имена
    selected_name, ok = QInputDialog.getItem(
        None,
        config.dlg_goto_title,
        config.dlg_goto_message,
        instance_names_for_dialog, # <--- Передаем список имен
        0,
        False
    )

    if ok and selected_name:
        # Пользователь выбрал имя, теперь находим соответствующий ID
        selected_id = name_to_id_map.get(selected_name)
        if not selected_id:
            tqdm.write(f"Критическая ошибка: не удалось найти ID для имени '{selected_name}'")
            sys.exit(1)

        #tqdm.write(f"Пользователь выбрал: '{selected_name}' (ID: {selected_id})")
        try:
            pysm_context.set_next_script(selected_id)
            print(f"Очередь выполнения скриптов изменена. Следующий скрипт <b>{selected_name}</b> <i>(id: {selected_id})</i><br>")
            sys.exit(0)
        except Exception as e:
            tqdm.write(f"Критическая ошибка при установке следующего скрипта: {e}")
            sys.exit(1)
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
    else:
        tqdm.write("Операция отменена пользователем. Выполнение будет остановлено.")
        sys.exit(1)


# 4. БЛОК: Точка входа (без изменений)
# ==============================================================================
if __name__ == "__main__":
    main()