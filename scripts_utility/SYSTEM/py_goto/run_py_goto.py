# run_gui_dialog_goto.py

"""
Тестовый и диагностический скрипт для PyScriptManager.

Назначение:
1. Продемонстрировать использование условных переходов.
2. Показать, как программно установить следующий скрипт для выполнения.



"""

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
import logging
from argparse import Namespace
from pathlib import Path

IS_MANAGED_RUN = False
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    #from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    print("Убедитесь, что структура папок верна и все зависимости установлены.", file=sys.stderr)
    sys.exit(1)
    
from _common import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_delete,
    icon_play,
    icon_save,
    icon_arrow_sub,
    icon_save_warning,
    icon_save_error
)


# Инициализируем глобальный логгер
logger = logging.getLogger(__name__)    



# 2. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Показывает диалог выбора следующего скрипта для выполнения."
    )
    parser.add_argument(
        "--dlg_goto_script_id",
        type=str,
        default=None,
        help="ID скрипта для прямого перехода без отображения диалога."
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

    log_level = pysm_context.get("sys_log_level", "INFO") if IS_MANAGED_RUN and pysm_context else "INFO"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s", stream=sys.stdout)

    config = get_config()

    if config.dlg_goto_script_id:
        # Режим прямого перехода без GUI
        selected_id = config.dlg_goto_script_id
        try:
            pysm_context.set_next_script(selected_id)
            logger.info(f"{icon_play} Порядок выполнения скриптов изменён")
            logger.debug(f"Следующий скрипт <i>id: {selected_id}</i><br>")
            sys.exit(0)
        except Exception as e:
            logger.error(f"{icon_error} Критическая ошибка при установке следующего скрипта: {e}")
            sys.exit(1)
    else:
        #print(f"{icon_error} В текущем наборе не найдено экземпляров скриптов. Переход невозможен.")
        sys.exit(0)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()