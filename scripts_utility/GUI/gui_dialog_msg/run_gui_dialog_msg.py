# run_gui_dialog_msg.py

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

IS_MANAGED_RUN = False

try:
    current_script_path = Path(__file__).resolve()
    # Предполагаем, что скрипт лежит в папке внутри проекта, поднимаемся к корню
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Импорт PySM
    from pysm_lib import pysm_context
    from pysm_lib import theme_api        
    from pysm_lib.context_variable_ops import format_error
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.input_processor import InputProcessor
    # ResourceNode, StandardTreeBuilder, DashboardBuilder удалены, так как не используются
    IS_MANAGED_RUN = True
except ImportError as e:
    pysm_context = None
    ConfigResolver = None
    InputProcessor = None
    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"
    print(f"Ошибка импорта pysm_lib: {e}", file=sys.stderr)
    # Здесь можно не выходить, а продолжить в автономном режиме, если логика позволяет

# Импорт иконок
from _common import (
    icon_save,
    icon_save_error,
    icon_save_warning    
)

# Импортируем PySide6
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox
except ImportError:
    print("Ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    sys.exit(1)


# 2. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config():
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Показывает диалоговое окно и сохраняет выбор пользователя."
    )
    
    parser.add_argument(
        "--dlg_msg_var", 
        type=str,
        help="Имя переменной контекста для сохранения результата. Поддерживается точечная нотация, например project.confirmation.",
        default="var_user_choice"
    )
    parser.add_argument(
        "--dlg_msg_type", 
        type=str,
        choices=['ok', 'yes_no', 'yes_no_cancel'],
        help="Тип диалога.",
        default="yes_no"
    )
    parser.add_argument(
        "--dlg_msg_title", 
        type=str,
        help="Заголовок окна.",
        default="Подтверждение"
    )
    parser.add_argument(
        "--dlg_msg_message", 
        type=str,
        help="Текст сообщения.",
        default="Вы уверены, что хотите продолжить?"
    )

    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        config = argparse.Namespace()
        config.dlg_msg_var = resolver.get("dlg_msg_var")
        config.dlg_msg_type = resolver.get("dlg_msg_type")
        config.dlg_msg_title = resolver.get("dlg_msg_title")
        config.dlg_msg_message = resolver.get("dlg_msg_message")
        return config
    else:
        return parser.parse_args()


# 3. БЛОК: Основная логика
# ==============================================================================
def main():
    config = get_config()

    # Инициализация приложения
    q_app = QApplication.instance() or QApplication(sys.argv)
    
    # Применяем тему, если доступна
    if IS_MANAGED_RUN:
        theme_api.apply_theme_to_app(q_app)
    
    msg_box = QMessageBox()
    # Окно поверх всех окон (важно для диалогов, вызываемых из других процессов)
    msg_box.setWindowFlag(Qt.WindowStaysOnTopHint)
    msg_box.setWindowTitle(config.dlg_msg_title)
    msg_box.setText(config.dlg_msg_message)

    # Настройка кнопок
    button_map = {
        'ok': QMessageBox.Ok,
        'yes_no': QMessageBox.Yes | QMessageBox.No,
        'yes_no_cancel': QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
    }
    default_button_map = {
        'ok': QMessageBox.Ok,
        'yes_no': QMessageBox.Yes,
        'yes_no_cancel': QMessageBox.Yes
    }
    
    msg_box.setStandardButtons(button_map.get(config.dlg_msg_type, QMessageBox.Ok))
    msg_box.setDefaultButton(default_button_map.get(config.dlg_msg_type, QMessageBox.Ok))
        
    # Показ окна
    result_code = msg_box.exec()

    # Интерпретация результата
    result_string_map = {
        QMessageBox.Ok: "ok",
        QMessageBox.Yes: "yes",
        QMessageBox.No: "no",
        QMessageBox.Cancel: "cancel",
    }
    result_string = result_string_map.get(result_code, "unknown")

    # Сохранение в контекст
    if IS_MANAGED_RUN and pysm_context:
        try:
            # 1. Запись переменной
            processor = InputProcessor(config, pysm_context, IS_MANAGED_RUN)
            processor.process(
                raw_value=result_string,
                var_name=config.dlg_msg_var,
                value_type="string",
            )
            
            # 2. Логирование в HTML
            # Получаем стиль из темы (или дефолтный белый, если темы нет)
            style_color = theme_api.get_dynamic_style("tooltip_script_args_block", default="color: #fff;")
            buttons_labels_map = {
                'ok': "[<i>OK</i>]",
                'yes_no': "[<i>YES</i>] / [<i>NO</i>]",
                'yes_no_cancel': "[<i>YES</i>] / [<i>NO</i>] / [<i>CANCEL</i>]"
            }            
            # Получаем строку кнопок, если тип неизвестен - выводим сам тип
            buttons_display = buttons_labels_map.get(config.dlg_msg_type, config.dlg_msg_type)
            
            msg_html = f"<b>СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЮ</b><br><br>{config.dlg_msg_message} {buttons_display}"
            msg_html += f"<br><br>{icon_save} Выбор пользователя [<b>{result_string.upper()}</b>] сохранён в переменную <b>{config.dlg_msg_var.upper()}</b>"
            
            # ИСПРАВЛЕНИЕ: Используем div вместо table-тегов, так как log_html оборачивает в div
            info_html = f"""
            <tr><td style="{style_color} padding: 10px;">{msg_html}</td></tr>
            """
            pysm_context.log_html(info_html)

        except Exception as e:
            logger.error(format_error(f"Ошибка при сохранении в контекст: {e}"))
            sys.exit(1)
    else:
        print(f"Автономный режим. Выбор: {result_string}")

    # Логика кода выхода (для управления потоком выполнения)
    logger.info("")
    exit_code = 0
    if result_string in ["cancel", "unknown"]:
        exit_code = 1
        
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
