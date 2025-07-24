# 1. БЛОК: Импорты и константы
# ==============================================================================
import argparse
import re
import sys

# Определяем, запущен ли скрипт под управлением PySM
IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None

# Импортируем PySide6
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox, QLineEdit
except ImportError:
    print("Ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    sys.exit(1)

# Новая структура для хранения пресетов валидации
VALIDATION_PRESETS = {
    "not_empty": {
        "pattern": r".+",
        "description": "Требуется любой непустой текст."
    },
    "integer": {
        "pattern": r"^-?\d+$",
        "description": "Требуется целое число (например, -10, 0, 123)."
    },
    "positive_integer": {
        "pattern": r"^\d+$",
        "description": "Требуется положительное целое число или ноль (например, 0, 5, 100)."
    },
    "float": {
        "pattern": r"^-?\d+(\.\d+)?$",
        "description": "Требуется число с плавающей точкой (например, -3.14, 10, 99.9)."
    },
    "email": {
        "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "description": "Требуется корректный адрес электронной почты."
    }
}


# 2. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config():
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Показывает диалог для ввода значения с гибкой валидацией."
    )
    
    parser.add_argument(
        "--dlg_input_var", type=str,
        help="Имя переменной для сохранения результата.",
        default="dlg_input_user_var"
    )
    parser.add_argument(
        "--dlg_input_msg", type=str,
        help="Текст-приглашение для ввода.",
        default="Введите значение:"
    )
    parser.add_argument(
        "--dlg_input_title", type=str,
        help="Заголовок диалогового окна.",
        default="Ввод значения"
    )
    parser.add_argument(
        "--dlg_input_dvalue", type=str,
        help="Значение в поле ввода по умолчанию.",
        default=""
    )
    parser.add_argument(
        "--dlg_input_valid_type", type=str,
        help="Тип валидации вводимого значения.",
        choices=["none", "custom"] + list(VALIDATION_PRESETS.keys()),
        default="none"
    )
    parser.add_argument(
        "--dlg_input_custom_regexp", type=str,
        help="Пользовательский шаблон регулярного выражения (если dlg_input_valid_type='custom')."
    )
    parser.add_argument(
        "--dlg_input_custom_regexp_desc", type=str,
        help="Описание для пользовательского шаблона (если dlg_input_valid_type='custom')."
    )

    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        config = argparse.Namespace()
        
        # Правильная итерация для создания объекта конфигурации
        for action in parser._actions:
            if action.dest != 'help':
                value = resolver.get(action.dest)
                setattr(config, action.dest, value)
                
        return config
    else:
        return parser.parse_args()


# 3. БЛОК: Основная логика
# ==============================================================================
def main():
    """Основная функция-оркестратор."""
    # 3.1. Получение конфигурации
    config = get_config()

    # 3.2. Определение шаблона и описания для валидации
    validation_pattern = None
    error_description = "Неизвестная ошибка валидации."
    
    if config.dlg_input_valid_type == "custom":
        validation_pattern = config.dlg_input_custom_regexp
        error_description = config.dlg_input_custom_regexp_desc or "Значение не соответствует заданному формату."
        if not validation_pattern:
            print("Предупреждение: выбран тип валидации 'custom', но не задан шаблон 'dlg_input_custom_regexp'. Валидация отключена.", file=sys.stderr)
    elif config.dlg_input_valid_type in VALIDATION_PRESETS:
        preset = VALIDATION_PRESETS[config.dlg_input_valid_type]
        validation_pattern = preset["pattern"]
        error_description = preset["description"]

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # 3.3. Определение начального значения для поля ввода
    # Сначала берем значение по умолчанию из аргументов
    initial_value = config.dlg_input_dvalue
    
    # Если запущен в PySM, пытаемся получить значение из контекста,
    # которое будет иметь более высокий приоритет.
    if IS_MANAGED_RUN and pysm_context:
        context_value = pysm_context.get(config.dlg_input_var)
        # Проверяем, что значение из контекста не None, чтобы не перезаписать
        # намеренно установленное значение по умолчанию пустой строкой из контекста.
        if context_value is not None:
            initial_value = str(context_value) # Приводим к строке на всякий случай
            print(f"Начальное значение переменной контекста <b>{config.dlg_input_var}</b> = <i>{initial_value}</i>")

    # 3.4. Инициализация GUI и цикл ввода/валидации
    q_app = QApplication.instance() or QApplication(sys.argv)
    user_input = initial_value # Используем подготовленное начальное значение
    ok_pressed = False
    
    while True:
        dialog = QInputDialog()
        dialog.setWindowTitle(config.dlg_input_title)
        dialog.setLabelText(config.dlg_input_msg)
        dialog.setTextValue(user_input) # Устанавливаем начальное/текущее значение
        dialog.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        
        # QInputDialog.getText() - более простой статический метод, вернемся к нему
        text, ok = QInputDialog.getText(
            None, 
            config.dlg_input_title, 
            config.dlg_input_msg, 
            QLineEdit.Normal, 
            user_input # Передаем начальное значение сюда
        )
        
        if not ok:
            ok_pressed = False
            break
            
        user_input = text
        
        if not validation_pattern or re.fullmatch(validation_pattern, user_input):
            ok_pressed = True
            break
        else:
            msg_box = QMessageBox(QMessageBox.Icon.Warning, "Неверный формат", error_description)
            msg_box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msg_box.exec()
            # После ошибки оставляем введенное неверное значение в поле для исправления
            
    # 3.5. Обработка результата и завершение
    if not ok_pressed:
        print("Операция отменена пользователем. Выполнение прервано.")
        sys.exit(1)
        
    #print(f"Пользователь ввел: '{user_input}'")
    
    if IS_MANAGED_RUN and pysm_context:
        try:
            pysm_context.set(config.dlg_input_var, user_input)
            print("\n<b>Переменная контекста успешно сохранена.</b>")
            print(f"<b>{config.dlg_input_var}</b> = <i>{user_input}")

        except Exception as e:
            print(f"\nКритическая ошибка при сохранении данных в контекст: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nЗапуск в автономном режиме, результат в контекст не сохраняется.")

    # 3.6. Успешное завершение
    print("\nСкрипт успешно завершен.<br>")
    sys.exit(0)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()