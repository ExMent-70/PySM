# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys

# Определяем, запущен ли скрипт под управлением PySM
IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    # Создаем заглушки для автономного запуска
    pysm_context = None
    ConfigResolver = None

# Импортируем PySide6 с проверкой на его наличие
try:
    # --- ИЗМЕНЕНИЕ: Добавляем импорт Qt для флагов окна ---
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox
except ImportError:
    # Это сообщение будет видно только при прямом запуске,
    # т.к. PySM поставляется с PySide6.
    print("Ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    print("Пожалуйста, установите его: pip install PySide6", file=sys.stderr)
    sys.exit(1)


# 2. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config():
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Показывает диалоговое окно и сохраняет выбор пользователя."
    )
    
    # --- ИЗМЕНЕНИЯ: Аргументы приведены в полное соответствие с script_passport.json ---
    
    parser.add_argument(
        "--dlg_msg_var", 
        type=str,
        # help теперь соответствует description из паспорта
        help="Имя переменной, в которую будет сохранен результат выбора пользователя (например, 'user_choice'). Результатом будет строка: 'ok', 'yes', 'no' или 'cancel'.",
        # Добавлен default из паспорта, required=True убран
        default="var_user_choice"
    )
    parser.add_argument(
        "--dlg_msg_type", 
        type=str,
        choices=['ok', 'yes_no', 'yes_no_cancel'],
        help="Выберите набор кнопок для отображения в диалоговом окне.",
        default="yes_no"
    )
    parser.add_argument(
        "--dlg_msg_title", 
        type=str,
        help="Текст, который будет отображаться в заголовке диалогового окна.",
        default="Подтверждение"
    )
    parser.add_argument(
        "--dlg_msg_message", 
        type=str,
        help="Основной текст вопроса или сообщения, который будет показан пользователю. Можно использовать несколько строк.",
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
        # При автономном запуске стандартный argparse теперь будет использовать
        # те же значения по умолчанию, что и в паспорте.
        return parser.parse_args()


# 3. БЛОК: Основная логика
# ==============================================================================
def main():
    """Основная функция-оркестратор."""
    # 3.1. Получение конфигурации
    config = get_config()

    # 3.2. Инициализация GUI и создание QMessageBox
    # Если приложение уже запущено (в среде PySM), используем его экземпляр.
    # Иначе — создаем новый.
    q_app = QApplication.instance() or QApplication(sys.argv)
    
    msg_box = QMessageBox()
    # --- ИЗМЕНЕНИЕ: Устанавливаем флаг, чтобы окно было поверх всех остальных ---
    msg_box.setWindowFlag(Qt.WindowStaysOnTopHint)
    msg_box.setWindowTitle(config.dlg_msg_title)
    msg_box.setText(config.dlg_msg_message)

    # 3.3. Настройка кнопок в зависимости от типа диалога
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
    
    msg_box.setStandardButtons(button_map.get(config.dlg_msg_type, QMessageBox.NoButton))
    msg_box.setDefaultButton(default_button_map.get(config.dlg_msg_type))
        
    # 3.4. Показ окна и обработка результата
    result_code = msg_box.exec()

    # Карта для преобразования кода ответа PySide6 в строку
    result_string_map = {
        QMessageBox.Ok: "ok",
        QMessageBox.Yes: "yes",
        QMessageBox.No: "no",
        QMessageBox.Cancel: "cancel",
    }
    result_string = result_string_map.get(result_code, "unknown")
    
    #print(f"Пользователь выбрал: '{result_string}'")

    # 3.5. Сохранение результата в контекст (если возможно)
    if IS_MANAGED_RUN and pysm_context:
        try:
            # Используем актуальный метод set()
            pysm_context.set(config.dlg_msg_var, result_string)
            print("\n<b>Переменная контекста успешно сохранена.</b>")
            print(f"<b>{config.dlg_msg_var}</b> = <i>{result_string}<br>")

        except Exception as e:
            # В случае ошибки при записи в контекст, выводим ее в stderr
            # и завершаем скрипт с кодом ошибки.
            print(f"Критическая ошибка при сохранении данных в контекст: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Запуск в автономном режиме, результат в контекст не сохраняется.")

    # 3.6. Определение кода завершения скрипта
    # Операция считается "отмененной", если пользователь нажал Cancel,
    # "No" в диалоге yes/no, или просто закрыл окно.
    # В этих случаях возвращаем код 1, в остальных — 0.
    
    exit_code = 0  # Код по умолчанию (успех)
    
    if config.dlg_msg_type == 'yes_no' and result_string == 'no':
        exit_code = 1
    elif result_string in ["cancel", "unknown"]:
        exit_code = 1
        
    print(f"Скрипт завершается с кодом выхода: {exit_code}<br>")
    sys.exit(exit_code)


# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()