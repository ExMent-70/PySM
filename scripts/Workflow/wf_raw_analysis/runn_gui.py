# run_gui.py

import sys
import logging
from pathlib import Path
import toml

from PySide6.QtWidgets import QApplication, QMessageBox

# Импортируем только то, что нужно для запуска
from main import setup_logging
from gui_lib.main_window import MainWindow, DEFAULT_CONFIG_PATH
from gui_lib.ui_texts import UI_TEXTS

# --- Запуск приложения ---
if __name__ == "__main__":
    # Проверка версии Python
    if sys.version_info < (3, 8):
        print(UI_TEXTS["msg_python_version_required"])
        # Показать сообщение об ошибке, если возможно
        try:
            app_temp = QApplication([])
            QMessageBox.critical(
                None,
                UI_TEXTS["msg_python_version_error"],
                UI_TEXTS["msg_python_version_required"],
            )
        except Exception:
            pass  # Если GUI не доступен, просто выходим
        sys.exit(1)
        
    initial_log_level = "INFO"
    if DEFAULT_CONFIG_PATH.is_file():
        try:
            config_for_log_setup = toml.load(DEFAULT_CONFIG_PATH)
            log_level_from_config = config_for_log_setup.get("logging_level")
            if isinstance(log_level_from_config, str) and log_level_from_config.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                initial_log_level = log_level_from_config.upper()
        except Exception as cfg_err:
            print(f"Warning: Could not read {DEFAULT_CONFIG_PATH.name} for logger setup: {cfg_err}", file=sys.stderr)
    
    setup_logging(log_level_str=initial_log_level, log_file="face_processing_gui.log")
    logging.captureWarnings(True)

    app = QApplication(sys.argv)
    
    try:
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())
    except Exception as main_err:
        logging.critical("Critical unhandled error in GUI main thread!", exc_info=True)
        QMessageBox.critical(None, UI_TEXTS["msg_gui_error_critical"], UI_TEXTS["msg_gui_unhandled_error"].format(main_err))
        sys.exit(1)