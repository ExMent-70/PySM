1# main.py

import sys
import os
import logging
import pathlib
from PySide6.QtWidgets import QApplication

from pysm_lib.app_controller import AppController
from pysm_lib.config_manager import ConfigManager
from pysm_lib.gui.main_window import MainWindow
from pysm_lib.locale_manager import LocaleManager
from pysm_lib.app_constants import APPLICATION_ROOT_DIR




# 1. Блок без изменений
def get_logging_level_from_string(level_str: str, default_level=logging.INFO) -> int:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_str.upper(), default_level)


# 2. Блок без изменений
def setup_logging(log_level_val: int, log_to_console: bool = True):
    #app_root_dir = pathlib.Path(__file__).parent.resolve()
    app_root_dir = pathlib.Path.cwd().resolve()
    log_file_path = app_root_dir / "app_script_manager.log"
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)-8s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)-8s - %(name)s - %(message)s")
    logger_instance = logging.getLogger("PyScriptManager")
    logger_instance.setLevel(log_level_val)
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()
    try:
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level_val)
        logger_instance.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(
            f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось настроить файловый логгер: {e}\n"
        )
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level_val)
        logger_instance.addHandler(console_handler)
    logger_instance.info(
        f"Логирование настроено. Уровень: {logging.getLevelName(logger_instance.level)}."
    )

# 3. Блок с изменениями (основная функция)
if __name__ == "__main__":
    # --- НАЧАЛО ИСПРАВЛЕНИЙ ---
    # РЕШЕНИЕ ПРОБЛЕМЫ С "DESKTOP NOT AVAILABLE"
    # Это обходной путь для ошибки, возникающей из-за того, что
    # внешний скрипт переопределяет переменную окружения USERPROFILE.
    # Мы создаем "фальшивую" папку Desktop до инициализации QApplication.
    try:
        user_profile = os.environ.get('USERPROFILE')
        if user_profile:
            desktop_path = pathlib.Path(user_profile) / 'Desktop'
            desktop_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Игнорируем возможные ошибки (например, нет прав на запись),
        # так как это не критично для основной логики.
        print(f"Предупреждение: не удалось создать папку Desktop: {e}", file=sys.stderr)
    # --- КОНЕЦ ИСПРАВЛЕНИЙ ---    
    #APP_ROOT = pathlib.Path(__file__).parent.resolve()
    APP_ROOT = pathlib.Path.cwd().resolve()

    try:
        config_path = APPLICATION_ROOT_DIR / "config.toml"
        config_manager = ConfigManager(config_path=config_path)

        log_level_str = config_manager.log_level
        numeric_log_level = get_logging_level_from_string(log_level_str, logging.INFO)

        language = config_manager.language
        locale_manager = LocaleManager(language_code=language)

    except Exception as e_cfg_init:
        sys.stderr.write(
            f"КРИТ. ОШИБКА инициализации: {e_cfg_init}. Уровень лога: INFO.\n"
        )
        numeric_log_level = logging.INFO
        config_manager = None
        locale_manager = LocaleManager()

    setup_logging(log_level_val=numeric_log_level, log_to_console=True)
    main_logger = logging.getLogger("PyScriptManager.Main")
    # --- СТРОКА ИЗМЕНЕНА ---
    main_logger.info(locale_manager.get("main.log_info.app_starting"))

    app = QApplication(sys.argv)

    controller_kwargs = {"locale_manager": locale_manager}
    if config_manager:
        controller_kwargs["config_manager_instance"] = config_manager

    controller = AppController(**controller_kwargs)

    window = MainWindow(app_controller=controller, locale_manager=locale_manager)

    # --- СТРОКА ИЗМЕНЕНА ---
    main_logger.info(locale_manager.get("main.log_info.showing_main_window"))
    window.show()
    exit_code = app.exec()
    # --- СТРОКА ИЗМЕНЕНА ---
    main_logger.info(locale_manager.get("main.log_info.app_exit", code=exit_code))
    sys.exit(exit_code)
