# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
import os
import logging
import pathlib  # Добавляем импорт pathlib, если его не было

# Определяем, запущен ли скрипт под управлением PySM
IS_MANAGED_RUN = False
try:
    from pysm_lib import pysm_context
    from pysm_lib.context_variable_ops import (
        format_error,
        format_success,
        read_context_value,
        write_context_value,
    )
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    # Создаем заглушки для автономного запуска
    pysm_context = None
    ConfigResolver = None
    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"
    def format_success(var_name: str, value) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"

# Импортируем PySide6 с проверкой на его наличие
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QFileDialog
except ImportError:
    print("Ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    print("Пожалуйста, установите его: pip install PySide6", file=sys.stderr)
    sys.exit(1)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# 2. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config():
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Показывает диалог выбора файла/папки и сохраняет путь в контекст."
    )
    
    # Аргументы синхронизированы с паспортом скрипта
    parser.add_argument(
        "--dlg_open_var", 
        type=str,
        help="Имя переменной, в которую будет сохранен выбранный путь. Поддерживается точечная нотация, например project.output_path.",
        default="dlg_open_user_var"
    )
    parser.add_argument(
        "--dlg_open_type", 
        type=str,
        choices=['file', 'directory'],
        help="Тип диалога: 'file' для выбора файла, 'directory' для выбора папки.",
        default="file"
    )
    parser.add_argument(
        "--dlg_open_title", 
        type=str,
        help="Текст заголовка диалогового окна.",
        default="Выберите путь"
    )
    parser.add_argument(
        "--dlg_open_filter", 
        type=str,
        help="Фильтр файлов для диалога (например, 'Изображения (*.png *.jpg)'). Актуально только для выбора файла.",
        default="Все файлы (*.*)"
    )

    parser.add_argument("--dlg_path", type=str, help="Путь к файлу или папке", default=""
        )


    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()
  


# 3. БЛОК: Основная логика (С ИСПРАВЛЕНИЯМИ)
# ==============================================================================
def main():
    """Основная функция-оркестратор."""
    config = get_config()

    if not IS_MANAGED_RUN or not pysm_context:
        logger.critical(format_error("Этот скрипт может быть запущен только в среде PySM"))
        sys.exit(1)
    
    # Теперь эта строка не должна вызывать ошибку
    q_app = QApplication.instance() or QApplication(sys.argv)
    
    # Логика определения начальной директории
    initial_dir = ""
    existing_path = read_context_value(pysm_context, config.dlg_open_var)
    existing_path_str = existing_path.value if existing_path.exists else None
    if existing_path_str and os.path.exists(existing_path_str):
        if os.path.isfile(existing_path_str):
            initial_dir = os.path.dirname(existing_path_str)
        else:
            initial_dir = existing_path_str
    else:
        collection_dir_value = read_context_value(pysm_context, "pysm_info.collection_dir")
        collection_dir = collection_dir_value.value if collection_dir_value.exists else None
        if collection_dir and os.path.isdir(collection_dir):
            initial_dir = collection_dir


    selected_path = ""
    parent_widget = None

    logger.info(f"<b>{config.dlg_open_title}</b>")
    if config.dlg_open_type == 'file':
        selected_path, _ = QFileDialog.getOpenFileName(
            parent=parent_widget,
            caption=config.dlg_open_title,
            dir=initial_dir if initial_dir else ".",
            filter=config.dlg_open_filter
        )
    elif config.dlg_open_type == 'directory':
        selected_path = QFileDialog.getExistingDirectory(
            parent=parent_widget,
            caption=config.dlg_open_title,
            dir=initial_dir if initial_dir else "."
        )

    if not selected_path:
        logger.critical(format_error("Операция отменена пользователем<br>"))
        sys.exit(1)
    
    # Логика сохранения результата в контекст
    path_type = "dir_path" if config.dlg_open_type == 'directory' else 'file_path'        
    try:
        write_context_value(
            pysm_context,
            config.dlg_open_var,
            selected_path,
            var_type=path_type,
        )
        logger.info(format_success(config.dlg_open_var, selected_path))
        #s = f"<b>{config.dlg_open_var}</b> = <i>{selected_path} (тип: {path_type})</i><br>"
        #print(f"<b>{config.dlg_open_var}</b> = <i>{selected_path} (тип: {path_type})</i><br>")
        
        """
        link_path = selected_path if os.path.isdir(selected_path) else os.path.dirname(selected_path)
        
        pysm_context.log_link(
            url_or_path=str(link_path),
            text=s,                
            #text=f"Открыть папку <i>{link_path}</i>",
        )                  
        logger.info(" ")
        """
    except Exception as e:
        logger.critical(format_error(f"Критическая ошибка при записи в контекст: {e}"))
        sys.exit(1)


    sys.exit(0)

# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()
