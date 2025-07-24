# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
import os

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
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QFileDialog
except ImportError:
    print("Ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    print("Пожалуйста, установите его: pip install PySide6", file=sys.stderr)
    sys.exit(1)


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
        help="Имя переменной, в которую будет сохранен выбранный путь.",
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

    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        config = argparse.Namespace()
        config.dlg_open_var = resolver.get("dlg_open_var")
        config.dlg_open_type = resolver.get("dlg_open_type")
        config.dlg_open_title = resolver.get("dlg_open_title")
        config.dlg_open_filter = resolver.get("dlg_open_filter")
        return config
    else:
        return parser.parse_args()


# 3. БЛОК: Основная логика
# ==============================================================================
def main():
    """Основная функция-оркестратор."""
    config = get_config()
    
    q_app = QApplication.instance() or QApplication(sys.argv)
    
    initial_dir = ""
    if IS_MANAGED_RUN and pysm_context:
        existing_path_str = pysm_context.get(config.dlg_open_var)
        if existing_path_str and os.path.exists(existing_path_str):
            if os.path.isfile(existing_path_str):
                initial_dir = os.path.dirname(existing_path_str)
            else:
                initial_dir = existing_path_str
        else:
            collection_dir = pysm_context.get_structured("pysm_info.collection_dir")
            if collection_dir and os.path.isdir(collection_dir):
                initial_dir = collection_dir
            print(f"Начальная папка: {initial_dir}")

    dialog = QFileDialog(None) 
    dialog.setWindowTitle(config.dlg_open_title)
    if initial_dir:
        dialog.setDirectory(initial_dir)
    
    dialog.setWindowFlag(Qt.WindowStaysOnTopHint, True)

    if config.dlg_open_type == 'file':
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(config.dlg_open_filter)
        print("Открытие диалога выбора файла...")
    elif config.dlg_open_type == 'directory':
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        print("Открытие диалога выбора папки...")
    #print(f"{config.dlg_open_title}")

    selected_path = ""
    if dialog.exec() == QFileDialog.DialogCode.Accepted:
        selected_files = dialog.selectedFiles()
        if selected_files:
            selected_path = selected_files[0]

    if not selected_path:
        print("\nОперация отменена пользователем.", file=sys.stderr)
        sys.exit(1)
    
    #print(f"Выбран путь: {selected_path}")

    # --- ИСПОЛЬЗУЕМ НОВЫЙ, УЛУЧШЕННЫЙ API ---
    if IS_MANAGED_RUN and pysm_context:
        path_type = "dir_path" if config.dlg_open_type == 'directory' else 'file_path'        
        try:
            # Одна простая и понятная строка решает все наши проблемы
            pysm_context.set(
                key=config.dlg_open_var,
                value=selected_path,
                var_type=path_type
            )
            print("\n<b>Переменная контекста успешно сохранена.</b>")
            print(f"<b>{config.dlg_open_var}</b> = <i>{selected_path} (тип: {path_type})</i><br>")
            link_path = selected_path if config.dlg_open_type == 'directory' else selected_path.parent
            
            pysm_context.log_link(
                url_or_path=str(link_path), # Передаем строку, а не объект Path
                text=f"Открыть папку <i>{link_path}</i>",
            )                  
            print(" ", file=sys.stderr)


        except Exception as e:
            print(f"\nКритическая ошибка при записи в контекст: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nЗапуск в автономном режиме, результат в контекст не сохраняется.")


    sys.exit(0)

# 4. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()