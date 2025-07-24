# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import sys
from typing import Dict, Any

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
    from PySide6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,
        QHeaderView, QDialogButtonBox, QMessageBox, QPushButton
    )
except ImportError:
    print("Ошибка: для работы этого скрипта требуется PySide6.", file=sys.stderr)
    sys.exit(1)


# 2. БЛОК: GUI-компонент (диалоговое окно)
# ==============================================================================
class ContextEditorDialog(QDialog):
    def __init__(self, initial_data: Dict[str, Any], mode: str, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.mode = mode
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Ключ", "Значение", "Действие"])

        header = self.table.horizontalHeader()
        header.setStyleSheet("""
            QHeaderView::section { background-color: #E8E8E8; font-weight: bold; border: none; padding: 4px; }
        """)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        layout.addWidget(self.table)

        self.add_button = QPushButton("Добавить переменную")
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        if self.mode == 'edit':
            layout.addWidget(self.add_button)
            self.add_button.clicked.connect(self.add_table_row)
            self.button_box.accepted.connect(self.validate_and_accept)
        else: # view mode
            self.add_button.hide()
            self.button_box.accepted.connect(self.accept)
            self.button_box.button(QDialogButtonBox.Cancel).hide()

        layout.addWidget(self.button_box)
        self.button_box.rejected.connect(self.reject)

        self._populate_table(initial_data)

    def _populate_table(self, data: Dict[str, Any]):
        filtered_data = {k: v for k, v in data.items() if k != "pysm_info"}
        for key, value in sorted(filtered_data.items()):
            self.add_table_row(key, str(value))

    def add_table_row(self, key: str = "", value: str = ""):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        
        key_item = QTableWidgetItem(key)
        value_item = QTableWidgetItem(value)
        
        if self.mode == 'view':
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)

        self.table.setItem(row_position, 0, key_item)
        self.table.setItem(row_position, 1, value_item)

        if self.mode == 'edit':
            delete_button = QPushButton("Удалить")
            delete_button.clicked.connect(lambda: self.table.removeRow(self.table.indexAt(delete_button.pos()).row()))
            self.table.setCellWidget(row_position, 2, delete_button)
        
        self.table.scrollToBottom()

    def validate_and_accept(self):
        keys = set()
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            key = key_item.text().strip()
            if not key:
                self._show_validation_error("Имя ключа не может быть пустым.", key_item)
                return
            if key in keys:
                self._show_validation_error(f"Ключ '{key}' должен быть уникальным.", key_item)
                return
            keys.add(key)
        self.accept()

    def _show_validation_error(self, message: str, item_to_focus: QTableWidgetItem):
        msg_box = QMessageBox(QMessageBox.Icon.Warning, "Ошибка валидации", message)
        msg_box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        msg_box.exec()
        self.table.setCurrentItem(item_to_focus)

    def get_values(self) -> Dict[str, str]:
        data = {}
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).text().strip()
            value = self.table.item(row, 1).text()
            if key: # Сохраняем только строки с непустыми ключами
                data[key] = value
        return data


# 3. БЛОК: Определение и получение конфигурации
# ==============================================================================
def get_config():
    """Определяет аргументы командной строки и получает их значения."""
    parser = argparse.ArgumentParser(description="Редактор/просмотрщик переменных контекста.")
    parser.add_argument(
        "--__edit",
        action='store_true',
        help="Открыть в режиме редактирования. По умолчанию - режим просмотра."
    )
    parser.add_argument(
        "--__title",
        type=str,
        help="Пользовательский заголовок для окна."
    )
    
    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser)
        config = argparse.Namespace()
        config.__edit = resolver.get("__edit", default=False)
        # Динамический заголовок в зависимости от режима
        default_title = "Редактирование контекста" if config.__edit else "Просмотр контекста"
        config.__title = resolver.get("__title", default=default_title)
        return config
    else:
        # Для автономного запуска
        args = parser.parse_args()
        if not args.__title:
            args.__title = "Редактирование контекста" if args.__edit else "Просмотр контекста"
        return args


# 4. БЛОК: Основная логика
# ==============================================================================
def main():
    """Основная функция-оркестратор."""
    # 4.1. Проверка на запуск в управляемой среде
    if not IS_MANAGED_RUN or not pysm_context:
        print("Ошибка: Этот скрипт предназначен для запуска только в среде PySM.", file=sys.stderr)
        sys.exit(1)
    
    # 4.2. Получение конфигурации
    config = get_config()
    mode_string = 'edit' if config.__edit else 'view'
    print(f"Запуск в режиме '{mode_string}'...")
    
    # 4.3. Получение данных и запуск диалога
    initial_context = pysm_context.get_all()
    q_app = QApplication.instance() or QApplication(sys.argv)
    dialog = ContextEditorDialog(initial_context, mode_string, config.__title)
    dialog_result = dialog.exec()
    
    # 4.4. Обработка результата
    if config.__edit:
        if dialog_result == QDialog.Accepted:
            new_data = dialog.get_values()
            print("Сохранение новых данных в контекст...")
            
            # ЗАМЕНА _write_data НА ПУБЛИЧНОЕ API
            # 1. Удаляем все пользовательские переменные
            pysm_context.remove()
            # 2. Обновляем контекст новыми данными
            pysm_context.update(new_data)

            print(f"Контекст успешно обновлен. Новых пар ключ-значение: {len(new_data)}.")
            sys.exit(0) # Успешное завершение
        else:
            print("Операция отменена пользователем. Изменения не сохранены.")
            sys.exit(1) # Завершение с кодом отмены
    else: # view mode
        print("Окно просмотра закрыто.")
        sys.exit(0) # Успешное завершение


# 5. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()