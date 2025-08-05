# run_wf_photo_select.py

# 1. БЛОК: Импорты и константы
# ==============================================================================
import argparse
import logging
import os
import pathlib
import re
import shutil
import sys
from argparse import Namespace
from collections import defaultdict
from typing import List, Dict, Any, Set, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Попытка импорта библиотек PySM
try:
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib import pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver = None
    pysm_context = None
    try:
        from tqdm import tqdm
    except ImportError:
        class TqdmMock:
            def __init__(self, i=None, **kwargs): self.iterable = i or []
            def __iter__(self): return iter(self.iterable)
            @staticmethod
            def write(s, **kwargs): print(s)
        tqdm = TqdmMock

# Опциональные импорты для ipymarkup
try:
    from ipymarkup.palette import Palette, Color, material
    from ipymarkup.span import format_span_box_markup
except ImportError:
    format_span_box_markup, Palette, Color, material = None, None, None, None
    logger.warning("Библиотека 'ipymarkup' не найдена. Визуализация текста будет недоступна.")

# Импорты для GUI на PySide6
try:
    from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, Signal, QThread, QSize
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                   QHBoxLayout, QSplitter, QLabel, QTextEdit,
                                   QTreeView, QPushButton, QHeaderView, QMessageBox,
                                   QStatusBar, QStyle, QTextBrowser, QTabWidget)
except ImportError:
    print("Ошибка: PySide6 не найден. Установите: pip install pyside6", file=sys.stderr)
    sys.exit(1)

# Константы
RAW_EXTENSIONS = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.rw2'}
CAPTURE_FOLDER = "Capture"
SELECTS_FOLDER = "Selects"


# 2. БЛОК: Вспомогательные функции
# ==============================================================================
def construct_session_paths() -> Dict[str, Optional[pathlib.Path]]:
    """
    Формирует пути к исходной и целевой папкам на основе переменных контекста.
    """
    if not IS_MANAGED_RUN or not pysm_context:
        return {"source": None, "dest": None}

    photo_session = pysm_context.get("wf_photo_session", "")
    session_name = pysm_context.get("wf_session_name", "")
    session_path_str = pysm_context.get("wf_session_path", "")

    if not all([session_name, session_path_str, photo_session]):
        logger.error("Одна или несколько переменных контекста (wf_session_path, wf_session_name, wf_photo_session) не найдены.")
        return {"source": None, "dest": None}

    base_path = pathlib.Path(session_path_str) / session_name
    source_dir = base_path / CAPTURE_FOLDER / photo_session
    dest_dir = base_path / SELECTS_FOLDER / photo_session

    return {"source": source_dir, "dest": dest_dir}


# 3. БЛОК: Получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """
    Получает конфигурацию из командной строки.
    Параметр __fs_source_dir был удален, пути формируются из контекста.
    """
    parser = argparse.ArgumentParser(description="Выборка файлов по номерам из текста.")
    parser.add_argument("--__fs_mode", type=str, choices=["copy", "move"], default="copy", help="Режим операции.")
    
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    
    return parser.parse_args()


# 4. БЛОК: Иерархическая модель данных для TreeView
# ==============================================================================
class TreeItem:
    """Элемент древовидной структуры."""
    def __init__(self, data: List, parent: 'TreeItem' = None):
        self._data = data
        self._parent = parent
        self._children: List['TreeItem'] = []
    def child(self, row: int) -> Optional['TreeItem']: return self._children[row] if 0 <= row < len(self._children) else None
    def childCount(self) -> int: return len(self._children)
    def columnCount(self) -> int: return len(self._data)
    def data(self, column: int) -> Any: return self._data[column] if 0 <= column < len(self._data) else None
    def parent(self) -> Optional['TreeItem']: return self._parent
    def row(self) -> int: return self._parent._children.index(self) if self._parent else 0
    def appendChild(self, item: 'TreeItem'): self._children.append(item)

class FileTreeModel(QAbstractItemModel):
    """Модель для отображения иерархических данных о файлах с иконками и поддержкой сортировки."""
    def __init__(self, data: List[Dict[str, Any]] = None, icons: Dict = None, parent=None):
        super().__init__(parent)
        self._root_item = TreeItem(["Номер", "Статус", "Файл"])
        self.icons = icons or {}
        self.setup_model_data(data or [])
    def setup_model_data(self, model_data: List[Dict[str, Any]]):
        self.beginResetModel()
        self._root_item = TreeItem(["Номер", "Статус", "Файл"])
        sorted_data = sorted(model_data, key=lambda x: (x.get('status', '') != 'Найден', x['number']))
        for item_data in sorted_data:
            number, status = item_data['number'], item_data.get('status', 'Ожидает')
            files, base_path = item_data.get('files', []), item_data.get('base_path')
            if not files:
                parent_item = TreeItem([number, status, ""], self._root_item)
            else:
                main_file = files[0]
                parent_item = TreeItem([number, status, str(main_file.relative_to(base_path))], self._root_item)
                for other_file in files[1:]:
                    parent_item.appendChild(TreeItem(["", "", str(other_file.relative_to(base_path))], parent_item))
            self._root_item.appendChild(parent_item)
        self.endResetModel()
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid(): return None
        item = index.internalPointer()
        if role == Qt.DecorationRole and index.column() == 2 and item.data(2):
            return self.icons.get("folder") if item.parent() == self._root_item else self.icons.get("file")
        if role == Qt.DisplayRole: return item.data(index.column())
        if role == Qt.BackgroundRole and index.column() == 1:
            status = item.data(1)
            if status == "Найден": return QColor("#d4edda")
            if status == "Не найден": return QColor("#f8d7da")
        return None
    def headerData(self, section: int, o: Qt.Orientation, role: int):
        if o == Qt.Horizontal and role == Qt.DisplayRole: return self._root_item.data(section)
        return None
    def index(self, row: int, col: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, col, parent): return QModelIndex()
        parent_item = parent.internalPointer() if parent.isValid() else self._root_item
        child_item = parent_item.child(row)
        return self.createIndex(row, col, child_item) if child_item else QModelIndex()
    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid(): return QModelIndex()
        child_item = index.internalPointer()
        parent_item = child_item.parent()
        if parent_item == self._root_item: return QModelIndex()
        return self.createIndex(parent_item.row(), 0, parent_item)
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return (parent.internalPointer() if parent.isValid() else self._root_item).childCount()
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int: return self._root_item.columnCount()
    def sort(self, column: int, order: Qt.SortOrder):
        self.layoutAboutToBeChanged.emit()
        key_func = lambda item: (int(item.data(0)),) if column == 0 else (item.data(column),)
        self._root_item._children.sort(key=key_func, reverse=(order == Qt.DescendingOrder))
        self.layoutChanged.emit()


# 5. БЛОК: Рабочий поток (Worker)
# ==============================================================================
class Worker(QThread):
    parse_finished = Signal(list, str)
    search_finished = Signal(list)
    operation_finished = Signal(str, int, int)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.task: str = ""
        self.text_to_parse: str = ""
        self.numbers_to_find: List[str] = []
        self.source_dir: Optional[pathlib.Path] = None
        self.dest_dir: Optional[pathlib.Path] = None
        self.operation_mode: str = "copy"
        self.files_to_process: List[pathlib.Path] = []
        self._init_palette()

    def _init_palette(self):
        self.palette = None
        if Palette and Color and material:
            try:
                num_color = Color('Номер', background=material('Blue', '50'), border=material('Blue', '100'), text=material('Blue', '900'))
                self.palette = Palette([num_color])
            except Exception as e:
                logger.error(f"Ошибка при создании палитры ipymarkup: {e}")

    def run(self):
        try:
            if self.task == "parse": self._task_parse_text()
            elif self.task == "search": self._task_search_files()
            elif self.task == "operation": self._task_execute_operation()
        except Exception as e:
            logger.error(f"Критическая ошибка в потоке: {e}", exc_info=True)
            self.error_occurred.emit(f"Критическая ошибка в потоке: {e}")

    def _task_parse_text(self):
        logger.info("Извлечение 4-х значных номеров фотографий из текста...")
        matches = list(re.finditer(r'(\d{4})', self.text_to_parse))
        numbers: Set[str] = {match.group(1) for match in matches}
        sorted_numbers = sorted(list(numbers))
        logger.info(f"Найдено <b>{len(sorted_numbers)}</b> уникальных номеров:")
        logger.info(f"<i>{sorted_numbers}</i>\n")

        markup_html = ""
        if format_span_box_markup and self.palette:
            try:
                spans = [(match.start(), match.end(), "") for match in matches]
                markup_html = "".join(format_span_box_markup(self.text_to_parse, spans, palette=self.palette))
            except Exception as e:
                logger.error(f"Ошибка генерации разметки ipymarkup: {e}")
                markup_html = f"<p style='color: red;'>Ошибка при генерации разметки: {e}</p>"
        else:
            markup_html = "<p style='color: orange;'>Библиотека 'ipymarkup' не найдена, визуализация недоступна.</p>"

        self.parse_finished.emit(sorted_numbers, markup_html)

    def _task_search_files(self):
        source_path = self.source_dir
        logger.info(f"Поиск файлов для <b>{len(self.numbers_to_find)}</b> номеров в: <b>{source_path}</b>")
        
        all_files_in_dir = []
        select_folder_path = self.dest_dir
        for p in source_path.rglob("*"):
            if select_folder_path and select_folder_path in p.parents: continue
            if p.is_file(): all_files_in_dir.append(p)
        all_files_in_dir.sort(key=lambda x: x.name)
        logger.info(f"Всего файлов для анализа (исключая {SELECTS_FOLDER}): {len(all_files_in_dir)}")

        existing_basenames = {p.stem for p in all_files_in_dir}
        filtered_numbers = [num for num in self.numbers_to_find if any(num in name for name in existing_basenames)]
        logger.info(f"Уникальные номера после фильтрации:")
        logger.info(f"<i>{filtered_numbers}</i>")

        files_by_basename = defaultdict(list)
        for file_path in all_files_in_dir:
            files_by_basename[file_path.stem].append(file_path)

        final_data = []
        for number in self.numbers_to_find:
            if number not in filtered_numbers:
                final_data.append({"number": number, "status": "Не найден"})
                continue
            found_files = [f for stem, files in files_by_basename.items() if number in stem for f in files]
            if found_files:
                sorted_files = sorted(list(set(found_files)), key=lambda p: (p.suffix.lower() not in RAW_EXTENSIONS, p.name))
                final_data.append({"number": number, "status": "Найден", "files": sorted_files, "base_path": source_path})
            else:
                final_data.append({"number": number, "status": "Не найден"})
        logger.info("Поиск файлов завершен.\n")
        self.search_finished.emit(final_data)

    def _task_execute_operation(self):
        source_path = self.source_dir
        dest_root = self.dest_dir
        op_name = "Копирование" if self.operation_mode == "copy" else "Перемещение"
        logger.info(f"{op_name} файлов в папку {dest_root}")
        op_func = shutil.copy2 if self.operation_mode == "copy" else shutil.move
        with tqdm(self.files_to_process, desc=op_name, unit="файл") as progress_bar:
            for src_file in progress_bar:
                relative_path = src_file.relative_to(source_path)
                dest_file = dest_root / relative_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                op_func(src_file, dest_file)
        self.operation_finished.emit(self.operation_mode, len(self.files_to_process), len(self.files_to_process))


# 6. БЛОК: Главное окно приложения
# ==============================================================================
class FileSelectorWindow(QMainWindow):
    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.final_status = 1
        
        paths = construct_session_paths()
        self.source_dir = paths.get("source")
        self.dest_dir = paths.get("dest")

        self._load_icons()
        self._init_ui()
        self._init_worker()
        
        if IS_MANAGED_RUN and (not self.source_dir or not self.dest_dir):
            QMessageBox.critical(self, "Критическая ошибка", "Не удалось сформировать пути. Проверьте переменные контекста:\n'wf_session_path', 'wf_session_name', 'wf_photo_session'.")
            self._set_ui_busy(True, "Ошибка: пути не сформированы.")
        elif not self.source_dir:
             QMessageBox.critical(self, "Критическая ошибка", "Исходная папка не указана (требуется запуск из PySM с заданным контекстом).")
             self._set_ui_busy(True, "Ошибка: исходная папка не найдена.")

    def _load_icons(self):
        style = self.style()
        self.icons = {"folder": style.standardIcon(QStyle.SP_DirIcon), "file": style.standardIcon(QStyle.SP_FileIcon)}

    def _init_worker(self):
        self.worker = Worker()
        self.worker.parse_finished.connect(self._on_parse_finished)
        self.worker.search_finished.connect(self._on_search_finished)
        self.worker.operation_finished.connect(self._on_operation_finished)
        self.worker.error_occurred.connect(self._on_worker_error)

    def _init_ui(self):
        win_title = "Выборка файлов по номерам"
        if IS_MANAGED_RUN and self.source_dir:
            win_title += f" - [{self.source_dir.parent.name}/{self.source_dir.name}]"
        self.setWindowTitle(win_title)
        self.resize(1000, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)
        self.left_tabs = QTabWidget()
        left_layout.addWidget(self.left_tabs)

        input_tab = QWidget()
        input_tab_layout = QVBoxLayout(input_tab)
        input_tab_layout.addWidget(QLabel("Вставьте текст со списком номеров:"))
        self.text_input = QTextEdit()
        input_tab_layout.addWidget(self.text_input)
        self.left_tabs.addTab(input_tab, "Ввод")

        markup_tab = QWidget()
        markup_tab_layout = QVBoxLayout(markup_tab)
        self.markup_browser = QTextBrowser()
        self.markup_browser.setOpenExternalLinks(True)
        markup_tab_layout.addWidget(self.markup_browser)
        self.left_tabs.addTab(markup_tab, "Результат разбора")
        self.left_tabs.setStyleSheet(
            """
            /* Стиль для неактивной вкладки */
            QTabBar::tab:!selected {
                background-color: #ffc300;
                color: #000000;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
            }

            /* Стиль для неактивной вкладки при наведении */
            QTabBar::tab:!selected:hover {
                color: #ffffff;
                font: bold;
            }

            /* Стиль для активной (выбранной) вкладки */
            QTabBar::tab:selected {
                background-color: #ffffff;
                font: bold;                
                border: 1px solid #c0c0c0;
                border-bottom: 1px solid white; /* "Сливается" с фоном */
                margin-bottom: -1px; /* Сдвиг для эффекта слияния */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
            }
            """
        )            

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Результаты:"))
        self.tree_view = QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setIconSize(QSize(24, 24))
        self.tree_view.setSortingEnabled(True)
        h_header = self.tree_view.header()
        h_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        h_header.setSectionResizeMode(2, QHeaderView.Stretch)
        right_layout.addWidget(self.tree_view)
        
        splitter.addWidget(left_panel); splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])

        bottom_layout = QHBoxLayout()
        self.parse_button = QPushButton("1. Извлечь номера")
        self.search_button = QPushButton("2. Найти файлы")
        self.action_button = QPushButton("3. Выполнить")
        self.parse_button.clicked.connect(self._start_parse)
        self.search_button.clicked.connect(self._start_search)
        self.action_button.clicked.connect(self._start_operation)
        self.search_button.setEnabled(False); self.action_button.setEnabled(False)
        bottom_layout.addWidget(self.parse_button); bottom_layout.addWidget(self.search_button)
        bottom_layout.addStretch(); bottom_layout.addWidget(self.action_button)
        main_layout.addLayout(bottom_layout)
        
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Готово к работе.")

    def _set_ui_busy(self, is_busy: bool, message: str = ""):
        self.parse_button.setEnabled(not is_busy)
        self.search_button.setEnabled(False)
        self.action_button.setEnabled(False)
        self.statusBar().showMessage(message)

    def _start_parse(self):
        if not self.source_dir or not self.source_dir.is_dir():
            QMessageBox.critical(self, "Ошибка", f"Исходная папка не найдена или не является директорией:\n{self.source_dir}"); return
        text = self.text_input.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Внимание", "Поле для текста пустое."); return
        self._set_ui_busy(True, "Извлечение номеров...")
        self.worker.task = "parse"; self.worker.text_to_parse = text; self.worker.start()

    def _on_parse_finished(self, numbers: List[str], markup_html: str):
        self._set_ui_busy(False)
        self.tree_model = FileTreeModel(data=[{"number": num, "status": "Ожидает"} for num in numbers], icons=self.icons)
        self.tree_view.setModel(self.tree_model)
        
        self.markup_browser.setHtml(markup_html)
        if numbers:
            self.left_tabs.setCurrentIndex(1)
        
        if numbers:
            self.search_button.setEnabled(True)
            self.statusBar().showMessage(f"Найдено {len(numbers)} номеров. Готово к поиску файлов.")
        else:
            self.statusBar().showMessage("В тексте не найдено 4-значных номеров.")

    def _start_search(self):
        if not hasattr(self, 'tree_model'): return
        numbers_in_model = [self.tree_model.index(i, 0).data() for i in range(self.tree_model.rowCount())]
        if not numbers_in_model: return
        self._set_ui_busy(True, "Поиск файлов на диске...")
        self.worker.task = "search"
        self.worker.source_dir = self.source_dir
        self.worker.dest_dir = self.dest_dir
        self.worker.numbers_to_find = numbers_in_model
        self.worker.start()

    def _on_search_finished(self, results: List[Dict[str, Any]]):
        self._set_ui_busy(False)
        self.search_button.setEnabled(True)
        self.tree_model = FileTreeModel(results, icons=self.icons)
        self.tree_view.setModel(self.tree_model)
        for i in range(self.tree_model.columnCount()): self.tree_view.resizeColumnToContents(i)
        all_files = [f for item in results if item.get('files') for f in item['files']]
        if all_files:
            op_mode = getattr(self.config, '__fs_mode', 'copy')
            op_name = "Копировать" if op_mode == 'copy' else "Переместить"
            self.action_button.setText(f"3. {op_name} ({len(all_files)} файлов)")
            self.action_button.setEnabled(True)
            self.statusBar().showMessage(f"Поиск завершен. Найдено {len(all_files)} файлов.")
        else:
            self.statusBar().showMessage("Поиск завершен. Файлы не найдены.")

    def _start_operation(self):
        if not hasattr(self, 'tree_model'): return
        files_to_process = []
        root = self.tree_model._root_item
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(1) == 'Найден': files_to_process.extend(self.get_files_from_item(item))
        if not files_to_process: return
        op_mode = getattr(self.config, '__fs_mode', 'copy')
        op_name_verb = "скопировать" if op_mode == 'copy' else "ПЕРЕМЕСТИТЬ"
        if QMessageBox.question(self, "Подтверждение", f"Вы уверены, что хотите {op_name_verb} {len(files_to_process)} файлов в '{self.dest_dir.name}'?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No: return
        self._set_ui_busy(True, "Выполнение операции...")
        self.worker.task = "operation"
        self.worker.source_dir = self.source_dir
        self.worker.dest_dir = self.dest_dir
        self.worker.operation_mode = op_mode
        self.worker.files_to_process = list(set(files_to_process))
        self.worker.start()

    def get_files_from_item(self, item: TreeItem) -> List[pathlib.Path]:
        files = []
        base_dir = self.source_dir
        if file_path_str := item.data(2): files.append(base_dir / file_path_str)
        for i in range(item.childCount()):
            if child_file_path_str := item.child(i).data(2): files.append(base_dir / child_file_path_str)
        return files

    def _on_operation_finished(self, mode: str, processed_count: int, total_files: int):
        self._set_ui_busy(False)
        self.search_button.setEnabled(True)
        self.action_button.setEnabled(True)
        op_name_noun = "Копирование" if mode == "copy" else "Перемещение"
        if processed_count == total_files:
            self.final_status = 0
            QMessageBox.information(self, "Успех", f"{op_name_noun} успешно завершено.\nОбработано файлов: {processed_count}")
        else:
            QMessageBox.warning(self, "Завершено с ошибками", f"{op_name_noun} завершено.\nОбработано {processed_count} из {total_files} файлов.")
        self.statusBar().showMessage("Готово к работе.")

    def _on_worker_error(self, message: str):
        self._set_ui_busy(False); self.search_button.setEnabled(True)
        QMessageBox.critical(self, "Ошибка в фоновом потоке", message)
        self.statusBar().showMessage("Произошла ошибка.")

    def add_final_links(self):
        if IS_MANAGED_RUN and self.final_status == 0 and pysm_context:
            try:
                if self.dest_dir and self.dest_dir.exists(): 
                    pysm_context.log_link(url_or_path=str(self.dest_dir), text=f"<br>Открыть папку с отобранными файлами ('{self.dest_dir.name}')")
                if self.source_dir:
                    pysm_context.log_link(url_or_path=str(self.source_dir), text="Открыть исходную папку")
            except Exception as e:
                logger.error(f"Не удалось сгенерировать ссылки: {e}")

    def closeEvent(self, event):
        self.add_final_links(); event.accept()


# 7. БЛОК: Точка входа
# ==============================================================================
def main():
    config = get_config()
    app = QApplication.instance() or QApplication(sys.argv)
    window = FileSelectorWindow(config)
    window.show()
    exit_code = app.exec()
    sys.exit(window.final_status if exit_code == 0 else exit_code)

if __name__ == "__main__":
    main()