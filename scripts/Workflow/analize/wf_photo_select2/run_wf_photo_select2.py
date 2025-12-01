# run_wf_photo_select2.py

"""
Интерактивный скрипт для выборки, тегирования и переименования фотографий.

Предназначен для запуска в среде PyScriptManager.
1.  Извлекает 4-значные номера из текста.
2.  Ищет файлы в папке Capture.
3.  Для найденных файлов:
    - Обновляет XMP (Локация + Ключевые слова).
    - ОПЦИОНАЛЬНО: Переименовывает исходные файлы по шаблону "Имя-Номер".
4.  Ведет журнал операций.
"""

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import json
import logging
import pathlib
import re
import sys
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

# Добавляем путь к корню проекта для импорта _common
try:
    current_script_path = pathlib.Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from _common.xmp_editor import XmpEditor
except ImportError as e:
    print(f"Критическая ошибка импорта общих модулей: {e}", file=sys.stderr)
    sys.exit(1)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Импорты PySM и сторонних библиотек
try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    from pysm_lib.pysm_theme_api import format_ipymarkup_box, set_widget_class
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context, theme_api, format_ipymarkup_box = (None, None, None, None)
    try:
        from tqdm import tqdm
    except ImportError:
        class TqdmMock:
            def __init__(self, i=None, **kwargs): self.iterable = i or []
            def __iter__(self): return iter(self.iterable)
            @staticmethod
            def write(s, **kwargs): print(s)
        tqdm = TqdmMock

try:
    import ijson
except ImportError:
    print("Ошибка: ijson не найден. Установите: pip install ijson", file=sys.stderr)
    sys.exit(1)

try:
    from ipymarkup.palette import Palette
except ImportError:
    Palette = None

try:
    from PySide6.QtCore import QAbstractItemModel, QModelIndex, QSize, Qt, Signal, QThread
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
        QMainWindow, QMessageBox, QPushButton, QSplitter, QStatusBar, QStyle,
        QTabWidget, QTextBrowser, QTextEdit, QTreeView, QVBoxLayout, QWidget
    )
except ImportError:
    print("Ошибка: PySide6 не найден. Установите: pip install pyside6", file=sys.stderr)
    sys.exit(1)


# Константы
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2"}
CAPTURE_FOLDER = "Capture"

# 2. БЛОК: Вспомогательные функции
# ==============================================================================
def construct_session_paths() -> Dict[str, Optional[pathlib.Path]]:
    """Формирует пути к папке с исходными фото."""
    if not IS_MANAGED_RUN or not pysm_context:
        return {"source": None}

    photo_session = pysm_context.get("wf_photo_session", "")
    session_name = pysm_context.get("wf_session_name", "")
    session_path_str = pysm_context.get("wf_session_path", "")

    if not all([session_name, session_path_str, photo_session]):
        logger.error("Не найдены переменные контекста wf_session_*.")
        return {"source": None}

    base_path = pathlib.Path(session_path_str) / session_name
    source_dir = base_path / CAPTURE_FOLDER / photo_session
    return {"source": source_dir}


def construct_context_paths() -> Dict[str, Optional[pathlib.Path]]:
    """Формирует пути к файлам данных (JSON)."""
    if not IS_MANAGED_RUN or not pysm_context:
        return {"info_faces_path": None, "operations_log_path": None}

    photo_session = pysm_context.get("wf_photo_session")
    session_name = pysm_context.get("wf_session_name")
    session_path_str = pysm_context.get("wf_session_path")
    psd_path_str = pysm_context.get("wf_psd_path")

    if not all([session_path_str, session_name, photo_session, psd_path_str]):
        return {"info_faces_path": None, "operations_log_path": None}

    base_path = pathlib.Path(session_path_str) / session_name
    base_psd_path = pathlib.Path(psd_path_str) / session_name

    info_faces_path = base_path / "Output" / f"Analysis_{photo_session}" / "info_portrait_faces.json"
    operations_log_path = base_psd_path / "operations_log.json"

    return {"info_faces_path": info_faces_path, "operations_log_path": operations_log_path}


def load_faces_data(json_path: pathlib.Path) -> Dict[str, str]:
    """Загружает маппинг '4 цифры -> Имя'."""
    if not json_path or not json_path.exists():
        logger.warning(f"Файл {json_path} не найден.")
        return {}

    logger.info(f"Загрузка имен из {json_path.name}...")
    faces_map = {}
    try:
        with open(json_path, "rb") as f:
            parser = ijson.kvitems(f, "")
            for filename_key, data in parser:
                match = re.search(r"(\d{4})", filename_key)
                if not match: continue
                number = match.group(1)
                try:
                    child_name = data.get("faces", [{}])[0].get("child_name")
                    if child_name and number not in faces_map:
                        faces_map[number] = child_name
                except (IndexError, KeyError, TypeError):
                    continue
        return faces_map
    except Exception as e:
        logger.error(f"Ошибка чтения JSON: {e}")
        return {}


def read_operations_log(log_path: pathlib.Path) -> Dict:
    if not log_path or not log_path.exists(): return {}
    try:
        with open(log_path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

def write_operations_log(log_path: pathlib.Path, data: Dict):
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка записи лога: {e}")


# 3. БЛОК: Модель данных GUI (TreeModel)
# ==============================================================================
class TreeItem:
    def __init__(self, data: List, parent: "TreeItem" = None):
        self._data = data
        self._parent = parent
        self._children: List["TreeItem"] = []

    def child(self, row: int): return self._children[row] if 0 <= row < len(self._children) else None
    def childCount(self): return len(self._children)
    def columnCount(self): return len(self._data)
    def data(self, column: int): return self._data[column] if 0 <= column < len(self._data) else None
    def parent(self): return self._parent
    def row(self): return self._parent._children.index(self) if self._parent else 0
    def appendChild(self, item: "TreeItem"): self._children.append(item)


class FileTreeModel(QAbstractItemModel):
    def __init__(self, data=None, icons=None, colors=None, parent=None):
        super().__init__(parent)
        self.headers = ["№", "Номер", "Статус", "Имя (из базы)", "Файл"]
        self._root_item = TreeItem(self.headers)
        self.icons = icons or {}
        self.colors = colors or {}
        self.setup_model_data(data or [])

    def setup_model_data(self, model_data: List[Dict]):
        self.beginResetModel()
        self._root_item = TreeItem(self.headers)
        sorted_data = sorted(model_data, key=lambda x: (x.get("status", "") != "Найден", x["number"]))

        for idx, item_data in enumerate(sorted_data, 1):
            number = item_data["number"]
            status = item_data.get("status", "Ожидает")
            person_name = item_data.get("person_name", "")
            files = item_data.get("files", [])
            base_path = item_data.get("base_path")
            
            row_data = [str(idx), number, status, person_name, ""]
            if files and base_path:
                row_data[4] = f"({len(files)} файлов)"
            
            parent_item = TreeItem(row_data, self._root_item)
            
            if files and base_path:
                for fpath in files:
                    rel_path = str(fpath.relative_to(base_path))
                    parent_item.appendChild(TreeItem(["", "", "", "", rel_path], parent_item))
            
            self._root_item.appendChild(parent_item)
        self.endResetModel()

    def data(self, index, role):
        if not index.isValid(): return None
        item = index.internalPointer()
        col = index.column()

        if role == Qt.DisplayRole: return item.data(col)
        
        if role == Qt.DecorationRole and col == 4:
            if item.parent() == self._root_item: return self.icons.get("folder")
            return self.icons.get("file")

        if col == 2:
            status = item.data(2)
            is_found = status == "Найден"
            is_not_found = status == "Не найден"
            if role == Qt.BackgroundRole:
                return self.colors.get("found_bg") if is_found else (self.colors.get("not_found_bg") if is_not_found else None)
            if role == Qt.ForegroundRole:
                return self.colors.get("found_fg") if is_found else (self.colors.get("not_found_fg") if is_not_found else None)
        return None

    def index(self, row, col, parent=QModelIndex()):
        if not self.hasIndex(row, col, parent): return QModelIndex()
        parent_item = parent.internalPointer() if parent.isValid() else self._root_item
        child_item = parent_item.child(row)
        return self.createIndex(row, col, child_item) if child_item else QModelIndex()

    def parent(self, index):
        if not index.isValid(): return QModelIndex()
        child_item = index.internalPointer()
        parent_item = child_item.parent()
        if parent_item == self._root_item: return QModelIndex()
        return self.createIndex(parent_item.row(), 0, parent_item)

    def rowCount(self, parent=QModelIndex()):
        return (parent.internalPointer() if parent.isValid() else self._root_item).childCount()

    def columnCount(self, parent=QModelIndex()):
        return self._root_item.columnCount()
    
    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole: return self._root_item.data(section)
        return None


# 4. БЛОК: Воркер (фоновые задачи)
# ==============================================================================
class Worker(QThread):
    parse_finished = Signal(list, str)
    search_finished = Signal(list)
    operation_finished = Signal(int, int) # processed, total
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.task = ""
        self.text_to_parse = ""
        self.source_dir: Optional[pathlib.Path] = None
        self.info_faces_path: Optional[pathlib.Path] = None
        self.target_location_name = "" 
        self.numbers_to_find = []
        self.search_results = []
        self.faces_data = {}
        self.log_data = {}
        self.markup_color = None
        self.do_rename = True # По умолчанию включено

    def run(self):
        try:
            if self.task == "parse": self._task_parse()
            elif self.task == "search": self._task_search()
            elif self.task == "tagging": self._task_tagging_and_renaming()
        except Exception as e:
            logger.error(f"Error in worker: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

    def _task_parse(self):
        logger.info("Парсинг текста...")
        matches = list(re.finditer(r"(\d{4})", self.text_to_parse))
        numbers = sorted(list({m.group(1) for m in matches}))
        
        html = ""
        if format_ipymarkup_box and Palette:
            palette = Palette([self.markup_color]) if self.markup_color else None
            spans = [(m.start(), m.end(), "") for m in matches]
            try:
                html = format_ipymarkup_box(self.text_to_parse, spans, palette=palette)
            except Exception: html = "<p>Ошибка визуализации</p>"
        else:
            html = "<p>Визуализация недоступна</p>"
        
        self.parse_finished.emit(numbers, html)

    def _task_search(self):
        logger.info(f"Поиск файлов в {self.source_dir}...")
        self.faces_data = load_faces_data(self.info_faces_path)
        
        files_map = defaultdict(list)
        for f in self.source_dir.rglob("*"):
            if f.is_file():
                files_map[f.stem].append(f) 
        
        results = []
        for num in self.numbers_to_find:
            found_files = []
            for stem, files in files_map.items():
                if num in stem:
                    found_files.extend(files)
            
            found_files = sorted(list(set(found_files)))
            
            if found_files:
                results.append({
                    "number": num,
                    "status": "Найден",
                    "files": found_files,
                    "base_path": self.source_dir,
                    "person_name": self.faces_data.get(num, "")
                })
            else:
                results.append({"number": num, "status": "Не найден"})
        
        self.search_results = results
        self.search_finished.emit(results)

    def _task_tagging_and_renaming(self):
        """Тегирование XMP и переименование исходных файлов (если включено)."""
        logger.info(f"Начало обработки. Локация: {self.target_location_name}. Переименование: {self.do_rename}")
        
        items_to_process = [item for item in self.search_results if item.get("status") == "Найден"]
        total_items = len(items_to_process)
        processed_count = 0
        new_log_entries = defaultdict(list)
        
        loc_keyword = f"loc_{self.target_location_name}"

        with tqdm(items_to_process, desc="Обработка", unit="группа") as pbar:
            for item in pbar:
                number = item["number"]
                person_name = item.get("person_name")
                files = item["files"] 

                if person_name:
                    new_stem = f"{person_name}-{number}"
                else:
                    new_stem = None

                # Шаг 1: Обновление XMP
                for file_path in files:
                    try:
                        xmp_path = file_path.parent / f"{file_path.stem}.xmp"
                        editor = XmpEditor(xmp_path)
                        
                        editor.set_simple_field("Iptc4xmpCore", "Location", self.target_location_name)
                        editor.update_bag("dc", "subject", [loc_keyword], sort=True, append=True)
                        editor.update_bag("lightroom", "hierarchicalSubject", [loc_keyword], sort=True, append=True)
                        editor.save()
                        
                    except Exception as e:
                        tqdm.write(f"[WARN] Ошибка XMP для {file_path.name}: {e}")

                # Шаг 2: Переименование (опционально)
                if self.do_rename and new_stem:
                    renamed_files_log = []
                    for file_path in files:
                        if not file_path.exists(): continue
                        
                        target_name = f"{new_stem}{file_path.suffix}"
                        target_path = file_path.parent / target_name
                        
                        if target_path == file_path:
                            if file_path.suffix.lower() in RAW_EXTENSIONS:
                                renamed_files_log.append(target_name)
                            continue

                        if target_path.exists():
                            tqdm.write(f"[SKIP] Файл существует: {target_name}")
                            continue
                        
                        try:
                            file_path.rename(target_path)
                            if target_path.suffix.lower() in RAW_EXTENSIONS:
                                renamed_files_log.append(target_name)
                        except Exception as e:
                            tqdm.write(f"[ERR] Ошибка переименования {file_path.name}: {e}")
                    
                    if renamed_files_log:
                        new_log_entries[person_name if person_name else "Unknown"].extend(renamed_files_log)
                    processed_count += 1
                else:
                    # Если переименование отключено или нет имени, просто считаем обработанным
                    processed_count += 1

        self.log_data = new_log_entries
        self.operation_finished.emit(processed_count, total_items)


# 5. БЛОК: Главное окно
# ==============================================================================
class FileSelectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.final_status = 1
        
        paths = construct_session_paths()
        self.source_dir = paths.get("source")
        
        c_paths = construct_context_paths()
        self.info_faces_path = c_paths.get("info_faces_path")
        self.log_file_path = c_paths.get("operations_log_path")
        
        self.operations_log = read_operations_log(self.log_file_path)
        
        self._load_presets()
        self._init_theme()
        self._init_ui()
        self._init_worker()

        if IS_MANAGED_RUN and not self.source_dir:
            self._set_ui_busy(True, "Ошибка: пути сессии не сформированы.")

    def _load_presets(self):
        self.session_presets = []
        if IS_MANAGED_RUN and pysm_context:
            photo_session = pysm_context.get("wf_photo_session", "")
            key = f"sys_location_name_{photo_session}" if photo_session else "sys_location_name"
            presets = pysm_context.get(key, [])
            if isinstance(presets, dict): self.session_presets = list(presets.keys())
            elif isinstance(presets, list): self.session_presets = presets

    def _init_theme(self):
        self.colors = {}
        if IS_MANAGED_RUN and theme_api:
            self.ipymarkup_color = theme_api.get_ipymarkup_color(
                "markup_number",
                defaults={
                    "background-color": "#e3f2fd", "border-color": "#bbdefb", "color": "#000000"
                }
            )
            self.colors["found_bg"] = theme_api.get_qcolor("table_status_found", "background-color", "#d4edda")
            self.colors["found_fg"] = theme_api.get_qcolor("table_status_found", "color", "#155724")
            self.colors["not_found_bg"] = theme_api.get_qcolor("table_status_not_found", "background-color", "#f8d7da")
            self.colors["not_found_fg"] = theme_api.get_qcolor("table_status_not_found", "color", "#721c24")
        else:
            self.ipymarkup_color = None
            self.colors = {"found_bg": QColor("#d4edda"), "found_fg": QColor("#155724"),
                           "not_found_bg": QColor("#f8d7da"), "not_found_fg": QColor("#721c24")}

    def _init_ui(self):
        title = "Выборка: Тегирование и Переименование"
        if self.source_dir: title += f" - [{self.source_dir.name}]"
        self.setWindowTitle(title)
        self.resize(1200, 750)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Левая панель
        left_panel = QTabWidget()
        input_tab = QWidget()
        l_in = QVBoxLayout(input_tab)
        l_in.addWidget(QLabel("Вставьте текст с номерами:"))
        self.text_input = QTextEdit()
        l_in.addWidget(self.text_input)
        left_panel.addTab(input_tab, "Ввод")
        
        self.markup_browser = QTextBrowser()
        self.markup_browser.setOpenExternalLinks(True)
        left_panel.addTab(self.markup_browser, "Разбор")
        
        # Правая панель
        right_panel = QWidget()
        l_right = QVBoxLayout(right_panel)
        l_right.addWidget(QLabel("Результаты поиска:"))
        self.tree_view = QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        l_right.addWidget(self.tree_view)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        # Нижняя панель
        bot_layout = QHBoxLayout()
        self.btn_parse = QPushButton("1. Извлечь номера")
        self.btn_search = QPushButton("2. Найти файлы")
        self.btn_action = QPushButton("3. Выполнить")
        if IS_MANAGED_RUN: set_widget_class(self.btn_action, "primary")

        self.combo_presets = QComboBox()
        self.combo_presets.setMinimumWidth(250)
        if self.session_presets: self.combo_presets.addItems(self.session_presets)
        else:
            self.combo_presets.addItem("Нет пресетов")
            self.combo_presets.setEnabled(False)

        # Чекбокс переименования
        self.chk_rename = QCheckBox("Переименовать файлы")
        self.chk_rename.setChecked(True)

        bot_layout.addWidget(self.btn_parse)
        bot_layout.addWidget(self.btn_search)
        bot_layout.addStretch() # Растягиваем пространство между кнопками слева и настройками справа
        
        bot_layout.addWidget(self.chk_rename)
        
        bot_layout.addSpacing(40) # <-- ИЗМЕНЕНИЕ: Добавлен отступ между чекбоксом и локацией
        
        bot_layout.addWidget(QLabel("Локация (тег):"))
        bot_layout.addWidget(self.combo_presets)
        bot_layout.addWidget(self.btn_action)
        layout.addLayout(bot_layout)

        self.setStatusBar(QStatusBar())
        self.btn_search.setEnabled(False)
        self.btn_action.setEnabled(False)
        
        self.btn_parse.clicked.connect(self.start_parse)
        self.btn_search.clicked.connect(self.start_search)
        self.btn_action.clicked.connect(self.start_tagging)

        self.icons = {
            "folder": self.style().standardIcon(QStyle.SP_DirIcon),
            "file": self.style().standardIcon(QStyle.SP_FileIcon)
        }

    def _init_worker(self):
        self.worker = Worker()
        self.worker.markup_color = self.ipymarkup_color
        self.worker.parse_finished.connect(self.on_parse_done)
        self.worker.search_finished.connect(self.on_search_done)
        self.worker.operation_finished.connect(self.on_tagging_done)
        self.worker.error_occurred.connect(lambda msg: self._set_ui_busy(False, f"Ошибка: {msg}"))

    def _set_ui_busy(self, busy: bool, msg: str = ""):
        self.btn_parse.setEnabled(not busy)
        self.btn_search.setEnabled(False if busy else True)
        self.btn_action.setEnabled(False if busy else True)
        self.statusBar().showMessage(msg)

    # Слоты
    def start_parse(self):
        txt = self.text_input.toPlainText()
        if not txt.strip(): return
        self._set_ui_busy(True, "Парсинг...")
        self.worker.task = "parse"
        self.worker.text_to_parse = txt
        self.worker.start()

    def on_parse_done(self, numbers, html):
        self._set_ui_busy(False, f"Найдено номеров: {len(numbers)}")
        self.markup_browser.setHtml(html)
        self.tree_view.setModel(FileTreeModel([{"number": n} for n in numbers], self.icons, self.colors))
        self.btn_search.setEnabled(bool(numbers))
        
    def start_search(self):
        if not self.source_dir: 
            QMessageBox.warning(self, "Ошибка", "Не найден путь к папке Capture.")
            return

        model = self.tree_view.model()
        if not model: return
        
        if self.worker.isRunning(): self.worker.wait()
        
        nums = [model.index(i, 1).data() for i in range(model.rowCount())] 
        self._set_ui_busy(True, "Поиск файлов...")
        self.worker.task = "search"
        self.worker.source_dir = self.source_dir
        self.worker.info_faces_path = self.info_faces_path
        self.worker.numbers_to_find = nums
        self.worker.start()

    def on_search_done(self, results):
        count = sum(len(r.get("files", [])) for r in results)
        self._set_ui_busy(False, f"Найдено файлов: {count}")
        self.tree_view.setModel(FileTreeModel(results, self.icons, self.colors))
        self.tree_view.collapseAll()
        for i in range(5): self.tree_view.resizeColumnToContents(i)
        self.btn_action.setEnabled(count > 0)
        btn_text = f"3. Обработать ({count} файлов)"
        self.btn_action.setText(btn_text)

    def start_tagging(self):
        loc = self.combo_presets.currentText()
        if not loc: return
        
        rename_files = self.chk_rename.isChecked()
        
        msg = f"1. Запись локации '{loc}' в XMP.\n"
        if rename_files:
            msg += "2. ПЕРЕИМЕНОВАНИЕ файлов (Имя-Номер).\n"
        else:
            msg += "2. (Переименование отключено).\n"
            
        msg += "\nПродолжить?"
        
        reply = QMessageBox.question(self, "Подтверждение", msg, QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return

        self._set_ui_busy(True, "Выполнение...")
        self.worker.task = "tagging"
        self.worker.target_location_name = loc
        self.worker.do_rename = rename_files
        self.worker.start()

    def on_tagging_done(self, processed, total):
        self._set_ui_busy(False, "Готово!")
        self.final_status = 0
        self._update_log_file()
        QMessageBox.information(self, "Успех", f"Успешно обработано групп: {processed} из {total}.")

    def _update_log_file(self):
        log_data = self.worker.log_data
        session_name = self.worker.target_location_name
        if not log_data: return

        for person, files in log_data.items():
            if person not in self.operations_log: self.operations_log[person] = {}
            if session_name not in self.operations_log[person]:
                self.operations_log[person][session_name] = {"timestamp": "", "files": []}
            
            current_files = set(self.operations_log[person][session_name]["files"])
            for f in files: current_files.add(f)
            
            self.operations_log[person][session_name]["files"] = sorted(list(current_files))
            self.operations_log[person][session_name]["timestamp"] = datetime.utcnow().isoformat()

        write_operations_log(self.log_file_path, self.operations_log)


# 6. БЛОК: Запуск
# ==============================================================================
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    if IS_MANAGED_RUN and theme_api: theme_api.apply_theme_to_app(app)
    
    win = FileSelectorWindow()
    win.show()
    
    code = app.exec()
    sys.exit(win.final_status if code == 0 else code)

if __name__ == "__main__":
    main()