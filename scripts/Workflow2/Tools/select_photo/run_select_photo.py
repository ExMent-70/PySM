# run_wf_photo_copy.py

"""
Интерактивный скрипт для пакетной обработки фотографий.

Основные функции:
1.  **Парсинг:** Извлекает номера фотографий из произвольного текста.
2.  **Поиск:** Находит соответствующие файлы (RAW, JPG, XMP) в исходной папке.
3.  **Обработка:**
    -   Обновляет метаданные XMP (локация, ключевые слова).
    -   Копирует файлы в целевую структуру с сохранением иерархии папок.
    -   Опционально переименовывает файлы при копировании.
4.  **Синхронизация:** Обновляет файл соответствия портретов и групповых фото.
"""

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import json
import logging
import pathlib
import re
import shutil
import sys
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    current_script_path = pathlib.Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # JsonDataManager удален
    from _common.xmp_editor import XmpEditor
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr); sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    from pysm_lib.pysm_theme_api import format_ipymarkup_box, set_widget_class
    IS_MANAGED_RUN = True
except ImportError:
    (IS_MANAGED_RUN, ConfigResolver, pysm_context, theme_api, format_ipymarkup_box, set_widget_class) = (False, None, None, None, None, None)
    try: from tqdm import tqdm
    except ImportError:
        class TqdmMock:
            def __init__(self, i=None, **kwargs): self.iterable = i or []
            def __iter__(self): return iter(self.iterable)
            @staticmethod
            def write(s, **kwargs): print(s)
        tqdm = TqdmMock

try: import ijson
except ImportError: print("Ошибка: ijson не найден.", file=sys.stderr); sys.exit(1)
try: from ipymarkup.palette import Palette
except ImportError: Palette = None
try:
    from PySide6.QtCore import QAbstractItemModel, QModelIndex, QSize, Qt, Signal, QThread
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
        QMainWindow, QMessageBox, QPushButton, QSplitter, QStatusBar,
        QStyle, QTabWidget, QTextBrowser, QTextEdit, QTreeView, QVBoxLayout, QWidget,
    )
except ImportError:
    print("Ошибка: PySide6 не найден.", file=sys.stderr); sys.exit(1)

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2"}
CAPTURE_FOLDER = "Capture"
SELECTS_FOLDER = "Selects"


# 2. БЛОК: Получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """
    Парсит аргументы командной строки и разрешает конфигурацию через PySM Context.

    Returns:
        Namespace: Объект с конфигурационными параметрами (пути, флаги).
    """
    parser = argparse.ArgumentParser(description="Пакетная обработка файлов по номерам.")
    
    # --- ИЗМЕНЕНО: Новые аргументы путей ---
    parser.add_argument(
        "--analysis_dir", 
        type=str, 
        required=True, 
        help="Путь к папке с результатами анализа (содержащей info_faces.json)."
    )
    # ---------------------------------------
    
    parser.add_argument("--source_dir", type=str, required=True, help="Папка с исходными файлами.")
    parser.add_argument("--dest_dir", type=str, required=False, help="Корневая папка для результатов (обязательна для копирования).")
    parser.add_argument("--min_digits", type=int, default=4, help="Мин. длина номера.")
    parser.add_argument("--max_digits", type=int, default=4, help="Макс. длина номера.")
    parser.add_argument("--exclude_dirs", nargs="+", default=["CaptureOne"], help="Папки для исключения.")
    
    # Флаги операций
    parser.add_argument("--update-xmp", action="store_true", help="Активировать обновление XMP-тегов.")
    parser.add_argument("--rename-files", action="store_true", help="Активировать переименование файлов при копировании.")
    parser.add_argument("--copy-files", action="store_true", help="Активировать копирование файлов в папку назначения.")

    if IS_MANAGED_RUN and ConfigResolver: return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


# 3. БЛОК: Вспомогательные функции
# ==============================================================================
def _read_json_safely(json_path: pathlib.Path) -> Dict:
    """
    Безопасно читает JSON файл.

    Args:
        json_path (pathlib.Path): Путь к файлу.

    Returns:
        Dict: Данные JSON или пустой словарь в случае ошибки.
    """
    if not json_path or not json_path.exists(): return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка чтения {json_path.name}: {e}"); return {}

def _write_json_safely(json_path: pathlib.Path, data: Dict):
    """
    Безопасно записывает словарь в JSON файл.

    Args:
        json_path (pathlib.Path): Путь для сохранения.
        data (Dict): Данные для записи.
    """
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Файл успешно сохранен: {json_path.name}")
    except Exception as e:
        logger.error(f"Ошибка записи {json_path.name}: {e}")


# 4. БЛОК: Модель данных GUI
# ==============================================================================
class TreeItem:
    """
    Узел древовидной структуры данных для модели Qt.
    """
    def __init__(self, data: List, parent: "TreeItem" = None):
        """
        Args:
            data (List): Данные столбцов для этого узла.
            parent (TreeItem, optional): Родительский узел.
        """
        self._data = data
        self._parent = parent
        self._children: List["TreeItem"] = []

    def child(self, row: int) -> Optional["TreeItem"]:
        """Возвращает дочерний элемент по индексу строки."""
        return self._children[row] if 0 <= row < len(self._children) else None

    def childCount(self) -> int:
        """Возвращает количество дочерних элементов."""
        return len(self._children)

    def columnCount(self) -> int:
        """Возвращает количество столбцов данных."""
        return len(self._data)

    def data(self, column: int) -> Any:
        """Возвращает данные конкретного столбца."""
        return self._data[column] if 0 <= column < len(self._data) else None

    def parent(self) -> Optional["TreeItem"]:
        """Возвращает родительский узел."""
        return self._parent

    def row(self) -> int:
        """Возвращает индекс строки этого узла в родителе."""
        return self._parent._children.index(self) if self._parent else 0

    def appendChild(self, item: "TreeItem"):
        """Добавляет дочерний узел."""
        self._children.append(item)


class ResultTreeModel(QAbstractItemModel):
    """
    Модель данных Qt (AbstractItemModel) для отображения иерархического списка найденных файлов.
    Колонки: №, Номер, Имя, Локация, Статус, Новое имя, Файлы.
    """
    def __init__(self, data=None, icons=None, colors=None, parent=None):
        super().__init__(parent)
        self.headers = ["№", "Номер", "Имя", "Локация", "Статус", "Новое имя (основа)", "Файлы"]
        self._root_item = TreeItem(self.headers)
        self.icons = icons or {}
        self.colors = colors or {}
        self.setup_model_data(data or [])

    def setup_model_data(self, model_data: List[Dict]):
        """
        Инициализирует структуру дерева на основе списка словарей с данными.
        
        Args:
            model_data (List[Dict]): Список результатов поиска.
        """
        self.beginResetModel()
        self._root_item = TreeItem(self.headers)
        sorted_data = sorted(model_data, key=lambda x: int(x.get("number", 0)))
        
        for idx, item_data in enumerate(sorted_data, 1):
            files = item_data.get("files", [])
            base_path = item_data.get("base_path")
            files_count_str = f"({len(files)} файлов)" if files else ""
            row_data = [
                str(idx), 
                item_data.get("number", ""),
                item_data.get("child_name", ""),
                item_data.get("location_name", ""),
                item_data.get("status", "Ожидает"),
                item_data.get("new_stem", ""),
                files_count_str,
            ]
            parent_item = TreeItem(row_data, self._root_item)

            if files and base_path:
                for fpath in files:
                    # Вычисляем относительный путь для отображения
                    try:
                        rel_path = fpath.relative_to(base_path)
                    except ValueError:
                        rel_path = fpath.name
                    child_row_data = ["", "", "", "", "", "", str(rel_path)]
                    parent_item.appendChild(TreeItem(child_row_data, parent_item))

            self._root_item.appendChild(parent_item)
        self.endResetModel()

    def data(self, index: QModelIndex, role: int) -> Any:
        """Возвращает данные для ячейки таблицы в зависимости от роли (текст, цвет, иконка)."""
        if not index.isValid(): return None
        item = index.internalPointer()
        col = index.column()

        if role == Qt.DisplayRole:
            return item.data(col)

        if role == Qt.DecorationRole and col == 6:
            if item.parent() == self._root_item and item.childCount() > 0:
                return self.icons.get("folder")
            elif item.parent() != self._root_item:
                return self.icons.get("file")

        if col == 4: # Колонка статуса
            status = item.data(4)
            is_found = status == "Найден"
            is_not_found = status == "Не найден"
            if role == Qt.BackgroundRole:
                if is_found: return self.colors.get("found_bg")
                if is_not_found: return self.colors.get("not_found_bg")
            if role == Qt.ForegroundRole:
                if is_found: return self.colors.get("found_fg")
                if is_not_found: return self.colors.get("not_found_fg")

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
        if parent.column() > 0: return 0
        parent_item = parent.internalPointer() if parent.isValid() else self._root_item
        return parent_item.childCount()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.headers)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole: return self._root_item.data(section)
        return None

    def sort(self, column: int, order: Qt.SortOrder):
        """Сортирует таблицу по выбранной колонке."""
        self.layoutAboutToBeChanged.emit()
        reverse_order = order == Qt.DescendingOrder

        def sort_key(item):
            data = item.data(column)
            if data is None: return ""
            if column in [0, 1]: # Числовая сортировка для № и Номера
                try: return int(data)
                except (ValueError, TypeError): return 0
            return str(data)

        self._root_item._children.sort(key=sort_key, reverse=reverse_order)
        self.layoutChanged.emit()


# 5. БЛОК: Воркер
# ==============================================================================
class Worker(QThread):
    """
    Фоновый рабочий поток для выполнения длительных операций (парсинг, поиск, копирование),
    чтобы не блокировать графический интерфейс.
    """
    parse_finished = Signal(list, str)
    search_finished = Signal(list)
    processing_finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.task = ""
        self.source_dir: Optional[pathlib.Path] = None
        self.dest_dir: Optional[pathlib.Path] = None
        self.metadata = {}
        self.numbers_to_find = []
        self.search_results = []
        self.text_to_parse = ""
        self.min_digits = 4
        self.max_digits = 4
        self.markup_color = None
        self.palette = None
        self.exclude_dirs: List[str] = []
        self.do_update_xmp = False
        self.do_rename_files = False
        self.do_copy_files = False

    def run(self):
        """Точка входа потока. Распределяет задачи."""
        try:
            if self.task == "parse": self._task_parse()
            elif self.task == "search": self._task_search()
            elif self.task == "process": self._task_process_files()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _init_palette(self):
        """Инициализирует палитру цветов для ipymarkup."""
        if Palette and self.markup_color:
            try:
                self.palette = Palette([self.markup_color])
            except Exception as e:
                logger.error(f"Ошибка при создании палитры ipymarkup: {e}")

    def _task_parse(self):
        """Задача: Извлечение номеров из текста и генерация HTML с подсветкой."""
        self._init_palette()
        logger.info("Парсинг текста...")
        min_d, max_d = self.min_digits, self.max_digits
        if min_d > max_d: min_d, max_d = max_d, min_d
        regex = r"(\d{%d,%d})" % (min_d, max_d)
        matches = list(re.finditer(regex, self.text_to_parse))
        numbers = sorted(list({m.group(1) for m in matches}))
        html = ""
        if format_ipymarkup_box and self.palette:
            spans = [(m.start(), m.end(), "") for m in matches]
            html = format_ipymarkup_box(self.text_to_parse, spans, palette=self.palette)
        self.parse_finished.emit(numbers, html)

    def _task_search(self):
        """Задача: Поиск файлов в исходной папке по номерам."""
        logger.info(f"Поиск файлов в {self.source_dir}...")
        logger.info(f"Исключая папки: {self.exclude_dirs}")
        all_files_in_dir = []
        excluded_paths = [self.source_dir / name for name in self.exclude_dirs]
        for f in self.source_dir.rglob("*"):
            if not f.is_file(): continue
            is_excluded = any(ex_path in f.parents for ex_path in excluded_paths)
            if not is_excluded: all_files_in_dir.append(f)
        logger.info(f"Всего файлов для анализа: {len(all_files_in_dir)}")

        number_to_metadata = {}
        # Unified Storage: метаданные уже содержат все файлы
        for filename, data in self.metadata.items():
            # Извлекаем номер из имени файла (IMG_1234.jpg -> 1234)
            match = re.search(r"(\d{4})", filename)
            if match and match.group(1) not in number_to_metadata:
                number_to_metadata[match.group(1)] = data

        files_map = defaultdict(list)
        for f in all_files_in_dir:
            match = re.search(r"(\d{4})", f.stem)
            if match: files_map[match.group(1)].append(f)
        
        results = []
        for num in self.numbers_to_find:
            item_data = number_to_metadata.get(num, {})
            found_files = files_map.get(num, [])
            status = "Найден" if found_files else "Не найден"

            # Извлечение имени ребенка (Unified Storage: может быть в первом лице)
            child_name = ""
            faces = item_data.get("faces", [])
            if faces and isinstance(faces[0], dict):
                # Пробуем найти child_name в лице
                child_name = faces[0].get("child_name", "")
            
            # Извлечение локации (Unified Storage: лежит в корне записи)
            location_name = item_data.get("location_name", "Не определена")
            
            new_stem = ""
            if found_files:
                if child_name: new_stem = f"{child_name}-{num}"
                else: new_stem = found_files[0].stem

            results.append({
                "number": num, "status": status, "files": sorted(list(set(found_files))),
                "child_name": child_name, 
                "location_name": location_name,
                "base_path": self.source_dir, "new_stem": new_stem
            })
        self.search_results = results
        self.search_finished.emit(results)

    def _task_process_files(self):
        """
        Задача: Выполняет основные операции (XMP, Копирование).
        Критически важно: при копировании сохраняет структуру папок относительно source_dir.
        """
        tasks = []
        if self.do_update_xmp: tasks.append("Обновление XMP")
        if self.do_copy_files: tasks.append("Копирование")
        if not tasks:
            self.processing_finished.emit("Нет задач для выполнения.")
            return

        logger.info(f"Начало обработки. Задачи: {', '.join(tasks)}.")
        items_to_process = [item for item in self.search_results if item["status"] == "Найден"]
        
        with tqdm(items_to_process, desc="Обработка", unit="группа") as pbar:
            for item in pbar:
                files = item.get("files", [])
                
                # --- Шаг 1: Обновление XMP (если включено) ---
                if self.do_update_xmp:
                    location = item.get("location_name", "unknown")
                    loc_keyword = f"loc_{location}"
                    for file_path in files:
                        if file_path.suffix.lower() in [".xmp", *RAW_EXTENSIONS]:
                            try:
                                xmp_path = file_path.with_suffix(".xmp")
                                if (
                                    not xmp_path.exists()
                                    and file_path.suffix.lower() in RAW_EXTENSIONS
                                ):
                                    logger.debug(f"Создание XMP для {file_path.name}")

                                editor = XmpEditor(xmp_path)
                                editor.set_simple_field(
                                    "Iptc4xmpCore", "Location", location
                                )
                                editor.update_bag(
                                    "dc", "subject", [loc_keyword], append=True, sort=True
                                )
                                editor.update_bag(
                                    "lightroom",
                                    "hierarchicalSubject",
                                    [loc_keyword],
                                    append=True,
                                    sort=True,
                                )
                                editor.save()
                            except Exception as e:
                                tqdm.write(f"[ОШИБКА XMP] {file_path.name}: {e}")

                # --- Шаг 2: Копирование с сохранением структуры (если включено) ---
                if self.do_copy_files:
                    if not self.dest_dir:
                        tqdm.write("[КРИТИЧЕСКАЯ ОШИБКА] Копирование включено, но не указана папка назначения (dest_dir).")
                        break 

                    location = item.get("location_name", "unknown")
                    child_name = item.get("child_name")
                    number = item.get("number")
                    
                    target_location_root = self.dest_dir / str(location)

                    new_stem = None
                    if self.do_rename_files and child_name:
                        new_stem = f"{child_name}-{number}"

                    for src_file in files:
                        try:
                            # КЛЮЧЕВОЙ МОМЕНТ: Вычисляем путь относительно корня поиска
                            relative_path = src_file.relative_to(self.source_dir)
                        except ValueError:
                            relative_path = pathlib.Path(src_file.name)
                        
                        # Воссоздаем структуру папок
                        dest_folder = target_location_root / relative_path.parent
                        dest_folder.mkdir(parents=True, exist_ok=True)
                        
                        dest_name = f"{new_stem}{src_file.suffix}" if new_stem else src_file.name
                        dest_file = dest_folder / dest_name
                        
                        if dest_file.exists():
                            tqdm.write(f"[ПРОПУСК] Файл {dest_file.name} уже существует.")
                            continue
                        try:
                            shutil.copy2(src_file, dest_file)
                        except Exception as e:
                            tqdm.write(f"[ОШИБКА] Не удалось скопировать {src_file.name}: {e}")
        
        self.processing_finished.emit(f"Обработка завершена. Обработано групп: {len(items_to_process)}.")


# 6. БЛОК: Главное окно
# ==============================================================================
class SorterWindow(QMainWindow):
    """
    Главное окно приложения на PySide6.
    """
    def __init__(self, config: Namespace):
        """
        Инициализирует окно, загружает данные и настраивает UI.
        
        Args:
            config (Namespace): Конфигурация запуска.
        """
        super().__init__()
        self.config = config
        self.final_status = 1
        self.metadata = {}
        self.matches_data = {}

        try:
            analysis_dir = pathlib.Path(config.analysis_dir)
            faces_json_path = analysis_dir / "info_faces.json"
            
            if not faces_json_path.exists():
                raise FileNotFoundError(f"Файл {faces_json_path} не найден.")
                
            # Загрузка единого JSON
            self.metadata = _read_json_safely(faces_json_path)
            
            matches_path = analysis_dir / "matches_portrait_to_group.json"
            if matches_path.exists():
                self.matches_data = _read_json_safely(matches_path)
                self.matches_json_path = matches_path
            else:
                self.matches_data = {}
                self.matches_json_path = None
                
        except Exception as e:
            QMessageBox.critical(self, "Критическая ошибка", f"Не удалось загрузить файлы данных:\n{e}")
            self.setEnabled(False)

        self._load_theme_colors()
        self._init_ui()
        self._init_worker()

    def _load_theme_colors(self):
        """Загружает цвета из темы PySM или использует дефолтные."""
        if IS_MANAGED_RUN and theme_api:
            self.ipymarkup_color = theme_api.get_ipymarkup_color("markup_number", defaults={"background-color": "#e3f2fd", "border-color": "#bbdefb", "color": "#000000"})
            self.colors = {
                "found_bg": theme_api.get_qcolor("table_status_found", "background-color", "#d4edda"),
                "found_fg": theme_api.get_qcolor("table_status_found", "color", "#155724"),
                "not_found_bg": theme_api.get_qcolor("table_status_not_found", "background-color", "#f8d7da"),
                "not_found_fg": theme_api.get_qcolor("table_status_not_found", "color", "#721c24"),
            }
        else:
            self.ipymarkup_color = None
            self.colors = {"found_bg": QColor("#d4edda"), "found_fg": QColor("#155724"), "not_found_bg": QColor("#f8d7da"), "not_found_fg": QColor("#721c24")}

    def _init_ui(self):
        """Настраивает графический интерфейс (кнопки, таблицы, вкладки)."""
        title = "Сортировка по локациям"
        source_dir = pathlib.Path(self.config.source_dir)
        if source_dir: title += f" - [{source_dir.name}]"
        self.setWindowTitle(title)
        self.resize(1200, 750)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.left_tabs = QTabWidget()
        input_tab = QWidget()
        l_in = QVBoxLayout(input_tab)
        l_in.addWidget(QLabel("Вставьте текст с номерами:"))
        self.text_input = QTextEdit()
        l_in.addWidget(self.text_input)
        self.left_tabs.addTab(input_tab, "Ввод")
        self.markup_browser = QTextBrowser(); self.markup_browser.setOpenExternalLinks(True)
        self.left_tabs.addTab(self.markup_browser, "Разбор")

        right_panel = QWidget()
        l_right = QVBoxLayout(right_panel)
        l_right.addWidget(QLabel("Результаты анализа:"))
        self.tree_view = QTreeView(); self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setSortingEnabled(True)
        l_right.addWidget(self.tree_view)
        l_right.addWidget(QLabel("Найденные номера для копирования:"))
        self.numbers_line_edit = QLineEdit(); self.numbers_line_edit.setReadOnly(True)
        l_right.addWidget(self.numbers_line_edit)

        splitter.addWidget(self.left_tabs)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        bottom_layout = QHBoxLayout()
        self.btn_parse = QPushButton("1. Извлечь номера")
        self.btn_search = QPushButton("2. Найти файлы")
        self.btn_copy = QPushButton("3. Копировать файлы")
        set_widget_class(self.btn_copy, "primary")
        self.chk_sync_matches = QCheckBox("Синхронизировать matches.json")
        self.chk_sync_matches.setChecked(False)

        # Новая кнопка "Закрыть"
        self.btn_close = QPushButton("Закрыть")
        set_widget_class(self.btn_close, "danger")
        self.btn_close.clicked.connect(self.close)

        bottom_layout.addWidget(self.btn_parse)
        bottom_layout.addWidget(self.btn_search)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.chk_sync_matches)
        bottom_layout.addWidget(self.btn_copy)
        bottom_layout.addWidget(self.btn_close)
        layout.addLayout(bottom_layout)

        self.btn_search.setEnabled(False); self.btn_copy.setEnabled(False)
        self.setStatusBar(QStatusBar())

        self.btn_parse.clicked.connect(self.start_parse)
        self.btn_search.clicked.connect(self.start_search)
        self.btn_copy.clicked.connect(self.start_processing)

        self.icons = {"folder": self.style().standardIcon(QStyle.SP_DirIcon), "file": self.style().standardIcon(QStyle.SP_FileIcon)}

    def _init_worker(self):
        """Инициализирует фоновый поток (Worker)."""
        self.worker = Worker()
        self.worker.markup_color = self.ipymarkup_color
        self.worker.parse_finished.connect(self.on_parse_finished)
        self.worker.search_finished.connect(self.on_search_finished)
        self.worker.processing_finished.connect(self.on_processing_finished)
        self.worker.error_occurred.connect(lambda msg: self.statusBar().showMessage(f"Ошибка: {msg}"))

    def start_parse(self):
        """Запускает процесс парсинга текста."""
        text = self.text_input.toPlainText()
        if not text.strip(): return
        self.numbers_line_edit.clear()
        self.statusBar().showMessage("Парсинг...")
        self.worker.task = "parse"
        self.worker.text_to_parse = text
        self.worker.min_digits = self.config.min_digits
        self.worker.max_digits = self.config.max_digits
        self.worker.start()

    def on_parse_finished(self, numbers, html):
        """Обработчик завершения парсинга."""
        self.statusBar().showMessage(f"Найдено номеров: {len(numbers)}")
        self.markup_browser.setHtml(html)
        self.numbers_line_edit.setText(" ".join(numbers))
        self.btn_search.setEnabled(bool(numbers))
        self.tree_view.setModel(ResultTreeModel([]))
        if numbers:
            self.left_tabs.setCurrentIndex(1)

    def start_search(self):
        """Запускает поиск файлов по извлеченным номерам."""
        if not self.isEnabled(): return
        numbers = self.numbers_line_edit.text().split()
        if not numbers: return
        self.statusBar().showMessage("Поиск и анализ...")
        self.worker.task = "search"
        self.worker.source_dir = pathlib.Path(self.config.source_dir)
        self.worker.metadata = self.metadata
        self.worker.numbers_to_find = numbers
        self.worker.exclude_dirs = getattr(self.config, "exclude_dirs", [])
        self.worker.start()

    def on_search_finished(self, results):
        """Обработчик завершения поиска."""
        self.statusBar().showMessage(f"Анализ завершен. Найдено групп: {len(results)}")
        model = ResultTreeModel(results, self.icons, getattr(self, "colors", {}))
        self.tree_view.setModel(model)
        self.tree_view.collapseAll() # Сворачиваем строки по умолчанию
        for i in range(model.columnCount()):
            self.tree_view.resizeColumnToContents(i)
        self.tree_view.header().setSectionResizeMode(5, QHeaderView.Stretch)
        found_items_count = sum(1 for r in results if r["status"] == "Найден")
        self.btn_copy.setEnabled(found_items_count > 0)
        self.btn_copy.setText(f"3. Копировать ({found_items_count} групп)")

    def start_processing(self): 
        """
        Запускает основные операции в фоновом потоке: XMP, Копирование, Переименование.
        """
        if not self.worker.search_results:
            QMessageBox.warning(self, "Внимание", "Нет файлов для обработки."); return

        tasks = []
        do_update_xmp = getattr(self.config, "update_xmp", False)
        do_rename_files = getattr(self.config, "rename_files", False)
        do_copy_files = getattr(self.config, "copy_files", False)

        if do_update_xmp: tasks.append("- Обновление XMP-тегов")
        if do_copy_files:
            if do_rename_files: tasks.append("- Копирование с переименованием")
            else: tasks.append("- Копирование без переименования")
        
        if not tasks:
            QMessageBox.information(self, "Информация", "Ни одна из операций (обновление XMP, копирование) не была включена в параметрах запуска.")
            return

        msg = "Будут выполнены следующие операции:\n\n" + "\n".join(tasks) + "\n\nПродолжить?"
        if QMessageBox.question(self, "Подтверждение", msg, QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
            return

        self.statusBar().showMessage("Выполнение операций...")
        self.btn_copy.setEnabled(False)
        
        self.worker.task = "process"
        self.worker.search_results = self.worker.search_results
        self.worker.do_update_xmp = do_update_xmp
        self.worker.do_rename_files = do_rename_files
        self.worker.do_copy_files = do_copy_files
        
        if do_copy_files:
            if not self.config.dest_dir:
                QMessageBox.critical(self, "Ошибка", "Операция копирования включена, но не указана папка назначения (--dest_dir).")
                return
            self.worker.dest_dir = pathlib.Path(self.config.dest_dir)
            
        self.worker.start()

    def on_processing_finished(self, message: str):
        """
        Обрабатывает завершение всех операций. Очищает интерфейс для новой работы.
        
        Args:
            message (str): Сообщение о результате.
        """
        self.statusBar().showMessage(message)
        self.final_status = 0
        
        if self.chk_sync_matches.isChecked():
            self.sync_matches_file()

        # --- ОЧИСТКА ИНТЕРФЕЙСА ---
        self.markup_browser.clear()
        self.numbers_line_edit.clear()
        self.tree_view.setModel(ResultTreeModel([]))
        self.btn_search.setEnabled(False)
        self.btn_copy.setEnabled(False)
        self.left_tabs.setCurrentIndex(0) # Переключение на вкладку "Ввод"
        # ---------------------------

        QMessageBox.information(self, "Успех", "Все операции успешно завершены.")
        
        if IS_MANAGED_RUN and pysm_context and getattr(self.config, "copy_files", False):
            dest_dir = pathlib.Path(self.config.dest_dir)
            if dest_dir.exists():
                pysm_context.log_link(
                    url_or_path=str(dest_dir),
                    text=f"<br>Открыть папку с результатами ('{dest_dir.name}')",
                )

    def sync_matches_file(self):
        """Координирует процесс синхронизации файла matches_portrait_to_group.json."""
        logger.info("Синхронизация matches_portrait_to_group.json...")
        if not self.matches_data or not self.matches_json_path:
            logger.warning("Файл matches.json пуст или не загружен. Синхронизация пропущена.")
            return

        selected_numbers, found_numbers_map, regex = self._prepare_sync_data()

        for cluster_id, data in self.matches_data.items():
            child_name = data.get("child_name")
            if not child_name: continue

            current_photos = data.get("group_photos", [])
            new_group_photos = self._sync_person_photos(
                child_name, current_photos, selected_numbers, found_numbers_map, regex
            )
            self.matches_data[cluster_id]["group_photos"] = new_group_photos

        _write_json_safely(self.matches_json_path, self.matches_data)

    def _prepare_sync_data(self) -> Tuple[Set[str], Dict[str, str], re.Pattern]:
        """
        Подготавливает данные для синхронизации и создает динамический RegExp.
        
        Returns:
            Tuple[Set[str], Dict[str, str], re.Pattern]: Выбранные номера, карта найденных, скомпилированный regex.
        """
        selected_numbers = set(self.numbers_line_edit.text().split())
        found_numbers_map = {
            item['number']: item['child_name']
            for item in self.worker.search_results
            if item['status'] == 'Найден' and item['child_name']
        }
        min_d, max_d = self.config.min_digits, self.config.max_digits
        if min_d > max_d: min_d, max_d = max_d, min_d
        regex_pattern = r"(\d{%d,%d})" % (min_d, max_d)
        return selected_numbers, found_numbers_map, re.compile(regex_pattern)

    def _sync_person_photos(self, child_name: str, current_photos: List[Dict], selected_numbers: Set[str], found_numbers_map: Dict[str, str], regex: re.Pattern) -> List[Dict]:
        """
        Выполняет очистку и дополнение списка фотографий для одного человека.
        
        Args:
            child_name (str): Имя человека.
            current_photos (List[Dict]): Текущий список фото из JSON.
            selected_numbers (Set[str]): Набор номеров, выбранных пользователем.
            found_numbers_map (Dict[str, str]): Маппинг "Номер -> Имя" из текущего поиска.
            regex (re.Pattern): Регулярное выражение для извлечения номера из имени файла.

        Returns:
            List[Dict]: Обновленный список фотографий.
        """
        new_group_photos = []
        existing_photo_numbers = set()

        # 1. Очистка: оставляем только те, что есть в selected_numbers
        for photo_info in current_photos:
            match = regex.search(photo_info.get("filename", ""))
            if match and match.group(1) in selected_numbers:
                new_group_photos.append(photo_info)
                existing_photo_numbers.add(match.group(1))

        # 2. Дополнение: добавляем найденные в текущем сеансе
        for number, name in found_numbers_map.items():
            if name == child_name and number not in existing_photo_numbers:
                new_photo_info = {"filename": f"IMG_{number}.jpg", "min_distance": 0.0, "num_faces": 1}
                new_group_photos.append(new_photo_info)

        new_group_photos.sort(key=lambda p: p.get("filename", ""))
        return new_group_photos


# 7. БЛОК: Точка входа
# ==============================================================================
def main():
    config = get_config()
    app = QApplication.instance() or QApplication(sys.argv)
    if IS_MANAGED_RUN and theme_api:
        theme_api.apply_theme_to_app(app)

    window = SorterWindow(config)
    window.show()

    exit_code = app.exec()
    sys.exit(window.final_status if exit_code == 0 else exit_code)

if __name__ == "__main__":
    main()