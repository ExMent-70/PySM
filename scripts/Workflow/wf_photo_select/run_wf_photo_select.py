# run_wf_photo_select.py

"""
Интерактивный скрипт для выборки и организации фотографий.

Предназначен для запуска в среде PyScriptManager. Скрипт предоставляет
графический интерфейс (GUI) на PySide6, который позволяет пользователю
вставить текст, содержащий 4-значные номера фотографий.

Основной функционал:
1.  Извлекает уникальные 4-значные номера из предоставленного текста.
2.  Рекурсивно ищет в исходной папке фотосессии все файлы (RAW, XMP и т.д.),
    соответствующие этим номерам.
3.  Используя внешний JSON-файл ('info_portrait_faces.json'), сопоставляет
    номера фотографий с именами людей.
4.  Позволяет пользователю выбрать имя "сессии копирования" из списка,
    предоставленного переменной контекста 'sys_location_name_{wf_photo_session}'.
5.  Копирует или перемещает найденные файлы в целевую папку, создавая
    подпапку с именем сессии и переименовывая файлы по шаблону
    '<Имя Человека>-<Номер>.<расширение>'.
6.  Ведет постоянный журнал всех операций в файле 'operations_log.json'.

Все длительные операции (поиск по диску, файловые операции) выполняются
в фоновом потоке, чтобы интерфейс оставался отзывчивым.
"""

# 1. БЛОК: Импорты и константы
# ==============================================================================
import argparse
import json
import logging
import os
import pathlib
import re
import shutil
import sys
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Попытка импорта библиотек PySM
try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    from pysm_lib.pysm_theme_api import format_ipymarkup_box, set_widget_class
    
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context, theme_api, format_ipymarkup_box = (
        None,
        None,
        None,
        None,
    )
    try:
        from tqdm import tqdm
    except ImportError:

        class TqdmMock:
            def __init__(self, i=None, **kwargs):
                self.iterable = i or []

            def __iter__(self):
                return iter(self.iterable)

            @staticmethod
            def write(s, **kwargs):
                print(s)

        tqdm = TqdmMock

# Импорт ijson для эффективной работы с JSON
try:
    import ijson
except ImportError:
    print("Ошибка: ijson не найден. Установите: pip install ijson", file=sys.stderr)
    sys.exit(1)


# Опциональные импорты для ipymarkup
try:
    from ipymarkup.palette import Palette
except ImportError:
    Palette = None
    logger.warning(
        "Библиотека 'ipymarkup' не найдена. Визуализация текста будет недоступна."
    )

# Импорты для GUI на PySide6
try:
    from PySide6.QtCore import (
        QAbstractItemModel,
        QModelIndex,
        QSize,
        Qt,
        Signal,
        QThread,
    )
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplitter,
        QStatusBar,
        QStyle,
        QTabWidget,
        QTextBrowser,
        QTextEdit,
        QTreeView,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("Ошибка: PySide6 не найден. Установите: pip install pyside6", file=sys.stderr)
    sys.exit(1)

# Константы
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2"}
CAPTURE_FOLDER = "Capture"
SELECTS_FOLDER = "Selects"


# 2. БЛОК: Вспомогательные функции
# ==============================================================================
def construct_session_paths() -> Dict[str, Optional[pathlib.Path]]:
    """Формирует пути к исходной и целевой папкам на основе переменных контекста."""
    if not IS_MANAGED_RUN or not pysm_context:
        return {"source": None, "dest": None}

    photo_session = pysm_context.get("wf_photo_session", "")
    session_name = pysm_context.get("wf_session_name", "")
    session_path_str = pysm_context.get("wf_session_path", "")

    if not all([session_name, session_path_str, photo_session]):
        logger.error(
            "Одна или несколько переменных контекста (wf_session_path, "
            "wf_session_name, wf_photo_session) не найдены."
        )
        return {"source": None, "dest": None}

    base_path = pathlib.Path(session_path_str) / session_name
    source_dir = base_path / CAPTURE_FOLDER / photo_session
    dest_dir = base_path / SELECTS_FOLDER / photo_session
    return {"source": source_dir, "dest": dest_dir}


def construct_context_paths() -> Dict[str, Optional[pathlib.Path]]:
    """Формирует пути к файлам данных на основе переменных контекста PySM."""
    if not IS_MANAGED_RUN or not pysm_context:
        logger.error(
            "Скрипт запущен без окружения PySM, автоматическое формирование "
            "путей к файлам данных невозможно."
        )
        return {"info_faces_path": None, "operations_log_path": None}

    photo_session = pysm_context.get("wf_photo_session")
    session_name = pysm_context.get("wf_session_name")
    session_path_str = pysm_context.get("wf_session_path")
    psd_path_str = pysm_context.get("wf_psd_path")

    if not all([session_path_str, session_name, photo_session, psd_path_str]):
        logger.error(
            "Критическая ошибка: Одна или несколько переменных контекста (wf_... ) не найдены."
        )
        return {"info_faces_path": None, "operations_log_path": None}

    base_path = pathlib.Path(session_path_str) / session_name
    base_psd_path = pathlib.Path(psd_path_str) / session_name

    info_faces_path = (
        base_path / "Output" / f"Analysis_{photo_session}" / "info_portrait_faces.json"
    )
    operations_log_path = base_psd_path / "operations_log.json"

    return {
        "info_faces_path": info_faces_path,
        "operations_log_path": operations_log_path,
    }


def load_faces_data(json_path: pathlib.Path) -> Dict[str, str]:
    """Эффективно загружает данные "4-значный номер -> имя человека" из JSON."""
    if not json_path or not json_path.exists():
        name = json_path.name if json_path else "info_portrait_faces.json"
        logger.warning(f"Файл {name} не найден. Переименование будет невозможно.")
        return {}

    logger.info(f"Загрузка данных из {json_path.name}...")
    faces_map = {}
    try:
        with open(json_path, "rb") as f:
            parser = ijson.kvitems(f, "")
            for filename_key, data in parser:
                match = re.search(r"(\d{4})", filename_key)
                if not match:
                    continue
                number = match.group(1)
                try:
                    child_name = data.get("faces", [{}])[0].get("child_name")
                    if child_name and number not in faces_map:
                        faces_map[number] = child_name
                except (IndexError, KeyError, TypeError):
                    continue
        logger.info(f"Загружено {len(faces_map)} уникальных записей 'номер -> имя'.")
        return faces_map
    except Exception as e:
        logger.error(f"Ошибка при чтении {json_path.name}: {e}")
        return {}


def read_operations_log(log_path: pathlib.Path) -> Dict:
    """Безопасно читает и парсит файл журнала операций."""
    if not log_path or not log_path.exists():
        return {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Ошибка чтения файла журнала {log_path.name}: {e}")
        return {}


def write_operations_log(log_path: pathlib.Path, data: Dict):
    """Записывает обновленные данные в файл журнала операций."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Журнал операций успешно сохранен в {log_path.name}")
    except Exception as e:
        logger.error(f"Не удалось сохранить журнал операций: {e}")


# 3. БЛОК: Получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """Определяет и парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Выборка файлов по номерам из текста.")
    parser.add_argument(
        "--__fs_mode",
        type=str,
        choices=["copy", "move"],
        default="copy",
        help="Режим операции.",
    )
    parser.add_argument(
        "--__fs_on_conflict",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="Действие при конфликте имен файлов в целевой папке.",
    )
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


# 4. БЛОК: Иерархическая модель данных для TreeView
# ==============================================================================
class TreeItem:
    """Простой элемент древовидной структуры для модели Qt."""

    def __init__(self, data: List, parent: "TreeItem" = None):
        self._data = data
        self._parent = parent
        self._children: List["TreeItem"] = []

    def child(self, row: int) -> Optional["TreeItem"]:
        return self._children[row] if 0 <= row < len(self._children) else None

    def childCount(self) -> int:
        return len(self._children)

    def columnCount(self) -> int:
        return len(self._data)

    def data(self, column: int) -> Any:
        return self._data[column] if 0 <= column < len(self._data) else None

    def parent(self) -> Optional["TreeItem"]:
        return self._parent

    def row(self) -> int:
        return self._parent._children.index(self) if self._parent else 0

    def appendChild(self, item: "TreeItem"):
        self._children.append(item)


class FileTreeModel(QAbstractItemModel):
    """Модель Qt для отображения иерархических данных о файлах."""

    def __init__(
        self,
        data: List[Dict[str, Any]] = None,
        icons: Dict = None,
        color_found_bg: QColor = None,
        color_found_fg: QColor = None,
        color_not_found_bg: QColor = None,
        color_not_found_fg: QColor = None,
        parent=None,
    ):
        super().__init__(parent)
        self.headers = [
            "Номер",
            "Статус",
            "Найденное имя",
            "Новое имя (основа)",
            "Файл",
        ]
        self._root_item = TreeItem(self.headers)
        self.icons = icons or {}
        self.color_found_bg = color_found_bg or QColor("#d4edda")
        self.color_found_fg = color_found_fg or QColor("#155724")
        self.color_not_found_bg = color_not_found_bg or QColor("#f8d7da")
        self.color_not_found_fg = color_not_found_fg or QColor("#721c24")
        self.setup_model_data(data or [])

    def setup_model_data(self, model_data: List[Dict[str, Any]]):
        """Перестраивает модель на основе новых данных."""
        self.beginResetModel()
        self._root_item = TreeItem(self.headers)
        sorted_data = sorted(
            model_data, key=lambda x: (x.get("status", "") != "Найден", x["number"])
        )

        for item_data in sorted_data:
            number = item_data["number"]
            status = item_data.get("status", "Ожидает")
            person_name = item_data.get("person_name", "")
            new_stem = item_data.get("new_stem", "")
            files = item_data.get("files", [])
            base_path = item_data.get("base_path")
            row_data = [number, status, person_name, new_stem, ""]
            if files:
                main_file = files[0]
                row_data[4] = str(main_file.relative_to(base_path))
            parent_item = TreeItem(row_data, self._root_item)
            if files:
                for other_file in files[1:]:
                    child_path = str(other_file.relative_to(base_path))
                    parent_item.appendChild(TreeItem(["", "", "", "", child_path], parent_item))
            self._root_item.appendChild(parent_item)
        self.endResetModel()

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        """Возвращает данные для отображения элемента."""
        if not index.isValid():
            return None
        item = index.internalPointer()
        col = index.column()
        if role == Qt.DecorationRole and col == 4 and item.data(4):
            return (
                self.icons.get("folder")
                if item.parent() == self._root_item
                else self.icons.get("file")
            )
        if role == Qt.DisplayRole:
            return item.data(col)

        if col == 1:
            status = item.data(1)
            is_found = status == "Найден"
            is_not_found = status == "Не найден"

            if role == Qt.BackgroundRole:
                if is_found:
                    return self.color_found_bg
                if is_not_found:
                    return self.color_not_found_bg
            if role == Qt.ForegroundRole:
                if is_found:
                    return self.color_found_fg
                if is_not_found:
                    return self.color_not_found_fg

        return None

    def headerData(self, section: int, o: Qt.Orientation, role: int):
        """Возвращает данные для заголовков таблицы."""
        if o == Qt.Horizontal and role == Qt.DisplayRole:
            return self._root_item.data(section)
        return None

    def index(self, row: int, col: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        """Возвращает индекс элемента модели."""
        if not self.hasIndex(row, col, parent):
            return QModelIndex()
        parent_item = parent.internalPointer() if parent.isValid() else self._root_item
        child_item = parent_item.child(row)
        return self.createIndex(row, col, child_item) if child_item else QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Возвращает родительский индекс."""
        if not index.isValid():
            return QModelIndex()
        child_item = index.internalPointer()
        parent_item = child_item.parent()
        if parent_item == self._root_item:
            return QModelIndex()
        return self.createIndex(parent_item.row(), 0, parent_item)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Возвращает количество строк."""
        return (
            parent.internalPointer() if parent.isValid() else self._root_item
        ).childCount()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Возвращает количество колонок."""
        return self._root_item.columnCount()

    def sort(self, column: int, order: Qt.SortOrder):
        """Сортирует данные в модели по указанной колонке."""
        self.layoutAboutToBeChanged.emit()
        reverse_order = order == Qt.DescendingOrder

        def sort_key(item):
            data = item.data(column)
            if data is None:
                return ""
            if column == 0:
                try:
                    return int(data)
                except (ValueError, TypeError):
                    return 0
            return str(data)

        self._root_item._children.sort(key=sort_key, reverse=reverse_order)
        self.layoutChanged.emit()


# 5. БЛОК: Рабочий поток (Worker)
# ==============================================================================
class Worker(QThread):
    """Выполняет все длительные операции в фоновом потоке, чтобы GUI не зависал."""

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
        self.info_faces_path: Optional[pathlib.Path] = None
        self.operation_mode: str = "copy"
        self.on_conflict_mode: str = "skip"
        self.session_copy_name: str = ""
        self.search_results: List[Dict[str, Any]] = []
        self.faces_data: Dict[str, str] = {}
        self.log_data: Dict[str, List[str]] = {}
        self.markup_color: Optional["Color"] = None
        self.palette: Optional[Palette] = None
        self._init_palette()

    def _init_palette(self):
        """Инициализирует палитру для подсветки текста."""
        if Palette and self.markup_color:
            try:
                self.palette = Palette([self.markup_color])
            except Exception as e:
                logger.error(f"Ошибка при создании палитры ipymarkup: {e}")

    def run(self):
        """Основной метод, запускающий выполнение задачи в потоке."""
        try:
            if self.task == "parse":
                self._task_parse_text()
            elif self.task == "search":
                self._task_search_files()
            elif self.task == "operation":
                self._task_execute_operation()
        except Exception as e:
            logger.error(f"Критическая ошибка в потоке: {e}", exc_info=True)
            self.error_occurred.emit(f"Критическая ошибка в потоке: {e}")

    def _task_parse_text(self):
        """Извлекает 4-значные номера из текста и генерирует HTML с подсветкой."""
        self._init_palette()
        logger.info("Извлечение 4-х значных номеров фотографий из текста...")
        matches = list(re.finditer(r"(\d{4})", self.text_to_parse))
        numbers: Set[str] = {match.group(1) for match in matches}
        sorted_numbers = sorted(list(numbers))
        logger.info(f"Найдено <b>{len(sorted_numbers)}</b> уникальных номеров:")
        logger.info(f"<i>{sorted_numbers}</i>\n")
        markup_html = ""
        if format_ipymarkup_box and self.palette:
            try:
                spans = [(match.start(), match.end(), "") for match in matches]
                markup_html = format_ipymarkup_box(
                    self.text_to_parse, spans, palette=self.palette
                )
            except Exception as e:
                logger.error(f"Ошибка генерации разметки ipymarkup: {e}")
                markup_html = (
                    f"<p style='color: red;'>Ошибка при генерации разметки: {e}</p>"
                )
        else:
            markup_html = "<p style='color: orange;'>Библиотека 'ipymarkup' не найдена, визуализация недоступна.</p>"
        self.parse_finished.emit(sorted_numbers, markup_html)

    def _task_search_files(self):
        """Ищет файлы по номерам и сопоставляет их с именами из JSON."""
        source_path = self.source_dir
        logger.info(
            f"Поиск файлов для <b>{len(self.numbers_to_find)}</b> номеров в: <b>{source_path}</b>"
        )
        self.faces_data = load_faces_data(self.info_faces_path)
        all_files_in_dir = []
        for p in source_path.rglob("*"):
            if self.dest_dir and self.dest_dir in p.parents:
                continue
            if p.is_file():
                all_files_in_dir.append(p)
        logger.info(
            f"Всего файлов для анализа (исключая {SELECTS_FOLDER}): {len(all_files_in_dir)}"
        )
        files_by_basename = defaultdict(list)
        for file_path in all_files_in_dir:
            files_by_basename[file_path.stem].append(file_path)
        final_data = []
        for number in self.numbers_to_find:
            found_files = [
                f for stem, files in files_by_basename.items() if number in stem for f in files
            ]
            if found_files:
                sorted_files = sorted(
                    list(set(found_files)),
                    key=lambda p: (p.suffix.lower() not in RAW_EXTENSIONS, p.name),
                )
                person_name = self.faces_data.get(number)
                if person_name:
                    new_stem = f"{person_name}-{number}"
                else:
                    new_stem = sorted_files[0].stem
                final_data.append(
                    {
                        "number": number,
                        "status": "Найден",
                        "files": sorted_files,
                        "base_path": source_path,
                        "person_name": person_name,
                        "new_stem": new_stem,
                    }
                )
            else:
                final_data.append({"number": number, "status": "Не найден"})
        logger.info("Поиск файлов завершен.\n")
        self.search_results = final_data
        self.search_finished.emit(final_data)

    def _task_execute_operation(self):
        """Выполняет копирование/перемещение и переименование файлов."""
        source_path = self.source_dir
        dest_root = self.dest_dir
        op_name = "Копирование" if self.operation_mode == "copy" else "Перемещение"
        logger.info(
            f"{op_name} файлов для сессии '{self.session_copy_name}' в папку {dest_root}"
        )
        files_to_process = [
            file
            for item in self.search_results
            if item.get("status") == "Найден"
            for file in item.get("files", [])
        ]
        if not files_to_process:
            self.operation_finished.emit(self.operation_mode, 0, 0)
            return

        file_to_item_map = {
            file: item
            for item in self.search_results
            if item.get("status") == "Найден"
            for file in item.get("files", [])
        }
        op_func = shutil.copy2 if self.operation_mode == "copy" else shutil.move
        new_log_entries = defaultdict(list)

        with tqdm(files_to_process, desc=op_name, unit="файл") as progress_bar:
            for src_file in progress_bar:
                item = file_to_item_map.get(src_file)
                if not item:
                    tqdm.write(
                        f"[ПРЕДУПРЕЖДЕНИЕ] Не найдены метаданные для файла {src_file.name}"
                    )
                    continue

                new_stem = item.get("new_stem")
                person_name = item.get("person_name", "Без имени")
                new_name = new_stem + src_file.suffix
                relative_path = src_file.relative_to(source_path)
                dest_file = dest_root / relative_path.parent / new_name

                if dest_file.exists():
                    if self.on_conflict_mode == "skip":
                        tqdm.write(
                            f"[ПРОПУСК] Файл {dest_file.name} уже существует в целевой папке."
                        )
                        continue

                try:
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    op_func(src_file, dest_file)
                    if src_file.suffix.lower() in RAW_EXTENSIONS:
                        new_log_entries[person_name].append(new_name)
                except Exception as e:
                    tqdm.write(f"[ОШИБКА] Не удалось обработать {src_file.name}: {e}")

        self.log_data = new_log_entries
        self.operation_finished.emit(
            self.operation_mode, len(files_to_process), len(files_to_process)
        )


# 6. БЛОК: Главное окно приложения
# ==============================================================================
class FileSelectorWindow(QMainWindow):
    """Главное окно приложения с графическим интерфейсом."""

    def __init__(self, config: Namespace):
            """Инициализирует окно, пути, виджеты и обработчики."""
            super().__init__()
            self.config = config
            self.final_status = 1

            session_paths = construct_session_paths()
            self.source_dir = session_paths.get("source")
            self.base_dest_dir = session_paths.get("dest")
            self.final_dest_dir: Optional[pathlib.Path] = None

            context_paths = construct_context_paths()
            self.info_faces_path = context_paths.get("info_faces_path")
            self.log_file_path = context_paths.get("operations_log_path")

            self.operations_log = read_operations_log(self.log_file_path)
            self.session_presets = []
            if IS_MANAGED_RUN and pysm_context:
                # --- ИСПРАВЛЕНИЕ: Имя переменной приведено к единообразию ---
                photo_session = pysm_context.get("wf_photo_session","")
                current_location_name = "sys_location_name"
                if photo_session != "":                       
                    current_location_name = "sys_location_name"+"_"+photo_session
                presets = pysm_context.get(current_location_name, [])
                
                if isinstance(presets, dict):
                    # Если это новый формат (словарь), извлекаем ключи
                    self.session_presets = list(presets.keys())
                    logger.info(f"Загружено {len(self.session_presets)} пресетов локаций из словаря.")
                elif isinstance(presets, list):
                    # Оставляем обработку старого формата (списка) для обратной совместимости
                    self.session_presets = presets
                    logger.info(f"Загружено {len(self.session_presets)} пресетов локаций из списка.")
                else:
                    logger.warning("Переменная контекста 'sys_location_name' имеет неожиданный тип.")

            self._load_theme_colors()
            self._load_icons()
            self._init_ui()
            self._init_worker()

            if IS_MANAGED_RUN and (not self.source_dir or not self.base_dest_dir):
                QMessageBox.critical(
                    self, "Критическая ошибка", "Не удалось сформировать пути для сессии..."
                )
                self._set_ui_busy(True, "Ошибка: пути сессии не сформированы.")

    def _load_theme_colors(self):
        """Загружает цвета из API тем с резервными значениями."""
        if IS_MANAGED_RUN and theme_api:
            self.ipymarkup_color = theme_api.get_ipymarkup_color(
                "markup_number",
                defaults={
                    "background-color": "#e3f2fd",
                    "border-color": "#bbdefb",
                    "color": "#000000",
                },
            )
            self.color_found_bg = theme_api.get_qcolor(
                "table_status_found", "background-color", "#d4edda"
            )
            self.color_found_fg = theme_api.get_qcolor(
                "table_status_found", "color", "#155724"
            )
            self.color_not_found_bg = theme_api.get_qcolor(
                "table_status_not_found", "background-color", "#f8d7da"
            )
            self.color_not_found_fg = theme_api.get_qcolor(
                "table_status_not_found", "color", "#721c24"
            )
        else:
            # Резервные значения для автономного запуска
            self.ipymarkup_color = None
            self.color_found_bg = QColor("#d4edda")
            self.color_found_fg = QColor("#155724")
            self.color_not_found_bg = QColor("#f8d7da")
            self.color_not_found_fg = QColor("#721c24")

    def _load_icons(self):
        """Загружает стандартные иконки для использования в GUI."""
        style = self.style()
        self.icons = {
            "folder": style.standardIcon(QStyle.SP_DirIcon),
            "file": style.standardIcon(QStyle.SP_FileIcon),
        }

    def _init_worker(self):
        """Инициализирует фоновый поток и связывает его сигналы со слотами."""
        self.worker = Worker()
        self.worker.markup_color = self.ipymarkup_color
        self.worker.parse_finished.connect(self._on_parse_finished)
        self.worker.search_finished.connect(self._on_search_finished)
        self.worker.operation_finished.connect(self._on_operation_finished)
        self.worker.error_occurred.connect(self._on_worker_error)

    def _init_ui(self):
        """Создает и настраивает все виджеты графического интерфейса."""
        win_title = "Выборка и переименование файлов по номерам"
        if IS_MANAGED_RUN and self.source_dir:
            win_title += f" - [{self.source_dir.parent.name}/{self.source_dir.name}]"
        self.setWindowTitle(win_title)
        self.resize(1200, 750)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
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
            "QTabBar::tab:!selected { background-color: #ffc300; color: #000000; padding: 6px 12px; } QTabBar::tab:selected { font: bold; }"
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
        h_header.setSectionResizeMode(4, QHeaderView.Stretch)
        right_layout.addWidget(self.tree_view)

        right_layout.addWidget(QLabel("Найденные номера для копирования:"))
        self.numbers_line_edit = QLineEdit()
        self.numbers_line_edit.setReadOnly(True)
        right_layout.addWidget(self.numbers_line_edit)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 10, 0, 5)
        self.parse_button = QPushButton("1. Извлечь номера")
        self.search_button = QPushButton("2. Найти файлы")
        self.action_button = QPushButton("3. Выполнить")
        
        if IS_MANAGED_RUN: # Применяем класс только в управляемом режиме
            set_widget_class(self.action_button, "primary")
        
        self.parse_button.clicked.connect(self._start_parse)
        self.search_button.clicked.connect(self._start_search)
        self.action_button.clicked.connect(self._start_operation)
        self.search_button.setEnabled(False)
        self.action_button.setEnabled(False)

        self.session_combo = QComboBox()
        self.session_combo.setEditable(False)
        self.session_combo.setMinimumWidth(250)
        if self.session_presets:
            self.session_combo.addItems(self.session_presets)
        else:
            self.session_combo.addItem("Нет доступных сессий")
            self.session_combo.setEnabled(False)

        bottom_layout.addWidget(self.parse_button)
        bottom_layout.addWidget(self.search_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(QLabel("Имя сессии копирования:"))
        bottom_layout.addWidget(self.session_combo)
        bottom_layout.addWidget(self.action_button)

        main_layout.addLayout(bottom_layout)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Готово к работе.")

    def _set_ui_busy(self, is_busy: bool, message: str = ""):
        """Блокирует/разблокирует интерфейс на время выполнения операции."""
        self.parse_button.setEnabled(not is_busy)
        self.search_button.setEnabled(False)
        self.action_button.setEnabled(False)
        self.statusBar().showMessage(message)

    def _start_parse(self):
        """Запускает задачу парсинга текста в фоновом потоке."""
        if not self.source_dir or not self.source_dir.is_dir():
            QMessageBox.critical(self, "Ошибка", f"Исходная папка не найдена:\n{self.source_dir}")
            return
        text = self.text_input.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Внимание", "Поле для текста пустое.")
            return

        self.numbers_line_edit.clear()

        self._set_ui_busy(True, "Извлечение номеров...")
        self.worker.task = "parse"
        self.worker.text_to_parse = text
        self.worker.start()

    def _on_parse_finished(self, numbers: List[str], markup_html: str):
        """Слот: обрабатывает результаты парсинга текста."""
        self._set_ui_busy(False)
        self.tree_model = FileTreeModel(
            data=[{"number": num} for num in numbers],
            icons=self.icons,
            color_found_bg=self.color_found_bg,
            color_found_fg=self.color_found_fg,
            color_not_found_bg=self.color_not_found_bg,
            color_not_found_fg=self.color_not_found_fg,
        )
        self.tree_view.setModel(self.tree_model)
        self.markup_browser.setHtml(markup_html)

        if numbers:
            self.numbers_line_edit.setText(" ".join(numbers))
            self.left_tabs.setCurrentIndex(1)
            self.search_button.setEnabled(True)
            self.statusBar().showMessage(
                f"Найдено {len(numbers)} номеров. Готово к поиску файлов."
            )
        else:
            self.statusBar().showMessage("В тексте не найдено 4-значных номеров.")

    def _start_search(self):
        """Запускает задачу поиска файлов в фоновом потоке."""
        if not hasattr(self, "tree_model"):
            return
        numbers_in_model = [
            self.tree_model.index(i, 0).data() for i in range(self.tree_model.rowCount())
        ]
        if not numbers_in_model:
            return
        self._set_ui_busy(True, "Поиск файлов и имен на диске...")
        self.worker.task = "search"
        self.worker.source_dir = self.source_dir
        self.worker.numbers_to_find = numbers_in_model
        self.worker.info_faces_path = self.info_faces_path
        self.worker.start()

    def _on_search_finished(self, results: List[Dict[str, Any]]):
        """Слот: обрабатывает результаты поиска файлов."""
        self._set_ui_busy(False)
        self.search_button.setEnabled(True)
        self.tree_model = FileTreeModel(
            results,
            icons=self.icons,
            color_found_bg=self.color_found_bg,
            color_found_fg=self.color_found_fg,
            color_not_found_bg=self.color_not_found_bg,
            color_not_found_fg=self.color_not_found_fg,
        )
        self.tree_view.setModel(self.tree_model)
        for i in range(self.tree_model.columnCount()):
            self.tree_view.resizeColumnToContents(i)
        all_files = [f for item in results if item.get("files") for f in item["files"]]
        if all_files:
            op_mode = getattr(self.config, "__fs_mode", "copy")
            op_name = (
                "Копировать и переименовать"
                if op_mode == "copy"
                else "Переместить и переименовать"
            )
            self.action_button.setText(f"3. {op_name} ({len(all_files)} файлов)")
            self.action_button.setEnabled(True)
            self.statusBar().showMessage(
                f"Поиск завершен. Найдено {len(all_files)} файлов."
            )
        else:
            self.statusBar().showMessage("Поиск завершен. Файлы не найдены.")

    def _start_operation(self):
        """Запускает файловую операцию в фоновом потоке."""
        if not self.worker.search_results:
            QMessageBox.warning(self, "Внимание", "Нет файлов для обработки.")
            return
        session_name = self.session_combo.currentText().strip()
        if not session_name or not self.session_presets:
            QMessageBox.warning(
                self, "Внимание", "Необходимо выбрать имя сессии копирования из списка."
            )
            return

        self.final_dest_dir = self.base_dest_dir / session_name

        op_mode = getattr(self.config, "__fs_mode", "copy")
        op_name_verb = (
            "скопировать и переименовать"
            if op_mode == "copy"
            else "ПЕРЕМЕСТИТЬ и переименовать"
        )
        files_count = sum(
            len(item.get("files", []))
            for item in self.worker.search_results
            if item.get("status") == "Найден"
        )
        msg = (
            f"Вы уверены, что хотите {op_name_verb} {files_count} файлов "
            f"в папку:\n\n{self.final_dest_dir}?"
        )
        if (
            QMessageBox.question(
                self, "Подтверждение", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            == QMessageBox.No
        ):
            return

        self._set_ui_busy(True, "Выполнение операции...")
        self.worker.task = "operation"
        self.worker.operation_mode = op_mode
        self.worker.session_copy_name = session_name
        self.worker.on_conflict_mode = getattr(self.config, "__fs_on_conflict", "skip")
        self.worker.dest_dir = self.final_dest_dir
        self.worker.start()

    def _update_log_file(self):
        """Обновляет файл журнала операций на основе результатов работы Worker."""
        log_data = self.worker.log_data
        session_name = self.worker.session_copy_name
        if not log_data or not session_name:
            return

        logger.info("Обновление журнала операций...")
        for person_name, new_files in log_data.items():
            if person_name not in self.operations_log:
                self.operations_log[person_name] = {}
            if session_name not in self.operations_log[person_name]:
                self.operations_log[person_name][session_name] = {
                    "timestamp": "",
                    "files": [],
                }

            existing_files = set(
                self.operations_log[person_name][session_name]["files"]
            )
            for f in new_files:
                if f not in existing_files:
                    self.operations_log[person_name][session_name]["files"].append(f)

            self.operations_log[person_name][session_name][
                "timestamp"
            ] = datetime.utcnow().isoformat()
            self.operations_log[person_name][session_name]["files"].sort()

        write_operations_log(self.log_file_path, self.operations_log)

    def _on_operation_finished(self, mode: str, processed_count: int, total_files: int):
        """Слот: обрабатывает завершение файловой операции."""
        self._set_ui_busy(False)
        self.search_button.setEnabled(True)
        self.action_button.setEnabled(True)

        if processed_count == total_files:
            self.final_status = 0
            self._update_log_file()
            QMessageBox.information(
                self, "Успех", f"Операция успешно завершена.\nОбработано файлов: {processed_count}"
            )
        else:
            QMessageBox.warning(
                self,
                "Завершено с ошибками",
                f"Операция завершена.\nОбработано {processed_count} из {total_files} "
                "файлов. Журнал не обновлен.",
            )
        self.statusBar().showMessage("Готово к работе.")

    def _on_worker_error(self, message: str):
        """Слот: обрабатывает ошибку, возникшую в фоновом потоке."""
        self._set_ui_busy(False)
        self.search_button.setEnabled(True)
        QMessageBox.critical(self, "Ошибка в фоновом потоке", message)
        self.statusBar().showMessage("Произошла ошибка.")

    def add_final_links(self):
        """Добавляет ссылки на папки в лог PySM при успешном завершении."""
        if IS_MANAGED_RUN and self.final_status == 0 and pysm_context:
            try:
                if self.final_dest_dir and self.final_dest_dir.exists():
                    pysm_context.log_link(
                        url_or_path=str(self.final_dest_dir),
                        text=f"<br>Открыть папку сессии ('{self.final_dest_dir.name}')",
                    )
                if self.source_dir:
                    pysm_context.log_link(
                        url_or_path=str(self.source_dir), text="Открыть исходную папку"
                    )
                if self.log_file_path and self.log_file_path.exists():
                    pysm_context.log_link(
                        url_or_path=str(self.log_file_path),
                        text="Открыть файл журнала операций",
                    )
            except Exception as e:
                logger.error(f"Не удалось сгенерировать ссылки: {e}")

    def closeEvent(self, event):
        """Перехватывает событие закрытия окна для добавления ссылок."""
        self.add_final_links()
        event.accept()


# 7. БЛОК: Точка входа
# ==============================================================================
def main():
    """Основная функция, запускающая приложение."""
    config = get_config()
    app = QApplication.instance() or QApplication(sys.argv)
    if IS_MANAGED_RUN and theme_api:
        theme_api.apply_theme_to_app(app)
    window = FileSelectorWindow(config)
    window.show()
    exit_code = app.exec()
    sys.exit(window.final_status if exit_code == 0 else exit_code)


if __name__ == "__main__":
    main()