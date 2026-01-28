#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cluster_editor.py
=====================
Модуль для редактирования кластеров изображений с графическим интерфейсом на основе PySide6.

Основные возможности:
- Режимы: Лица (face), Локации (location), Сопоставление (matches).
- Визуализация: Галерея, Просмотрщик, Панель деталей лиц.
- Редактирование: Drag & Drop, Переименование, Объединение.
- Сопоставление: Ручная привязка групповых фото к портретам.
- Экспорт: Настройка качества, DPI, размеров, водяных знаков.
"""

# --- 1. ИМПОРТЫ И НАСТРОЙКА ОКРУЖЕНИЯ ---
# ==============================================================================
import sys
import os
import logging
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# PySide6
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QInputDialog, QProgressBar, QMessageBox, QLineEdit, QMenu,
    QListWidget, QListWidgetItem, QDialog, QCheckBox, QFileDialog,
    QSplitter, QSlider
)
from PySide6.QtGui import QPixmap, QAction, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer, QSize

# Pillow (Опционально, но критично для нарезки лиц и экспорта)
try:
    from PIL import Image, ImageQt, ImageOps
except ImportError:
    Image = None

# Внутренние модули и PySM
IS_MANAGED_RUN = False
try:
    current_script_dir = Path(__file__).resolve().parent
    if str(current_script_dir) not in sys.path: sys.path.insert(0, str(current_script_dir))
    project_root = current_script_dir.parent
    if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_theme_api import set_widget_class
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True

    # Библиотека редактора
    from _lib.editor_viewer import ImageViewer
    from _lib.editor_workers import ChunkedImageLoader, ExportWorker # Используем новые воркеры
    from _lib.editor_delegates import ClusterItemDelegate, ImageItemDelegate, FACE_SIZE, FACE_MIN, FACE_MAX
    from _lib.editor_widgets import ImageDragListWidget, ClusterDropListWidget, FaceDetailsWidget
    from _lib.editor_dialogs import EnhanceSettingsDialog, RenameDialog, FaceSelectorDialog
    from _lib.data_manager import ClusterDataManager
    from _lib.data_models import Face

except ImportError as e:
    print(f"Критическая ошибка импорта внутренних модулей: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)


# --- 2. ГЛАВНЫЙ КЛАСС ОКНА ---
# ==============================================================================
class MainWindow(QWidget):
    
    # --- 2.1 Инициализация и Конфигурация ---
    # ----------------------------------------
    def __init__(self, portrait_json_path: Path, group_json_path: Path, mode: str):
        super().__init__()
        self.mode = mode
        
        # Основные пути
        self.portrait_json_path = portrait_json_path
        self.group_json_path_initial = group_json_path
        
        # Базовая директория (для диалогов открытия файлов)
        self.data_dir = self.portrait_json_path.parent
        
        # Директории с изображениями (раздельные для портретов и групп)
        self.portrait_images_dir: Path = self.data_dir / "JPG"
        
        # Инициализация сессии на основе исходного group_json
        group_analysis_dir = self.group_json_path_initial.parent
        group_output_dir = group_analysis_dir.parent
        group_session_dir = group_output_dir.parent
        
        self.photo_session = group_analysis_dir.name.replace("Analysis_", "")
        self.session_name = group_session_dir.name

        self.group_images_dir: Optional[Path] = group_analysis_dir / "JPG"
        self.current_group_json_path: Optional[Path] = self.group_json_path_initial

        # Конфигурация режимов
        if self.mode == 'face':
            self.mode_config = {
                "mode_name": "face",
                "window_title_template": "Редактор кластеров [по Лицам] - {}",
                "name_prefix_logic": lambda cid: f"{int(cid):02d}-" if str(cid).isdigit() else "",
            }
        elif self.mode == 'location':
            self.mode_config = {
                "mode_name": "location",
                "window_title_template": "Редактор кластеров [по Локациям] - {}",
                "name_prefix_logic": lambda cid: "",
            }
        elif self.mode == 'matches':
             self.mode_config = {
                "mode_name": "matches",
                "window_title_template": "Просмотр совпадений [Портреты -> Группы] - {}",
                "name_prefix_logic": lambda cid: f"{int(cid):02d}-" if str(cid).isdigit() else "",
            }
        else:
            raise ValueError(f"Неизвестный режим работы: {self.mode}")

        self.setWindowTitle(self.mode_config["window_title_template"].format(self.photo_session))

        # Загрузка предопределенных имен
        self.predefined_cluster_names: List[str] = []
        try:
            current_dir = Path(__file__).resolve().parent
            predefined_names_path = current_dir / "predefined_names.json"
            if predefined_names_path.exists():
                with open(predefined_names_path, 'r', encoding='utf-8') as f:
                    self.predefined_cluster_names = json.load(f)
                    logger.info(f"Загружено {len(self.predefined_cluster_names)} предопределенных имен.")
        except Exception as e:
            logger.error(f"Ошибка загрузки predefined_names.json: {e}")

        # Инициализация менеджера данных
        group_json_for_dm = self.group_json_path_initial if self.mode != 'matches' else None
        self.data_manager = ClusterDataManager(self.portrait_json_path, group_json_for_dm)

        # Состояние UI
        self.active_cluster_id: Optional[str] = None
        self.preview_pixmaps: Dict[str, QPixmap] = {} # Кэш превью для левой панели
        self.image_pixmap_cache: Dict[str, QPixmap] = {} # Кэш галереи

        # Потоки загрузки
        self.loader_thread = None
        self.loader_worker = None
        
        self.cluster_delegate = ClusterItemDelegate(parent=self)
        self.image_delegate = ImageItemDelegate(parent=self)

        self.init_ui()
        self._load_and_display_data()

        # Автозагрузка данных для режима matches
        if self.mode == 'matches':
            if self.group_json_path_initial and self.group_json_path_initial.is_file():
                logger.info("Автоматическая загрузка данных о совпадениях по умолчанию...")
                self._load_and_process_group_data(self.group_json_path_initial)
            else:
                logger.warning("Файл info_group_faces.json по умолчанию не найден.")

    def _center_on_screen(self):
        """Центрирует окно на экране."""
        try:
            screen_geometry = self.screen().geometry()
            window_geometry = self.frameGeometry()
            window_geometry.moveCenter(screen_geometry.center())
            self.move(window_geometry.topLeft())
        except Exception:
            pass

    # --- 2.2 Построение Интерфейса (UI) ---
    # --------------------------------------
    def init_ui(self):
        """Создает и компонует виджеты главного окна."""
        self.setGeometry(0, 0, 1450, 900)

        main_layout = QVBoxLayout(self)
        
        # Сплиттер для трех колонок
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        # --- КОЛОНКА 1: Список Кластеров ---
        left_panel_widget = QWidget()
        left_layout = QVBoxLayout(left_panel_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        left_title = QLabel(f"Фотосессия: {self.photo_session} (cписок кластеров)")
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Поиск кластера...")
        self.search_bar.textChanged.connect(self._on_search_text_changed)

        self.cluster_list_widget = ClusterDropListWidget(self)
        self.cluster_list_widget.setObjectName("clusterListWidget")
        self.cluster_list_widget.setItemDelegate(self.cluster_delegate)
        self.cluster_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.cluster_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.cluster_list_widget.setMovement(QListWidget.Movement.Static)
        self.cluster_list_widget.setSpacing(10)
        self.cluster_list_widget.itemDoubleClicked.connect(self._rename_cluster_action)
        self.cluster_list_widget.currentItemChanged.connect(self._on_cluster_selected)
        self.cluster_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cluster_list_widget.customContextMenuRequested.connect(self.show_cluster_context_menu)

        # Drag & Drop настройки
        if self.mode == 'matches':
            self.cluster_list_widget.setAcceptDrops(True)
            self.cluster_list_widget.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        else:
            self.cluster_list_widget.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        
        self.cluster_list_widget.itemsDropped.connect(self._handle_drop)
        self.cluster_list_widget.viewport().setAcceptDrops(True)
        self.cluster_list_widget.setDropIndicatorShown(True)

        # Кнопки
        self.export_button = QPushButton("Экспорт")
        export_menu = QMenu(self)
        export_all = export_menu.addAction("Экспортировать всё")
        export_active = export_menu.addAction("Экспортировать активный кластер")
        self.export_button.setMenu(export_menu)
        export_all.triggered.connect(self._on_export_all_triggered)
        export_active.triggered.connect(self._on_export_active_triggered)

        self.save_button = QPushButton("Сохранить изменения")
        self.save_button.clicked.connect(lambda: self._save_changes(silent=False))

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.export_button)
        buttons_layout.addWidget(self.save_button)

        left_layout.addWidget(left_title)
        left_layout.addWidget(self.search_bar)
        left_layout.addWidget(self.cluster_list_widget, 1)
        left_layout.addLayout(buttons_layout)

        # --- КОЛОНКА 2: Галерея ---
        center_panel_widget = QWidget()
        center_layout = QVBoxLayout(center_panel_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)

        self.right_panel_label = QLabel("Галерея")

        self.image_list_widget = ImageDragListWidget(self)
        self.image_list_widget.setObjectName("imageListWidget")
        self.image_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list_widget.setSpacing(10)
        self.image_list_widget.setItemDelegate(self.image_delegate)

        if self.mode == 'matches':
            self.image_list_widget.setDragEnabled(True)
            self.image_list_widget.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        else:
            self.image_list_widget.setDragDropMode(QListWidget.DragDropMode.DragOnly)

        self.image_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list_widget.itemDoubleClicked.connect(self._open_image_viewer)
        
        # Обновление панели лиц при выборе фото
        self.image_list_widget.currentItemChanged.connect(self._update_face_panel)

        center_layout.addWidget(self.right_panel_label)
        center_layout.addWidget(self.image_list_widget, 1)

        # --- КОЛОНКА 3: Лица на фото ---
        right_panel_widget = QWidget()
        right_layout = QVBoxLayout(right_panel_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        right_title = QLabel("Лица на фото")
        right_layout.addWidget(right_title)

        self.face_details_widget = FaceDetailsWidget(self)
        if IS_MANAGED_RUN:
            set_widget_class(self.face_details_widget, "face-panel")
        
        right_layout.addWidget(self.face_details_widget, 1)

        self.face_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.face_size_slider.setRange(FACE_MIN, FACE_MAX) 
        self.face_size_slider.setValue(FACE_SIZE)      
        self.face_size_slider.setToolTip("Размер миниатюр")
        self.face_size_slider.valueChanged.connect(self._on_face_size_changed)
        
        right_layout.addWidget(self.face_size_slider)

        # --- Сборка ---
        splitter.addWidget(left_panel_widget)
        splitter.addWidget(center_panel_widget)
        splitter.addWidget(right_panel_widget)
        
        splitter.setStretchFactor(0, 32)
        splitter.setStretchFactor(1, 45)
        splitter.setStretchFactor(2, 23)

        self.status_progress_bar = QProgressBar()
        self.status_progress_bar.setTextVisible(True)

        main_layout.addWidget(self.status_progress_bar)
        self._center_on_screen()

    # --- 3. РАБОТА С ДАННЫМИ (Загрузка, Перезагрузка) ---
    # ----------------------------------------------------
    def _load_and_display_data(self):
        """Первичная загрузка данных при старте."""
        success, message = self.data_manager.load_data()
        if not success:
            QMessageBox.critical(self, "Ошибка загрузки данных", message)
            return
        self._refresh_left_panel()

    def _load_and_process_group_data(self, group_json_path: Path):
        """Динамическая загрузка данных групповых фото (для режима matches)."""
        if self.data_manager.reload_group_data(group_json_path):
            self.current_group_json_path = group_json_path
            
            # Обновление путей сессии
            group_analysis_dir = self.current_group_json_path.parent
            group_output_dir = group_analysis_dir.parent
            group_session_dir = group_output_dir.parent
            
            self.photo_session = group_analysis_dir.name.replace("Analysis_", "")
            self.session_name = group_session_dir.name
            self.group_images_dir = group_analysis_dir / "JPG"
            
            self.setWindowTitle(self.mode_config["window_title_template"].format(self.photo_session))
            
            logger.info(f"Успешно загружен файл: {group_json_path.name}")
            self._refresh_left_panel()
        else:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{group_json_path}")

    @Slot()
    def _load_group_data_action(self):
        """Слот для пункта меню 'Загрузить групповые данные...'."""
        start_dir = str(self.data_dir)
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл info_group_faces.json", start_dir, "JSON files (info_group_faces.json)"
        )
        if filepath:
            self._load_and_process_group_data(Path(filepath))

    # --- 4. ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ (Получение путей и данных) ---
    # -----------------------------------------------------------
    def _get_image_path(self, filename: str, image_type: str) -> Path:
        """Возвращает корректный путь к файлу в зависимости от его типа."""
        if image_type == 'portrait':
            return self.portrait_images_dir / filename
        elif image_type == 'group' and self.group_images_dir:
            return self.group_images_dir / filename
        return self.portrait_images_dir / filename

    def _get_clusters_from_model(self) -> Dict[str, List[Face]]:
        if self.mode == 'matches':
            # В matches используем логику формирования имен как в face mode
            face_mode_config = {
                "mode_name": "face",
                "name_prefix_logic": lambda cid: f"{int(cid):02d}-" if str(cid).isdigit() else "",
            }
            return self.data_manager.get_clusters(face_mode_config)
        return self.data_manager.get_clusters(self.mode_config)

    def _get_files_for_cluster_for_viewer(self, cluster_id: str) -> List[str]:
        if self.mode == 'matches':
            return self.data_manager.get_group_matches_for_cluster(cluster_id)
        config_map = { 'face': {'mode_name': 'face'}, 'location': {'mode_name': 'location'} }
        return self.data_manager.get_files_for_cluster(config_map[self.mode], cluster_id)

    def _get_cluster_item_data_by_id(self, cluster_id: str) -> Optional[Dict]:
        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole)["id"] == cluster_id:
                return item.data(Qt.ItemDataRole.UserRole)
        return None
        
    def _get_item_by_cluster_id(self, cluster_id: str) -> Optional[QListWidgetItem]:
        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole)["id"] == cluster_id:
                return item
        return None

    # --- 5. ОБНОВЛЕНИЕ ИНТЕРФЕЙСА (Отрисовка панелей) ---
    # ----------------------------------------------------
    def _refresh_left_panel(self):
        """Перерисовывает список кластеров (левая панель)."""
        active_id_before_refresh = self.active_cluster_id
        self.cluster_list_widget.clear()
        self.preview_pixmaps.clear()
        clusters = self._get_clusters_from_model()

        sort_key_func = lambda x: int(x) if x.isdigit() else (9998 if x == "-1" else 9999)
        if self.mode == 'location':
            sort_key_func = lambda x: x

        sorted_labels = sorted(clusters.keys(), key=sort_key_func)

        # --- НАЧАЛО ИЗМЕНЕНИЯ: Всегда показываем кластер ошибок в режиме matches ---
        if self.mode == 'matches':
            error_files_count = len(self.data_manager.get_files_for_cluster({}, "error_matches"))
            
            # Создаем и добавляем элемент БЕЗУСЛОВНО
            item_data = { 
                "id": "error_matches", 
                "name": "⚠️ Неопознанные", 
                "count": error_files_count,
                "pixmap": QPixmap(), 
                "is_changed": False 
            }
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            
            # Меняем подсказку в зависимости от состояния
            if error_files_count == 0:
                item.setToolTip("Все лица распознаны. Перетащите сюда фото, чтобы отменить сопоставление.")
            else:
                item.setToolTip("Файлы, где автоматика не смогла опознать лица")
            
            self.cluster_list_widget.addItem(item)
            
            if active_id_before_refresh == "error_matches":
                self.cluster_list_widget.setCurrentItem(item)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---



        item_to_select = None
        if active_id_before_refresh == "error_matches" and self.cluster_list_widget.count() > 0:
             item_to_select = self.cluster_list_widget.item(0)

        for label in sorted_labels:
            if self.mode == 'matches' and label in ["-1", "group"]:
                continue

            faces = clusters[label]

            # Формирование имени кластера
            if self.mode == 'matches':
                raw_name = self.data_manager._cluster_id_to_name_cache.get(label)
                if not raw_name and faces: raw_name = faces[0].child_name
                if not raw_name: raw_name = f"Кластер {label}"
                
                clean_name = raw_name
                if clean_name and '-' in clean_name and clean_name.split('-', 1)[0].isdigit():
                    clean_name = clean_name.split('-', 1)[-1]
                
                prefix = self.mode_config["name_prefix_logic"](label)
                cluster_name = prefix + clean_name
            else:
                cluster_name = faces[0].effective_name if faces else f"Кластер {label}"

            # Превью всегда из портретов
            preview_path = Path()
            if faces:
                preview_path = self._get_image_path(faces[0].filename, 'portrait')

            from _lib.editor_delegates import PREVIEW_SIZE
            pixmap = QPixmap(str(preview_path))
            if not pixmap.isNull():
                pixmap = pixmap.scaled(PREVIEW_SIZE, PREVIEW_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
            self.preview_pixmaps[label] = pixmap

            # Подсчет количества фото
            if self.mode == 'matches':
                count = len(self.data_manager.get_group_matches_for_cluster(label))
            else:
                count = len(self.data_manager.get_files_for_cluster(self.mode_config, label))

            item_data = { 
                "id": label, "name": cluster_name, "count": count, "pixmap": pixmap, 
                "is_changed": self.data_manager.is_cluster_changed(self.mode_config["mode_name"], label) 
            }
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            self.cluster_list_widget.addItem(item)

            if label == active_id_before_refresh:
                item_to_select = item

        # Добавление новых пустых кластеров
        for new_cluster in self.data_manager.newly_created_clusters:
            if new_cluster["id"] in clusters: continue
            item_data = {
                "id": new_cluster["id"], "name": new_cluster["name"], "count": 0,
                "pixmap": QPixmap(), "is_changed": True
            }
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            self.cluster_list_widget.addItem(item)
            if new_cluster["id"] == active_id_before_refresh:
                item_to_select = item

        if item_to_select:
            self.cluster_list_widget.setCurrentItem(item_to_select)
        elif self.cluster_list_widget.count() > 0:
            self.cluster_list_widget.setCurrentRow(0)

    def _render_gallery(self, cluster_id: str):
        """Отрисовывает галерею (центральная панель)."""
        self._stop_loader_if_running()

        cluster_data = self._get_cluster_item_data_by_id(cluster_id)
        if not cluster_data:
            self.image_list_widget.clear()
            self.right_panel_label.setText("Кластер не найден")
            return

        label_text = f"Групповые фото: {cluster_data['name']}" if self.mode == 'matches' else f"Кластер: {cluster_data['name']}"
        self.right_panel_label.setText(f"{label_text} ({cluster_data['count']} фото)")

        self.image_list_widget.clear()

        if self.mode == 'matches':
            files_to_show = self.data_manager.get_group_matches_for_cluster(cluster_id)
        else:
            files_to_show = self.data_manager.get_files_for_cluster(self.mode_config, cluster_id)

        if not files_to_show:
            return

        cached_items = []
        uncached_tasks = []
        
        for filename in files_to_show:
            if filename in self.image_pixmap_cache:
                cached_items.append((filename, self.image_pixmap_cache[filename]))
            else:
                uncached_tasks.append({ "filename": filename, "cluster_id": cluster_id })
        
        # Показываем кэшированные
        self._add_images_to_gallery(cached_items)

        # Запускаем загрузку новых
        if uncached_tasks:
            image_type = 'group' if self.mode == 'matches' else 'portrait'
            for task in uncached_tasks:
                task["full_path"] = self._get_image_path(task["filename"], image_type)
            self._start_chunked_loader(uncached_tasks)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_cluster_selected(self, current_item: QListWidgetItem, previous_item: Optional[QListWidgetItem] = None):
        if not current_item:
            self._stop_loader_if_running()
            self.image_list_widget.clear()
            self.right_panel_label.setText("Кластер")
            self.active_cluster_id = None
            return

        cluster_data = current_item.data(Qt.ItemDataRole.UserRole)
        cluster_id = cluster_data["id"]
        if self.active_cluster_id == cluster_id:
            return

        self._stop_loader_if_running()
        self.active_cluster_id = cluster_id
        self._render_gallery(cluster_id)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _update_face_panel(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Обновляет правую панель с лицами."""
        self.face_details_widget.clear()
        if not current: return

        filename = current.data(Qt.ItemDataRole.UserRole)["filename"]
        record = self.data_manager.records.get(filename)
        if not record or not record.faces: return

        image_type = 'group' if self.mode == 'matches' else 'portrait'
        if self.mode == 'face' and self.active_cluster_id == 'group': image_type = 'group'
             
        full_path = self._get_image_path(filename, image_type)
        if not full_path.exists(): return

        try:
            if Image:
                pil_img = Image.open(str(full_path)).convert("RGBA")
                img_w, img_h = pil_img.size
            else: return 

            for i, face in enumerate(record.faces):
                bbox = face.bbox
                if len(bbox) != 4: continue
                v1, v2, v3, v4 = map(int, bbox)
                x1, y1, x2, y2 = v1, v2, v3, v4
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                
                padding = int(max(x2-x1, y2-y1) * 0.3)
                cx1 = max(0, x1 - padding); cy1 = max(0, y1 - padding)
                cx2 = min(img_w, x2 + padding); cy2 = min(img_h, y2 + padding)
                
                if cx2 <= cx1 or cy2 <= cy1: continue

                face_crop = pil_img.crop((cx1, cy1, cx2, cy2))
                qim = ImageQt.ImageQt(face_crop)
                pixmap = QPixmap.fromImage(qim)
                
                border_color = Qt.GlobalColor.lightGray
                status_text = ""
                
                if self.mode == 'matches':
                    matched_cluster = face.extra_data.get('matched_portrait_cluster_label')
                    if matched_cluster is None:
                        border_color = QColor(255, 50, 50) # Красный
                        status_text = "Не опознан"
                    elif str(matched_cluster) == str(self.active_cluster_id):
                        border_color = QColor(50, 205, 50) # Зеленый
                        status_text = "Этот кластер"
                    else:
                        border_color = QColor(100, 149, 237) # Синий
                        other_name = face.extra_data.get('matched_child_name', f"Cluster {matched_cluster}")
                        status_text = f"{other_name}"
                
                painter = QPainter(pixmap)
                pen = QPen(border_color)
                pen_width = max(4, int(min(pixmap.width(), pixmap.height()) * 0.05))
                pen.setWidth(pen_width)
                painter.setPen(pen)
                painter.drawRect(0, 0, pixmap.width(), pixmap.height())
                painter.end()
                
                item = QListWidgetItem()
                item.setIcon(pixmap)
                label_text = f"Лицо #{i+1}"
                if status_text: label_text += f"\n{status_text}"
                item.setText(label_text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.face_details_widget.addItem(item)

        except Exception as e:
            logger.error(f"Ошибка при отображении лиц: {e}")

    @Slot(int)
    def _on_face_size_changed(self, value: int):
        self.face_details_widget.setIconSize(QSize(value, value))

    # --- 6. АСИНХРОННАЯ ЗАГРУЗКА (WORKERS) ---
    # -----------------------------------------
    def _start_chunked_loader(self, tasks: List[Dict]):
        self.status_progress_bar.setRange(0, len(tasks))
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setFormat("Загрузка галереи... %p%")

        self.loader_thread = QThread(self)
        self.loader_worker = ChunkedImageLoader(tasks, self.image_pixmap_cache)
        self.loader_worker.moveToThread(self.loader_thread)
        
        self.loader_worker.chunk_ready.connect(self._on_chunk_ready)
        self.loader_worker.progress_updated.connect(self.status_progress_bar.setValue)
        self.loader_worker.finished.connect(self._on_loader_finished)
        
        self.loader_thread.started.connect(self.loader_worker.run)
        self.loader_thread.start()

    def _stop_loader_if_running(self):
        if hasattr(self, 'loader_thread') and self.loader_thread and self.loader_thread.isRunning():
            if hasattr(self, 'loader_worker') and self.loader_worker:
                self.loader_worker.requestInterruption()
            try:
                self.loader_worker.chunk_ready.disconnect()
                self.loader_worker.finished.disconnect()
            except Exception: pass
            self.loader_thread.quit()
            if not self.loader_thread.wait(1000): self.loader_thread.terminate()
            self.loader_worker = None; self.loader_thread = None

    @Slot(list)
    def _on_chunk_ready(self, items: List[Tuple[str, Any]]): # QImage не импортирован, но передается
        pixmap_items = []
        for filename, qimage in items:
            pixmap = QPixmap.fromImage(qimage)
            self.image_pixmap_cache[filename] = pixmap
            pixmap_items.append((filename, pixmap))
        self._add_images_to_gallery(pixmap_items)

    def _add_images_to_gallery(self, items: List[Tuple[str, QPixmap]]):
        for filename, pixmap in items:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
            item.setData(Qt.ItemDataRole.UserRole, {"filename": filename})
            self.image_list_widget.addItem(item)

    @Slot()
    def _on_loader_finished(self):
        self.status_progress_bar.reset(); self.status_progress_bar.setFormat("")
        self._stop_loader_if_running()

    # --- 7. СОБЫТИЯ И ДЕЙСТВИЯ (Drag&Drop, Меню) ---
    # -----------------------------------------------
    @Slot(str, str, list)
    def _handle_drop(self, source_id: str, target_id: str, filenames: List[str]):
        """Обрабатывает Drag & Drop."""
        
        # --- Режим MATCHES ---
        if self.mode == 'matches':
            target_cluster_data = self._get_cluster_item_data_by_id(target_id)
            if not target_cluster_data and target_id != "error_matches":
                return 

            # Привязка (Error -> Portrait)
            if source_id == "error_matches" and target_id != "error_matches":
                existing_matches = set(self.data_manager.get_group_matches_for_cluster(target_id))

                for fname in filenames:
                    if fname in existing_matches:
                        logger.info(f"Файл {fname} уже привязан к кластеру. Пропуск.")
                        continue

                    record = self.data_manager.records.get(fname)
                    if not record or not record.faces: continue
                    
                    unmatched_faces_indices = [
                        i for i, f in enumerate(record.faces) 
                        if f.extra_data.get('matched_portrait_cluster_label') is None
                    ]
                    
                    if not unmatched_faces_indices: continue 
                    target_face_idx = unmatched_faces_indices[0] 
                    
                    # ВСЕГДА показываем диалог
                    candidate_faces = [record.faces[i] for i in unmatched_faces_indices]
                    full_image_path = self._get_image_path(fname, 'group')
                    instruction = (f"Пожалуйста, укажите, кто на этом фото — <span style='font-size:12pt; font-weight:bold; color:#4CAF50;'>{target_cluster_data['name']}</span>?")
                    
                    dialog = FaceSelectorDialog(full_image_path, candidate_faces, self, instruction_text=instruction)
                    dialog.setWindowTitle(f"Ручное сопоставление")
                    
                    if dialog.exec() == QDialog.Accepted:
                        local_idx = dialog.get_selected_index()
                        target_face_idx = unmatched_faces_indices[local_idx]
                        self.data_manager.assign_manual_match(fname, target_id, target_cluster_data["name"], target_face_idx)
                    else:
                        continue 

            # Отвязка (Portrait -> Error)
            elif target_id == "error_matches" and source_id != "error_matches":
                for fname in filenames:
                    self.data_manager.unassign_manual_match(fname, source_id)

            self._refresh_left_panel()
            if self.active_cluster_id in [source_id, target_id]:
                 self._render_gallery(self.active_cluster_id)
            return 
        
        # --- Режимы FACE / LOCATION ---
        target_cluster_data = self._get_cluster_item_data_by_id(target_id)
        if not target_cluster_data: return

        face_selection_map = {}
        files_to_process = []

        if self.mode == 'face' and target_id not in ["group", "-1"]:
            for fname in filenames:
                record = self.data_manager.records.get(fname)
                if not record or not record.faces: continue
                
                if len(record.faces) > 1:
                    current_image_type = 'group' if source_id == 'group' else 'portrait'
                    full_image_path = self._get_image_path(fname, current_image_type)
                    instruction = (f"На фото несколько лиц.<br>Выберите то, которое должно стать <b>эталоном</b> для:<br><span style='font-size:12pt; font-weight:bold; color:#2196F3;'>{target_cluster_data['name']}</span>")
                    
                    dialog = FaceSelectorDialog(full_image_path, record.faces, self, instruction_text=instruction)
                    if dialog.exec() == QDialog.Accepted:
                        face_selection_map[fname] = dialog.get_selected_index()
                        files_to_process.append(fname)
                else:
                    face_selection_map[fname] = 0
                    files_to_process.append(fname)
        else:
            files_to_process = filenames

        if not files_to_process: return

        self.data_manager.move_images_to_cluster(
            self.mode_config, target_id, target_cluster_data["name"], 
            files_to_process, face_selection_map
        )

        active_id_before_refresh = self.active_cluster_id
        self._refresh_left_panel()
        if active_id_before_refresh:
            current_item = self._get_item_by_cluster_id(active_id_before_refresh)
            if current_item:
                self.cluster_list_widget.setCurrentItem(current_item)
                self._render_gallery(active_id_before_refresh)
            else:
                self.image_list_widget.clear(); self.right_panel_label.setText("Кластер")

    def show_cluster_context_menu(self, pos):
        menu = QMenu(self)
        if self.mode == 'matches':
            load_action = menu.addAction("Загрузить групповые данные (info_group_faces.json)...")
            load_action.triggered.connect(self._load_group_data_action)
        else:
            item = self.cluster_list_widget.itemAt(pos)
            create_action = menu.addAction("Создать кластер")
            menu.addSeparator()
            rename_action = menu.addAction("Переименовать")
            delete_action = menu.addAction("Удалить кластер")

            if not item:
                rename_action.setEnabled(False); delete_action.setEnabled(False)
            else:
                cluster_data = item.data(Qt.ItemDataRole.UserRole)
                is_empty = cluster_data.get("count", 0) == 0
                is_special = cluster_data.get("id") in ["-1", "group"]
                rename_action.setEnabled(not is_special)
                delete_action.setEnabled(is_empty and not is_special)

            action = menu.exec(self.cluster_list_widget.mapToGlobal(pos))
            if action == create_action: self._create_cluster_action()
            elif action == rename_action and item: self._rename_cluster_action(item)
            elif action == delete_action and item: self._delete_cluster_action(item)
            return
        menu.exec(self.cluster_list_widget.mapToGlobal(pos))

    @Slot(QListWidgetItem)
    def _open_image_viewer(self, item: QListWidgetItem):
        current_filename = item.data(Qt.ItemDataRole.UserRole)["filename"]
        all_filenames = self._get_files_for_cluster_for_viewer(self.active_cluster_id)
        try:
            current_index = all_filenames.index(current_filename)
        except ValueError: return

        image_type_for_viewer = 'group' if self.mode == 'matches' else 'portrait'
        image_paths = [self._get_image_path(fname, image_type_for_viewer) for fname in all_filenames]
        viewer = ImageViewer(image_paths, all_filenames, current_index, self)
        viewer.exec()

    @Slot(str)
    def _on_search_text_changed(self, text: str):
        search_text = text.strip().lower()
        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            cluster_name = item.data(Qt.ItemDataRole.UserRole)["name"]
            item.setHidden(search_text not in cluster_name.lower())

    def _handle_rename(self, cluster_id: str, new_name: str):
        self.data_manager.rename_cluster(self.mode_config, cluster_id, new_name)
        self._refresh_left_panel()

    def _create_cluster_action(self):
        new_name, ok = QInputDialog.getText(self, "Создание кластера", "Имя нового кластера:")
        if ok and new_name.strip():
            self.data_manager.create_cluster(self.mode_config, new_name)
            self._refresh_left_panel()

    def _delete_cluster_action(self, item: QListWidgetItem):
        cluster_data = item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(self, "Подтверждение", f"Удалить кластер '{cluster_data['name']}'?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.data_manager.delete_newly_created_cluster(cluster_data["id"])
            self._refresh_left_panel()

    @Slot(QListWidgetItem)
    def _rename_cluster_action(self, item: QListWidgetItem):
        cluster_data = item.data(Qt.ItemDataRole.UserRole)
        if self.mode == 'matches' or (self.mode == 'face' and cluster_data["id"] in ["group", "-1"]): return
        current_name = cluster_data["name"].split('-', 1)[-1]
        
        if self.mode == 'location':
            dialog = RenameDialog(self.predefined_cluster_names, current_name, self)
            if dialog.exec() == QDialog.Accepted:
                self._handle_rename(cluster_data["id"], dialog.get_selected_name())
        else:
            new_name, ok = QInputDialog.getText(self, "Переименование", "Новое имя (без префикса):", text=current_name)
            if ok and new_name.strip(): self._handle_rename(cluster_data["id"], new_name.strip())

    # --- 8. ЭКСПОРТ И СОХРАНЕНИЕ ---
    # -------------------------------
    @Slot()
    def _on_export_all_triggered(self):
        ids = [cid for cid in self._get_clusters_from_model().keys() if cid not in ["-1", "group"]]
        if ids: self._start_export(ids)
        else: QMessageBox.information(self, "Инфо", "Нет кластеров для экспорта.")

    @Slot()
    def _on_export_active_triggered(self):
        if self.active_cluster_id and self.active_cluster_id not in ["-1", "group"]:
            self._start_export([self.active_cluster_id])
        else: QMessageBox.warning(self, "Внимание", "Выберите кластер для экспорта.")

    def _start_export(self, cluster_ids: List[str]):
        if self.current_group_json_path:
            group_analysis_dir = self.current_group_json_path.parent
            base_output_dir = group_analysis_dir.parent / self.session_name / f"Выбор_Фото_{self.photo_session}_{self.mode}"
        else:
            base_output_dir = self.data_dir.parent / self.session_name / f"Выбор_Фото_{self.photo_session}_{self.mode}"
        
        logger.info(f"Экспорт в: {base_output_dir}")

        tasks = []
        for cid in cluster_ids:
            cdata = self._get_cluster_item_data_by_id(cid)
            if not cdata: continue
            
            out_folder = base_output_dir / cdata["name"]
            img_type = 'group' if self.mode == 'matches' else 'portrait'
            
            fnames = self.data_manager.get_group_matches_for_cluster(cid) if self.mode == 'matches' else self.data_manager.get_files_for_cluster(self.mode_config, cid)
            
            for fname in fnames:
                tasks.append({
                    "source_path": self._get_image_path(fname, img_type),
                    "output_path": out_folder / Path(fname).name,
                    "child_name": cdata["name"].split('-', 1)[-1].strip()
                })

        if not tasks:
            QMessageBox.information(self, "Инфо", "Нет файлов для экспорта.")
            return

        dialog = EnhanceSettingsDialog(tasks[0]["source_path"], self)
        if dialog.exec() != QDialog.Accepted: return
        
        settings = dialog.get_export_settings()
        
        self.status_progress_bar.setRange(0, len(tasks)); self.status_progress_bar.setValue(0); self.status_progress_bar.setFormat("Экспорт... %p%")
        
        self.export_worker = ExportWorker(
            tasks=tasks, num_threads=os.cpu_count() or 4, 
            enhancement_factors=settings["factors"], target_size=(settings["width"], settings["height"]),
            target_dpi=(settings["dpi"], settings["dpi"]), quality=settings["quality"], 
            apply_watermarks=settings["watermarks"]
        )
        self.export_thread = QThread()
        self.export_worker.moveToThread(self.export_thread)
        self.export_worker.progress_updated.connect(self.status_progress_bar.setValue)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_thread.started.connect(self.export_worker.run)
        self.export_thread.start()

    @Slot(str)
    def _on_export_finished(self, message: str):
        self.status_progress_bar.reset(); self.status_progress_bar.setFormat("")
        QMessageBox.information(self, "Экспорт завершен", message)
        if hasattr(self, 'export_thread'): self.export_thread.quit(); self.export_thread.wait()

# analize/cluster_editor/run_cluster_editor.py -> class MainWindow

    def _perform_save(self) -> bool:
        """
        Выполняет сохранение данных JSON и, в режиме 'location',
        обновляет переменную контекста PySM.

        Returns:
            bool: True в случае успеха, иначе False.
        """
        # 1. Сохраняем данные на диск
        if not self.data_manager.save_data():
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить JSON.")
            return False

        # 2. Обновляем UI
        self._refresh_left_panel()

# --- ИСПРАВЛЕНИЕ: Восстановлена логика сохранения в контекст ---
        # 3. В режиме 'location' обновляем системную переменную контекста
        if self.mode == 'location' and IS_MANAGED_RUN:
            try:
                location_previews: Dict[str, str] = {}
                # Получаем актуальные кластеры из менеджера данных
                clusters = self.data_manager.get_clusters(self.mode_config)
                
                # Заполняем словарь: Имя Локации -> Имя файла-представителя
                for cluster_id, faces in clusters.items():
                    if faces:  # Убеждаемся, что кластер не пустой
                        location_name = faces[0].effective_name
                        first_filename = faces[0].filename
                        if location_name and first_filename:
                            location_previews[location_name] = Path(first_filename).name

                # Добавляем системные/резервные имена (пустые, если их нет в реальных данных)
                additional_system_names = [
                    "portrait_A6",
                    "portrait_A5",
                    "portrait_A4"
                ]
                for name in additional_system_names:
                    if name not in location_previews:
                        location_previews[name] = ""
                
                # Формируем имя переменной: sys_location_name_ИМЯ_СЕССИИ
                current_location_name = f"sys_location_name_{self.photo_session}"
                
                # Сохраняем в контекст PySM
                pysm_context.set(current_location_name, location_previews)
                logger.info(f"Словарь локаций сохранен в переменную контекста: '{current_location_name}'.")
                
            except Exception as e:
                logger.error(f"Ошибка при сохранении контекста PySM: {e}")
                # Не блокируем сохранение файла, если контекст не записался, но логируем ошибку
# --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        return True

    def _save_changes(self, silent: bool = False) -> bool:
        if self.mode == 'matches':
            if self.current_group_json_path:
                base_dir = self.current_group_json_path.parent
                matches_path = base_dir / "matches_portrait_to_group.json"
                error_path = base_dir / "error_matches.json"
                if not silent: logger.info(f"Сохранение в: {base_dir}")
            else:
                if not silent: QMessageBox.critical(self, "Ошибка", "Нет файла групп.")
                return False

            ok, msg = self.data_manager.save_matches_mode_data(matches_path, error_path)
            if ok:
                if not silent: QMessageBox.information(self, "Успех", msg)
                return True
            else:
                QMessageBox.critical(self, "Ошибка", msg)
                return False

        if not self.data_manager.has_changes():
            if not silent: QMessageBox.information(self, "Инфо", "Нет изменений.")
            return True

        if not silent:
            if QMessageBox.question(self, "Сохранение", "Сохранить изменения?", QMessageBox.Save | QMessageBox.Cancel) != QMessageBox.Save:
                return False

        if self._perform_save():
            if not silent: QMessageBox.information(self, "Успех", "Сохранено.")
            return True
        else:
            return False

    def closeEvent(self, event):
        self._stop_loader_if_running()
        
        if not self.data_manager.has_changes():
            event.accept(); return

        reply = QMessageBox.question(self, "Выход", "Сохранить изменения перед выходом?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if reply == QMessageBox.Cancel: event.ignore()
        elif reply == QMessageBox.Discard: event.accept()
        elif reply == QMessageBox.Save:
            if self._save_changes(silent=True): event.accept()
            else: event.ignore()


# --- 9. ТОЧКА ВХОДА ---
# ==============================================================================
def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Редактор кластеров изображений.")
    arg_prefix = "ce_"
    parser.add_argument(f"--{arg_prefix}portrait_json", type=str, default="", help="Путь к info_portrait_faces.json")
    parser.add_argument(f"--{arg_prefix}group_json", type=str, default="", help="Путь к info_group_faces.json")
    parser.add_argument("--mode", type=str, choices=["face", "location", "matches"], default="face", help="Режим работы")
    return ConfigResolver(parser).resolve_all()

if __name__ == "__main__":
    cli_config = get_config()
    arg_prefix = "ce_"
    
    log_level = "INFO"
    if IS_MANAGED_RUN and pysm_context:
        log_level = pysm_context.get("sys_log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s", stream=sys.stdout)

    app = QApplication.instance() or QApplication(sys.argv)
    if IS_MANAGED_RUN: theme_api.apply_theme_to_app(app)

    try:
        portrait_json_str = getattr(cli_config, f"{arg_prefix}portrait_json")
        group_json_str = getattr(cli_config, f"{arg_prefix}group_json")
        if not portrait_json_str or not group_json_str: raise ValueError("Пути не указаны.")
        
        portrait_json_path = Path(portrait_json_str)
        group_json_path = Path(group_json_str)
        if not portrait_json_path.is_file(): raise FileNotFoundError(f"Нет файла: {portrait_json_path}")
        if not group_json_path.is_file(): raise FileNotFoundError(f"Нет файла: {group_json_path}")

    except Exception as e:
        QMessageBox.critical(None, "Ошибка запуска", str(e)); sys.exit(1)
    
    try:
        window = MainWindow(portrait_json_path, group_json_path, cli_config.mode)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)
        QMessageBox.critical(None, "Критическая ошибка", f"{traceback.format_exc()}")
        sys.exit(1)