#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cluster_editor.py
Refactored Version using Strategy Pattern.
"""

import sys
import os
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QInputDialog, QProgressBar, QMessageBox, QLineEdit, QMenu,
    QListWidget, QListWidgetItem, QDialog, QSplitter, QSlider, QTextEdit, QGroupBox, QFileDialog
)
from PySide6.QtGui import QPixmap, QColor, QImage, QPainter, QPen
from PySide6.QtCore import Qt, Slot, QThread, QSize, QTimer

try:
    from PIL import Image, ImageQt
except ImportError:
    Image = None

IS_MANAGED_RUN = False
try:
    current_script_dir = Path(__file__).resolve().parent
    if str(current_script_dir) not in sys.path: sys.path.insert(0, str(current_script_dir))
    project_root = current_script_dir.parent
    if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_theme_api import set_widget_class
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder, DashboardBuilder
    IS_MANAGED_RUN = True

    from _lib.editor_viewer import ImageViewer
    from _lib.editor_workers import ChunkedImageLoader, ExportWorker
    from _lib.editor_delegates import ClusterItemDelegate, ImageItemDelegate, THUMBNAIL_SIZE, FACE_SIZE, FACE_SIZE_PORTRAIT, FACE_MIN, FACE_MAX, PREVIEW_SIZE
    from _lib.editor_widgets import ImageDragListWidget, ClusterDropListWidget, FaceDetailsWidget
    from _lib.editor_dialogs import EnhanceSettingsDialog, RenameDialog, FaceSelectorDialog
    from _lib.data_manager import ClusterDataManager
    from _lib.data_models import Face

except ImportError as e:
    print(f"Критическая ошибка импорта внутренних модулей: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)


class MainWindow(QWidget):
    
    def __init__(self, working_dir: Path, reference_dir: Optional[Path], mode: str, num_workers: int):
        super().__init__()
        self.mode = mode # Сохраняем для специфичных UI-проверок (если остались)
        self.num_workers = num_workers
        self.working_dir = working_dir
        self.reference_dir = reference_dir if reference_dir else working_dir
        
        self.working_images_dir = self.working_dir / "JPG"
        self.reference_images_dir = self.reference_dir / "JPG"
        
        self.session_name = working_dir.parent.parent.name 
        self.photo_session = working_dir.name.replace("Analysis_", "")
        self.export_dir = ""
        self.export_end = False

        # 1. Инициализация Data Manager (здесь же создается Strategy)
        self.data_manager = ClusterDataManager(self.working_dir, self.reference_dir, mode=mode)
        
        # 2. Настройка окна через стратегию
        self.setWindowTitle(self.data_manager.strategy.get_window_title(self.photo_session))

        self.predefined_cluster_names: List[str] = []
        try:
            predefined_names_path = current_script_dir / "predefined_names.json"
            if predefined_names_path.exists():
                with open(predefined_names_path, 'r', encoding='utf-8') as f:
                    self.predefined_cluster_names = json.load(f)
        except Exception as e:
            logger.error(f"Error loading names: {e}")

        self.active_cluster_id: Optional[str] = None
        self.image_pixmap_cache: Dict[str, QPixmap] = {} 
        # ДОБАВЛЕНО: Словарь для быстрого поиска ячейки по имени файла/ключу
        self.gallery_items_map: Dict[str, QListWidgetItem] = {}        

        self.loader_thread = None
        self.cluster_delegate = ClusterItemDelegate(parent=self)
        self.image_delegate = ImageItemDelegate(parent=self)

        self.init_ui()
        self._load_and_display_data()

    def _get_image_path(self, filename: str) -> Path:
        # 1. Проверяем working dir
        p1 = self.working_images_dir / filename
        if p1.exists(): return p1
        
        # 2. Если режим matches и reference_dir отличается, проверяем там
        if self.mode == 'matches' and self.reference_dir != self.working_dir:
            p2 = self.reference_images_dir / filename
            if p2.exists(): return p2
        return p1

    def _get_cluster_item_data_by_id(self, cluster_id: str) -> Optional[Dict]:
        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data["id"] == cluster_id: return data
        return None

    def _center_on_screen(self):
        try:
            screen_geometry = self.screen().geometry()
            window_geometry = self.frameGeometry()
            window_geometry.moveCenter(screen_geometry.center())
            self.move(window_geometry.topLeft())
        except Exception: pass

    def init_ui(self):
        self.setGeometry(0, 0, 1420, 900)
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        # 1. LEFT PANEL
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Заголовок левой панели определяем по режиму
        left_label_text = "Список кластеров"
        if self.mode == 'matches': left_label_text = "Эталоны (Портреты)"
        elif self.mode == 'cleaning': left_label_text = "Технические группы"
        
        left_layout.addWidget(QLabel(f"{self.photo_session}: {left_label_text}"))
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Поиск...")
        self.search_bar.textChanged.connect(self._on_search_text_changed)
        left_layout.addWidget(self.search_bar)

        self.cluster_list_widget = ClusterDropListWidget(self)
        self.cluster_list_widget.setItemDelegate(self.cluster_delegate)
        self.cluster_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.cluster_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.cluster_list_widget.setSpacing(10)
        self.cluster_list_widget.itemDoubleClicked.connect(self._rename_cluster_action)
        self.cluster_list_widget.currentItemChanged.connect(self._on_cluster_selected)
        self.cluster_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cluster_list_widget.customContextMenuRequested.connect(self.show_cluster_context_menu)
        self.cluster_list_widget.setAcceptDrops(True)
        self.cluster_list_widget.setDragDropMode(QListWidget.DragDropMode.DropOnly) 
        self.cluster_list_widget.itemsDropped.connect(self._handle_drop)

        left_layout.addWidget(self.cluster_list_widget, 1)

        btn_layout = QHBoxLayout()
        # Кнопка Экспорт (не нужна в Cleaning)
        if self.mode != 'cleaning':
            self.export_button = QPushButton("Экспорт")
            export_menu = QMenu(self)
            export_menu.addAction("Все кластеры").triggered.connect(self._on_export_all_triggered)
            export_menu.addAction("Активный кластер").triggered.connect(self._on_export_active_triggered)
            self.export_button.setMenu(export_menu)
            btn_layout.addWidget(self.export_button)

        self.save_button = QPushButton("Сохранить")
        if self.mode == 'cleaning':
            self.save_button.setText("Удалить мусор и Сохранить")
            self.save_button.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.save_button.clicked.connect(lambda: self._save_changes(silent=False))
        btn_layout.addWidget(self.save_button)
        
        left_layout.addLayout(btn_layout)

        # 2. CENTER PANEL
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        self.gallery_label = QLabel("Галерея")
        center_layout.addWidget(self.gallery_label)

        self.image_list_widget = ImageDragListWidget(self)
        self.image_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list_widget.setSpacing(10)
        self.image_list_widget.setItemDelegate(self.image_delegate)
        self.image_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list_widget.setDragEnabled(True)
        self.image_list_widget.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        self.image_list_widget.itemDoubleClicked.connect(self._open_image_viewer)
        self.image_list_widget.currentItemChanged.connect(self._update_face_panel)
        
        self.image_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_list_widget.customContextMenuRequested.connect(self.show_gallery_context_menu)
        center_layout.addWidget(self.image_list_widget, 1)

        # 3. RIGHT PANEL (МОДИФИЦИРОВАНО)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        
        if self.data_manager.strategy.show_face_details_panel():
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            right_layout.setContentsMargins(0, 0, 0, 0)
            
            # --- СЕКЦИЯ 1: Информация о фото ---
            # group_photo = QGroupBox("Информация о фото")
            # group_photo_layout = QVBoxLayout(group_photo)
            # group_photo_layout.setContentsMargins(0, 5, 0, 0)
            
            self.photo_info_label = QLabel("Информация о фото")
            right_layout.addWidget(self.photo_info_label)
            
            self.photo_info_viewer = QTextEdit()
            self.photo_info_viewer.setReadOnly(True)
            right_layout.addWidget(self.photo_info_viewer, 15) # Stretch factor 2
            
            # --- СЕКЦИЯ 2: Список лиц ---
            right_layout.addWidget(QLabel("Лица на фото"))
            self.face_details_widget = FaceDetailsWidget(self, mode=self.mode)
            if IS_MANAGED_RUN: set_widget_class(self.face_details_widget, "face-panel")
            
            # Подключаем клик по лицу
            self.face_details_widget.itemClicked.connect(self._on_face_item_clicked)
            # Подключаем двойной клик (Просмотр)
            self.face_details_widget.itemDoubleClicked.connect(self._on_face_item_double_clicked) # <--- NEW
            self.face_details_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.face_details_widget.customContextMenuRequested.connect(self.show_face_details_context_menu)

            right_layout.addWidget(self.face_details_widget, 51) # Stretch factor 4
            
            # Слайдер размера            
            self.face_size_slider = QSlider(Qt.Orientation.Horizontal)
            self.face_size_slider.setRange(FACE_MIN, FACE_MAX)
            if self.mode == 'face':
                self.face_size_slider.setValue(FACE_SIZE_PORTRAIT)
            else:
                self.face_size_slider.setValue(FACE_SIZE)
                
            self.face_size_slider.valueChanged.connect(self._on_face_size_changed)
            right_layout.addWidget(self.face_size_slider)
            
            # --- СЕКЦИЯ 3: Информация о лице ---
            right_layout.addWidget(QLabel("Информация о выбранном лице"))
            self.face_info_viewer = QTextEdit()
            self.face_info_viewer.setReadOnly(True)
            right_layout.addWidget(self.face_info_viewer, 24) # Stretch factor 2

            splitter.addWidget(right_widget)
            
            # Настройка пропорций сплиттера (Лево, Центр, Право)
            splitter.setStretchFactor(0, 31)
            splitter.setStretchFactor(1, 46)
            splitter.setStretchFactor(2, 23)
        else:
            splitter.setStretchFactor(0, 35)
            splitter.setStretchFactor(1, 65)

        self.status_bar = QProgressBar()
        self.status_bar.setTextVisible(True)
        main_layout.addWidget(self.status_bar)
        self._center_on_screen()

    def _load_and_display_data(self):
        success, msg = self.data_manager.load_data()
        if not success:
            QMessageBox.critical(self, "Ошибка загрузки", msg)
            return
        
        # Legacy support for location covers via context
        if self.mode == 'location' and IS_MANAGED_RUN:
            var_name = f"sys_location_name_{self.photo_session}"
            covers_data = pysm_context.get(var_name)
            if covers_data and isinstance(covers_data, dict):
                self.data_manager.ingest_location_covers(covers_data)
        
        self._refresh_left_panel()

    def _refresh_left_panel(self):
        active_id = self.active_cluster_id
        self.cluster_list_widget.clear()
        
        clusters = self.data_manager.get_clusters()
        
        def sort_key(cid):
            if cid == "trash": return -2 
            if cid == "error_matches": return -1
            if cid.lstrip('-').isdigit(): return int(cid)
            if cid == "group": return 9997
            if cid == "-1": return 9999
            return 9998
            
        sorted_ids = sorted(clusters.keys(), key=sort_key)

        # Добавляем "Спец" кластеры, если их нет, но они нужны по режиму
        if self.mode == 'cleaning' and "trash" not in sorted_ids:
            self._add_cluster_item("trash", "🗑️ КОРЗИНА", [], is_special=True)
        
        if self.mode == 'matches':
            err_count = len(self.data_manager.get_files_for_cluster({}, "error_matches"))
            self._add_cluster_item("error_matches", f"⚠️ Неопознанные ({err_count})", [], is_special=True)

        for cid in sorted_ids:
            if self.mode == 'matches' and cid in ["-1", "group", "trash"]: continue
            faces = clusters[cid]
            is_new = any(c['id'] == cid for c in self.data_manager.newly_created_clusters)
            
            # Скрываем пустые (кроме новых и спец)
            if not faces and not is_new and cid not in ["trash", "error_matches"]: continue
            
            if faces: name = faces[0].effective_name
            else: name = f"Cluster {cid}"

            if cid == "trash": name = "🗑️ КОРЗИНА"
            
            self._add_cluster_item(cid, name, faces, is_special=(cid in ["trash", "error_matches", "group"]))

        # Восстановление выделения
        if active_id:
            for i in range(self.cluster_list_widget.count()):
                item = self.cluster_list_widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole)["id"] == active_id:
                    self.cluster_list_widget.setCurrentItem(item)
                    break
        elif self.cluster_list_widget.count() > 0:
            self.cluster_list_widget.setCurrentRow(0)

    def _add_cluster_item(self, cid: str, name: str, faces: List, is_special: bool = False):
        pixmap = QPixmap()
        fname = None
        best_face = None

        # --- ИЗМЕНЕНИЕ: Приоритет выбора файла ---
        if self.mode == 'matches' and faces:
            # В режиме Matches faces содержит Портреты (эталоны). 
            # Берем файл оттуда напрямую.
            fname = faces[0].filename
        elif self.mode == 'location':
             # В Location спрашиваем DataManager (поддержка ручных обложек)
             fname = self.data_manager.get_representative_file({}, cid)
        else:
             # Cleaning / Face
            if faces:
                if self.mode in ['cleaning', 'face']:
                    best_face = max(faces, key=lambda f: f.extra_data.get('det_score', 0.0) if f.extra_data else 0.0)
                    fname = best_face.filename
                else:
                    fname = faces[0].filename

        if fname:
            # 2. Логика отображения
            # В режиме Cleaning всегда делаем кроп с помощью PIL
            if self.mode == 'cleaning' and best_face:
                    full_path = self._get_image_path(fname)
                    if full_path.exists() and Image:
                        try:
                            with Image.open(str(full_path)) as pil_img:
                                bbox = best_face.bbox
                                if len(bbox) == 4:
                                    x1, y1, x2, y2 = map(int, bbox)
                                    if x1 > x2: x1, x2 = x2, x1
                                    if y1 > y2: y1, y2 = y2, y1
                                    
                                    w, h = x2 - x1, y2 - y1
                                    pad = int(max(w, h) * 0.4)
                                    cx1 = max(0, x1 - pad)
                                    cy1 = max(0, y1 - pad)
                                    cx2 = min(pil_img.width, x2 + pad)
                                    cy2 = min(pil_img.height, y2 + pad)
                                    
                                    crop = pil_img.crop((cx1, cy1, cx2, cy2))
                                    
                                    # Цвет: Конвертация в RGBA перед созданием QImage
                                    if crop.mode != "RGBA": crop = crop.convert("RGBA")
                                    data = crop.tobytes("raw", "RGBA")
                                    qim = QImage(data, crop.width, crop.height, QImage.Format.Format_RGBA8888).copy()
                                    pixmap = QPixmap.fromImage(qim)
                        except Exception as e:
                            logger.error(f"Preview crop error: {e}")
            
            # РЕЖИМ MATCHES (Эталоны) и остальные
            else:
                # В режиме Matches (разные папки) fname может быть из reference
                # Используем _get_image_path который сам проверит и там и там
                path = self._get_image_path(fname)
                if path.exists():
                    pixmap = QPixmap(str(path))
        
        if not pixmap.isNull():
            pixmap = pixmap.scaled(PREVIEW_SIZE, PREVIEW_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        if self.mode == 'matches':
            count = len(self.data_manager.get_group_matches_for_cluster(cid))
        else:
            count = len(self.data_manager.get_files_for_cluster({}, cid))

        item_data = {
            "id": cid, "name": name, "count": count, "pixmap": pixmap,
            "is_changed": self.data_manager.is_cluster_changed(self.mode, cid)
        }
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, item_data)
        
        if is_special:
            if cid == "trash": item.setBackground(QColor("#fff0f0"))
            if cid == "error_matches": item.setBackground(QColor("#fff8e1"))
            
        self.cluster_list_widget.addItem(item)


    @Slot(QListWidgetItem)
    def _on_face_item_double_clicked(self, item):
        """
        Открывает просмотрщик изображений.
        Собирает карту подсветок (highlights_map) для всех фото в галерее,
        где встречается выбранный человек (по cluster_label или match_id).
        """
        if not item: return
        
        # 1. Определяем текущее фото и лицо
        current_photo_item = self.image_list_widget.currentItem()
        if not current_photo_item: return
        fname = current_photo_item.data(Qt.ItemDataRole.UserRole)["filename"]
        
        record = self.data_manager.records.get(fname)
        if not record: return
        
        face_idx = item.data(Qt.ItemDataRole.UserRole)
        if face_idx is None or face_idx >= len(record.faces): return
        
        selected_face = record.faces[face_idx]
        
        # 2. Определяем критерии поиска (независимо от режима)
        # Если у лица есть ID сопоставления - используем его (наивысший приоритет)
        target_match_id = selected_face.extra_data.get('matched_portrait_cluster_label')
        
        # Если нет матча, но есть ID кластера (распознанное лицо) - используем его
        target_cluster_id = selected_face.cluster_label
            
        # 3. Подготавливаем список файлов
        if self.mode == 'matches':
             files = self.data_manager.get_group_matches_for_cluster(self.active_cluster_id)
        else:
             files = self.data_manager.get_files_for_cluster({}, self.active_cluster_id)
        
        # 4. Строим карту подсветок {filename: bbox}
        highlights_map = {}
        
        # Если у лица нет никаких ID (оно совсем неизвестное), подсвечиваем только на текущем фото
        if target_match_id is None and target_cluster_id is None:
            highlights_map[fname] = selected_face.bbox
        else:
            # Иначе ищем этого человека на всех фото в текущей галерее
            for f_name in files:
                rec = self.data_manager.records.get(f_name)
                if not rec: continue
                
                found_bbox = None
                for f in rec.faces:
                    # Приоритет 1: Совпадение по Match ID
                    if target_match_id is not None:
                        if f.extra_data.get('matched_portrait_cluster_label') == target_match_id:
                            found_bbox = f.bbox
                            break # Нашли - выходим из цикла лиц, идем к след. файлу
                    
                    # Приоритет 2: Совпадение по Cluster ID (если не ищем по Match ID)
                    elif target_cluster_id is not None:
                        if f.cluster_label == target_cluster_id:
                            found_bbox = f.bbox
                            break
                
                if found_bbox:
                    highlights_map[f_name] = found_bbox

        # Если вдруг на текущем фото лицо не нашлось через цикл (редкий кейс рассинхрона),
        # добавляем его принудительно, чтобы рамка точно была при открытии
        if fname not in highlights_map:
            highlights_map[fname] = selected_face.bbox

        if fname in files:
            idx = files.index(fname)
            paths = [self.working_images_dir / f for f in files]
            
            # 5. Запускаем вьювер с картой
            ImageViewer(paths, files, idx, self, highlights_map=highlights_map).exec()

    @Slot(str, str, list)
    def _handle_drop(self, source_id, target_id, filenames):
        """
        Обработчик сигнала сброса.
        Используем таймер, чтобы дать UI время завершить визуальную операцию Drag&Drop
        перед тем, как начинать тяжелую обработку и показывать диалоги.
        """
        QTimer.singleShot(30, lambda: self._process_drop_logic(source_id, target_id, filenames))

    def _process_drop_logic(self, source_id, target_id, filenames):
        """
        Основная логика обработки перемещения (вынесена из _handle_drop).
        """
        target_data = self._get_cluster_item_data_by_id(target_id)
        target_name = target_data["name"] if target_data else ""
        
        face_selection = {}
        valid_files = []

        # 1. Matches Mode
        if self.mode == 'matches':
            if target_id == "error_matches":
                # Unassign (Сброс соответствия)
                for f in filenames: 
                    self.data_manager.unassign_manual_match(f, source_id)
                self._refresh_left_panel()
                if self.active_cluster_id == source_id:
                    self._render_gallery(self.active_cluster_id)
                return
            else:
                # Assign to target (Назначение соответствия)
                for fname in filenames:
                    record = self.data_manager.records.get(fname)
                    if not record: continue
                    
                    # Собираем кандидатов (индекс, лицо)
                    candidates = []
                    for i, f in enumerate(record.faces):
                        if f.extra_data.get('matched_portrait_cluster_label') is None:
                            candidates.append((i, f))
                    
                    idx = -1
                    
                    if not candidates:
                        continue

                    # Если кандидатов несколько - спрашиваем пользователя
                    if len(candidates) > 1:
                        full_path = self._get_image_path(fname)
                        faces_to_show = [c[1] for c in candidates]
                        
                        dlg = FaceSelectorDialog(full_path, faces_to_show, self, 
                                                 f"Кто на фото - <b>{target_name}</b>?<br>(Показаны только неопознанные)")
                        
                        if dlg.exec() == QDialog.Accepted:
                            local_idx = dlg.get_selected_index()
                            if 0 <= local_idx < len(candidates):
                                idx = candidates[local_idx][0]
                        else:
                            continue 
                    else:
                        # Если кандидат один - берем автоматически
                        idx = candidates[0][0]

                    if idx != -1:
                        face_selection[fname] = idx
                        valid_files.append(fname)

        # 2. Cleaning Mode
        elif self.mode == 'cleaning':
            for fname in filenames:
                record = self.data_manager.records.get(fname)
                if not record: continue
                
                target_idx = -1
                for i, f in enumerate(record.faces):
                    current_sid = "trash" if f.is_trash else str(f.temp_cluster_label)
                    if current_sid == source_id:
                        target_idx = i
                        break
                
                if target_idx != -1:
                    face_selection[fname] = target_idx
                    valid_files.append(fname)
        
        # 3. Face Mode
        elif self.mode == 'face':
            for fname in filenames:
                record = self.data_manager.records.get(fname)
                if not record: continue
                
                if record.face_count > 1:
                    full_path = self._get_image_path(fname)
                    dlg = FaceSelectorDialog(full_path, record.faces, self)
                    if dlg.exec() == QDialog.Accepted:
                        face_selection[fname] = dlg.get_selected_index()
                        valid_files.append(fname)
                else:
                    face_selection[fname] = 0
                    valid_files.append(fname)

        # 4. Location Mode
        elif self.mode == 'location':
            valid_files = filenames

        # Execute Move
        if valid_files:
            self.data_manager.move_images_to_cluster(
                {}, target_id, target_name, valid_files, face_selection
            )
            self._refresh_left_panel()
            
            if self.active_cluster_id == source_id:
                self._render_gallery(source_id)
            if self.mode == 'matches' and self.active_cluster_id == target_id:
                self._render_gallery(target_id)

    def _save_changes(self, silent=False):
        self._stop_loader()
        
        # Специфичное подтверждение для Cleaning
        if self.mode == 'cleaning':
            if QMessageBox.warning(self, "Подтверждение очистки", 
                                   "Внимание! Все лица и файлы, находящиеся в 'Корзине', будут удалены БЕЗВОЗВРАТНО.\nПродолжить?",
                                   QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return False
        
        # Единая точка сохранения
        if self.data_manager.save_data():
            # Обновление контекста (если нужно для Legacy)
            if self.mode == 'location' and IS_MANAGED_RUN:
                self._update_pysm_context()
                
            if not silent:
                msg = "Мусор удален, данные обновлены." if self.mode == 'cleaning' else "Сохранено."
                QMessageBox.information(self, "Успех", msg)
            
            self._refresh_left_panel() # Перезагрузка UI (важно для Cleaning, чтобы убрать удаленное)
            return True
        else:
            if not silent: QMessageBox.critical(self, "Ошибка", "Ошибка при сохранении.")
            return False

    # --- UI Helpers ---
    
    @Slot(int)
    def _on_face_size_changed(self, value: int):
        if hasattr(self, 'face_details_widget'):
            self.face_details_widget.setIconSize(QSize(value, value))
            self.face_details_widget.setGridSize(QSize(value + 20, value + 60))

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_cluster_selected(self, current: QListWidgetItem, prev: QListWidgetItem):
        if not current:
            self._stop_loader()
            self.image_list_widget.clear()
            self.gallery_label.setText("Галерея")
            self.active_cluster_id = None
            return

        cid = current.data(Qt.ItemDataRole.UserRole)["id"]
        if self.active_cluster_id == cid: return

        self._stop_loader()
        self.active_cluster_id = cid
        self._render_gallery(cid)

    def _render_gallery(self, cluster_id: str):
        self._stop_loader()
        cdata = self._get_cluster_item_data_by_id(cluster_id)
        if not cdata: return
        
        self.gallery_label.setText(f"Галерея: {cdata['name']} ({cdata['count']} фото)")
        self.image_list_widget.clear()
        
        self.gallery_items_map.clear()

        if self.mode == 'matches':
             filenames = self.data_manager.get_group_matches_for_cluster(cluster_id)
        else:
             filenames = self.data_manager.get_files_for_cluster({}, cluster_id)

        if not filenames: return

        tasks = []
        placeholder = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        placeholder.fill(QColor("#3e3e3e")) 
        
        for fname in filenames:
            record = self.data_manager.records.get(fname)
            if not record: continue
            
            full_path = self.working_images_dir / fname
            current_keys = []
            
            if self.mode == 'cleaning':
                target_faces = []
                for i, f in enumerate(record.faces):
                    if cluster_id == "trash":
                        if f.is_trash: target_faces.append(i)
                    else:
                        if str(f.temp_cluster_label) == cluster_id: target_faces.append(i)
                
                for idx in target_faces:
                    cache_key = f"{fname}::{idx}"
                    current_keys.append((cache_key, idx)) 
            else:
                current_keys.append((fname, None))

            for cache_key, face_idx in current_keys:
                display_name = cache_key.split("::")[0]
                item = QListWidgetItem(display_name)
                
                # --- ИЗМЕНЕНИЕ: Формируем расширенный словарь данных ---
                user_data = {"filename": display_name}
                if face_idx is not None:
                    user_data["face_index"] = face_idx # Сохраняем индекс лица для cleaning
                
                item.setData(Qt.ItemDataRole.UserRole, user_data)
                
                if cache_key in self.image_pixmap_cache:
                    item.setData(Qt.ItemDataRole.DecorationRole, self.image_pixmap_cache[cache_key])
                else:
                    item.setData(Qt.ItemDataRole.DecorationRole, placeholder)
                    task = {
                        "filename": fname,
                        "cache_key": cache_key,
                        "full_path": full_path
                    }
                    if face_idx is not None: 
                        task["bbox"] = record.faces[face_idx].bbox
                        task["draw_face_rect"] = True
                    
                    tasks.append(task)
                
                self.image_list_widget.addItem(item)
                self.gallery_items_map[cache_key] = item

        if tasks:
            self._start_loader(tasks)

    def _start_loader(self, tasks: List[Dict]):
        self.status_bar.setRange(0, len(tasks))
        self.status_bar.setValue(0)
        self.loader_thread = QThread()
        self.loader_worker = ChunkedImageLoader(tasks, self.image_pixmap_cache, self.num_workers)
        self.loader_worker.moveToThread(self.loader_thread)
        self.loader_worker.chunk_ready.connect(self._on_chunk_ready)
        self.loader_worker.progress_updated.connect(self.status_bar.setValue)
        self.loader_worker.finished.connect(self._on_loader_finished)
        self.loader_thread.started.connect(self.loader_worker.run)
        self.loader_thread.start()

    def _stop_loader(self):
        if self.loader_thread and self.loader_thread.isRunning():
            if self.loader_worker: self.loader_worker.requestInterruption()
            self.loader_thread.quit()
            self.loader_thread.wait()
        self.loader_thread = None; self.loader_worker = None

    @Slot(list)
    def _on_chunk_ready(self, items: List):
        """
        Вызывается, когда поток загрузил пачку картинок.
        Обновляет иконки в уже созданных ячейках.
        """
        for key, qimage in items:
            pixmap = QPixmap.fromImage(qimage)
            self.image_pixmap_cache[key] = pixmap
            
            # Находим нужную ячейку по ключу и обновляем картинку
            if key in self.gallery_items_map:
                item = self.gallery_items_map[key]
                # setIcon требует QIcon, setData(DecorationRole) принимает QPixmap
                # Делегат использует DecorationRole
                item.setData(Qt.ItemDataRole.DecorationRole, pixmap)

    @Slot()
    def _on_loader_finished(self):
        self.status_bar.reset()
        self._stop_loader()


    @Slot(QListWidgetItem)
    def _update_face_panel(self, current: QListWidgetItem, prev=None):
        """Обновляет правую панель при выборе фото в галерее."""
        if self.mode == 'cleaning': return 

        # Очистка панелей
        if hasattr(self, 'face_details_widget'): self.face_details_widget.clear()
        if hasattr(self, 'photo_info_viewer'): self.photo_info_viewer.clear()
        if hasattr(self, 'face_info_viewer'): self.face_info_viewer.clear()
        
        if not current: return
        
        fname = current.data(Qt.ItemDataRole.UserRole)["filename"]
        record = self.data_manager.records.get(fname)
        if not record: return
        
        # 1. Заполнение информации о ФОТО
        info_html = f"""
        <style>td {{ padding-right: 10px; }}</style>
        <table>
        <tr><td><b>Файл:</b></td><td>{fname}</td></tr>
        <tr><td><b>Размер:</b></td><td>{record.original_shape[1]} x {record.original_shape[0]}</td></tr>
        <tr><td><b>Лиц найдено:</b></td><td>{record.face_count}</td></tr>
        <tr><td><b>Тип:</b></td><td>{record.image_type}</td></tr>
        """
        
        if record.location_name:
            info_html += f"<tr><td><b>Локация:</b></td><td>{record.location_name} (ID: {record.location_cluster})</td></tr>"
            
        info_html += "</table>"
        self.photo_info_viewer.setHtml(info_html)

        # 2. Отрисовка списка лиц (существующий код)
        full_path = self._get_image_path(fname)
        if not full_path.exists(): return
        
        try:
            if Image:
                pil_img = Image.open(str(full_path))
                w, h = pil_img.size
                for i, face in enumerate(record.faces):
                    bbox = face.bbox
                    if len(bbox) != 4: continue
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    if x1 > x2: x1, x2 = x2, x1
                    if y1 > y2: y1, y2 = y2, y1
                    pad = int(max(x2-x1, y2-y1)*0.3)
                    cx1 = max(0, x1-pad); cy1 = max(0, y1-pad)
                    cx2 = min(w, x2+pad); cy2 = min(h, y2+pad)
                    
                    crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    if crop.mode != "RGBA": crop = crop.convert("RGBA")
                    
                    qim = ImageQt.ImageQt(crop)
                    pixmap = QPixmap.fromImage(qim)
                    
                    # Цветные рамки для Matches
                    txt_color = "#dcdcdc"
                    border_color = None

                    if self.mode == 'matches':
                        matched_id = face.extra_data.get('matched_portrait_cluster_label')
                        painter = QPainter(pixmap)
                        pen_width = max(3, int(min(pixmap.width(), pixmap.height()) * 0.04))
                        
                        if matched_id is None:
                            color = QColor(255, 50, 50) 
                            status_text = "Не опознан"
                            txt_color = "#ff6666"
                        elif str(matched_id) == str(self.active_cluster_id):
                            color = QColor(50, 205, 50) 
                            status_text = "(Этот кластер)"
                            txt_color = "#66ff66"
                        else:
                            color = QColor(65, 105, 225) 
                            cname = face.extra_data.get('matched_child_name', f"ID {matched_id}")
                            status_text = f"{cname}"
                            txt_color = "#66ccff"
                        
                        pen = QPen(color); pen.setWidth(pen_width)
                        painter.setPen(pen)
                        offset = pen_width // 2
                        painter.drawRect(offset, offset, pixmap.width() - pen_width, pixmap.height() - pen_width)
                        painter.end()
                        
                        txt = f"Лицо #{i+1}\n{status_text}"
                    else:
                        txt = f"Лицо #{i+1}"
                        # Добавляем инфо, если есть имя
                        if face.child_name:
                             txt += f"\n{face.child_name}"

                    item = QListWidgetItem()
                    item.setIcon(pixmap)
                    item.setText(txt)
                    # Сохраняем индекс лица в списке record.faces для дальнейшего доступа
                    item.setData(Qt.ItemDataRole.UserRole, i) 
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.face_details_widget.addItem(item)
                    
        except Exception as e:
            logger.error(f"Error face panel: {e}")

    @Slot(QListWidgetItem)
    def _on_face_item_clicked(self, item):
        """
        Отображает информацию о лице.
        Приоритет: Портрет -> Match -> Пусто. TempID игнорируется.
        """
        if not item: return
        
        current_photo_item = self.image_list_widget.currentItem()
        if not current_photo_item: return
        fname = current_photo_item.data(Qt.ItemDataRole.UserRole)["filename"]
        
        record = self.data_manager.records.get(fname)
        if not record: return
        
        face_idx = item.data(Qt.ItemDataRole.UserRole)
        if face_idx is None or face_idx >= len(record.faces): return
        
        face = record.faces[face_idx]
        
        # --- ЛОГИКА ОТОБРАЖЕНИЯ ---
        display_status = "Не определено"
        display_cluster_id = "None"
        display_name = "---"
        match_distance = None

        # 1. Если это Портрет (есть cluster_label)
        if face.cluster_label is not None:
            display_status = "Портрет (Cluster)"
            display_cluster_id = str(face.cluster_label)
            display_name = face.child_name or "---"
            
        # 2. Если есть Матч (matched_portrait_cluster_label)
        elif face.extra_data.get('matched_portrait_cluster_label') is not None:
            display_status = "Сопоставление (Match)"
            display_cluster_id = str(face.extra_data.get('matched_portrait_cluster_label'))
            display_name = face.extra_data.get('matched_child_name') or "---"
            match_distance = face.extra_data.get('match_distance')
            
        # 3. Иначе - данные не заполняются (Temp ID игнорируем)
        else:
            display_status = "Не опознан"

        # Формируем HTML
        html = f"""
        <style>td {{ padding-right: 8px; }}</style>
        <table>
        <tr><td><b>Внутренний ID:</b></td><td>{face.face_index}</td></tr>
        <tr><td><b>Позиция:</b></td><td>{face_idx}</td></tr>
        <tr><td><b>Score (Детекция):</b></td><td>{face.extra_data.get('det_score', 0.0):.4f}</td></tr>
        """
        
        # Атрибуты
        age = face.extra_data.get('age_insight')
        gender = face.extra_data.get('gender_insight')
        if age is not None: html += f"<tr><td><b>Возраст (AI):</b></td><td>{age}</td></tr>"
        if gender is not None: html += f"<tr><td><b>Пол (AI):</b></td><td>{'М' if gender==1 else 'Ж'}</td></tr>"

        html += "<tr><td colspan='2'><hr></td></tr>"
        
        html += f"<tr><td><b>Cluster ID:</b></td><td>{display_cluster_id}</td></tr>"
        html += f"<tr><td><b>Имя:</b></td><td>{display_name}</td></tr>"
        
        if match_distance is not None:
             html += f"<tr><td><b>Дистанция:</b></td><td>{match_distance:.4f}</td></tr>"
        
        # Temp ID показываем только для отладки в скобках, или убираем совсем?
        # По задаче "Temp ID использовать не нужно", поэтому не выводим.

        html += "</table>"
        self.face_info_viewer.setHtml(html)


    @Slot(str)
    def _on_search_text_changed(self, text):
        search = text.lower()
        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)["name"].lower()
            item.setHidden(search not in name)

    @Slot(QListWidgetItem)
    def _rename_cluster_action(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        cid = data["id"]
        # Matches mode rename is disabled in strategy, check here or call anyway
        if self.mode == 'matches': return 
        if cid in ["trash", "error_matches"]: return
        
        current_name = self.data_manager.strategy._strip_name_prefix(data["name"])
        new_name = None
        
        if self.mode == 'location':
            dialog = RenameDialog(self.predefined_cluster_names, current_name, self)
            if dialog.exec() == QDialog.Accepted:
                new_name = dialog.get_selected_name()
        else:
            text, ok = QInputDialog.getText(self, "Переименование", "Имя:", text=current_name)
            if ok: new_name = text
            
        if new_name and new_name.strip():
            self.data_manager.rename_cluster({}, cid, new_name.strip())
            self._refresh_left_panel()

    def show_cluster_context_menu(self, pos):
        item = self.cluster_list_widget.itemAt(pos)
        menu = QMenu()
        
        if self.mode == 'cleaning':
            act_empty = menu.addAction("Очистить корзину (удалить навсегда)")
            
            # --- ИСПРАВЛЕНИЕ: Явное преобразование в bool ---
            is_trash = bool(item and item.data(Qt.ItemDataRole.UserRole)["id"] == "trash")
            act_empty.setEnabled(is_trash)
            
            if act_empty.isEnabled():
                act_empty.triggered.connect(lambda: self._save_changes(silent=False))
        elif self.mode == 'matches':
            # В режиме Matches добавляем возможность сменить рабочую папку (группы),
            # оставив эталоны (портреты) загруженными.
            action_load = menu.addAction("📂 Открыть другую съемку (JSON)...")
            action_load.triggered.connect(self._load_other_session)
            
            # Разделитель, если кликнули по элементу
            if item: menu.addSeparator()
        elif self.mode != 'matches':
            menu.addAction("Создать кластер").triggered.connect(self._create_cluster)
            if item:
                menu.addAction("Переименовать").triggered.connect(lambda: self._rename_cluster_action(item))
                
        menu.exec(self.cluster_list_widget.mapToGlobal(pos))

    def _load_other_session(self):
        """
        Позволяет выбрать другой JSON файл и переключить рабочую директорию,
        сохранив текущие эталоны (в режиме matches).
        """
        # 1. Проверка на несохраненные данные
        if self.data_manager.has_changes():
            reply = QMessageBox.question(self, "Смена сессии", 
                                         "Есть несохраненные изменения. Сохранить перед переключением?", 
                                         QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel: return
            if reply == QMessageBox.Save:
                self._save_changes(silent=False)

        # 2. Диалог выбора файла
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите файл данных (info_faces.json / info_group_faces.json)",
            str(self.working_dir),
            "JSON Files (info_faces.json)"
        )
        
        if not file_path: return
        
        new_path = Path(file_path)
        
        # 3. Обновление путей в Main Window
        self.working_dir = new_path.parent
        self.working_images_dir = self.working_dir / "JPG"
        
        # Обновляем имя сессии в заголовке
        self.photo_session = self.working_dir.name.replace("Analysis_", "")
        self.setWindowTitle(self.data_manager.strategy.get_window_title(self.photo_session))
        
        # 4. Обновление Data Manager
        self.data_manager.switch_working_session(new_path)
        
        # 5. Перезагрузка данных
        self._load_and_display_data()        


    def show_gallery_context_menu(self, pos):
        item = self.image_list_widget.itemAt(pos)
        if not item: return
        menu = QMenu()
        
        if self.mode == 'location':
            action = menu.addAction("📸 Сделать обложкой локации")
            action.triggered.connect(lambda: self._set_cover_action(item))
            
        if not menu.isEmpty():
            menu.exec(self.image_list_widget.mapToGlobal(pos))

    # ====== КОНТЕКСТНОЕ МЕНЮ ПАНЕЛИ ЛИЦ ======
    """
    Группа методов для работы с контекстным меню для панели лиц (правая панель).
    Позволяет перейти к кластеру, которому принадлежит лицо.
    Работает в режиме Face и matches
    """
    def show_face_details_context_menu(self, pos):
        """
        Контекстное меню для панели лиц (правая панель).
        Позволяет перейти к кластеру, которому принадлежит лицо.
        """
        # --- ИЗМЕНЕНИЕ: Отключаем меню для Cleaning и Location ---
        if self.mode in ['cleaning', 'location']:
            return

        item = self.face_details_widget.itemAt(pos)
        if not item: return

        # 1. Получаем данные лица
        current_photo_item = self.image_list_widget.currentItem()
        if not current_photo_item: return
        fname = current_photo_item.data(Qt.ItemDataRole.UserRole)["filename"]
        
        record = self.data_manager.records.get(fname)
        if not record: return
        
        face_idx = item.data(Qt.ItemDataRole.UserRole)
        if face_idx is None or face_idx >= len(record.faces): return
        
        face = record.faces[face_idx]
        
        # 2. Определяем Target ID (с учетом улучшенной логики)
        target_id = None
        
        # Шаг А: Проверяем прямую привязку к кластеру (для портретов)
        if face.cluster_label is not None:
            target_id = face.cluster_label
            
        # Шаг Б: Если нет прямой привязки, проверяем Сопоставление (Matches)
        elif face.extra_data.get('matched_portrait_cluster_label') is not None:
            target_id = face.extra_data.get('matched_portrait_cluster_label')

        # 3. Создаем меню
        menu = QMenu()
        action_open = menu.addAction("📂 Перейти к кластеру")
        
        if target_id is not None:
            target_id_str = str(target_id)
            action_open.triggered.connect(lambda: self._activate_cluster_by_id(target_id_str))
        else:
            action_open.setEnabled(False)
            action_open.setText("Кластер не определен")

        menu.exec(self.face_details_widget.mapToGlobal(pos))

    def _activate_cluster_by_id(self, cluster_id: str):
        """
        Находит кластер в левой панели и делает его активным.
        """
        # Сначала сбрасываем фильтр поиска, если кластер скрыт фильтром
        if self.search_bar.text():
            self.search_bar.clear()
            
        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            
            # Сравниваем ID как строки
            if str(data["id"]) == str(cluster_id):
                self.cluster_list_widget.setCurrentItem(item)
                self.cluster_list_widget.scrollToItem(item)
                # Фокус на список, чтобы можно было сразу листать клавиатурой
                self.cluster_list_widget.setFocus()
                return
        
        # Если не нашли
        QMessageBox.information(self, "Поиск", f"Кластер с ID {cluster_id} не найден в текущем списке.")

    # ====== КОНТЕКСТНОЕ МЕНЮ ПАНЕЛИ ЛИЦ ======


    def _set_cover_action(self, item):
        fname = item.data(Qt.ItemDataRole.UserRole)["filename"]
        cid = self.active_cluster_id
        if not cid: return
        self.data_manager.set_location_cover(cid, fname)
        self._refresh_left_panel()

    def _create_cluster(self):
        name, ok = QInputDialog.getText(self, "Новый кластер", "Имя:")
        if ok and name:
            self.data_manager.create_cluster({}, name)
            self._refresh_left_panel()

    @Slot(QListWidgetItem)
    def _open_image_viewer(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        fname = data["filename"]
        
        # --- НОВАЯ ЛОГИКА ДЛЯ CLEANING ---
        if self.mode == 'cleaning':
            # В этом режиме мы открываем конкретное фото для проверки контекста
            face_idx = data.get("face_index")
            
            record = self.data_manager.records.get(fname)
            if not record: return
            
            bbox = None
            if face_idx is not None and face_idx < len(record.faces):
                bbox = record.faces[face_idx].bbox
            
            full_path = self._get_image_path(fname)
            
            # В cleaning открываем вьювер только для одного фото (без навигации по кропам),
            # так как навигация по "кропам" в полноэкранном режиме не интуитивна.
            if full_path.exists():
                ImageViewer([full_path], [fname], 0, self, highlight_bbox=bbox).exec()
            return

        # --- СТАНДАРТНАЯ ЛОГИКА (Matches, Face, Location) ---
        if self.mode == 'matches':
             files = self.data_manager.get_group_matches_for_cluster(self.active_cluster_id)
        else:
             files = self.data_manager.get_files_for_cluster({}, self.active_cluster_id)
            
        if fname in files:
            idx = files.index(fname)
            paths = [self.working_images_dir / f for f in files]
            ImageViewer(paths, files, idx, self).exec()

    @Slot()
    def _on_export_all_triggered(self):
        clusters = self.data_manager.get_clusters()
        ids = [c for c in clusters if c not in ["trash", "-1", "group", "error_matches"]]
        self._start_export(ids)

    @Slot()
    def _on_export_active_triggered(self):
        if self.active_cluster_id and self.active_cluster_id not in ["trash", "-1", "group"]:
            self._start_export([self.active_cluster_id])

    def _start_export(self, cluster_ids):
        # NOTE: Export logic remains mostly same, but we can clean it up later in Optimization phase
        out_dir = self.working_dir.parent / self.session_name / f"Выбор_Фото_{self.photo_session}_{self.mode}"
        self.export_dir = out_dir
        self.export_end = False

        tasks = []
        
        for cid in cluster_ids:
            cdata = self._get_cluster_item_data_by_id(cid)
            if not cdata: continue
            
            # Use strategy to clean name
            cname = self.data_manager.strategy._strip_name_prefix(cdata["name"])
            
            files = self.data_manager.get_files_for_cluster({}, cid)
            for fname in files:
                faces_bboxes = []
                record = self.data_manager.records.get(fname)
                if record:
                    for face in record.faces:
                        if face.bbox and len(face.bbox) == 4:
                            faces_bboxes.append(face.bbox)

                tasks.append({
                    "source_path": self.working_images_dir / fname,
                    "output_path": out_dir / cname / fname,
                    "child_name": cname,
                    "faces_bboxes": faces_bboxes
                })
        
        if not tasks: return
        preview_bboxes = tasks[0].get("faces_bboxes", [])
        dlg = EnhanceSettingsDialog(tasks[0]["source_path"], preview_bboxes, self)
        if dlg.exec() != QDialog.Accepted: return
        settings = dlg.get_export_settings()
        
        self.status_bar.setFormat("Экспорт... %p%")
        self.status_bar.setRange(0, len(tasks))
        
        self.export_thread = QThread()
        # TODO: Move to ProcessPool in optimization phase
        self.export_worker = ExportWorker(tasks, self.num_workers, settings["factors"], 
                                          (settings["width"], settings["height"]),
                                          (settings["dpi"], settings["dpi"]),
                                          settings["quality"], settings["watermarks"])
        self.export_worker.moveToThread(self.export_thread)
        self.export_worker.progress_updated.connect(self.status_bar.setValue)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_thread.started.connect(self.export_worker.run)
        self.export_thread.start()

    @Slot(str)
    def _on_export_finished(self, message: str):
        self.status_bar.reset(); 
        self.status_bar.setFormat("")
        QMessageBox.information(self, "Экспорт завершен", message)
        self.export_end = True

        if hasattr(self, 'export_thread'): 
            self.export_thread.quit(); 
            self.export_thread.wait()

    def _update_pysm_context(self):
        # Helper to update legacy context if managed run
        if self.mode == 'location' and IS_MANAGED_RUN:
            try:
                location_previews = self.data_manager.get_location_covers_dict()
                for name in ["portrait_A6", "portrait_A5", "portrait_A4"]:
                    if name not in location_previews:
                        location_previews[name] = ""
                var_name = f"sys_location_name_{self.photo_session}"
                pysm_context.set(var_name, location_previews)
            except Exception as e:
                logger.error(f"Context update error: {e}")


    def _log_final_report(self):
            """Формирует и выводит финальный отчет перед закрытием."""
            if not IS_MANAGED_RUN or not pysm_context:
                return

            try:
                # 1. Инициализация
                tv_builder = StandardTreeBuilder(icon_size=28)
                report_folder = [] # Создаем пустой список

                # 2. Наполнение списка
                if self.reference_dir != self.working_dir:
                    root_node_ref = ResourceNode(self.reference_dir.name, Path(self.reference_dir), "folder", "Папка референсной фотосессии с эталонными портретами")
                    report_folder.append(root_node_ref) # <--- ИСПРАВЛЕНО: append
                    
                root_node_target = ResourceNode(self.working_dir.name, Path(self.working_dir), "folder", "Целевая папка текущей фотосессии")
                report_folder.append(root_node_target) # <--- ИСПРАВЛЕНО: append

                # Проверяем атрибуты через getattr на случай, если экспорт не запускался и переменные не созданы
                if getattr(self, 'export_end', False) and hasattr(self, 'export_dir'):
                    root_node_export = ResourceNode(Path(self.export_dir).name, Path(self.export_dir), "folder", "Папка с экспортированными файлами JPG")
                    report_folder.append(root_node_export) # <--- ИСПРАВЛЕНО: append

                # 3. Передача списка в билдер
                # Передаем сам список report_folder
                tv_builder.add_section("<br>Рабочие папки и файлы", report_folder)
                
                # 4. Вывод в лог
                pysm_context.log_html(tv_builder.get_html())
            except Exception as e:
                logger.error(f"Ошибка при формировании финального отчета: {e}")



    def closeEvent(self, event):
        # --- ИСПРАВЛЕНИЕ: Останавливаем потоки ПЕРЕД любой логикой выхода ---
        self._stop_loader()
      
        # Если идет экспорт, его тоже по-хорошему надо остановить, 
        # но ProcessPoolExecutor сложно убить мгновенно. 
        # Оставим на совести пользователя (или можно добавить проверку isRunning).
        if hasattr(self, 'export_thread') and self.export_thread and self.export_thread.isRunning():
             # Можно добавить предупреждение: "Идет экспорт, прервать?"
             pass
        if self.data_manager.has_changes():
            reply = QMessageBox.question(self, "Выход", "Сохранить изменения?", 
                                         QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if reply == QMessageBox.Save:
                # _save_changes внутри себя тоже вызывает _stop_loader, это нормально (дубль не страшен)
                if self._save_changes(silent=True):
                    self._log_final_report()
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.Discard:
                self._log_final_report()
                event.accept()
            else:
                event.ignore()
        else:
            self._log_final_report()
            event.accept()

def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Редактор кластеров.")
    p = "ce_"
    parser.add_argument(f"--{p}working_dir", type=str, required=True, help="Папка с данными")
    parser.add_argument(f"--{p}reference_dir", type=str, default=None, help="Папка с эталонами (для matches)")
    parser.add_argument("--all_threads", type=int, dest="all_threads", default=0, help="Количество потоков (0=авто).")
    parser.add_argument("--mode", type=str, choices=["face", "location", "matches", "cleaning"], default="face")
    return ConfigResolver(parser).resolve_all()

if __name__ == "__main__":
    print("<b>ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ</b>")
    print("<i>Инициализация...</i><br>")

    cli_config = get_config()
    arg_prefix = "ce_"
    
    log_level = "INFO"
    if IS_MANAGED_RUN and pysm_context:
        log_level = pysm_context.get("sys_log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s", stream=sys.stdout)

    app = QApplication.instance() or QApplication(sys.argv)
    if IS_MANAGED_RUN: theme_api.apply_theme_to_app(app)

    try:   
        w_dir = Path(getattr(cli_config, f"{arg_prefix}working_dir"))
        r_dir_str = getattr(cli_config, f"{arg_prefix}reference_dir")
        r_dir = Path(r_dir_str) if r_dir_str else None
        
        if not w_dir.exists(): raise FileNotFoundError(f"Нет папки: {w_dir}")

        num_workers = cli_config.all_threads or (os.cpu_count() or 8)    

        window = MainWindow(w_dir, r_dir, cli_config.mode, num_workers)
        window.show()
     
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Ошибка запуска: {e}", exc_info=True)
        QMessageBox.critical(None, "Ошибка", str(e))
        sys.exit(1)