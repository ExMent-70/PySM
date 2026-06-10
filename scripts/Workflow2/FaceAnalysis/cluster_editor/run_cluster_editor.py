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
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMainWindow,
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
    from pysm_lib.pysm_icons import icons as pysm_icons    
    from pysm_lib.window_state_manager import WindowStateManager
    
    IS_MANAGED_RUN = True

    from _lib.editor_viewer import ImageViewer
    from _lib.editor_workers import ChunkedImageLoader, ExportWorker
    from _lib.editor_delegates import ClusterItemDelegate, ImageItemDelegate, THUMBNAIL_SIZE, FACE_SIZE, FACE_SIZE_PORTRAIT, FACE_MIN, FACE_MAX, PREVIEW_SIZE
    from _lib.editor_widgets import ImageDragListWidget, ClusterDropListWidget, FaceDetailsWidget
    from _lib.editor_dialogs import EnhanceSettingsDialog, RenameDialog, FaceSelectorDialog
    from _lib.data_manager import ClusterDataManager
    from _lib.data_models import Face
    from _lib.editor_ui import EditorUIBuilder
    from _lib.editor_filters import GalleryFilterManager
    from _lib.editor_menus import EditorMenuManager

except ImportError as e:
    print(f"Критическая ошибка импорта внутренних модулей: {e}", file=sys.stderr)
    pysm_icons = None
    sys.exit(1)

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    
    def __init__(self, working_dir: Path, reference_dir: Optional[Path], mode: str, num_workers: int, export_dir: str, win_state_var_name: str):
        super().__init__()
        self.mode = mode # Сохраняем для специфичных UI-проверок (если остались)
        self.num_workers = num_workers
        self.working_dir = working_dir
        
        self.win_state_var_name = win_state_var_name

        self.reference_dir = reference_dir if reference_dir else working_dir
        
        self.working_images_dir = self.working_dir / "JPG"
        self.reference_images_dir = self.reference_dir / "JPG"
        
        self.session_name = working_dir.parent.parent.name 
        self.photo_session = working_dir.name.replace("Analysis_", "")
       
        exp_dir = Path(export_dir) if export_dir else self.working_dir.parent / self.session_name       
        self.export_dir = exp_dir / f"Выбор_Фото_{self.photo_session}_{self.mode}"  
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

        # --- ИЗМЕНЕНИЕ: Внешний билдер вместо внутреннего метода ---
        self.menu_manager = EditorMenuManager(self)     # <--- ДОБАВЛЕНО
        self.filter_manager = GalleryFilterManager(self)
        
        # Внешний билдер вместо внутреннего метода
        EditorUIBuilder.build_ui(self)
        self.filter_manager.bind_ui()        
        try:
            screen_geometry = self.screen().geometry()
            window_geometry = self.frameGeometry()
            window_geometry.moveCenter(screen_geometry.center())
            self.move(window_geometry.topLeft())
        except Exception:
            pass

        
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
        
        # --- ИСПРАВЛЕНИЕ: Блокируем сигналы на всё время пересборки панели, 
        # чтобы clear() не стирал active_cluster_id и центральную галерею! ---
        self.cluster_list_widget.blockSignals(True)
        
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

        if self.mode == 'cleaning' and "trash" not in sorted_ids:
            self._add_cluster_item("trash", "🗑️ КОРЗИНА",[], is_special=True)
        
        if self.mode == 'matches':
            err_count = len(self.data_manager.get_files_for_cluster(dict(), "error_matches"))
            self._add_cluster_item("error_matches", f"⚠️ Неопознанные ({err_count})",[], is_special=True)

        for cid in sorted_ids:
            if self.mode == 'matches' and cid in["-1", "group", "trash"]: continue
            faces = clusters[cid]
            is_new = any(c['id'] == cid for c in self.data_manager.newly_created_clusters)
            
            if not faces and not is_new and cid not in ["trash", "error_matches"]: continue
            
            if faces: name = faces[0].effective_name
            else: name = f"Cluster {cid}"

            if cid == "trash": name = "🗑️ КОРЗИНА"
            
            self._add_cluster_item(cid, name, faces, is_special=(cid in["trash", "error_matches", "group"]))

        # --- ВОССТАНОВЛЕНИЕ ВЫДЕЛЕНИЯ ---
        found = False
        if active_id:
            for i in range(self.cluster_list_widget.count()):
                item = self.cluster_list_widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole)["id"] == active_id:
                    self.cluster_list_widget.setCurrentItem(item)
                    found = True
                    break
                    
        # Если кластер исчез (мы перетащили из него последнюю фотографию)
        if not found and self.cluster_list_widget.count() > 0:
            self.cluster_list_widget.setCurrentRow(0)
            new_item = self.cluster_list_widget.currentItem()
            if new_item:
                self.active_cluster_id = new_item.data(Qt.ItemDataRole.UserRole)["id"]
                self._render_gallery(self.active_cluster_id) # Отрисовываем новый кластер
        elif not found:
            self.active_cluster_id = None
            self.image_list_widget.clear()
            self.gallery_label.setText("Галерея")

        # --- Снимаем блокировку сигналов ---
        self.cluster_list_widget.blockSignals(False)

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

        count = len(self.data_manager.get_files_for_cluster(dict(), cid))

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
        Открывает просмотрщик изображений (Smart Viewer).
        Определяет сценарий (Опознанное vs Неопознанное) и передает индекс.
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
        
        # Проверяем, является ли лицо "Опознанным"
        is_recognized = False
        rec_id = face.extra_data.get('matched_portrait_cluster_label')
        if rec_id is None:
            rec_id = face.cluster_label
            
        if rec_id is not None and str(rec_id) not in ("-1", "trash", "None"):
            is_recognized = True

        # Если лицо опознано, передаем его индекс во Viewer (Сценарий 1)
        # Если не опознано, передаем None, чтобы Viewer отрисовал все лица на фото (Сценарий 2)
        target_idx = face_idx if is_recognized else None
        
        ImageViewer(self.data_manager, fname, parent=self, target_face_index=target_idx).exec()


    @Slot(str, str, list)
    def _handle_drop(self, source_id, target_id, filenames):
        """
        Обработчик сигнала сброса.
        Используем таймер, чтобы дать UI время завершить визуальную операцию Drag&Drop
        перед тем, как начинать тяжелую обработку и показывать диалоги.
        """
        QTimer.singleShot(30, lambda: self._process_drop_logic(source_id, target_id, filenames))

    def _process_drop_logic(self, source_id, target_id, raw_filenames):
        """
        Основная логика обработки перемещения (вынесена из _handle_drop).
        """
        target_data = self._get_cluster_item_data_by_id(target_id)
        target_name = target_data["name"] if target_data else ""
        
        face_selection = {}
        valid_files =[]

        # --- ИСПРАВЛЕНИЕ: Нормализация путей и извлечение индексов лиц ---
        filenames =[]
        parsed_indices = {}
        for raw_fname in raw_filenames:
            if "::" in raw_fname:
                fname, idx_str = raw_fname.split("::", 1)
                idx = int(idx_str)
                if fname not in parsed_indices:
                    parsed_indices[fname] = []
                parsed_indices[fname].append(idx)
            else:
                fname = raw_fname
                
            if fname not in filenames:
                filenames.append(fname)

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
                    candidates =[]
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
                
                # --- ИСПРАВЛЕНИЕ: Используем массив индексов, переданный через Drag&Drop ---
                if fname in parsed_indices:
                    # Теперь face_selection хранит СПИСОК индексов для этого файла
                    face_selection[fname] = parsed_indices[fname] 
                    valid_files.append(fname)
                else:
                    # Старый fallback на всякий случай
                    target_idx = -1
                    for i, f in enumerate(record.faces):
                        current_sid = "trash" if f.is_trash else str(f.temp_cluster_label)
                        if current_sid == source_id:
                            target_idx = i
                            break
                    if target_idx != -1:
                        face_selection[fname] = [target_idx]
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
                dict(), target_id, target_name, valid_files, face_selection
            )
            self._refresh_left_panel()
            
            if self.active_cluster_id == source_id:
                # ИЗМЕНЕНО: Добавлен preserve_state
                self._render_gallery(source_id, preserve_state=True) 
                
            if self.mode == 'matches' and self.active_cluster_id == target_id:
                # ИЗМЕНЕНО: Добавлен preserve_state
                self._render_gallery(target_id, preserve_state=True)


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

    def _render_gallery(self, cluster_id: str, preserve_state: bool = False):
        saved_scroll = 0
        saved_row = 0
        
        # --- ДОБАВЛЕНО: Запоминаем текущий скролл и строку перед очисткой ---
        if preserve_state and self.image_list_widget.count() > 0:
            saved_scroll = self.image_list_widget.verticalScrollBar().value()
            selected = self.image_list_widget.selectedItems()
            if selected:
                # Если выделено несколько, берем верхний, чтобы после удаления оказаться на "следующем"
                saved_row = min([self.image_list_widget.row(i) for i in selected])
            else:
                saved_row = self.image_list_widget.currentRow()
                if saved_row < 0: saved_row = 0

        self._stop_loader()
        
        if not preserve_state: # Очищаем только если мы не пытаемся сохранить текущее состояние (например, после удаления фото)
            self.image_pixmap_cache.clear()       
        
        
        cdata = self._get_cluster_item_data_by_id(cluster_id)
        if not cdata: return
        
        # --- 1. Читаем текущие состояния кнопок и полей фильтров ---
        has_filters = self.filter_manager.has_active_filters()
        
        # Очищаем виджеты
        self.image_list_widget.clear()
        self.gallery_items_map.clear()

        filenames = self.data_manager.get_files_for_cluster(dict(), cluster_id)
        if not filenames: 
            self.gallery_label.setText(f"Галерея: {cdata['name']} (0 фото)")
            return

        tasks =[]
        placeholder = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        placeholder.fill(QColor("#3e3e3e")) 
        
        visible_count = 0
        
        # --- 2. Перебираем файлы и создаем только те, что прошли фильтр ---
        for fname in filenames:
            record = self.data_manager.records.get(fname)
            if not record: continue
            
            full_path = self.working_images_dir / fname
            current_keys =[]
            
            if self.mode == 'cleaning':
                target_faces =[]
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
                
                user_data = {"filename": display_name, "overlays": list()}
                if face_idx is not None:
                    user_data["face_index"] = face_idx 
                
                face_for_icon = None
                if face_idx is not None and face_idx < len(record.faces):
                    face_for_icon = record.faces[face_idx]
                elif record.face_count == 1 and record.faces:
                    face_for_icon = record.faces[0]
                    
                if face_for_icon:
                    gender = face_for_icon.extra_data.get('gender_faceonnx')
                    if gender == 'Male':
                        user_data["overlays"].append("GENDER_MALE")
                    elif gender == 'Female':
                        user_data["overlays"].append("GENDER_FEMALE")
                        
                    eye_left = face_for_icon.extra_data.get('eye_left_state')
                    eye_right = face_for_icon.extra_data.get('eye_right_state')
                    if eye_left == 'Closed' or eye_right == 'Closed':
                        user_data["overlays"].append("EYE_CLOSED")
                        
                    kp_analysis = face_for_icon.extra_data.get('keypoint_analysis', dict())
                    mouth_state = kp_analysis.get('mouth_state')
                    if mouth_state in ("open", "slightly_open", "wide_open"):
                        user_data["overlays"].append("MOUTH_OPEN")

                    beauty = face_for_icon.extra_data.get('beauty_faceonnx')
                    if beauty is not None:
                        try:
                            user_data["beauty_score"] = int(float(beauty))
                        except (ValueError, TypeError):
                            pass
                
                user_data["face_count"] = record.face_count
                
                # --- 3. ЛОГИКА ФИЛЬТРАЦИИ НА ЭТАПЕ СОЗДАНИЯ ---
                if has_filters and not self.filter_manager.passes(user_data):
                    continue # Элемент не прошел фильтр -> пропускаем
                
                visible_count += 1

                # Добавляем только прошедшие элементы (используем стандартный DecorationRole)
                item = QListWidgetItem(display_name)
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

        # Запускаем лоадер (он загрузит только видимые фото!)
        if tasks:
            self._start_loader(tasks)

        # --- 4. Обновляем заголовок ---
        if has_filters:
            self.gallery_label.setText(f"Галерея: {cdata['name']} (Показано {visible_count} из {cdata['count']})")
        else:
            self.gallery_label.setText(f"Галерея: {cdata['name']} ({cdata['count']} фото)")

        # --- 5. УМНЫЙ АВТОВЫБОР С УЧЕТОМ СОХРАНЕНИЯ ПОЗИЦИИ ---
        if self.image_list_widget.count() == 0:
            self.image_list_widget.setCurrentItem(None)
        else:
            target_row = 0
            if preserve_state:
                # Корректируем цель, если элементов стало меньше
                target_row = min(saved_row, self.image_list_widget.count() - 1)
                
            visible_found = False
            
            # Ищем вниз первую видимую ячейку
            for i in range(target_row, self.image_list_widget.count()):
                if not self.image_list_widget.item(i).isHidden():
                    self.image_list_widget.setCurrentRow(i)
                    visible_found = True
                    break
            
            # Если внизу ничего нет (например, перетащили последние элементы списка), ищем вверх
            if not visible_found and preserve_state:
                for i in range(target_row - 1, -1, -1):
                    if not self.image_list_widget.item(i).isHidden():
                        self.image_list_widget.setCurrentRow(i)
                        visible_found = True
                        break
                        
            if not visible_found:
                self.image_list_widget.setCurrentItem(None)
                
            # Восстанавливаем прокрутку с микро-задержкой (чтобы Qt успел расставить элементы по сетке)
            if preserve_state:
                QTimer.singleShot(10, lambda: self.image_list_widget.verticalScrollBar().setValue(saved_scroll))

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
        worker = getattr(self, 'loader_worker', None)
        if worker:
            try:
                worker.finished.disconnect(self._on_loader_finished)
            except Exception:
                pass
            worker.requestInterruption()
            
        thread = getattr(self, 'loader_thread', None)
        if thread:
            if thread.isRunning():
                thread.quit()
                thread.wait() # Гарантированно ждем завершения C++ потока
            thread.deleteLater() # Безопасное удаление объекта в движке Qt
            
        self.loader_thread = None
        self.loader_worker = None
        
        if hasattr(self, 'status_bar'):
            self.status_bar.reset()

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

        # --- Автовыбор первого лица в правой панели ---
        if hasattr(self, 'face_details_widget') and self.face_details_widget.count() > 0:
            first_face_item = self.face_details_widget.item(0)
            self.face_details_widget.setCurrentItem(first_face_item)
            # Принудительно вызываем обновление информации о лице (т.к. программный выбор не эмулирует клик мыши)
            self._on_face_item_clicked(first_face_item)            

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
        age = face.extra_data.get('age_faceonnx')
        gender = face.extra_data.get('gender_faceonnx')
        if age is not None: html += f"<tr><td><b>Возраст (AI):</b></td><td>{age}</td></tr>"
        if gender is not None:     
            # Добавлена визуализация пола из FaceONNX
            html += f"<tr><td><b>Пол (FaceONNX):</b></td><td>{gender}</td></tr>"

        # --- ДОБАВЛЕНО: Beauty в правую панель ---
        beauty_score = face.extra_data.get('beauty_faceonnx')
        if beauty_score is not None:
             try:
                 html += f"<tr><td><b>Красота (Beauty):</b></td><td>{int(float(beauty_score))}</td></tr>"
             except (ValueError, TypeError):
                 pass

        # --- ДОБАВЛЕНО: Глаза и рот ---
        eye_left = face.extra_data.get('eye_left_state')
        eye_right = face.extra_data.get('eye_right_state')
        kp_analysis = face.extra_data.get('keypoint_analysis', dict())
        mouth_state = kp_analysis.get('mouth_state')

        if eye_left or eye_right:
            html += f"<tr><td><b>Глаза (Л / П):</b></td><td>{eye_left or '?'} / {eye_right or '?'}</td></tr>"
        if mouth_state:
            html += f"<tr><td><b>Рот:</b></td><td>{mouth_state}</td></tr>"

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
    def _open_image_viewer(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        fname = data["filename"]
        
        # --- ЛОГИКА ДЛЯ CLEANING (Оставляем рамки) ---
        if self.mode == 'cleaning':
            # В cleaning открываем с выделением конкретного лица
            face_idx = data.get("face_index")
            ImageViewer(self.data_manager, fname, parent=self, target_face_index=face_idx, draw_boxes=True).exec()
            return

        # --- СТАНДАРТНАЯ ЛОГИКА (Галерея) ---
        # Открываем чистое фото БЕЗ рамок для детального рассмотрения
        ImageViewer(self.data_manager, fname, parent=self, draw_boxes=False).exec()

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
                    "output_path": self.export_dir / cname / fname,
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
        self.status_bar.reset()
        self.status_bar.setFormat("")
     
        QMessageBox.information(self, "Экспорт завершен", message)
        self.export_end = True

        thread = getattr(self, 'export_thread', None)
        if thread: 
            if thread.isRunning():
                thread.quit()
                thread.wait()
            thread.deleteLater()
            self.export_thread = None

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
        # --- ДОБАВЛЕНО: Сохранение состояния окна и сплиттеров ---
        if IS_MANAGED_RUN and pysm_context and self.win_state_var_name and WindowStateManager:
            mode_var_name = f"{self.win_state_var_name}.{self.mode}"
            window_state = WindowStateManager.save_state(
                window=self,
                splitters={'main': self.main_splitter}
            )
            pysm_context.set_structured(mode_var_name, window_state)
        # --- ИСПРАВЛЕНИЕ: Останавливаем загрузчик фото перед выходом ---
        self._stop_loader()
      
        # --- ИСПРАВЛЕНИЕ: Безопасное завершение экспорта (предотвращает зомби-процессы) ---
        if hasattr(self, 'export_thread') and self.export_thread and self.export_thread.isRunning():
            reply = QMessageBox.question(self, "Прерывание", 
                                         "В данный момент выполняется экспорт фотографий.\nПрервать процесс и закрыть программу?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            else:
                if hasattr(self, 'export_worker') and self.export_worker:
                    self.export_worker.requestInterruption()
                self.export_thread.quit()
                self.export_thread.wait()
                self.export_thread.deleteLater() # <--- ДОБАВЛЕНО
                self.export_thread = None                 

        # Стандартная обработка несохраненных изменений
        if self.data_manager.has_changes():
            reply = QMessageBox.question(self, "Выход", "Сохранить изменения перед выходом?", 
                                         QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Save:
                if self._save_changes(silent=True):
                    self._log_final_report()
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.StandardButton.Discard:
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
    parser.add_argument(f"--{p}export_dir", type=str, default=None, help="Папка для экспорта фотографий с водяными знаками")
    parser.add_argument(f"--{p}win_state_var_name", type=str, default="", help="Имя переменной контекста для сохранения состояния окна")
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
        e_dir_str = getattr(cli_config, f"{arg_prefix}export_dir")  
        win_var = getattr(cli_config, f"{arg_prefix}win_state_var_name", "")        
        if not w_dir.exists(): raise FileNotFoundError(f"Нет папки: {w_dir}")

        num_workers = cli_config.all_threads or (os.cpu_count() or 8)    

        window = MainWindow(w_dir, r_dir, cli_config.mode, num_workers, e_dir_str, win_var)
        window.show()
     
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Ошибка запуска: {e}", exc_info=True)
        QMessageBox.critical(None, "Ошибка", str(e))
        sys.exit(1)