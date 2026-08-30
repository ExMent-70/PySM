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
import html as html_module
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
)
from PySide6.QtGui import QPixmap, QColor, QPainter, QPen
from PySide6.QtCore import Qt, Slot, QThread, QSize, QTimer

IS_MANAGED_RUN = False
try:
    current_script_dir = Path(__file__).resolve().parent
    if str(current_script_dir) not in sys.path: sys.path.insert(0, str(current_script_dir))
    face_analysis_root = current_script_dir.parent
    if str(face_analysis_root) not in sys.path:
        sys.path.insert(0, str(face_analysis_root))
    project_root = next(
        (
            parent
            for parent in current_script_dir.parents
            if (parent / "pysm_lib").is_dir()
        ),
        None,
    )
    if project_root is None:
        raise ImportError("Не найден корень PySM с папкой pysm_lib")
    if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder
    from pysm_lib.window_state_manager import WindowStateManager
    from pysm_lib.pysm_image_cache import AsyncImageResult, ImageRequest
    
    IS_MANAGED_RUN = True

    from _lib.editor_viewer import ImageViewer
    from _lib.editor_workers import DataLoadWorker
    from _lib.export_controller import ExportController
    from _lib.editor_delegates import (
        FACE_PIXMAP_ROLE,
        FACE_STATUS_COLOR_ROLE,
        PREVIEW_SIZE,
        THUMBNAIL_SIZE,
        ClusterItemDelegate,
        ImageItemDelegate,
    )
    from _lib.editor_dialogs import EnhanceSettingsDialog, FaceSelectorDialog
    from _lib.data_manager import ClusterDataManager
    from _lib.editor_ui import EditorUIBuilder
    from _lib.editor_filters import GalleryFilterManager
    from _lib.editor_menus import EditorMenuManager
    from _lib.photo_selection_filter import (
        extract_photo_numbers,
        load_selected_photo_numbers,
    )
    from _lib.image_requests import face_thumbnail_request, normalized_face_crop
    from _lib.image_pipeline import ImagePipelineController

except ImportError as e:
    print(f"Критическая ошибка импорта внутренних модулей: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)


def _safe_folder_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().rstrip(".")
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update({f"COM{i}" for i in range(1, 10)})
    reserved.update({f"LPT{i}" for i in range(1, 10)})
    if cleaned.split(".", 1)[0].upper() in reserved:
        cleaned = f"_{cleaned}"
    return cleaned or fallback


def _safe_export_path(root: Path, folder_name: str, filename: str) -> Path:
    """Build an export path and reject every escape from ``root``."""

    resolved_root = root.resolve()
    candidate = (resolved_root / folder_name / filename).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"Путь экспорта выходит за пределы каталога: {filename!r}")
    return candidate


def _export_folder_name(display_name: str, stable_id: object, fallback: str) -> str:
    """Keep human-readable export folders unique by their persistent ID."""

    return _safe_folder_name(f"{display_name} [{stable_id}]", fallback)


class MainWindow(QMainWindow):
    _GALLERY_PROGRESS_MAX = 1000
    _GALLERY_BUILD_PROGRESS_MAX = 200
    
    def __init__(self, working_dir: Path, reference_dir: Optional[Path], mode: str,
                 num_workers: int, export_dir: str, win_state_var_name: str,
                 student_list_file: Optional[Path] = None):
        super().__init__()
        self.mode = mode
        self.num_workers = num_workers
        self.working_dir = working_dir
        self.student_list_file = student_list_file
        
        self.win_state_var_name = win_state_var_name

        self.reference_dir = reference_dir if reference_dir else working_dir
        
        self.working_images_dir = self.working_dir / "JPG"
        self.reference_images_dir = self.reference_dir / "JPG"
        
        self.session_name = working_dir.parent.parent.name 
        self.photo_session = working_dir.name.replace("Analysis_", "")
       
        self._export_base_is_explicit = bool(export_dir)
        self.export_base_dir = (
            Path(export_dir) if export_dir else self.working_dir.parent / self.session_name
        )
        self.export_dir = self.export_base_dir / f"Выбор_Фото_{self.photo_session}_{self.mode}"
        self.export_end = False
        self._close_after_export = False
        self.data_load_thread = None
        self.data_load_worker = None
        self._pending_session_switch = None

        # 1. Инициализация Data Manager (здесь же создается Strategy)
        self.data_manager = ClusterDataManager(
            self.working_dir,
            self.reference_dir,
            mode=mode,
            student_list_file=student_list_file,
        )
        
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
        self.selected_photo_numbers: Optional[set[str]] = None
        # Быстрый поиск видимого элемента по ключу производного изображения.
        self.gallery_items_map: Dict[str, QListWidgetItem] = {}
        self.image_cache = None
        self.image_loader = None
        self.image_pipeline = ImagePipelineController(self)
        self.image_pipeline.image_ready.connect(self._on_gallery_image_ready)
        self._gallery_generation = 0
        self._gallery_thumbnail_channels: Dict[tuple[object, ...], Dict[str, Any]] = {}
        self._gallery_total_tasks = 0
        self._gallery_completed_tasks = 0
        self._cluster_cover_generation = 0
        self._cluster_cover_channels: Dict[tuple[object, ...], Dict[str, Any]] = {}
        self._face_panel_generation = 0
        self._face_panel_channels: Dict[tuple[object, ...], Dict[str, Any]] = {}
        self._defer_initial_gallery = True
        self._pending_initial_cluster_id: Optional[str] = None
        self._gallery_build_generation = 0
        self._gallery_build_state: Optional[Dict[str, Any]] = None
        self._gallery_build_batch_size = 80

        self.cluster_delegate = ClusterItemDelegate(parent=self)
        self.image_delegate = ImageItemDelegate(parent=self)

        self.menu_manager = EditorMenuManager(self)
        self.filter_manager = GalleryFilterManager(self)
        
        EditorUIBuilder.build_ui(self)
        self.filter_manager.bind_ui()
        self.export_controller = ExportController(self)
        self.export_controller.progress_updated.connect(self.status_bar.setValue)
        self.export_controller.finished.connect(self._on_export_finished)
        self.export_controller.stopped.connect(self._on_export_thread_stopped)
        self._reset_image_pipeline()
        try:
            screen_geometry = self.screen().geometry()
            window_geometry = self.frameGeometry()
            window_geometry.moveCenter(screen_geometry.center())
            self.move(window_geometry.topLeft())
        except Exception:
            pass

        
        QTimer.singleShot(0, self._start_initial_data_load)

    def _reset_image_pipeline(self) -> None:
        image_cache_root = self.working_dir / ".thumbnails" / "cluster_editor-v1"
        self.image_cache, self.image_loader = self.image_pipeline.reset(
            image_cache_root,
            self.num_workers,
        )

    def begin_working_session_switch(self, new_json_path: Path) -> tuple[bool, str]:
        """Validate another matches session off the GUI thread before adoption."""

        if self.data_load_thread is not None and self.data_load_thread.isRunning():
            return False, "Уже выполняется загрузка данных."
        new_working_dir = new_json_path.parent
        try:
            candidate = ClusterDataManager(
                new_working_dir,
                self.reference_dir,
                mode=self.mode,
                student_list_file=self.student_list_file,
            )
            candidate.switch_working_session(new_json_path)
        except Exception as exc:
            return False, str(exc)

        self._pending_session_switch = (candidate, new_json_path)
        self.centralWidget().setEnabled(False)
        self.gallery_label.setText("Загрузка новой сессии...")
        self._start_data_load(candidate, self._on_session_data_loaded)
        return True, ""

    @Slot(bool, str)
    def _on_session_data_loaded(self, success: bool, message: str) -> None:
        pending = self._pending_session_switch
        self._pending_session_switch = None
        self.centralWidget().setEnabled(True)
        if pending is None:
            return
        candidate, new_json_path = pending
        if not success:
            QMessageBox.critical(self, "Ошибка смены сессии", message)
            if self.active_cluster_id:
                self._render_gallery(self.active_cluster_id, preserve_state=True)
            return

        self._adopt_working_session(candidate, new_json_path)

    def _adopt_working_session(
        self,
        candidate: ClusterDataManager,
        new_json_path: Path,
    ) -> None:
        new_working_dir = new_json_path.parent

        self._stop_loader()
        self._cancel_cluster_cover_requests()
        self.data_manager = candidate
        self.working_dir = new_working_dir
        self.working_images_dir = self.working_dir / "JPG"
        self.session_name = self.working_dir.parent.parent.name
        self.photo_session = self.working_dir.name.replace("Analysis_", "")
        if not self._export_base_is_explicit:
            self.export_base_dir = self.working_dir.parent / self.session_name
        self.export_dir = (
            self.export_base_dir / f"Выбор_Фото_{self.photo_session}_{self.mode}"
        )
        self.active_cluster_id = None
        self.gallery_items_map.clear()
        self.search_bar.clear()
        self._reset_image_pipeline()
        self.setWindowTitle(self.data_manager.strategy.get_window_title(self.photo_session))
        if hasattr(self, "cluster_list_title"):
            self.cluster_list_title.setText(f"{self.photo_session}: Эталоны (Портреты)")
        self._reload_selected_photo_numbers(
            self.btn_filter_selected_photos.isChecked()
        )
        self._refresh_left_panel()

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

    def _start_initial_data_load(self) -> None:
        self.gallery_label.setText("Загрузка данных...")
        self._start_data_load(self.data_manager, self._on_initial_data_loaded)

    def _start_data_load(self, data_manager, result_slot) -> None:
        self.data_load_thread = QThread(self)
        self.data_load_worker = DataLoadWorker(data_manager)
        self.data_load_worker.moveToThread(self.data_load_thread)
        self.data_load_thread.started.connect(self.data_load_worker.run)
        self.data_load_worker.finished.connect(result_slot)
        self.data_load_worker.finished.connect(self.data_load_thread.quit)
        self.data_load_worker.finished.connect(self.data_load_worker.deleteLater)
        self.data_load_thread.finished.connect(self._on_data_load_thread_stopped)
        self.data_load_thread.finished.connect(self.data_load_thread.deleteLater)
        self.data_load_thread.start()

    @Slot(bool, str)
    def _on_initial_data_loaded(self, success: bool, msg: str) -> None:
        if not success:
            QMessageBox.critical(self, "Ошибка загрузки", msg)
            self.gallery_label.setText("Ошибка загрузки данных")
            return
        
        # Контракт обложек локаций хранится в структурированной переменной:
        # sys_location_name.{photo_session}
        if self.mode == 'location' and IS_MANAGED_RUN:
            var_name = f"sys_location_name.{self.photo_session}"
            covers_data = pysm_context.get_structured(var_name)
            if covers_data and isinstance(covers_data, dict):
                self.data_manager.ingest_location_covers(covers_data)
        
        self._refresh_left_panel()
        if self._pending_initial_cluster_id:
            QTimer.singleShot(0, self._activate_initial_cluster)

    @Slot()
    def _on_data_load_thread_stopped(self) -> None:
        self.data_load_thread = None
        self.data_load_worker = None

    def _refresh_left_panel(self):
        active_id = self.active_cluster_id
        
        # Сигналы блокируются, чтобы clear() не сбросил активную галерею.
        self.cluster_list_widget.blockSignals(True)
        self._cancel_cluster_cover_requests()
        
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
                selected_id = new_item.data(Qt.ItemDataRole.UserRole)["id"]
                if self._defer_initial_gallery:
                    self._pending_initial_cluster_id = selected_id
                    self.active_cluster_id = None
                else:
                    self.active_cluster_id = selected_id
                    self._render_gallery(self.active_cluster_id) # Отрисовываем новый кластер
        elif not found:
            self.active_cluster_id = None
            self.image_list_widget.clear()
            self.gallery_label.setText("Галерея")

        # --- Снимаем блокировку сигналов ---
        self.cluster_list_widget.blockSignals(False)

    def _activate_initial_cluster(self):
        self._defer_initial_gallery = False
        cluster_id = self._pending_initial_cluster_id
        self._pending_initial_cluster_id = None
        if not cluster_id:
            return

        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if item_data and item_data.get("id") == cluster_id:
                self.cluster_list_widget.blockSignals(True)
                self.cluster_list_widget.setCurrentItem(item)
                self.cluster_list_widget.blockSignals(False)
                self.active_cluster_id = cluster_id
                self._render_gallery(cluster_id)
                return

    def _add_cluster_item(self, cid: str, name: str, faces: List, is_special: bool = False):
        pixmap = QPixmap()
        fname = None
        best_face = None

        # Источник обложки зависит от режима редактора.
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

        count = len(self.data_manager.get_files_for_cluster(dict(), cid))

        item_data = {
            "id": cid, "name": name, "count": count, "pixmap": pixmap,
            "is_changed": self.data_manager.is_cluster_changed(self.mode, cid),
            "student_id": faces[0].student_id if faces else None,
        }
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, item_data)
        
        if is_special:
            if cid == "trash": item.setBackground(QColor("#fff0f0"))
            if cid == "error_matches": item.setBackground(QColor("#fff8e1"))
            
        self.cluster_list_widget.addItem(item)
        if fname:
            task = {
                "full_path": self._get_image_path(fname),
                "target_size": (PREVIEW_SIZE, PREVIEW_SIZE),
                "variant": "cluster_editor.cluster_cover.v1",
            }
            if self.mode == 'cleaning' and best_face:
                task["bbox"] = best_face.bbox
                task["crop_padding"] = 0.4
            self._request_cluster_cover(item, cid, task)


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
        
        ImageViewer(
            self.data_manager,
            fname,
            parent=self,
            target_face_index=target_idx,
            image_cache=self.image_cache,
            image_loader=self.image_loader,
        ).exec()


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
        target_display_name = target_data["name"] if target_data else ""
        target_name = target_display_name
        if self.mode in {"face", "matches"} and target_data:
            target_name = target_data.get("student_id") or ""
        
        face_selection = {}
        valid_files =[]

        # Элемент cleaning кодирует индекс лица после имени файла.
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
                        
                        dlg = FaceSelectorDialog(
                            full_path,
                            faces_to_show,
                            self,
                            f"Кто на фото - <b>{html_module.escape(target_display_name)}</b>?<br>(Показаны только неопознанные)",
                            image_cache=self.image_cache,
                            image_loader=self.image_loader,
                        )
                        
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
                
                if fname in parsed_indices:
                    face_selection[fname] = parsed_indices[fname] 
                    valid_files.append(fname)
                else:
                    # Совместимость с DnD-данными без индекса лица.
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
                    dlg = FaceSelectorDialog(
                        full_path,
                        record.faces,
                        self,
                        image_cache=self.image_cache,
                        image_loader=self.image_loader,
                    )
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
        # Специфичное подтверждение для Cleaning
        if self.mode == 'cleaning':
            if QMessageBox.warning(self, "Подтверждение очистки", 
                                   "Внимание! Все лица и файлы, находящиеся в 'Корзине', будут удалены БЕЗВОЗВРАТНО.\nПродолжить?",
                                   QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return False
            self._stop_loader()
        
        # Единая точка сохранения
        if self.data_manager.save_data():
            # Обновление контекста (если нужно для Legacy)
            if self.mode == 'location' and IS_MANAGED_RUN:
                if not self._update_pysm_context():
                    details = self.data_manager.last_error or "Не удалось сохранить обложки локаций."
                    QMessageBox.critical(self, "Ошибка сохранения", details)
                    return False
                self.data_manager.mark_location_covers_saved()
                
            if not silent:
                msg = "Мусор удален, данные обновлены." if self.mode == 'cleaning' else "Сохранено."
                QMessageBox.information(self, "Успех", msg)
            
            self._refresh_left_panel() # Перезагрузка UI (важно для Cleaning, чтобы убрать удаленное)
            return True
        else:
            details = self.data_manager.last_error or "Не удалось сохранить данные."
            QMessageBox.critical(self, "Ошибка сохранения", details)
            if self.mode == "cleaning" and self.active_cluster_id:
                self._render_gallery(self.active_cluster_id, preserve_state=True)
            return False

    # --- UI Helpers ---
    
    @Slot(int)
    def _on_face_size_changed(self, value: int):
        if hasattr(self, 'face_details_widget'):
            self.face_details_widget.setIconSize(QSize(value, value))
            self.face_details_widget.setGridSize(QSize(value + 20, value + 60))

    @Slot(bool)
    def _on_selected_photos_toggled(self, checked: bool):
        """Перезагружает общий выбор и обновляет текущую галерею."""

        self._reload_selected_photo_numbers(checked)

        if self.active_cluster_id is not None:
            self._render_gallery(self.active_cluster_id)

    def _reload_selected_photo_numbers(self, checked: bool) -> None:
        """Обновляет данные фильтра для текущей папки анализа."""

        self.selected_photo_numbers = None
        tooltip = "Только фотографии, выбранные пользователями"
        self.btn_filter_selected_photos.setToolTip(tooltip)
        if checked:
            selection_path = self.working_dir / "photo_selection.json"
            try:
                self.selected_photo_numbers = load_selected_photo_numbers(selection_path)
            except ValueError as exc:
                QMessageBox.warning(self, "Фильтр выбранных фотографий", str(exc))
                self.btn_filter_selected_photos.blockSignals(True)
                self.btn_filter_selected_photos.setChecked(False)
                self.btn_filter_selected_photos.blockSignals(False)
                return
            if self.selected_photo_numbers is None:
                self.btn_filter_selected_photos.setToolTip(
                    f"{tooltip}\nФайл не найден: {selection_path}"
                )
            else:
                self.btn_filter_selected_photos.setToolTip(
                    f"{tooltip}\n"
                    f"Загружено уникальных номеров: {len(self.selected_photo_numbers)}"
                )

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
        if preserve_state and self.image_list_widget.count() > 0:
            saved_scroll = self.image_list_widget.verticalScrollBar().value()
            selected = self.image_list_widget.selectedItems()
            if selected:
                saved_row = min([self.image_list_widget.row(item) for item in selected])
            else:
                saved_row = max(0, self.image_list_widget.currentRow())

        self._stop_loader()
        cdata = self._get_cluster_item_data_by_id(cluster_id)
        if not cdata:
            return

        has_gallery_filters = self.filter_manager.has_active_filters()
        has_selection_filter = (
            self.btn_filter_selected_photos.isChecked()
            and self.selected_photo_numbers is not None
        )
        has_filters = has_gallery_filters or has_selection_filter

        self.image_list_widget.clear()
        self.gallery_items_map.clear()

        filenames = self.data_manager.get_files_for_cluster(dict(), cluster_id)
        if not filenames:
            self.gallery_label.setText(f"Галерея: {cdata['name']} (0 фото)")
            return

        self._begin_gallery_loader()
        self._gallery_build_generation += 1
        generation = self._gallery_build_generation
        self._gallery_build_state = {
            "generation": generation,
            "cluster_id": cluster_id,
            "cdata": cdata,
            "filenames": filenames,
            "index": 0,
            "visible_count": 0,
            "has_gallery_filters": has_gallery_filters,
            "has_selection_filter": has_selection_filter,
            "has_filters": has_filters,
            "preserve_state": preserve_state,
            "saved_scroll": saved_scroll,
            "saved_row": saved_row,
            "placeholder": self._create_gallery_placeholder(),
        }
        self.gallery_label.setText(f"Галерея: {cdata['name']} (загрузка...)")
        QTimer.singleShot(0, self._process_gallery_build_batch)

    def _create_gallery_placeholder(self) -> QPixmap:
        placeholder = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        placeholder.fill(QColor("#3e3e3e"))
        return placeholder

    def _process_gallery_build_batch(self):
        state = self._gallery_build_state
        if not state:
            return
        if state["generation"] != self._gallery_build_generation:
            return

        filenames = state["filenames"]
        start_index = state["index"]
        end_index = min(start_index + self._gallery_build_batch_size, len(filenames))
        batch_tasks = []

        for fname in filenames[start_index:end_index]:
            batch_tasks.extend(self._add_gallery_file_items(state, fname))

        state["index"] = end_index
        self._update_gallery_progress()
        if batch_tasks:
            self._queue_gallery_tasks(batch_tasks)

        if end_index < len(filenames):
            QTimer.singleShot(0, self._process_gallery_build_batch)
            return

        self._finish_gallery_build(state)

    def _add_gallery_file_items(self, state: Dict[str, Any], fname: str) -> List[Dict[str, Any]]:
        if state["has_selection_filter"] and not (
            extract_photo_numbers(fname) & self.selected_photo_numbers
        ):
            return []

        record = self.data_manager.records.get(fname)
        if not record:
            return []

        full_path = self._get_image_path(fname)
        current_keys = []
        if self.mode == 'cleaning':
            target_faces = []
            for i, face in enumerate(record.faces):
                if state["cluster_id"] == "trash":
                    if face.is_trash:
                        target_faces.append(i)
                else:
                    if str(face.temp_cluster_label) == state["cluster_id"]:
                        target_faces.append(i)
            for idx in target_faces:
                current_keys.append((f"{fname}::{idx}", idx))
        else:
            current_keys.append((fname, None))

        tasks = []
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
            if state["has_gallery_filters"] and not self.filter_manager.passes(user_data):
                continue

            state["visible_count"] += 1
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, user_data)

            item.setData(Qt.ItemDataRole.DecorationRole, state["placeholder"])
            task = {
                "filename": fname,
                "cache_key": cache_key,
                "full_path": full_path,
                "source_size": self._source_size_from_record(record),
            }
            if face_idx is not None:
                task["bbox"] = record.faces[face_idx].bbox
                task["draw_face_rect"] = True
            tasks.append(task)

            self.image_list_widget.addItem(item)
            self.gallery_items_map[cache_key] = item

        return tasks

    @staticmethod
    def _source_size_from_record(record) -> Optional[tuple[int, int]]:
        shape = getattr(record, "original_shape", None)
        if not shape or len(shape) < 2:
            return None
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            return None
        return width, height

    def _finish_gallery_build(self, state: Dict[str, Any]):
        if state["generation"] != self._gallery_build_generation:
            return
        self._gallery_build_state = None

        if hasattr(self, "status_bar") and not self._export_is_running():
            if self._gallery_thumbnail_channels:
                self._update_gallery_progress()
            else:
                self.status_bar.reset()

        cdata = state["cdata"]
        visible_count = state["visible_count"]
        if state["has_filters"]:
            self.gallery_label.setText(f"Галерея: {cdata['name']} (Показано {visible_count} из {cdata['count']})")
        else:
            self.gallery_label.setText(f"Галерея: {cdata['name']} ({cdata['count']} фото)")

        if self.image_list_widget.count() == 0:
            self.image_list_widget.setCurrentItem(None)
        else:
            target_row = 0
            if state["preserve_state"]:
                target_row = min(state["saved_row"], self.image_list_widget.count() - 1)

            visible_found = False
            for i in range(target_row, self.image_list_widget.count()):
                if not self.image_list_widget.item(i).isHidden():
                    self.image_list_widget.setCurrentRow(i)
                    visible_found = True
                    break

            if not visible_found and state["preserve_state"]:
                for i in range(target_row - 1, -1, -1):
                    if not self.image_list_widget.item(i).isHidden():
                        self.image_list_widget.setCurrentRow(i)
                        visible_found = True
                        break

            if not visible_found:
                self.image_list_widget.setCurrentItem(None)

            if state["preserve_state"]:
                QTimer.singleShot(
                    10,
                    lambda: self.image_list_widget.verticalScrollBar().setValue(state["saved_scroll"]),
                )

    def _begin_gallery_loader(self):
        self._gallery_generation += 1
        self._gallery_completed_tasks = 0
        self._gallery_thumbnail_channels.clear()
        self._gallery_total_tasks = 0
        if hasattr(self, 'status_bar') and not self._export_is_running():
            self.status_bar.setRange(0, self._GALLERY_PROGRESS_MAX)
            self.status_bar.setValue(0)

    def _update_gallery_progress(self):
        if not hasattr(self, "status_bar") or self._export_is_running():
            return

        state = self._gallery_build_state
        if state is not None:
            total_files = max(1, len(state["filenames"]))
            processed_files = min(total_files, state["index"])
            value = round(
                self._GALLERY_BUILD_PROGRESS_MAX * processed_files / total_files
            )
        elif self._gallery_total_tasks > 0:
            completed_tasks = min(
                self._gallery_total_tasks,
                self._gallery_completed_tasks,
            )
            loading_range = (
                self._GALLERY_PROGRESS_MAX - self._GALLERY_BUILD_PROGRESS_MAX
            )
            value = self._GALLERY_BUILD_PROGRESS_MAX + round(
                loading_range * completed_tasks / self._gallery_total_tasks
            )
        else:
            value = self._GALLERY_PROGRESS_MAX

        self.status_bar.setValue(max(self.status_bar.value(), value))

    def _queue_gallery_tasks(self, tasks: List[Dict]):
        added = 0
        for task in tasks:
            request_data = self._gallery_request_for_task(task)
            if request_data is None:
                continue

            request, rect_norm = request_data
            cache_key = str(task.get("cache_key") or task.get("filename") or "")
            channel = ("cluster-gallery", id(self), self._gallery_generation, cache_key)
            self._gallery_thumbnail_channels[channel] = {
                "cache_key": cache_key,
                "rect_norm": rect_norm,
                "generation": self._gallery_generation,
            }
            self.image_loader.request(
                request,
                channel=channel,
                persist=True,
                disk_format="PNG",
            )
            added += 1

        self._gallery_total_tasks += added

    def _stop_loader(self):
        self._cancel_face_panel_requests()
        for channel in list(self._gallery_thumbnail_channels):
            self.image_loader.cancel(channel)
        self._gallery_thumbnail_channels.clear()
        self._gallery_generation += 1
        self._gallery_build_generation += 1
        self._gallery_build_state = None
        self._gallery_total_tasks = 0
        self._gallery_completed_tasks = 0
        
        if hasattr(self, 'status_bar') and not self._export_is_running():
            self.status_bar.reset()

    def _cancel_cluster_cover_requests(self) -> None:
        for channel in list(self._cluster_cover_channels):
            self.image_loader.cancel(channel)
        self._cluster_cover_channels.clear()
        self._cluster_cover_generation += 1

    def _request_cluster_cover(
        self,
        item: QListWidgetItem,
        cluster_id: str,
        task: Dict[str, Any],
    ) -> None:
        request_data = self._gallery_request_for_task(task)
        if request_data is None:
            return

        request, _ = request_data
        channel = ("cluster-cover", id(self), self._cluster_cover_generation, cluster_id)
        self._cluster_cover_channels[channel] = {
            "item": item,
            "generation": self._cluster_cover_generation,
        }
        self.image_loader.request(
            request,
            channel=channel,
            persist=True,
            disk_format="PNG",
        )

    def _gallery_request_for_task(
        self,
        task: Dict[str, Any],
    ) -> Optional[tuple[ImageRequest, Optional[tuple[float, float, float, float]]]]:
        full_path_value = task.get("full_path")
        if not full_path_value:
            return None
        full_path = Path(full_path_value)
        if not full_path.is_file():
            return None

        crop = None
        rect_norm = None
        bbox = task.get("bbox")
        if bbox and len(bbox) == 4:
            source_size = task.get("source_size")
            if source_size and len(source_size) >= 2:
                source_width, source_height = int(source_size[0]), int(source_size[1])
            else:
                source_width, source_height = self.image_cache.source_size(full_path)
            if source_width <= 0 or source_height <= 0:
                return None

            x1, y1, x2, y2 = map(int, bbox)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            x1 = max(0, min(source_width, x1))
            x2 = max(0, min(source_width, x2))
            y1 = max(0, min(source_height, y1))
            y2 = max(0, min(source_height, y2))
            pad_ratio = float(task.get("crop_padding", 0.5))
            crop = normalized_face_crop(
                (source_width, source_height),
                (x1, y1, x2, y2),
                padding=pad_ratio,
            )
            if crop is None:
                return None
            if task.get("draw_face_rect", False):
                crop_x1 = crop[0] * source_width
                crop_y1 = crop[1] * source_height
                crop_width = crop[2] * source_width
                crop_height = crop[3] * source_height
                rect_left = max(0.0, min(1.0, (x1 - crop_x1) / crop_width))
                rect_top = max(0.0, min(1.0, (y1 - crop_y1) / crop_height))
                rect_right = max(0.0, min(1.0, (x2 - crop_x1) / crop_width))
                rect_bottom = max(0.0, min(1.0, (y2 - crop_y1) / crop_height))
                rect_norm = (
                    rect_left,
                    rect_top,
                    max(0.0, rect_right - rect_left),
                    max(0.0, rect_bottom - rect_top),
                )

        target_size = tuple(task.get("target_size") or (THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        variant = str(task.get("variant") or "cluster_editor.gallery_thumbnail.v2")
        request = ImageRequest(
            full_path,
            target_size,
            mode="fit",
            crop=crop,
            allow_upscale=crop is not None,
            variant=variant,
        )
        return request, rect_norm

    def _draw_gallery_face_rect(
        self,
        pixmap: QPixmap,
        rect_norm: Optional[tuple[float, float, float, float]],
    ) -> QPixmap:
        if rect_norm is None or pixmap.isNull():
            return pixmap

        result = QPixmap(pixmap)
        painter = QPainter(result)
        pen = QPen(QColor(255, 165, 0))
        pen.setWidth(max(2, int(min(result.width(), result.height()) * 0.04)))
        painter.setPen(pen)
        x, y, width, height = rect_norm
        painter.drawRect(
            int(result.width() * x),
            int(result.height() * y),
            max(1, int(result.width() * width)),
            max(1, int(result.height() * height)),
        )
        painter.end()
        return result

    @Slot(object)
    def _on_gallery_image_ready(self, result: AsyncImageResult):
        channel = result.channel
        if (
            not isinstance(channel, tuple)
            or len(channel) != 4
            or channel[1] != id(self)
        ):
            return
        if channel[0] == "cluster-cover":
            self._on_cluster_cover_ready(result, channel)
            return
        if channel[0] == "face-panel":
            self._on_face_panel_image_ready(result, channel)
            return
        if channel[0] != "cluster-gallery":
            return

        task_data = self._gallery_thumbnail_channels.pop(channel, None)
        if task_data is None:
            return
        if task_data.get("generation") != self._gallery_generation:
            return

        self._gallery_completed_tasks += 1
        gallery_build_finished = self._gallery_build_state is None
        if gallery_build_finished:
            self._update_gallery_progress()

        if not result.image.isNull():
            pixmap = QPixmap.fromImage(result.image)
            pixmap = self._draw_gallery_face_rect(pixmap, task_data.get("rect_norm"))
            cache_key = task_data["cache_key"]
            item = self.gallery_items_map.get(cache_key)
            if item is not None:
                item.setData(Qt.ItemDataRole.DecorationRole, pixmap)

        if (
            gallery_build_finished
            and not self._gallery_thumbnail_channels
            and hasattr(self, "status_bar")
            and not self._export_is_running()
        ):
            self.status_bar.reset()

    def _on_cluster_cover_ready(self, result: AsyncImageResult, channel: tuple[object, ...]) -> None:
        task_data = self._cluster_cover_channels.pop(channel, None)
        if task_data is None:
            return
        if task_data.get("generation") != self._cluster_cover_generation:
            return
        if result.image.isNull():
            return

        item = task_data["item"]
        pixmap = QPixmap.fromImage(result.image)
        item_data = item.data(Qt.ItemDataRole.UserRole)
        if item_data is None:
            return
        item_data = dict(item_data)
        item_data["pixmap"] = pixmap
        item.setData(Qt.ItemDataRole.UserRole, item_data)
        list_widget = item.listWidget()
        if list_widget is not None:
            list_widget.viewport().update()

    def _cancel_face_panel_requests(self) -> None:
        for channel in list(self._face_panel_channels):
            self.image_loader.cancel(channel)
        self._face_panel_channels.clear()
        self._face_panel_generation += 1

    def _on_face_panel_image_ready(
        self,
        result: AsyncImageResult,
        channel: tuple[object, ...],
    ) -> None:
        task_data = self._face_panel_channels.pop(channel, None)
        if task_data is None:
            return
        if task_data["generation"] != self._face_panel_generation:
            return
        item = task_data["item"]
        if result.image.isNull() or item.listWidget() is None:
            return

        item.setData(FACE_PIXMAP_ROLE, QPixmap.fromImage(result.image))
        list_widget = item.listWidget()
        if list_widget is not None:
            list_widget.viewport().update(list_widget.visualItemRect(item))

    @Slot(QListWidgetItem)
    def _update_face_panel(self, current: QListWidgetItem, prev=None):
        """Обновляет правую панель при выборе фото в галерее."""
        if self.mode == 'cleaning':
            return

        self._cancel_face_panel_requests()
        if hasattr(self, 'face_details_widget'):
            self.face_details_widget.clear()
        if hasattr(self, 'photo_info_viewer'):
            self.photo_info_viewer.clear()
        if hasattr(self, 'face_info_viewer'):
            self.face_info_viewer.clear()
        if not current:
            return

        fname = current.data(Qt.ItemDataRole.UserRole)["filename"]
        record = self.data_manager.records.get(fname)
        if not record:
            return

        info_html = f"""
        <style>td {{ padding-right: 10px; }}</style>
        <table>
        <tr><td><b>Файл:</b></td><td>{html_module.escape(fname)}</td></tr>
        <tr><td><b>Размер:</b></td><td>{record.original_shape[1]} x {record.original_shape[0]}</td></tr>
        <tr><td><b>Лиц найдено:</b></td><td>{record.face_count}</td></tr>
        <tr><td><b>Тип:</b></td><td>{record.image_type}</td></tr>
        """
        
        if record.location_name:
            info_html += (
                "<tr><td><b>Локация:</b></td><td>"
                f"{html_module.escape(str(record.location_name))} "
                f"(ID: {html_module.escape(str(record.location_cluster))})</td></tr>"
            )
            
        info_html += "</table>"
        self.photo_info_viewer.setHtml(info_html)

        full_path = self._get_image_path(fname)
        if not full_path.is_file():
            return

        generation = self._face_panel_generation
        source_size = self._source_size_from_record(record)
        icon_extent = max(
            self.face_details_widget.iconSize().width(),
            self.face_details_widget.iconSize().height(),
        )
        target_size = (icon_extent, icon_extent)

        face_entries = self._ordered_face_panel_entries(record.faces)
        for display_index, (face_index, face) in enumerate(face_entries, start=1):
            border_color = None
            if self.mode == 'matches':
                matched_id = face.extra_data.get('matched_portrait_cluster_label')
                if matched_id is None:
                    status_text = "Не опознан"
                    border_color = "#ff3232"
                elif str(matched_id) == str(self.active_cluster_id):
                    status_text = "(Этот кластер)"
                    border_color = "#32cd32"
                else:
                    status_text = (
                        self.data_manager.student_label(face.student_id)
                        or f"ID кластера {matched_id}"
                    )
                    border_color = "#4169e1"
                text = f"Лицо #{display_index}\n{status_text}"
            else:
                text = f"Лицо #{display_index}"
                student_label = self.data_manager.student_label(face.student_id)
                if student_label:
                    text += f"\n{student_label}"

            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, face_index)
            item.setData(FACE_STATUS_COLOR_ROLE, border_color or "")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.face_details_widget.addItem(item)

            request = face_thumbnail_request(
                self.image_cache,
                full_path,
                face.bbox,
                target_size,
                padding=0.3,
                variant="cluster_editor.face_panel.v3",
                source_size=source_size,
            )
            if request is None:
                continue
            channel = ("face-panel", id(self), generation, f"{fname}::{face_index}")
            self._face_panel_channels[channel] = {
                "generation": generation,
                "item": item,
            }
            self.image_loader.request(
                request,
                channel=channel,
                persist=True,
                disk_format="PNG",
            )

        if self.face_details_widget.count() > 0:
            first_face_item = self.face_details_widget.item(0)
            self.face_details_widget.setCurrentItem(first_face_item)
            self._on_face_item_clicked(first_face_item)

    def _ordered_face_panel_entries(self, faces):
        """Возвращает лица в порядке панели, сохраняя их исходные индексы."""

        entries = list(enumerate(faces))
        if self.mode != "matches":
            return entries

        def sort_key(entry):
            original_index, face = entry
            matched_id = face.extra_data.get("matched_portrait_cluster_label")
            if self.active_cluster_id == "error_matches":
                return (0 if matched_id is None else 1, original_index)
            if matched_id is not None and str(matched_id) == str(self.active_cluster_id):
                return (0, original_index)
            return (1, original_index)

        return sorted(entries, key=sort_key)

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
        display_cluster_id = "None"
        display_name = self.data_manager.student_name(face.student_id) or "---"
        match_distance = None

        # 1. Если это Портрет (есть cluster_label)
        if face.cluster_label is not None:
            display_cluster_id = str(face.cluster_label)
            
        # 2. Если есть Матч (matched_portrait_cluster_label)
        elif face.extra_data.get('matched_portrait_cluster_label') is not None:
            display_cluster_id = str(face.extra_data.get('matched_portrait_cluster_label'))
            match_distance = face.extra_data.get('match_distance')
            
        # 3. Иначе - данные не заполняются (Temp ID игнорируем)
        else:
            display_cluster_id = "None"

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

        # Beauty score.
        beauty_score = face.extra_data.get('beauty_faceonnx')
        if beauty_score is not None:
             try:
                 html += f"<tr><td><b>Красота (Beauty):</b></td><td>{int(float(beauty_score))}</td></tr>"
             except (ValueError, TypeError):
                 pass

        # Состояние глаз и рта.
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
        html += f"<tr><td><b>Ученик:</b></td><td>{html_module.escape(str(display_name))}</td></tr>"
        html += f"<tr><td><b>student_id:</b></td><td>{html_module.escape(str(face.student_id or '---'))}</td></tr>"
        
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
            ImageViewer(
                self.data_manager,
                fname,
                parent=self,
                target_face_index=face_idx,
                draw_boxes=True,
                image_cache=self.image_cache,
                image_loader=self.image_loader,
            ).exec()
            return

        # --- СТАНДАРТНАЯ ЛОГИКА (Галерея) ---
        # Открываем чистое фото БЕЗ рамок для детального рассмотрения
        ImageViewer(
            self.data_manager,
            fname,
            parent=self,
            draw_boxes=False,
            image_cache=self.image_cache,
            image_loader=self.image_loader,
        ).exec()

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
        if self._export_is_running():
            QMessageBox.warning(self, "Экспорт", "Экспорт уже выполняется.")
            return
        self.export_end = False

        tasks = []
        
        for cid in cluster_ids:
            cdata = self._get_cluster_item_data_by_id(cid)
            if not cdata: continue
            
            student_id = cdata.get("student_id")
            if self.mode in {"face", "matches"}:
                display_name = self.data_manager.student_name(student_id)
                if not display_name:
                    logger.error(f"Экспорт кластера {cid} остановлен: отсутствует student_id.")
                    continue
            else:
                display_name = self.data_manager.strategy.normalize_cluster_name(
                    cdata["name"]
                )
            stable_id = student_id if self.mode in {"face", "matches"} else cid
            folder_name = _export_folder_name(
                display_name,
                stable_id,
                f"cluster_{cid}",
            )
            
            files = self.data_manager.get_files_for_cluster({}, cid)
            for fname in files:
                source_path = self._get_image_path(fname)
                if not source_path.is_file():
                    logger.error(f"Экспорт пропускает отсутствующий файл: {source_path}")
                    continue
                faces_bboxes = []
                record = self.data_manager.records.get(fname)
                if record:
                    for face in record.faces:
                        if face.bbox and len(face.bbox) == 4:
                            faces_bboxes.append(face.bbox)

                try:
                    output_path = _safe_export_path(
                        self.export_dir,
                        folder_name,
                        fname,
                    )
                except ValueError as exc:
                    logger.error(str(exc))
                    continue

                tasks.append({
                    "source_path": source_path,
                    "output_path": output_path,
                    "student_name": display_name,
                    "faces_bboxes": faces_bboxes
                })
        
        if not tasks:
            QMessageBox.warning(self, "Экспорт", "Нет доступных файлов для экспорта.")
            return
        preview_bboxes = tasks[0].get("faces_bboxes", [])
        dlg = EnhanceSettingsDialog(
            tasks[0]["source_path"],
            preview_bboxes,
            self,
            image_cache=self.image_cache,
            image_loader=self.image_loader,
        )
        if dlg.exec() != QDialog.Accepted: return
        settings = dlg.get_export_settings()
        
        self.status_bar.setFormat("Экспорт... %p%")
        self.status_bar.setRange(0, len(tasks))
        self.status_bar.setValue(0)
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(False)
        
        self.export_controller.start(
            tasks,
            self.num_workers,
            settings["factors"],
            (settings["width"], settings["height"]),
            (settings["dpi"], settings["dpi"]),
            settings["quality"],
            settings["watermarks"],
        )

    def _export_is_running(self) -> bool:
        return self.export_controller.is_running

    @Slot(str)
    def _on_export_finished(self, message: str):
        self.status_bar.reset()
        self.status_bar.setFormat("")
     
        if not self._close_after_export:
            QMessageBox.information(self, "Экспорт завершен", message)
        self.export_end = True

    @Slot()
    def _on_export_thread_stopped(self):
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(True)
        if self._gallery_build_state is not None or self._gallery_thumbnail_channels:
            self.status_bar.setRange(0, self._GALLERY_PROGRESS_MAX)
            self.status_bar.setValue(0)
            self._update_gallery_progress()
        if self._close_after_export:
            self._close_after_export = False
            QTimer.singleShot(0, self.close)

    def _update_pysm_context(self) -> bool:
        # Контракт обложек локаций хранится в структурированной переменной:
        # sys_location_name.{photo_session}
        if self.mode == 'location' and IS_MANAGED_RUN:
            try:
                location_previews = self.data_manager.get_location_covers_dict()
                for name in ["portrait_A6", "portrait_A5", "portrait_A4"]:
                    if name not in location_previews:
                        location_previews[name] = ""
                var_name = f"sys_location_name.{self.photo_session}"
                pysm_context.set_structured(var_name, location_previews)
                return True
            except Exception as e:
                self.data_manager.last_error = f"Context update error: {e}"
                logger.error(f"Context update error: {e}")
                return False
        return True


    def _log_final_report(self):
        """Формирует и выводит финальный отчет перед закрытием."""
        if not IS_MANAGED_RUN or not pysm_context:
            return

        try:
            tree_builder = StandardTreeBuilder(icon_size=28)
            resources = []
            if self.reference_dir != self.working_dir:
                resources.append(ResourceNode(
                    self.reference_dir.name,
                    Path(self.reference_dir),
                    "folder",
                    "Папка референсной фотосессии с эталонными портретами",
                ))
            resources.append(ResourceNode(
                self.working_dir.name,
                Path(self.working_dir),
                "folder",
                "Целевая папка текущей фотосессии",
            ))
            if self.export_end:
                resources.append(ResourceNode(
                    self.export_dir.name,
                    self.export_dir,
                    "folder",
                    "Папка с экспортированными файлами JPG",
                ))
            tree_builder.add_section("<br>Рабочие папки и файлы", resources)
            pysm_context.log_html(tree_builder.get_html())
        except Exception as exc:
            logger.error(f"Ошибка при формировании финального отчета: {exc}")

    def closeEvent(self, event):
        if self.data_load_thread is not None and self.data_load_thread.isRunning():
            QMessageBox.information(
                self,
                "Загрузка данных",
                "Дождитесь завершения загрузки данных.",
            )
            event.ignore()
            return
        if self._export_is_running():
            reply = QMessageBox.question(self, "Прерывание", 
                                         "В данный момент выполняется экспорт фотографий.\nПрервать процесс и закрыть программу?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._close_after_export = True
            self.export_controller.request_interruption()
            self.status_bar.setFormat("Завершение экспорта...")
            event.ignore()
            return

        # Стандартная обработка несохраненных изменений
        if self.data_manager.has_changes():
            reply = QMessageBox.question(self, "Выход", "Сохранить изменения перед выходом?", 
                                         QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Save:
                if self._save_changes(silent=True):
                    self._finalize_close(event)
                else:
                    event.ignore()
            elif reply == QMessageBox.StandardButton.Discard:
                self._finalize_close(event)
            else:
                event.ignore()
        else:
            self._finalize_close(event)

    def _finalize_close(self, event) -> None:
        if IS_MANAGED_RUN and pysm_context and self.win_state_var_name and WindowStateManager:
            try:
                mode_var_name = f"{self.win_state_var_name}.{self.mode}"
                window_state = WindowStateManager.save_state(
                    window=self,
                    splitters={'main': self.main_splitter},
                )
                pysm_context.set_structured(mode_var_name, window_state)
            except Exception as exc:
                logger.error(f"Не удалось сохранить состояние окна: {exc}")
        self._stop_loader()
        self._cancel_cluster_cover_requests()
        self.image_pipeline.shutdown()
        self._log_final_report()
        event.accept()

def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Редактор кластеров.")
    p = "ce_"
    parser.add_argument(f"--{p}working_dir", type=str, required=True, help="Папка с данными")
    parser.add_argument(f"--{p}reference_dir", type=str, default=None, help="Папка с эталонами (для matches)")
    parser.add_argument(
        f"--{p}student_list_file",
        type=str,
        required=True,
        help="Файл *.list — источник ФИО",
    )
    parser.add_argument(f"--{p}export_dir", type=str, default=None, help="Папка для экспорта фотографий с водяными знаками")
    parser.add_argument(f"--{p}win_state_var_name", type=str, default="", help="Имя переменной контекста для сохранения состояния окна")
    parser.add_argument("--all_threads", type=int, dest="all_threads", default=0, help="Количество потоков (0=авто).")
    parser.add_argument("--mode", type=str, choices=["face", "location", "matches", "cleaning"], default="face")

    
    return ConfigResolver(parser).resolve_all()

if __name__ == "__main__":
    cli_config = get_config()
    print("<b>ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ</b>")
    print("<i>Инициализация...</i><br>")
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
        list_file_str = getattr(cli_config, f"{arg_prefix}student_list_file", None)
        list_file = Path(list_file_str) if list_file_str else None
        if not w_dir.exists(): raise FileNotFoundError(f"Нет папки: {w_dir}")

        if cli_config.all_threads < 0:
            raise ValueError("--all_threads не может быть отрицательным")
        num_workers = max(1, cli_config.all_threads or (os.cpu_count() or 8))

        window = MainWindow(
            w_dir, r_dir, cli_config.mode, num_workers, e_dir_str, win_var, list_file
        )
        window.show()
     
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Ошибка запуска: {e}", exc_info=True)
        QMessageBox.critical(None, "Ошибка", str(e))
        sys.exit(1)
