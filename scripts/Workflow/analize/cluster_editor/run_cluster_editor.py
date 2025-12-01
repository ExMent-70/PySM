#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cluster_editor.py
=====================
Модуль для редактирования кластеров изображений с графическим интерфейсом на основе PySide6.
"""

import sys
import os
import logging
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QInputDialog, QProgressBar, QMessageBox, QLineEdit, QMenu,
    QListWidget, QListWidgetItem, QDialog, QCheckBox, QFileDialog
)
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer

# Внутренние модули
IS_MANAGED_RUN = False

try:
    current_script_dir = Path(__file__).resolve().parent
    if str(current_script_dir) not in sys.path: sys.path.insert(0, str(current_script_dir))
    project_root = current_script_dir.parent
    if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True

    from _lib.editor_viewer import ImageViewer
    from _lib.editor_workers import FileReaderWorker, GalleryLoadWorker, ExportWorker
    from _lib.editor_delegates import ClusterItemDelegate, ImageItemDelegate
    from _lib.editor_widgets import ImageDragListWidget, ClusterDropListWidget
    from _lib.editor_dialogs import EnhanceSettingsDialog, RenameDialog
    from _lib.data_manager import ClusterDataManager
    # --- ИСПРАВЛЕНИЕ: Добавлен недостающий импорт класса Face ---
    from _lib.data_models import Face

except ImportError as e:
    print(f"Ошибка импорта: {e}", file=sys.stderr)


logger = logging.getLogger(__name__)


class MainWindow(QWidget):
    def __init__(self, portrait_json_path: Path, group_json_path: Path, mode: str):
        super().__init__()
        self.mode = mode
        
        # Основные пути, полученные извне
        self.portrait_json_path = portrait_json_path
        self.group_json_path_initial = group_json_path
        
        # Производные пути
        self.data_dir = self.portrait_json_path.parent
        self.portrait_images_dir: Path = self.data_dir / "JPG"
        
# --- НАЧАЛО ИЗМЕНЕНИЯ: Переносим вычисление self.photo_session и self.session_name ВВЕРХ ---
        # Инициализируем photo_session и session_name на основе ИСХОДНОГО group_json_path
        group_analysis_dir = self.group_json_path_initial.parent
        group_output_dir = group_analysis_dir.parent
        group_session_dir = group_output_dir.parent
        self.photo_session = group_analysis_dir.name.replace("Analysis_", "")
        self.session_name = group_session_dir.name

        self.group_images_dir: Optional[Path] = group_analysis_dir / "JPG"
        self.current_group_json_path: Optional[Path] = self.group_json_path_initial
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

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

# --- НАЧАЛО ИЗМЕНЕНИЯ: Используем self.photo_session ---
        self.setWindowTitle(self.mode_config["window_title_template"].format(self.photo_session))
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

        self.predefined_cluster_names: List[str] = []
        try:
            current_dir = Path(__file__).resolve().parent
            predefined_names_path = current_dir / "predefined_names.json"
            if predefined_names_path.exists():
                with open(predefined_names_path, 'r', encoding='utf-8') as f:
                    self.predefined_cluster_names = json.load(f)
                    logger.info(f"Загружено {len(self.predefined_cluster_names)} предопределенных имен кластеров.")
            else:
                logger.warning(f"Файл 'predefined_names.json' не найден в {current_dir}.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке или парсинге predefined_names.json: {e}")

# --- НАЧАЛО ИЗМЕНЕНИЯ: Инициализация DataManager ---
        # DataManager инициализируется явными путями
        group_json_for_dm = self.group_json_path_initial if self.mode != 'matches' else None
        self.data_manager = ClusterDataManager(self.portrait_json_path, group_json_for_dm)
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

        self.active_cluster_id: Optional[str] = None
        self.preview_pixmaps: Dict[str, QPixmap] = {}
        self.image_pixmap_cache: Dict[str, QPixmap] = {}

        self.file_reader_thread = None
        self.file_reader_worker = None
        self.gallery_load_thread = None
        self.gallery_load_worker = None
        self.gallery_populator_timer = None
        self.gallery_item_iterator = None

        self.cluster_delegate = ClusterItemDelegate(parent=self)
        self.image_delegate = ImageItemDelegate(parent=self)

        self.init_ui()
        self._load_and_display_data()

# --- НАЧАЛО ИЗМЕНЕНИЯ: Автоматическая загрузка данных для режима 'matches' ---
        # Автоматическая загрузка групповых данных по умолчанию в режиме 'matches'
        if self.mode == 'matches':
            if self.group_json_path_initial and self.group_json_path_initial.is_file():
                logger.info("Автоматическая загрузка данных о совпадениях по умолчанию...")
                self._load_and_process_group_data(self.group_json_path_initial)
            else:
                logger.warning(
                    "Файл info_group_faces.json по умолчанию не найден или не указан. "
                    "Загрузите файл вручную через контекстное меню."
                )
# --- КОНЕЦ ИЗМЕНЕНИЯ ---    


    def _load_and_process_group_data(self, group_json_path: Path):
        """
        Универсальный метод для загрузки и обработки нового файла 
        info_group_faces.json.
        """
        if self.data_manager.reload_group_data(group_json_path):
# --- НАЧАЛО ИЗМЕНЕНИЯ ---
            self.current_group_json_path = group_json_path
            
            # Пересчитываем все зависимые пути и имена на основе нового файла
            group_analysis_dir = self.current_group_json_path.parent
            group_output_dir = group_analysis_dir.parent
            group_session_dir = group_output_dir.parent
            self.photo_session = group_analysis_dir.name.replace("Analysis_", "")
            self.session_name = group_session_dir.name
            
            self.group_images_dir = group_analysis_dir / "JPG"
            
            self.setWindowTitle(self.mode_config["window_title_template"].format(self.photo_session))
# --- КОНЕЦ ИЗМЕНЕНИЯ ---
            
            logger.info(f"Успешно загружен файл: {group_json_path.name}")
            logger.info(f"Папка с групповыми фото изменена на: {self.group_images_dir}")
            
            self._refresh_left_panel()
        else:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить или обработать файл:\n{group_json_path}")


    def _center_on_screen(self):
        try:
            screen_geometry = self.screen().geometry()
            window_geometry = self.frameGeometry()
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            self.move(window_geometry.topLeft())
        except Exception as e:
            logger.warning(f"Не удалось центрировать окно на экране: {e}")

    def init_ui(self):
        #self.setWindowTitle(self.mode_config["window_title"])
        self.setGeometry(0, 0, 1350, 900)

        main_layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, 1)

        left_panel_widget = QWidget()
        left_layout = QVBoxLayout(left_panel_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        left_title = QLabel(f"Фотосессия: {self.photo_session} (cписок кластеров)")

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Поиск...")
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

        if self.mode == 'matches':
            self.cluster_list_widget.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        else:
            self.cluster_list_widget.itemsDropped.connect(self._handle_drop)
            self.cluster_list_widget.setDragDropMode(QListWidget.DragDropMode.DropOnly)

        self.cluster_list_widget.viewport().setAcceptDrops(True)
        self.cluster_list_widget.setDropIndicatorShown(True)

        right_panel_widget = QWidget()
        right_layout = QVBoxLayout(right_panel_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.right_panel_label = QLabel("Кластер")

        self.image_list_widget = ImageDragListWidget(self)
        self.image_list_widget.setObjectName("imageListWidget")
        self.image_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list_widget.setSpacing(10)
        self.image_list_widget.setItemDelegate(self.image_delegate)

        if self.mode == 'matches':
            self.image_list_widget.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        else:
            self.image_list_widget.setDragDropMode(QListWidget.DragDropMode.DragOnly)

        self.image_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list_widget.itemDoubleClicked.connect(self._open_image_viewer)

        self.export_button = QPushButton("Экспорт")
        export_menu = QMenu(self)
        export_all = export_menu.addAction("Экспортировать всё")
        export_active = export_menu.addAction("Экспортировать активный кластер")
        self.export_button.setMenu(export_menu)
        export_all.triggered.connect(self._on_export_all_triggered)
        export_active.triggered.connect(self._on_export_active_triggered)

        self.save_button = QPushButton("Сохранить изменения")
        self.save_button.clicked.connect(self._save_changes)

        if self.mode == 'matches':
            self.save_button.setText("Сгенерировать matches.json")

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.export_button)
        buttons_layout.addWidget(self.save_button)

        self.watermarks_enabled_checkbox = QCheckBox("Наложить водяные знаки на изображение при экспорте")
        self.watermarks_enabled_checkbox.setChecked(True)

        left_layout.addWidget(left_title)
        left_layout.addWidget(self.search_bar)
        left_layout.addWidget(self.cluster_list_widget, 1)
        left_layout.addWidget(self.watermarks_enabled_checkbox)
        left_layout.addLayout(buttons_layout)

        right_layout.addWidget(self.right_panel_label)
        right_layout.addWidget(self.image_list_widget, 1)

        content_layout.addWidget(left_panel_widget, 5)
        content_layout.addWidget(right_panel_widget, 9)

        self.status_progress_bar = QProgressBar()
        self.status_progress_bar.setTextVisible(True)

        main_layout.addWidget(self.status_progress_bar)
        self._center_on_screen()

    def _load_and_display_data(self):
        success, message = self.data_manager.load_data()
        if not success:
            QMessageBox.critical(self, "Ошибка загрузки данных", message)
            return
        self._refresh_left_panel()

    def _get_clusters_from_model(self) -> Dict[str, List[Face]]:
        if self.mode == 'matches':
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


    # --- НАЧАЛО НОВОГО МЕТОДА ---
    def _get_image_path(self, filename: str, image_type: str) -> Path:
        """
        Возвращает полный путь к файлу изображения, используя корректную
        директорию в зависимости от типа (портрет или группа).
        """
        if image_type == 'portrait':
            return self.portrait_images_dir / filename
        elif image_type == 'group' and self.group_images_dir:
            return self.group_images_dir / filename
        # Фоллбэк, если group_images_dir еще не задан или тип неизвестен
        return self.portrait_images_dir / filename
    # --- КОНЕЦ НОВОГО МЕТОДА ---


    def _find_file_globally(self, filename: str) -> Optional[Path]:
        file_stem = Path(filename).stem
        found_files = list(self.flat_images_dir.glob(f"{file_stem}.*"))
        if found_files:
            return found_files[0]
        logger.warning(f"Файл '{filename}' не был найден в {self.flat_images_dir}.")
        return None

    def _get_cluster_item_data_by_id(self, cluster_id: str) -> Optional[Dict]:
        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole)["id"] == cluster_id:
                return item.data(Qt.ItemDataRole.UserRole)
        return None
        
    def _get_item_by_cluster_id(self, cluster_id: str) -> Optional[QListWidgetItem]:
            """Находит QListWidgetItem по его ID кластера."""
            for i in range(self.cluster_list_widget.count()):
                item = self.cluster_list_widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole)["id"] == cluster_id:
                    return item
            return None        

    @Slot(str)
    def _on_search_text_changed(self, text: str):
        search_text = text.strip().lower()
        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            cluster_name = item.data(Qt.ItemDataRole.UserRole)["name"]
            item.setHidden(search_text not in cluster_name.lower())

    def _refresh_left_panel(self):
        active_id_before_refresh = self.active_cluster_id
        self.cluster_list_widget.clear()
        self.preview_pixmaps.clear()
        clusters = self._get_clusters_from_model()

        sort_key_func = lambda x: int(x) if x.isdigit() else (9998 if x == "-1" else 9999)
        if self.mode == 'location':
            sort_key_func = lambda x: x

        sorted_labels = sorted(clusters.keys(), key=sort_key_func)

        item_to_select = None
        for label in sorted_labels:
            if self.mode == 'matches' and label in ["-1", "group"]:
                continue

            faces = clusters[label]
# --- НАЧАЛО ИЗМЕНЕНИЯ ---
            # В режиме 'matches' имя берется из кэша data_manager,
            # который заполняется либо из portrait_faces, либо при перезагрузке.
            if self.mode == 'matches':
                cluster_name = self.data_manager._cluster_id_to_name_cache.get(label)
                if not cluster_name and faces: # Если в кэше нет, берем из данных портрета
                    cluster_name = faces[0].child_name
                if not cluster_name: cluster_name = f"Кластер {label}" # Фоллбэк
            else:
                cluster_name = faces[0].effective_name if faces else f"Кластер {label}"

            preview_path = Path()
            if faces:
                # Превью для левой панели ВСЕГДА берем из портретов
                preview_path = self._get_image_path(faces[0].filename, 'portrait')
# --- КОНЕЦ ИЗМЕНЕНИЯ ---




            from _lib.editor_delegates import PREVIEW_SIZE
            pixmap = QPixmap(str(preview_path))
            if not pixmap.isNull():
                pixmap = pixmap.scaled(PREVIEW_SIZE, PREVIEW_SIZE, Qt.AspectRatioMode.KeepAspectRatio)

            self.preview_pixmaps[label] = pixmap
            
            

# --- НАЧАЛО ИЗМЕНЕНИЯ ---
            if self.mode == 'matches':
                count = len(self.data_manager.get_group_matches_for_cluster(label))
            else:
                count = len(self.data_manager.get_files_for_cluster(self.mode_config, label))
# --- КОНЕЦ ИЗМЕНЕНИЯ ---



            item_data = { "id": label, "name": cluster_name, "count": count,
                          "pixmap": pixmap, "is_changed": self.data_manager.is_cluster_changed(self.mode_config["mode_name"], label) }
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            self.cluster_list_widget.addItem(item)

            if label == active_id_before_refresh:
                item_to_select = item

        for new_cluster in self.data_manager.newly_created_clusters:
            if new_cluster["id"] in clusters:
                continue
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
        self._stop_gallery_load_if_running()
        self._stop_file_reader_if_running()
        self._stop_populator_timer()

        cluster_data = self._get_cluster_item_data_by_id(cluster_id)
        if not cluster_data:
            self.image_list_widget.clear()
            self.right_panel_label.setText("Кластер не найден")
            return

        if self.mode == 'matches':
             self.right_panel_label.setText(f"Групповые фото для кластера: {cluster_data['name']} ({cluster_data['count']} совпадений)")
        else:
             self.right_panel_label.setText(f"Кластер: {cluster_data['name']} ({cluster_data['count']} фото)")

        self.image_list_widget.clear()

        if self.mode == 'matches':
            files_to_show = self.data_manager.get_group_matches_for_cluster(cluster_id)
        else:
            files_to_show = self.data_manager.get_files_for_cluster(self.mode_config, cluster_id)

        if not files_to_show:
            return

        cached_items, uncached_tasks = [], []
        for filename in files_to_show:
            if filename in self.image_pixmap_cache:
                cached_items.append({
                    "filename": filename,
                    "pixmap": self.image_pixmap_cache[filename]
                })
            else:
                uncached_tasks.append({
                    "filename": filename,
                    "cluster_id": cluster_id
                })

        self._on_gallery_prepared(cached_items, uncached_tasks)

    @Slot(list, list)
    def _on_gallery_prepared(self, cached_items: List[Dict], uncached_tasks: List[Dict]):
        for item_data in cached_items:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.DecorationRole, item_data["pixmap"])
            item.setData(Qt.ItemDataRole.UserRole, {"filename": item_data["filename"]})
            self.image_list_widget.addItem(item)

# --- НАЧАЛО ИЗМЕНЕНИЯ ---
        if uncached_tasks:
            image_type_for_gallery = 'group' if self.mode == 'matches' else 'portrait'
            full_tasks = [
                {**task, "full_path": self._get_image_path(task["filename"], image_type_for_gallery)}
                for task in uncached_tasks
            ]
            self._start_file_reading(full_tasks)
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

    def _stop_file_reader_if_running(self):
        if hasattr(self, 'file_reader_thread') and self.file_reader_thread and self.file_reader_thread.isRunning():
            if hasattr(self, 'file_reader_worker') and self.file_reader_worker:
                self.file_reader_worker.requestInterruption()
            try:
                self.file_reader_worker.finished.disconnect(self._on_files_read)
            except (RuntimeError, TypeError):
                pass
            self.file_reader_thread.quit()
            if not self.file_reader_thread.wait(1000):
                self.file_reader_thread.terminate()
            self.file_reader_worker = None
            self.file_reader_thread = None

    def _start_file_reading(self, tasks: List[Dict]):
        self._stop_file_reader_if_running()

        self.file_reader_thread = QThread(self)
        self.file_reader_worker = FileReaderWorker(tasks)
        self.file_reader_worker.moveToThread(self.file_reader_thread)
        self.file_reader_worker.finished.connect(self._on_files_read)
        self.file_reader_thread.started.connect(self.file_reader_worker.run)
        self.file_reader_thread.start()

    @Slot(list)
    def _on_files_read(self, read_data_tasks: List[Dict]):
        if hasattr(self, 'file_reader_thread') and self.file_reader_thread:
            self.file_reader_thread.quit()
            self.file_reader_thread.wait()
            self.file_reader_worker = None
            self.file_reader_thread = None

        if read_data_tasks:
            self._start_gallery_load(read_data_tasks)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_cluster_selected(self, current_item: QListWidgetItem, previous_item: Optional[QListWidgetItem] = None):
        if not current_item:
            self._stop_file_reader_if_running()
            self._stop_gallery_load_if_running()
            self._stop_populator_timer()
            self.image_list_widget.clear()
            self.right_panel_label.setText("Кластер")
            self.active_cluster_id = None
            return

        cluster_data = current_item.data(Qt.ItemDataRole.UserRole)
        cluster_id = cluster_data["id"]
        if self.active_cluster_id == cluster_id:
            return

        self._stop_file_reader_if_running()
        self._stop_gallery_load_if_running()
        self._stop_populator_timer()
        self.active_cluster_id = cluster_id
        self._render_gallery(cluster_id)

    def _stop_gallery_load_if_running(self):
        if hasattr(self, 'gallery_load_thread') and self.gallery_load_thread and self.gallery_load_thread.isRunning():
            if hasattr(self, 'gallery_load_worker') and self.gallery_load_worker:
                self.gallery_load_worker.requestInterruption()
            try:
                self.gallery_load_worker.progress_updated.disconnect(self.status_progress_bar.setValue)
                self.gallery_load_worker.finished.disconnect(self._on_gallery_load_finished)
            except (RuntimeError, TypeError): pass
            self.gallery_load_thread.quit()
            if not self.gallery_load_thread.wait(1000): self.gallery_load_thread.terminate()
            self.gallery_load_worker = None; self.gallery_load_thread = None

    def _start_gallery_load(self, tasks: List[Dict]):
        self._stop_gallery_load_if_running()
        self.status_progress_bar.setRange(0, len(tasks))
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setFormat("Подготовка изображений... %p%")

        self.gallery_load_thread = QThread(self)
        self.gallery_load_worker = GalleryLoadWorker(tasks, self.image_pixmap_cache)
        self.gallery_load_worker.moveToThread(self.gallery_load_thread)
        self.gallery_load_worker.progress_updated.connect(self.status_progress_bar.setValue)
        self.gallery_load_worker.finished.connect(self._on_gallery_load_finished)
        self.gallery_load_thread.started.connect(self.gallery_load_worker.run)
        self.gallery_load_thread.start()

    @Slot(list)
    def _on_gallery_load_finished(self, processed_tasks: List[Dict]):
        self.status_progress_bar.reset(); self.status_progress_bar.setFormat("")
        if hasattr(self, 'gallery_load_thread') and self.gallery_load_thread:
            self.gallery_load_thread.quit(); self.gallery_load_thread.wait()
            self.gallery_load_worker = None; self.gallery_load_thread = None

        if not processed_tasks:
            return

        self.gallery_item_iterator = iter(processed_tasks)
        if not self.gallery_populator_timer:
            self.gallery_populator_timer = QTimer(self)
            self.gallery_populator_timer.timeout.connect(self._populate_gallery_chunk)
        self.gallery_populator_timer.start(0)

    def _stop_populator_timer(self):
        if self.gallery_populator_timer and self.gallery_populator_timer.isActive():
            self.gallery_populator_timer.stop()
        self.gallery_item_iterator = None

    def _populate_gallery_chunk(self):
        if not self.gallery_item_iterator:
            self._stop_populator_timer()
            return

        chunk_size = 50

        for _ in range(chunk_size):
            try:
                task = next(self.gallery_item_iterator)
                filename = task["filename"]

                pixmap = self.image_pixmap_cache.get(filename)
                if not pixmap:
                    continue

                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
                item.setData(Qt.ItemDataRole.UserRole, {"filename": filename})
                self.image_list_widget.addItem(item)

            except StopIteration:
                self._stop_populator_timer()
                return

    @Slot(QListWidgetItem)
    def _open_image_viewer(self, item: QListWidgetItem):
        current_filename = item.data(Qt.ItemDataRole.UserRole)["filename"]
        cluster_id = self.active_cluster_id

        all_filenames = self._get_files_for_cluster_for_viewer(cluster_id)

        try:
            current_index = all_filenames.index(current_filename)
        except ValueError:
            logger.warning(f"Файл {current_filename} не найден в списке файлов для просмотра.")
            return

# --- НАЧАЛО ИЗМЕНЕНИЯ ---
        image_type_for_viewer = 'group' if self.mode == 'matches' else 'portrait'
        image_paths = [self._get_image_path(fname, image_type_for_viewer) for fname in all_filenames]
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

        viewer = ImageViewer(image_paths, all_filenames, current_index, self)
        viewer.exec()


# analize/cluster_editor/run_cluster_editor.py -> class MainWindow

    def show_cluster_context_menu(self, pos):
# --- НАЧАло ИЗМЕНЕНИЯ ---
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
                rename_action.setEnabled(False)
                delete_action.setEnabled(False)
            else:
                cluster_data = item.data(Qt.ItemDataRole.UserRole)
                is_empty = cluster_data.get("count", 0) == 0
                is_special = cluster_data.get("id") in ["-1", "group"]

                rename_action.setEnabled(not is_special)
                delete_action.setEnabled(is_empty and not is_special)

            action = menu.exec(self.cluster_list_widget.mapToGlobal(pos))

            if action == create_action:
                self._create_cluster_action()
            elif action == rename_action and item:
                self._rename_cluster_action(item)
            elif action == delete_action and item:
                self._delete_cluster_action(item)
            return  # Завершаем, чтобы не вызывать exec() еще раз ниже
        
        # Этот вызов сработает только для режима 'matches'
        menu.exec(self.cluster_list_widget.mapToGlobal(pos))
# --- КОНЕЦ ИЗМЕНЕНИЯ ---



    @Slot()
    def _load_group_data_action(self):
        """
        Открывает диалог выбора 'info_group_faces.json' и инициирует
        перезагрузку данных.
        """
# --- НАЧАЛО ИЗМЕНЕНИЯ ---
        start_dir = str(self.data_dir)
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл с данными групповых фото",
            start_dir,
            "JSON files (info_group_faces.json)"
        )

        if filepath:
            self._load_and_process_group_data(Path(filepath))
# --- КОНЕЦ ИЗМЕНЕНИЯ ---



    @Slot(QListWidgetItem)
    def _rename_cluster_action(self, item: QListWidgetItem):
        cluster_data = item.data(Qt.ItemDataRole.UserRole)
        cluster_id = cluster_data["id"]
        if self.mode == 'face' and cluster_id in ["group", "-1"]:
            QMessageBox.information(self, "Инфо", "Этот кластер нельзя переименовать.")
            return

        current_name_no_prefix = cluster_data["name"].split('-', 1)[-1]
        new_name = ""
        ok = False

        if self.mode == 'location':
            dialog = RenameDialog(self.predefined_cluster_names, current_name_no_prefix, self)
            if dialog.exec() == QDialog.Accepted:
                new_name = dialog.get_selected_name()
                ok = True
        else:
            new_name, ok = QInputDialog.getText(
                self, "Переименование", "Новое имя (без префикса):",
                text=current_name_no_prefix
            )

        if ok and new_name.strip():
            self._handle_rename(cluster_id, new_name.strip())

    @Slot(str, str, list)
    def _handle_drop(self, source_id: str, target_id: str, filenames: List[str]):
        """Обрабатывает завершение операции Drag & Drop."""
        target_cluster_data = self._get_cluster_item_data_by_id(target_id)
        if not target_cluster_data:
            return

        # 1. Обновляем модель данных
        self.data_manager.move_images_to_cluster(
            self.mode_config, target_id, target_cluster_data["name"], filenames
        )

        # --- ИЗМЕНЕНИЕ: Явная и надежная логика обновления UI ---

        # 2. Запоминаем, какой кластер был активен (это кластер-источник)
        active_id_before_refresh = self.active_cluster_id

        # 3. Полностью перерисовываем левую панель (обновятся счетчики).
        #    Эта функция также попытается восстановить выделение на active_id_before_refresh.
        self._refresh_left_panel()

        # 4. Принудительно и безусловно перерисовываем правую панель для активного кластера.
        #    Это исправляет баг, когда галерея не обновлялась, т.к. ID активного кластера не менялся.
        if active_id_before_refresh:
            # Находим обновленный QListWidgetItem для нашего кластера
            current_item = self._get_item_by_cluster_id(active_id_before_refresh)
            if current_item:
                # Убеждаемся, что он все еще выбран
                self.cluster_list_widget.setCurrentItem(current_item)
                # И ГЛАВНОЕ: вызываем _render_gallery напрямую, минуя ошибочную проверку в слоте.
                self._render_gallery(active_id_before_refresh)
            else:
                # Если исходный кластер исчез (например, стал пустым и был удален), очищаем правую панель.
                self.image_list_widget.clear()
                self.right_panel_label.setText("Кластер")

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
        cluster_id = cluster_data["id"]
        cluster_name = cluster_data["name"]

        reply = QMessageBox.question(
            self, "Подтверждение", f"Вы уверены, что хотите удалить пустой кластер '{cluster_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.data_manager.delete_newly_created_cluster(cluster_id)
            logger.info(f"Кластер '{cluster_name}' (ID: {cluster_id}) удален.")
            self._refresh_left_panel()

    @Slot()
    def _on_export_all_triggered(self):
        portrait_ids = [cid for cid in self._get_clusters_from_model().keys() if cid not in ["-1", "group"]]
        if not portrait_ids:
            QMessageBox.information(self, "Инфо", "Нет кластеров для экспорта.")
            return
        self._start_export(portrait_ids)

    @Slot()
    def _on_export_active_triggered(self):
        active_id = self.active_cluster_id
        if active_id and active_id not in ["-1", "group"]:
            self._start_export([active_id])
        else:
            QMessageBox.warning(self, "Внимание", "Выберите портретный кластер для экспорта совпадений.")


# analize/cluster_editor/run_cluster_editor.py -> class MainWindow

    def _start_export(self, cluster_ids: List[str]):
# --- НАЧАЛО ИЗМЕНЕНИЯ ---
        # Всегда вычисляем путь экспорта на основе АКТУАЛЬНЫХ данных
        if self.current_group_json_path:
            group_analysis_dir = self.current_group_json_path.parent
            group_output_dir = group_analysis_dir.parent
            
            # self.session_name уже актуален, так что используем его
            base_output_dir = group_output_dir / self.session_name / f"Выбор_Фото_{self.photo_session}_{self.mode}"
        else:
            # Фоллбэк, если по какой-то причине групповой путь не задан
            base_output_dir = self.data_dir.parent / self.session_name / f"Выбор_Фото_{self.photo_session}_{self.mode}"

        logger.info(f"Экспорт будет выполнен в: {base_output_dir}")
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

        tasks = []
        for cid in cluster_ids:
            cluster_data = self._get_cluster_item_data_by_id(cid)
            if not cluster_data: continue

            disk_folder_name = cluster_data["name"]
            output_folder = base_output_dir / disk_folder_name

            # Определяем, из какой папки брать изображения для экспорта
            image_type_for_export = 'group' if self.mode == 'matches' else 'portrait'
            
            if self.mode == 'matches':
                filenames = self.data_manager.get_group_matches_for_cluster(cid)
            else:
                filenames = self.data_manager.get_files_for_cluster(self.mode_config, cid)

            for fname in filenames:
                tasks.append({
                    "source_path": self._get_image_path(fname, image_type_for_export),
                    "output_path": output_folder / Path(fname).name,
                    "child_name": cluster_data["name"].split('-', 1)[-1].strip()
                })

        if not tasks:
            QMessageBox.information(self, "Инфо", "Нет файлов для экспорта в выбранных кластерах.")
            return

        first_image_path = tasks[0]["source_path"]
        dialog = EnhanceSettingsDialog(first_image_path, self)
        if dialog.exec() != QDialog.Accepted: return

        enhancement_factors = dialog.get_enhancement_factors()
        logger.info(f"Параметры обработки фотографий перед экспортом:\n<i>{enhancement_factors}</i>")

        self.status_progress_bar.setRange(0, len(tasks))
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setFormat("Экспорт... %p%")

        apply_watermarks = self.watermarks_enabled_checkbox.isChecked()
        logger.info(f"Водяные знаки: {'Включены' if apply_watermarks else 'Выключены'}")

        self.export_worker = ExportWorker(tasks, os.cpu_count() or 4, enhancement_factors, apply_watermarks)
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
        if hasattr(self, 'export_thread'):
            self.export_thread.quit(); self.export_thread.wait()

    def _perform_save(self) -> bool:
            """
            Выполняет сохранение данных JSON и, в режиме 'location',
            обновляет переменную контекста PySM.

            Returns:
                bool: True в случае успеха, иначе False.
            """
            if not self.data_manager.save_data():
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить JSON.")
                return False

            self._refresh_left_panel()

            # --- ИЗМЕНЕНИЕ: Обновлена логика сохранения в контекст ---
            if self.mode == 'location' and IS_MANAGED_RUN:
                # 1. Создаем словарь для локаций с файлами-представителями
                location_previews: Dict[str, str] = {}
                
                # 2. Получаем актуальные кластеры
                clusters = self.data_manager.get_clusters(self.mode_config)
                
                # 3. Заполняем словарь реальными данными
                for cluster_id, faces in clusters.items():
                    if faces:  # Убеждаемся, что кластер не пустой
                        location_name = faces[0].effective_name
                        first_filename = faces[0].filename
                        if location_name and first_filename:
                            location_previews[location_name] = Path(first_filename).name

                # 4. Определяем список системных имен
                additional_system_names = [
                    "portrait_A6",
                    "portrait_A5",
                    "portrait_A4"
                ]
                
                # 5. Добавляем системные имена с пустым значением, если их еще нет
                for name in additional_system_names:
                    if name not in location_previews:
                        location_previews[name] = ""
                
                # 6. Сохраняем итоговый словарь в переменную контекста
                current_location_name = "sys_location_name_"+self.photo_session
                pysm_context.set(current_location_name, location_previews)
                logger.info("Словарь 'имя локации: файл-представитель' сохранен в 'sys_location_name'.")

            return True

    def _save_changes(self):
        if self.mode == 'matches':
# --- НАЧАЛО ИЗМЕНЕНИЯ ---
            if self.current_group_json_path:
                output_path = self.current_group_json_path.parent / "matches_portrait_to_group.json"
                logger.info(f"Файл совпадений будет сохранен в: {output_path.parent}")
            else:
                # Этот фоллбэк сработает, если в режиме matches изначально не было группового JSON
                # и пользователь ничего не загрузил.
                logger.warning("Путь к групповому JSON не определен. Сохранение в директорию по умолчанию.")
                output_path = self.data_dir / "matches_portrait_to_group.json"

            success, message = self.data_manager.generate_and_save_matches_json(output_path)
            if success:
                QMessageBox.information(self, "Успех", message)
            else:
                QMessageBox.critical(self, "Ошибка", message)
            return
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

        if not self.data_manager.has_changes():
            QMessageBox.information(self, "Инфо", "Нет изменений для сохранения.")
            return

        msg = "Сохранить все изменения в метаданных кластеров (JSON)?"
        reply = QMessageBox.question(self, "Сохранение", msg, QMessageBox.Save | QMessageBox.Cancel)

        if reply == QMessageBox.Save:
            if self._perform_save():
                logger.info("\n<b>Все внесенные изменения сохранены в JSON</b>")
                QMessageBox.information(self, "Успех", "Изменения успешно сохранены.")
            else:
                logger.error("Ошибка при сохранении изменений.")

    def closeEvent(self, event):
        self._stop_file_reader_if_running()
        self._stop_gallery_load_if_running()
        self._stop_populator_timer()

        if self.mode == 'matches' or not self.data_manager.has_changes():
            event.accept()
            return

        reply = QMessageBox.question(
            self, "Несохраненные изменения",
            "У вас есть несохраненные изменения. Хотите сохранить их перед выходом?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save
        )

        if reply == QMessageBox.Cancel:
            event.ignore()
        elif reply == QMessageBox.Discard:
            event.accept()
        elif reply == QMessageBox.Save:
            if self._perform_save():
                event.accept()
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить данные. Выход отменен.")
                event.ignore()


def get_config() -> argparse.Namespace:
    """
    Определяет и парсит аргументы скрипта с помощью PySM ConfigResolver.
    """
    parser = argparse.ArgumentParser(description="Редактор кластеров изображений.")
    arg_prefix = "ce_"

    parser.add_argument(
        f"--{arg_prefix}portrait_json", type=str, default="",
        help="Путь к файлу info_portrait_faces.json (эталонные данные)."
    )
    parser.add_argument(
        f"--{arg_prefix}group_json", type=str, default="",
        help="Путь к файлу info_group_faces.json (данные для анализа/по умолчанию)."
    )
    parser.add_argument(
        "--mode", type=str, choices=["face", "location", "matches"], default="face",
        help="Режим работы: 'face', 'location' или 'matches'."
    )

    
    return ConfigResolver(parser).resolve_all()



if __name__ == "__main__":
    cli_config = get_config()
    arg_prefix = "ce_"
    
    log_level = "INFO"
    if IS_MANAGED_RUN and pysm_context:
        log_level = pysm_context.get("sys_log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s", stream=sys.stdout
    )

    app = QApplication.instance() or QApplication(sys.argv)
    if IS_MANAGED_RUN:
        theme_api.apply_theme_to_app(app)

    try:
        portrait_json_str = getattr(cli_config, f"{arg_prefix}portrait_json")
        group_json_str = getattr(cli_config, f"{arg_prefix}group_json")

        if not portrait_json_str or not group_json_str:
            raise ValueError("Пути к portrait_json и group_json должны быть указаны.")

        portrait_json_path = Path(portrait_json_str)
        group_json_path = Path(group_json_str)

        if not portrait_json_path.is_file():
            raise FileNotFoundError(f"Файл с портретами не найден: {portrait_json_path}")
        if not group_json_path.is_file():
            raise FileNotFoundError(f"Файл с группами не найден: {group_json_path}")


    except (ValueError, FileNotFoundError) as e:
        msg = f"Ошибка проверки входных данных:\n{e}"
        logger.critical(msg)
        QMessageBox.critical(None, "Критическая ошибка", msg)
        sys.exit(1)
    
    try:

        window = MainWindow(
            portrait_json_path=portrait_json_path,
            group_json_path=group_json_path,
            mode=cli_config.mode
        )

        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Непредвиденная ошибка при запуске приложения: {e}", exc_info=True)
        QMessageBox.critical(None, "Непредвиденная ошибка", f"Произошла критическая ошибка:\n{traceback.format_exc()}")
        sys.exit(1)