#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cluster_editor.py
=====================
Модуль для редактирования кластеров изображений с графическим интерфейсом на основе PySide6.
"""

# 1. БЛОК: Импорты
import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QInputDialog, QProgressBar, QMessageBox, QLineEdit, QMenu,
    QListWidget, QListWidgetItem, QDialog
)
from PySide6.QtGui import QPixmap, QAction, QKeySequence, QDrag, QPainter, QColor
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread, QMimeData, QPoint

# Внутренние модули
IS_COMMON_AVAILABLE = False
IS_MANAGED_RUN = False


try:
    current_script_dir = Path(__file__).resolve().parent
    if str(current_script_dir) not in sys.path: sys.path.insert(0, str(current_script_dir))
    project_root = current_script_dir.parent
    if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
    
    from _common.json_data_manager import JsonDataManager
    IS_COMMON_AVAILABLE = True
    
    from _lib.editor_viewer import ImageViewer
    from _lib.editor_workers import GalleryLoadWorker, ExportWorker, MoveWorker
    from _lib.editor_delegates import ClusterItemDelegate, ImageItemDelegate
    from _lib.editor_widgets import ImageDragListWidget, ClusterDropListWidget   
    from _lib.editor_dialogs import EnhanceSettingsDialog    
    from _lib import editor_styles as styles

    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib import pysm_context

    IS_MANAGED_RUN = True
except ImportError as e:
    print(f"Ошибка импорта: {e}", file=sys.stderr)


# Инициализируем глобальный логгер, но пока не настраиваем его
logger = logging.getLogger(__name__)


# 2. БЛОК: Главное окно
class MainWindow(QWidget):
    """Главное окно приложения для редактирования кластеров изображений."""
    def __init__(self, data_dir: Path, images_dir: Path, sorted_dir: Path, photo_session: str, session_name: str):
        super().__init__()
        self.data_dir = data_dir
        self.flat_images_dir = images_dir
        self.sorted_dir = sorted_dir
        self.photo_session = photo_session
        self.session_name = session_name
        
        self.is_clustered_mode = self.sorted_dir.is_dir()
        logger.info(f"Базовый метод группировки изображений: {'<b>папки кластеров</b>' if self.is_clustered_mode else 'отсутствует (папка JPG)'}")
        logger.info(f"При изменении кластера файлы изображений {'<b>будут перемещаться физически</b>' if self.is_clustered_mode else 'не будут перемещаться'}\n")
        
        self.json_manager = JsonDataManager(
            self.data_dir / "info_portrait_faces.json",
            self.data_dir / "info_group_faces.json",
        )

        self.portrait_model: Dict[str, Any] = {}
        self.group_model: Dict[str, Any] = {}
        
        self.changed_cluster_ids: set = set()
        self.pending_moves: Dict[str, Dict[str, str]] = {}
        
        self.active_cluster_id: Optional[str] = None
        self.preview_pixmaps: Dict[str, QPixmap] = {}
        self.image_pixmap_cache: Dict[str, QPixmap] = {}

        self.init_ui()
        self._load_and_display_data()

    def _center_on_screen(self):
        """Центрирует окно на экране, на котором оно будет показано."""
        try:
            screen_geometry = self.screen().geometry()
            window_geometry = self.frameGeometry()
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            self.move(window_geometry.topLeft())
        except Exception as e:
            logger.warning(f"Не удалось центрировать окно на экране: {e}")

    def init_ui(self):
        """Инициализирует пользовательский интерфейс с кастомными виджетами для Drag & Drop."""
        self.setWindowTitle("Редактор кластеров")
        self.setGeometry(0, 0, 1350, 900)
        self.setStyleSheet(styles.MAIN_WINDOW_STYLE)

        main_layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, 1)

        # --- Левая панель ---
        left_panel_widget = QWidget()
        left_panel_widget.setAcceptDrops(False)
        left_layout = QVBoxLayout(left_panel_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        left_title = QLabel(f"Фотосессия: {self.photo_session} (cписок кластеров)")
        left_title.setStyleSheet(styles.TITLE_LABEL_STYLE)
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Поиск...")
        self.search_bar.setStyleSheet(styles.SEARCH_BAR_STYLE)
        self.search_bar.textChanged.connect(self._on_search_text_changed)
        
        self.cluster_list_widget = ClusterDropListWidget(self)
        self.cluster_list_widget.setItemDelegate(ClusterItemDelegate(self))
        self.cluster_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.cluster_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.cluster_list_widget.setMovement(QListWidget.Movement.Static)
        self.cluster_list_widget.setSpacing(10)
        self.cluster_list_widget.setStyleSheet(styles.LIST_WIDGET_STYLE)
        self.cluster_list_widget.verticalScrollBar().setStyleSheet(styles.SCROLLBAR_STYLE)
        self.cluster_list_widget.itemDoubleClicked.connect(self._rename_cluster_action)
        self.cluster_list_widget.currentItemChanged.connect(self._on_cluster_selected)
        self.cluster_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cluster_list_widget.customContextMenuRequested.connect(self.show_cluster_context_menu)
        self.cluster_list_widget.itemsDropped.connect(self._handle_drop)
        self.cluster_list_widget.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.cluster_list_widget.viewport().setAcceptDrops(True)
        self.cluster_list_widget.setDropIndicatorShown(False)
        
        # --- Правая панель ---
        right_panel_widget = QWidget()
        right_panel_widget.setAcceptDrops(False)
        right_layout = QVBoxLayout(right_panel_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        self.right_panel_label = QLabel("Кластер")
        self.right_panel_label.setStyleSheet(styles.TITLE_LABEL_STYLE)
        
        self.image_list_widget = ImageDragListWidget(self)
        self.image_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list_widget.setSpacing(10)
        self.image_list_widget.setItemDelegate(ImageItemDelegate(self))
        self.image_list_widget.setStyleSheet(styles.LIST_WIDGET_STYLE)
        self.image_list_widget.verticalScrollBar().setStyleSheet(styles.SCROLLBAR_STYLE)
        self.image_list_widget.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        self.image_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list_widget.itemDoubleClicked.connect(self._open_image_viewer)
        
        # --- Кнопки и прогресс-бар ---
        self.export_button = QPushButton("Экспорт")
        self.export_button.setStyleSheet(styles.BUTTON_STYLE)
        export_menu = QMenu(self)
        export_menu.setStyleSheet(styles.MENU_STYLE)
        export_all = export_menu.addAction("Экспортировать всё")
        export_active = export_menu.addAction("Экспортировать активный кластер")
        self.export_button.setMenu(export_menu)
        export_all.triggered.connect(self._on_export_all_triggered)
        export_active.triggered.connect(self._on_export_active_triggered)
        
        self.save_button = QPushButton("Сохранить изменения")
        self.save_button.setStyleSheet(styles.BUTTON_STYLE)
        self.save_button.clicked.connect(self._save_changes)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.export_button)
        buttons_layout.addWidget(self.save_button)
        
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.search_bar)
        left_layout.addWidget(self.cluster_list_widget, 1)
        left_layout.addLayout(buttons_layout)
        
        right_layout.addWidget(self.right_panel_label)
        right_layout.addWidget(self.image_list_widget, 1)
        
        content_layout.addWidget(left_panel_widget, 5)
        content_layout.addWidget(right_panel_widget, 9)
        
        self.status_progress_bar = QProgressBar()
        self.status_progress_bar.setStyleSheet(styles.PROGRESS_BAR_STYLE_INACTIVE)
        self.status_progress_bar.setTextVisible(True)
        
        main_layout.addWidget(self.status_progress_bar)
        self._center_on_screen()

    # --- Методы управления данными и состоянием ---
    def _load_and_display_data(self):
        if not self.json_manager.load_data():
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить JSON.")
            return
        self.portrait_model = self.json_manager.portrait_data
        self.group_model = self.json_manager.group_data
        self._refresh_left_panel()

    def _get_clusters_from_model(self) -> Dict[str, List[Dict]]:
        clusters: Dict[str, List[Dict]] = {}
        for filename, data in self.portrait_model.items():
            face = data["faces"][0]
            label = str(face.get("cluster_label", -1))
            if label not in clusters: clusters[label] = []
            child_name = face.get("child_name")
            if label == "-1": child_name = "99-Noise"
            elif child_name and child_name.startswith("Unknown"):
                if not child_name.startswith("98-"): child_name = f"98-{child_name}"
            elif child_name and label not in ["-1", "group"]:
                try:
                    prefix = f"{int(label):02d}-"
                    if not child_name.startswith(prefix):
                        child_name = prefix + child_name.split('-', 1)[-1]
                except (ValueError, TypeError): pass
            face_copy = face.copy(); face_copy["child_name"] = child_name; face_copy["filename"] = filename
            clusters[label].append(face_copy)
        if self.group_model:
            clusters["group"] = [{"filename": f, "child_name": "_Group_Photos"} for f in self.group_model.keys()]
        return clusters

    def _get_files_for_cluster(self, cluster_id: str) -> List[str]:
        if cluster_id == "group":
            return sorted(self.group_model.keys())
        return sorted([f["filename"] for f in self.portrait_model.values() if str(f["faces"][0].get("cluster_label", -1)) == cluster_id])

    def get_cluster_count(self, cluster_id: str) -> int:
        if cluster_id == "group":
            return len(self.group_model)
        return sum(1 for data in self.portrait_model.values() if str(data["faces"][0].get("cluster_label", -1)) == cluster_id)
        
    def _get_image_path(self, filename: str, cluster_data: Dict) -> Path:
        if not self.is_clustered_mode:
            return self.flat_images_dir / filename

        if filename in self.pending_moves:
            source_id = self.pending_moves[filename]["source_id"]
            source_cluster_data = self._get_cluster_item_data_by_id(source_id)
            if source_cluster_data:
                source_folder_name = source_cluster_data["name"]
                path = self.sorted_dir / source_folder_name / filename
                if path.is_file():
                    logger.debug(f"Файл '{filename}' ожидает перемещения. Найден в исходной папке: {source_folder_name}")
                    return path
            
        if cluster_data:
            current_folder_name = cluster_data["name"]
            path = self.sorted_dir / current_folder_name / filename
            if path.is_file():
                return path

        expected_name = cluster_data["name"] if cluster_data else "unknown_folder"
        logger.warning(f"Файл '{filename}' не найден ни в исходной, ни в целевой папке. Возвращается ожидаемый путь.")
        return self.sorted_dir / expected_name / filename

    def _find_file_globally(self, filename: str) -> Optional[Path]:
        file_stem = Path(filename).stem
        logger.debug(f"Ищем файл {filename}")

        if self.is_clustered_mode:
            for d in self.sorted_dir.iterdir():
                if d.is_dir():
                    found_files = list(d.glob(f"{file_stem}.*"))
                    if found_files:
                        logger.debug(f"Файл '{filename}' найден в папке кластера: {d.name}")
                        return found_files[0]
        else:
            found_files = list(self.flat_images_dir.glob(f"{file_stem}.*"))
            if found_files:
                logger.debug(f"Файл '{filename}' найден в 'плоской' папке: {self.flat_images_dir.name}")
                return found_files[0]

        logger.warning(f"Файл '{filename}' не был найден ни в одном из расположений.")
        return None
    
    def _get_cluster_item_data_by_id(self, cluster_id: str) -> Optional[Dict]:
        for i in range(self.cluster_list_widget.count()):
            item = self.cluster_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole)["id"] == cluster_id:
                return item.data(Qt.ItemDataRole.UserRole)
        return None

    # --- Методы-слоты и методы обновления UI ---
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
        sorted_labels = sorted(
            clusters.keys(),
            key=lambda x: int(x) if x not in ["-1", "group"] else (9998 if x == "-1" else 9999),
        )
        
        item_to_select = None
        for label in sorted_labels:
            files = clusters[label]
            cluster_name = files[0].get("child_name") if files else f"Кластер {label}"
            
            preview_path = Path()
            if files:
                item_data_for_path = {"id": label, "name": cluster_name}
                preview_path = self._get_image_path(files[0]["filename"], item_data_for_path)

            pixmap = QPixmap(str(preview_path))
            if not pixmap.isNull():
                pixmap = pixmap.scaled(styles.PREVIEW_SIZE, styles.PREVIEW_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
            
            self.preview_pixmaps[label] = pixmap
            
            item_data = { "id": label, "name": cluster_name, "count": len(files),
                          "pixmap": pixmap, "is_changed": label in self.changed_cluster_ids }
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            self.cluster_list_widget.addItem(item)
            
            if label == active_id_before_refresh:
                item_to_select = item

        if item_to_select:
            self.cluster_list_widget.setCurrentItem(item_to_select)
        elif self.cluster_list_widget.count() > 0:
            self.cluster_list_widget.setCurrentRow(0)

    def _render_gallery(self, cluster_id: str):
        self._stop_gallery_load_if_running()
        cluster_data = self._get_cluster_item_data_by_id(cluster_id)
        if not cluster_data:
            self.image_list_widget.clear()
            self.right_panel_label.setText("Кластер не найден")
            return
        self.right_panel_label.setText(f"Кластер: {cluster_data['name']} ({cluster_data['count']} фото)")
        self.image_list_widget.clear()
        files_to_show = self._get_files_for_cluster(cluster_id)
        files_for_new_pixmaps = []
        for filename in files_to_show:
            if filename in self.image_pixmap_cache:
                pixmap = self.image_pixmap_cache[filename]
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
                item.setData(Qt.ItemDataRole.UserRole, {"filename": filename})
                self.image_list_widget.addItem(item)
            else:
                files_for_new_pixmaps.append(filename)
        if files_for_new_pixmaps:
            tasks = [{"filename": fname, "cluster_id": cluster_id, 
                      "full_path": self._get_image_path(fname, cluster_data)} for fname in files_for_new_pixmaps]
            if tasks:
                self._start_gallery_load(tasks)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_cluster_selected(self, current_item: QListWidgetItem, previous_item: Optional[QListWidgetItem] = None):
        if not current_item:
            self._stop_gallery_load_if_running()
            self.image_list_widget.clear()
            self.right_panel_label.setText("Кластер")
            self.active_cluster_id = None
            return
        cluster_data = current_item.data(Qt.ItemDataRole.UserRole)
        cluster_id = cluster_data["id"]
        if self.active_cluster_id == cluster_id:
            return
        self._stop_gallery_load_if_running()
        self.active_cluster_id = cluster_id
        self._render_gallery(cluster_id)

    def _stop_gallery_load_if_running(self):
        if hasattr(self, 'gallery_load_thread') and self.gallery_load_thread and self.gallery_load_thread.isRunning():
            if hasattr(self, 'gallery_load_worker') and self.gallery_load_worker:
                self.gallery_load_worker.requestInterruption()
            try:
                self.gallery_load_worker.widget_ready.disconnect(self._add_gallery_item)
                self.gallery_load_worker.finished.disconnect(self._on_gallery_load_finished)
            except (RuntimeError, TypeError): pass
            self.gallery_load_thread.quit()
            if not self.gallery_load_thread.wait(1000):
                self.gallery_load_thread.terminate()
            self.gallery_load_worker = None
            self.gallery_load_thread = None

    def _start_gallery_load(self, tasks: List[Dict]):
        self._stop_gallery_load_if_running()
        self.status_progress_bar.setStyleSheet(styles.PROGRESS_BAR_STYLE_ACTIVE)
        self.status_progress_bar.setRange(0, len(tasks))
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setFormat("Загрузка изображений... %p%")
        
        self.gallery_load_thread = QThread(self)
        self.gallery_load_worker = GalleryLoadWorker(tasks)
        self.gallery_load_worker.moveToThread(self.gallery_load_thread)
        self.gallery_load_worker.widget_ready.connect(self._add_gallery_item)
        self.gallery_load_worker.finished.connect(self._on_gallery_load_finished)
        self.gallery_load_thread.started.connect(self.gallery_load_worker.run)
        self.gallery_load_thread.start()

    @Slot(str, str, Path, QPixmap)
    def _add_gallery_item(self, filename: str, cluster_id: str, full_path: Path, pixmap: QPixmap):
        if self.active_cluster_id != cluster_id: return
        if filename not in self.image_pixmap_cache:
            self.image_pixmap_cache[filename] = pixmap
        
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
        item.setData(Qt.ItemDataRole.UserRole, {"filename": filename})
        self.image_list_widget.addItem(item)
        self.status_progress_bar.setValue(self.status_progress_bar.value() + 1)

    @Slot()
    def _on_gallery_load_finished(self):
        self.status_progress_bar.setStyleSheet(styles.PROGRESS_BAR_STYLE_INACTIVE)
        self.status_progress_bar.reset()
        self.status_progress_bar.setFormat("")
        if hasattr(self, 'gallery_load_thread') and self.gallery_load_thread:
            self.gallery_load_thread.quit()
            self.gallery_load_thread.wait()
            self.gallery_load_worker = None
            self.gallery_load_thread = None

    @Slot(QListWidgetItem)
    def _rename_cluster_action(self, item: QListWidgetItem):
        cluster_data = item.data(Qt.ItemDataRole.UserRole)
        cluster_id = cluster_data["id"]
        if cluster_id in ["group", "-1"]:
            QMessageBox.information(self, "Инфо", "Этот кластер нельзя переименовать."); return
            
        current_name = cluster_data["name"].split('-', 1)[-1]
        new_name, ok = QInputDialog.getText(self, "Переименование", "Новое имя (без префикса):", text=current_name)
        if ok and new_name.strip():
            self._handle_rename(cluster_id, new_name.strip())

    @Slot(QListWidgetItem)
    def _open_image_viewer(self, item: QListWidgetItem):
        item_data = item.data(Qt.ItemDataRole.UserRole)
        current_filename = item_data["filename"]
        cluster_id = self.active_cluster_id
        cluster_data = self._get_cluster_item_data_by_id(cluster_id)
        if not cluster_data: return
        all_filenames = self._get_files_for_cluster(cluster_id)
        try:
            current_index = all_filenames.index(current_filename)
        except ValueError:
            logger.warning(f"Файл {current_filename} не найден в списке файлов кластера {cluster_id}.")
            return
        
        image_paths = [self._get_image_path(fname, cluster_data) for fname in all_filenames]
        viewer = ImageViewer(image_paths, all_filenames, current_index, styles.SCROLLBAR_STYLE, self)
        viewer.exec()
   
    def show_cluster_context_menu(self, pos):
        item = self.cluster_list_widget.itemAt(pos)
        if not item: return
        menu = QMenu(self)
        menu.setStyleSheet(styles.MENU_STYLE)
        rename_action = menu.addAction("Переименовать")
        
        action = menu.exec(self.cluster_list_widget.mapToGlobal(pos))
        if action == rename_action:
            self._rename_cluster_action(item)

    @Slot(str, str, list)
    def _handle_drop(self, source_id: str, target_id: str, filenames: List[str]):
        active_id_on_drop = self.active_cluster_id
        for filename in filenames:
            if source_id == "group": file_data = self.group_model.pop(filename, None)
            else: file_data = self.portrait_model.pop(filename, None)
            if file_data is None: continue
            
            face = file_data["faces"][0]
            target_cluster_data = self._get_cluster_item_data_by_id(target_id)
            if not target_cluster_data: continue
            
            if target_id == "group":
                face["cluster_label"], face["child_name"] = None, "_Group_Photos"
                self.group_model[filename] = file_data
            else:
                face["cluster_label"], face["child_name"] = int(target_id), target_cluster_data["name"]
                self.portrait_model[filename] = file_data
        
            if filename in self.pending_moves:
                self.pending_moves[filename]["target_id"] = target_id
            else:
                self.pending_moves[filename] = {"source_id": source_id, "target_id": target_id}
        self.changed_cluster_ids.add(source_id)
        self.changed_cluster_ids.add(target_id)
        self._refresh_left_panel()
        if active_id_on_drop:
            self._render_gallery(active_id_on_drop)

    def _handle_rename(self, cluster_id: str, new_name: str):
        old_item_data = self._get_cluster_item_data_by_id(cluster_id)
        if not old_item_data: return
        old_disk_name = old_item_data["name"]
        old_path = self.sorted_dir / old_disk_name
        try:
            prefix = f"{int(cluster_id):02d}-"
            final_new_name = prefix + new_name
        except (ValueError, TypeError): final_new_name = new_name
        for data in self.portrait_model.values():
            if str(data["faces"][0].get("cluster_label")) == cluster_id:
                data["faces"][0]["child_name"] = final_new_name
        self.changed_cluster_ids.add(cluster_id)
        if self.is_clustered_mode and old_path.is_dir():
            new_path = self.sorted_dir / final_new_name
            try:
                if old_path.resolve() != new_path.resolve():
                    old_path.rename(new_path)
                    logger.info(f"Папка переименована: {old_path.name} -> {new_path.name}")
            except OSError as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось переименовать папку:\n{e}")
        self._refresh_left_panel()

    @Slot()
    def _on_export_all_triggered(self):
        portrait_ids = [cid for cid in self._get_clusters_from_model().keys() if cid not in ["-1", "group"]]
        if not portrait_ids: QMessageBox.information(self, "Инфо", "Нет кластеров для экспорта."); return
        self._start_export(portrait_ids)
        
    @Slot()
    def _on_export_active_triggered(self):
        active_id = self.active_cluster_id
        if active_id and active_id not in ["-1", "group"]: self._start_export([active_id])
        else: QMessageBox.warning(self, "Внимание", "Выберите портретный кластер.")

    def _start_export(self, cluster_ids: List[str]):
        base_output_dir = self.data_dir.parent / self.session_name /f"Выбор_Фото_{self.photo_session}"
        tasks = []
        for cid in cluster_ids:
            cluster_data = self._get_cluster_item_data_by_id(cid)
            if not cluster_data: continue
            disk_folder_name = cluster_data["name"]
            output_folder = base_output_dir / disk_folder_name
            filenames = self._get_files_for_cluster(cid)
            for fname in filenames:
                tasks.append({
                    "source_path": self._get_image_path(fname, cluster_data),
                    "output_path": output_folder / Path(fname).name,
                    "child_name": cluster_data["name"].split('-', 1)[-1].strip()
                })
        
        if not tasks: QMessageBox.information(self, "Инфо", "Нет файлов для экспорта."); return
        
        first_image_path = tasks[0]["source_path"]
        dialog = EnhanceSettingsDialog(first_image_path, self)
        if dialog.exec() != QDialog.Accepted: return
        
        enhancement_factors = dialog.get_enhancement_factors()
        logger.info(f"Параметры обработки фотографий перед экспортом:")
        logger.info(f"<i>{enhancement_factors}<i>")
        
        self.status_progress_bar.setStyleSheet(styles.PROGRESS_BAR_STYLE_ACTIVE)
        self.status_progress_bar.setRange(0, len(tasks))
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setFormat("Экспорт... %p%")
        
        self.export_worker = ExportWorker(tasks, os.cpu_count() or 4, enhancement_factors)
        self.export_thread = QThread()
        self.export_worker.moveToThread(self.export_thread)
        self.export_worker.progress_updated.connect(self.status_progress_bar.setValue)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_thread.started.connect(self.export_worker.run)
        self.export_thread.start()
      
    @Slot(str)
    def _on_export_finished(self, message: str):
        self.status_progress_bar.setStyleSheet(styles.PROGRESS_BAR_STYLE_INACTIVE)
        self.status_progress_bar.reset()
        self.status_progress_bar.setFormat("")
        QMessageBox.information(self, "Экспорт завершен", message)
        if hasattr(self, 'export_thread'):
            self.export_thread.quit(); self.export_thread.wait()

    def _perform_save(self) -> bool:
        self.json_manager.portrait_data, self.json_manager.group_data = self.portrait_model, self.group_model
        if not self.json_manager.save_data():
            logger.error("Ошибка", "Не удалось сохранить JSON.", file=sys.stderr)
            return False
        
        if self.is_clustered_mode:
            self._process_pending_moves()
        else:
            self.pending_moves.clear()
            self.changed_cluster_ids.clear()
            self._refresh_left_panel()
        return True

    def _save_changes(self):
        if not self.changed_cluster_ids and not self.pending_moves:
            QMessageBox.information(self, "Инфо", "Нет изменений для сохранения.")
            return
        
        msg = "Сохранить все изменения в кластерах?"
        if self.is_clustered_mode and self.pending_moves:
            msg += f"\n\nБудет перемещено {len(self.pending_moves)} фото и все связанные с ними файлы."
        
        reply = QMessageBox.question(self, "Сохранение", msg, QMessageBox.Save | QMessageBox.Cancel)
        if reply == QMessageBox.Save and self._perform_save():
            logger.info("\n<b>Все внесенные изменения сохранены</b>")

    def closeEvent(self, event):
        self._stop_gallery_load_if_running()
        if not self.changed_cluster_ids and not self.pending_moves:
            event.accept()
            return
        reply = QMessageBox.question(
            self, "Несохраненные изменения",
            "У вас есть несохраненные изменения. Хотите сохранить их перед выходом?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save
        )
        if reply == QMessageBox.Save:
            if self._perform_save(): event.accept()
            else: event.ignore()
        elif reply == QMessageBox.Discard: event.accept()
        else: event.ignore()

    def _get_folder_for_cluster_id(self, cluster_id: str) -> Optional[Path]:
        cluster_data = self._get_cluster_item_data_by_id(cluster_id)
        if not cluster_data:
            logger.warning(f"Не удалось найти данные для cluster_id '{cluster_id}' при поиске папки.")
            return None
        path = self.sorted_dir / cluster_data["name"]
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError as e:
            logger.error(f"Не удалось создать/проверить папку кластера {path}: {e}")
            QMessageBox.critical(self, "Ошибка I/O", f"Не удалось создать папку:\n{path}\n\n{e}")
            return None

    def _process_pending_moves(self):
        if not self.pending_moves: return
        tasks = [{"filename": fname, **move_info} for fname, move_info in self.pending_moves.items()]
        
        self.status_progress_bar.setStyleSheet(styles.PROGRESS_BAR_STYLE_ACTIVE)
        self.status_progress_bar.setRange(0, len(tasks))
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setFormat(f"Перемещение {len(tasks)} фото...")

        self.total_moved = 0
        self.total_errors = 0
        
        self.move_thread = QThread(self)
        self.move_worker = MoveWorker(tasks, self)
        self.move_worker.moveToThread(self.move_thread)
        self.move_worker.task_finished.connect(self._on_move_task_finished)
        self.move_worker.finished.connect(self._on_move_all_finished)
        self.move_thread.started.connect(self.move_worker.run)
        self.move_thread.start()

    @Slot(int, int)
    def _on_move_task_finished(self, moved_count: int, error_count: int):
        self.total_moved += moved_count
        self.total_errors += error_count
        self.status_progress_bar.setValue(self.status_progress_bar.value() + 1)

    @Slot()
    def _on_move_all_finished(self):
        self.status_progress_bar.setStyleSheet(styles.PROGRESS_BAR_STYLE_INACTIVE)
        self.status_progress_bar.reset()
        self.status_progress_bar.setFormat("")
        
        msg = f"Изменения успешно сохранены.\n\nПеремещено {self.total_moved} файла(ов)."
        if self.total_errors > 0:
            msg += f"\nВозникло ошибок: {self.total_errors}."
        QMessageBox.information(self, "Результат перемещения", msg)

        self.pending_moves.clear()
        self.changed_cluster_ids.clear()
        self._refresh_left_panel()

        if hasattr(self, 'move_thread') and self.move_thread:
            self.move_thread.quit()
            self.move_thread.wait()
            self.move_worker = None
            self.move_thread = None

# 3. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":

    log_level = "INFO" 
    if IS_MANAGED_RUN and pysm_context:
        log_level = pysm_context.get("sys_log_level", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout
    )

    # Запущен редактор кластеров
    # Переменная для определения статуса экспорта. Экспорт не выполнялся
    export_status = 0
    pysm_context.set("var_jpg_move", export_status)

    if not IS_COMMON_AVAILABLE or not IS_MANAGED_RUN:
        msg = "Критическая ошибка: Скрипт требует запуска из среды PySM и наличия общих библиотек (_common)."
        logging.critical(msg)
        if 'QApplication' in locals() and QApplication.instance():
             QMessageBox.critical(None, "Ошибка запуска", msg)
        else:
            print(msg, file=sys.stderr)
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)

    session_path_str = pysm_context.get("wf_session_path")
    session_name = pysm_context.get("wf_session_name")
    photo_session = pysm_context.get("wf_photo_session")

    if not all([session_path_str, session_name, photo_session]):
        QMessageBox.critical(None, "Ошибка контекста", "Не удалось получить необходимые переменные (wf_...) из контекста PySM.")
        sys.exit(1)

    try:
        base_path = Path(session_path_str) / session_name
        data_dir = base_path / "Output" / f"Analysis_{photo_session}"
        images_dir = data_dir / "JPG"
        sorted_dir = base_path / "Output" / f"Claster_{photo_session}"
        
        if not data_dir.is_dir(): data_dir.mkdir(parents=True, exist_ok=True)
        if not images_dir.is_dir(): images_dir.mkdir(parents=True, exist_ok=True)

    except Exception as e:
        QMessageBox.critical(None, "Ошибка путей", f"Не удалось инициализировать директории:\n{e}")
        sys.exit(1)
    
    window = MainWindow(data_dir, images_dir, sorted_dir, photo_session, session_name)
    window.show()
    sys.exit(app.exec())