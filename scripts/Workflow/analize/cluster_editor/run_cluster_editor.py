# 3. БЛОК: run_cluster_editor.py (ПОЛНЫЙ ОБНОВЛЕННЫЙ КОД)
# ==============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cluster_editor.py
=====================
Модуль для редактирования кластеров изображений с графическим интерфейсом на основе PySide6.
"""

import sys
import re
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QInputDialog, QProgressBar, QMessageBox, QLineEdit, QMenu,
    QListWidget, QListWidgetItem, QDialog, QCheckBox
)
from PySide6.QtGui import QPixmap, QAction, QColor
from PySide6.QtCore import Qt, Signal, Slot, QThread

# Внутренние модули
IS_COMMON_AVAILABLE = False
IS_MANAGED_RUN = False


try:
    current_script_dir = Path(__file__).resolve().parent
    if str(current_script_dir) not in sys.path: sys.path.insert(0, str(current_script_dir))
    project_root = current_script_dir.parent
    if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

    from pysm_lib import pysm_context, theme_api
    IS_MANAGED_RUN = True

    from _lib.editor_viewer import ImageViewer
    from _lib.editor_workers import GalleryPrepareWorker, GalleryLoadWorker, ExportWorker
    from _lib.editor_delegates import ClusterItemDelegate, ImageItemDelegate
    from _lib.editor_widgets import ImageDragListWidget, ClusterDropListWidget
    from _lib.editor_dialogs import EnhanceSettingsDialog
    from _lib import editor_styles as styles
    # --- ИЗМЕНЕНИЕ: Импортируем новые классы ---
    from _lib.data_manager import ClusterDataManager
    from _lib.data_models import Face # Face нужен для type hinting

except ImportError as e:
    print(f"Ошибка импорта: {e}", file=sys.stderr)


logger = logging.getLogger(__name__)


def get_color_from_css(css_string: Optional[str], default_color: str) -> QColor:
    """Извлекает HEX/имя цвета из CSS-строки 'property: value;'."""
    if not css_string:
        return QColor(default_color)
    
    match = re.search(r":\s*(#[0-9a-fA-F]{3,6}\b|[a-zA-Z]+)", css_string)
    if match:
        color_val_str = match.group(1).strip()
        temp_color = QColor(color_val_str)
        if temp_color.isValid():
            return temp_color
            
    return QColor(default_color)

class MainWindow(QWidget):
    def __init__(self, data_dir: Path, images_dir: Path, photo_session: str, session_name: str, mode: str):
        super().__init__()
        self.data_dir = data_dir
        self.flat_images_dir = images_dir
        self.photo_session = photo_session
        self.session_name = session_name
        self.mode = mode

        if self.mode == 'face':
            self.mode_config = {
                "mode_name": "face", "json_field_id": "cluster_label", "json_field_name": "child_name",
                "window_title": f"Редактор кластеров [по Лицам] - {photo_session}",
                "name_prefix_logic": lambda cid: f"{int(cid):02d}-" if str(cid).isdigit() else "",
            }
        elif self.mode == 'location':
            self.mode_config = {
                "mode_name": "location", "json_field_id": "location_cluster", "json_field_name": "location_name",
                "window_title": f"Редактор кластеров [по Локациям] - {photo_session}",
                "name_prefix_logic": lambda cid: "",
            }
        else:
            raise ValueError(f"Неизвестный режим работы: {self.mode}")

        # --- ИЗМЕНЕНИЕ: Инициализируем DataManager и удаляем старые модели ---
        self.data_manager = ClusterDataManager(
            self.data_dir / "info_portrait_faces.json",
            self.data_dir / "info_group_faces.json",
        )
        
        self.active_cluster_id: Optional[str] = None
        self.preview_pixmaps: Dict[str, QPixmap] = {}
        self.image_pixmap_cache: Dict[str, QPixmap] = {}

        hover_css = theme_api.get_dynamic_style("delegate_hover_border", "color: #0078d7;")
        changed_css = theme_api.get_dynamic_style("delegate_changed_indicator", "color: #f0ad4e;")
        preview_bg_css = theme_api.get_dynamic_style("delegate_preview_background", "color: #e8e8e8;")
        secondary_text_css = theme_api.get_dynamic_style("delegate_secondary_text", "color: #555555;")

        hover_color = get_color_from_css(hover_css, "#0078d7")
        changed_color = get_color_from_css(changed_css, "#f0ad4e")
        preview_bg = get_color_from_css(preview_bg_css, "#e8e8e8")
        secondary_text = get_color_from_css(secondary_text_css, "#555555")

        self.cluster_delegate = ClusterItemDelegate(
            hover_border_color=hover_color, changed_indicator_color=changed_color,
            preview_bg_color=preview_bg, secondary_text_color=secondary_text, parent=self
        )
        self.image_delegate = ImageItemDelegate(hover_border_color=hover_color, parent=self)

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
        """Инициализирует пользовательский интерфейс."""
        self.setWindowTitle(self.mode_config["window_title"])
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
        self.cluster_list_widget.setItemDelegate(self.cluster_delegate)
        self.cluster_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.cluster_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.cluster_list_widget.setMovement(QListWidget.Movement.Static)
        self.cluster_list_widget.setSpacing(10)
        self.cluster_list_widget.itemDoubleClicked.connect(self._rename_cluster_action)
        self.cluster_list_widget.currentItemChanged.connect(self._on_cluster_selected)
        self.cluster_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cluster_list_widget.customContextMenuRequested.connect(self.show_cluster_context_menu)
        self.cluster_list_widget.itemsDropped.connect(self._handle_drop)
        self.cluster_list_widget.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.cluster_list_widget.viewport().setAcceptDrops(True)
        self.cluster_list_widget.setDropIndicatorShown(False)
        
        right_panel_widget = QWidget()
        right_layout = QVBoxLayout(right_panel_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        self.right_panel_label = QLabel("Кластер")
        
        self.image_list_widget = ImageDragListWidget(self)
        self.image_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list_widget.setSpacing(10)
        self.image_list_widget.setItemDelegate(self.image_delegate)
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

    # --- ИЗМЕНЕНИЕ: Упрощенные методы, делегирующие DataManager ---
    def _load_and_display_data(self):
        if not self.data_manager.load_data():
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить JSON.")
            return
        self._refresh_left_panel()

    def _get_clusters_from_model(self) -> Dict[str, List[Face]]:
        return self.data_manager.get_clusters(self.mode_config)

    def _get_files_for_cluster(self, cluster_id: str) -> List[str]:
        return self.data_manager.get_files_for_cluster(self.mode_config, cluster_id)

    def get_cluster_count(self, cluster_id: str) -> int:
        return len(self._get_files_for_cluster(cluster_id))
        
    def _get_image_path(self, filename: str) -> Path:
        return self.flat_images_dir / filename

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
        
        if self.mode_config["mode_name"] == 'face':
            sorted_labels = sorted(
                clusters.keys(), key=lambda x: int(x) if x.isdigit() else (9998 if x == "-1" else 9999),
            )
        else:
            sorted_labels = sorted(clusters.keys())

        item_to_select = None
        for label in sorted_labels:
            faces = clusters[label]
            cluster_name = faces[0].effective_name if faces else f"Кластер {label}"
            
            preview_path = Path()
            if faces:
                preview_path = self._get_image_path(faces[0].filename)

            pixmap = QPixmap(str(preview_path))
            if not pixmap.isNull():
                pixmap = pixmap.scaled(styles.PREVIEW_SIZE, styles.PREVIEW_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
            
            self.preview_pixmaps[label] = pixmap
            
            item_data = { "id": label, "name": cluster_name, "count": len(faces),
                          "pixmap": pixmap, "is_changed": False } # is_changed теперь в DataManager
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

    # 2. БЛОК: Метод _render_gallery (заменить в run_cluster_editor.py)
    # ==============================================================================
    def _render_gallery(self, cluster_id: str):
        self._stop_gallery_load_if_running()
        self._stop_gallery_prepare_if_running() # Останавливаем и подготовку

        cluster_data = self._get_cluster_item_data_by_id(cluster_id)
        if not cluster_data:
            self.image_list_widget.clear()
            self.right_panel_label.setText("Кластер не найден")
            return
            
        self.right_panel_label.setText(f"Кластер: {cluster_data['name']} ({cluster_data['count']} фото)")
        self.image_list_widget.clear()
        
        # Запускаем подготовку в фоновом потоке
        self.prepare_thread = QThread(self)
        self.prepare_worker = GalleryPrepareWorker(
            self.data_manager, self.mode_config, cluster_id, self.image_pixmap_cache
        )
        self.prepare_worker.moveToThread(self.prepare_thread)
        
        self.prepare_worker.prepared.connect(self._on_gallery_prepared)
        self.prepare_worker.finished.connect(self._on_gallery_prepare_finished)
        
        self.prepare_thread.started.connect(self.prepare_worker.run)
        self.prepare_thread.start()
        
    # 3. БЛОК: НОВЫЕ МЕТОДЫ (добавить в MainWindow в run_cluster_editor.py)
    # ==============================================================================
    def _stop_gallery_prepare_if_running(self):
        """Останавливает поток подготовки галереи, если он запущен."""
        if hasattr(self, 'prepare_thread') and self.prepare_thread and self.prepare_thread.isRunning():
            if hasattr(self, 'prepare_worker') and self.prepare_worker:
                self.prepare_worker.requestInterruption()
            try:
                self.prepare_worker.prepared.disconnect(self._on_gallery_prepared)
                self.prepare_worker.finished.disconnect(self._on_gallery_prepare_finished)
            except (RuntimeError, TypeError):
                pass
            self.prepare_thread.quit()
            if not self.prepare_thread.wait(1000):
                self.prepare_thread.terminate()
            self.prepare_worker = None
            self.prepare_thread = None

    @Slot(list, list)
    def _on_gallery_prepared(self, cached_items: List[Dict], uncached_tasks: List[Dict]):
        """
        Слот, который вызывается после завершения подготовки.
        Быстро заполняет галерею кэшированными элементами и запускает загрузку остальных.
        """
        # 1. Быстро добавляем все, что уже есть в кэше
        for item_data in cached_items:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.DecorationRole, item_data["pixmap"])
            item.setData(Qt.ItemDataRole.UserRole, {"filename": item_data["filename"]})
            self.image_list_widget.addItem(item)
            
        # 2. Если остались некэшированные - запускаем для них старый воркер
        if uncached_tasks:
            # Дополняем задачи полными путями
            full_tasks = [
                {**task, "full_path": self._get_image_path(task["filename"])}
                for task in uncached_tasks
            ]
            self._start_gallery_load(full_tasks)

    @Slot()
    def _on_gallery_prepare_finished(self):
        """Очищает ресурсы после завершения потока подготовки."""
        if hasattr(self, 'prepare_thread') and self.prepare_thread:
            self.prepare_thread.quit()
            self.prepare_thread.wait()
            self.prepare_worker = None
            self.prepare_thread = None


    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_cluster_selected(self, current_item: QListWidgetItem, previous_item: Optional[QListWidgetItem] = None):
        if not current_item:
            self._stop_gallery_prepare_if_running()
            self._stop_gallery_load_if_running()
            self.image_list_widget.clear()
            self.right_panel_label.setText("Кластер")
            self.active_cluster_id = None
            return
            
        cluster_data = current_item.data(Qt.ItemDataRole.UserRole)
        cluster_id = cluster_data["id"]
        if self.active_cluster_id == cluster_id:
            return
            
        self._stop_gallery_prepare_if_running()
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
            if not self.gallery_load_thread.wait(1000): self.gallery_load_thread.terminate()
            self.gallery_load_worker = None; self.gallery_load_thread = None

    def _start_gallery_load(self, tasks: List[Dict]):
        self._stop_gallery_load_if_running()
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
        self.status_progress_bar.reset(); self.status_progress_bar.setFormat("")
        if hasattr(self, 'gallery_load_thread') and self.gallery_load_thread:
            self.gallery_load_thread.quit(); self.gallery_load_thread.wait()
            self.gallery_load_worker = None; self.gallery_load_thread = None

    @Slot(QListWidgetItem)
    def _rename_cluster_action(self, item: QListWidgetItem):
        cluster_data = item.data(Qt.ItemDataRole.UserRole)
        cluster_id = cluster_data["id"]
        if self.mode_config["mode_name"] == 'face' and cluster_id in ["group", "-1"]:
            QMessageBox.information(self, "Инфо", "Этот кластер нельзя переименовать."); return
            
        current_name_no_prefix = cluster_data["name"].split('-', 1)[-1]
        new_name, ok = QInputDialog.getText(self, "Переименование", "Новое имя (без префикса):", text=current_name_no_prefix)
        if ok and new_name.strip():
            self._handle_rename(cluster_id, new_name.strip())

    @Slot(QListWidgetItem)
    def _open_image_viewer(self, item: QListWidgetItem):
        current_filename = item.data(Qt.ItemDataRole.UserRole)["filename"]
        cluster_id = self.active_cluster_id
        all_filenames = self._get_files_for_cluster(cluster_id)
        try:
            current_index = all_filenames.index(current_filename)
        except ValueError:
            logger.warning(f"Файл {current_filename} не найден в списке файлов кластера {cluster_id}.")
            return
        
        image_paths = [self._get_image_path(fname) for fname in all_filenames]
        viewer = ImageViewer(image_paths, all_filenames, current_index, styles.SCROLLBAR_STYLE, self)
        viewer.exec()
   
    def show_cluster_context_menu(self, pos):
        item = self.cluster_list_widget.itemAt(pos)
        menu = QMenu(self)
        
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

    @Slot(str, str, list)
    def _handle_drop(self, source_id: str, target_id: str, filenames: List[str]):
        target_cluster_data = self._get_cluster_item_data_by_id(target_id)
        if not target_cluster_data:
            return
        
        self.data_manager.move_images_to_cluster(self.mode_config, target_id, target_cluster_data["name"], filenames)
        self._refresh_left_panel()
        
        if self.active_cluster_id:
            if self._get_cluster_item_data_by_id(self.active_cluster_id):
                 self._render_gallery(self.active_cluster_id)
            else:
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
                    "source_path": self._get_image_path(fname),
                    "output_path": output_folder / Path(fname).name,
                    "child_name": cluster_data["name"].split('-', 1)[-1].strip()
                })
        
        if not tasks: QMessageBox.information(self, "Инфо", "Нет файлов для экспорта."); return
        
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
        """Сохраняет данные JSON."""
        if not self.data_manager.save_data():
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить JSON.")
            return False
        
        self._refresh_left_panel()
        return True

    def _save_changes(self):
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
        self._stop_gallery_load_if_running()
        if not self.data_manager.has_changes():
            event.accept()
            return

        reply = QMessageBox.question(
            self, "Несохраненные изменения",
            "У вас есть несохраненные изменения. Хотите сохранить их перед выходом?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save
        )
        
        if reply == QMessageBox.Cancel: event.ignore()
        elif reply == QMessageBox.Discard: event.accept()
        elif reply == QMessageBox.Save:
            if self._perform_save(): event.accept()
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить данные. Выход отменен.")
                event.ignore()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Редактор кластеров изображений.")
    parser.add_argument(
        "--mode", type=str, choices=["face", "location"], default="face",
        help="Режим кластеризации: 'face' (по лицам) или 'location' (по локациям)."
    )
    args = parser.parse_args()

    log_level = "INFO" 
    if IS_MANAGED_RUN and pysm_context:
        log_level = pysm_context.get("sys_log_level", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s", stream=sys.stdout
    )

    export_status = 0
    pysm_context.set("var_jpg_move", export_status)

    if not IS_MANAGED_RUN:
        msg = "Критическая ошибка: Скрипт требует запуска из среды PySM."
        logging.critical(msg)
        if 'QApplication' in locals() and QApplication.instance():
             QMessageBox.critical(None, "Ошибка запуска", msg)
        else:
            print(msg, file=sys.stderr)
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)
    theme_api.apply_theme_to_app(app)

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
        
        if not data_dir.is_dir(): data_dir.mkdir(parents=True, exist_ok=True)
        if not images_dir.is_dir(): images_dir.mkdir(parents=True, exist_ok=True)

    except Exception as e:
        QMessageBox.critical(None, "Ошибка путей", f"Не удалось инициализировать директории:\n{e}")
        sys.exit(1)
    
    window = MainWindow(data_dir, images_dir, photo_session, session_name, args.mode)
    window.show()
    sys.exit(app.exec())