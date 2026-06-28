# analize/cluster_editor/_lib/editor_menus.py
# -*- coding: utf-8 -*-

"""
Модуль управления контекстными меню и действиями (Actions) редактора.
Обеспечивает логику вызова меню для списков, галереи и панели лиц.
"""

import logging
from pathlib import Path

from PySide6.QtWidgets import QMenu, QMessageBox, QInputDialog, QFileDialog, QDialog
from PySide6.QtCore import Qt, QObject, Slot

from .editor_dialogs import RenameDialog, StudentSelectionDialog

logger = logging.getLogger(__name__)

class EditorMenuManager(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window

    @Slot(object)
    def rename_cluster_action(self, item):
        w = self.window
        data = item.data(Qt.ItemDataRole.UserRole)
        cid = data["id"]
        
        if w.mode == 'matches': return 
        if cid in ["trash", "error_matches"]: return
        if w.mode == 'face' and cid in ["group", "-1"]: return
        
        current_name = w.data_manager.strategy._strip_name_prefix(data["name"])
        new_name = None
        
        if w.mode == 'face':
            students = w.data_manager.available_students(except_cluster_id=cid)
            dialog = StudentSelectionDialog(students, data.get("student_id", ""), w)
            dialog.setWindowTitle("Назначить или сменить ученика")
            if dialog.exec() == QDialog.Accepted:
                new_name = dialog.selected_student_id()
        elif w.mode == 'location':
            dialog = RenameDialog(w.predefined_cluster_names, current_name, w)
            if dialog.exec() == QDialog.Accepted:
                new_name = dialog.get_selected_name()
        else:
            text, ok = QInputDialog.getText(w, "Переименование", "Имя:", text=current_name)
            if ok: new_name = text
            
        if new_name and new_name.strip():
            try:
                w.data_manager.rename_cluster(dict(), cid, new_name.strip())
                w._refresh_left_panel()
            except ValueError as exc:
                QMessageBox.warning(w, "Назначение ученика", str(exc))

    def show_cluster_context_menu(self, pos):
        w = self.window
        item = w.cluster_list_widget.itemAt(pos)
        menu = QMenu()
        
        if w.mode == 'cleaning':
            act_empty = menu.addAction("Очистить корзину (удалить навсегда)")
            is_trash = bool(item and item.data(Qt.ItemDataRole.UserRole)["id"] == "trash")
            act_empty.setEnabled(is_trash)
            
            if act_empty.isEnabled():
                act_empty.triggered.connect(lambda: w._save_changes(silent=False))
                
        elif w.mode == 'matches':
            action_load = menu.addAction("📂 Открыть другую съемку (JSON)...")
            action_load.triggered.connect(self._load_other_session)
            if item: menu.addSeparator()
            
        elif w.mode != 'matches':
            create_label = (
                "Создать кластер для ученика"
                if w.mode == 'face'
                else "Создать кластер"
            )
            menu.addAction(create_label).triggered.connect(self._create_cluster)
            if item:
                rename_label = (
                    "Назначить/сменить ученика"
                    if w.mode == 'face'
                    else "Переименовать"
                )
                menu.addAction(rename_label).triggered.connect(
                    lambda: self.rename_cluster_action(item)
                )
                if w.mode == 'face':
                    menu.addSeparator()
                    gender_menu = menu.addMenu("Принудительно установить пол")
                    gender_menu.addAction("Мужской (Male)").triggered.connect(lambda: self._set_cluster_gender(item, "Male"))
                    gender_menu.addAction("Женский (Female)").triggered.connect(lambda: self._set_cluster_gender(item, "Female"))                

        menu.exec(w.cluster_list_widget.mapToGlobal(pos))

    def _set_cluster_gender(self, item, gender: str):
        w = self.window
        if not item: return
        cid = item.data(Qt.ItemDataRole.UserRole)["id"]
        
        if cid in["group", "-1", "trash", "error_matches"]: 
            QMessageBox.warning(w, "Внимание", "Для этого кластера нельзя изменить пол.")
            return
            
        w.data_manager.set_cluster_gender(cid, gender)
        w._refresh_left_panel() 
        
        if w.active_cluster_id == cid:
            w._render_gallery(cid) 
            if hasattr(w, 'face_details_widget') and w.face_details_widget.count() > 0:
                current_face_item = w.face_details_widget.currentItem()
                if current_face_item:
                    w._on_face_item_clicked(current_face_item)

    def _load_other_session(self):
        w = self.window
        if w.data_manager.has_changes():
            reply = QMessageBox.question(w, "Смена сессии", 
                                         "Есть несохраненные изменения. Сохранить перед переключением?", 
                                         QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel: return
            if reply == QMessageBox.StandardButton.Save:
                w._save_changes(silent=False)

        file_path, _ = QFileDialog.getOpenFileName(
            w, 
            "Выберите файл данных (info_faces.json / info_group_faces.json)",
            str(w.working_dir),
            "JSON Files (info_faces.json)"
        )
        
        if not file_path: return
        
        new_path = Path(file_path)
        w.working_dir = new_path.parent
        w.working_images_dir = w.working_dir / "JPG"
        w.photo_session = w.working_dir.name.replace("Analysis_", "")
        w.setWindowTitle(w.data_manager.strategy.get_window_title(w.photo_session))
        w.data_manager.switch_working_session(new_path)
        w._reload_selected_photo_numbers(w.btn_filter_selected_photos.isChecked())
        w._load_and_display_data()        

    def _create_cluster(self):
        w = self.window
        new_name = None
        
        # --- ИЗМЕНЕНИЕ: В режиме location используем диалог с выпадающим списком ---
        if w.mode == 'face':
            available_students = w.data_manager.available_students()
            if not available_students:
                QMessageBox.information(
                    w,
                    "Создание кластера",
                    "В файле *.list нет свободных записей учеников.",
                )
                return
            dialog = StudentSelectionDialog(available_students, parent=w)
            dialog.setWindowTitle("Ученик для нового кластера")
            if dialog.exec() == QDialog.Accepted:
                new_name = dialog.selected_student_id()
        elif w.mode == 'location':
            dialog = RenameDialog(w.predefined_cluster_names, "", w)
            dialog.setWindowTitle("Новый кластер локации")
            if dialog.exec() == QDialog.Accepted:
                new_name = dialog.get_selected_name()
        # --- Стандартная логика (для face и cleaning) ---
        else:
            text, ok = QInputDialog.getText(w, "Новый кластер", "Имя:")
            if ok: new_name = text
            
        # Создаем кластер, если пользователь ввел/выбрал имя и не нажал Отмена
        if new_name and new_name.strip():
            try:
                w.data_manager.create_cluster(dict(), new_name.strip())
                w._refresh_left_panel()
            except ValueError as exc:
                QMessageBox.warning(w, "Создание кластера", str(exc))

    def show_gallery_context_menu(self, pos):
        w = self.window
        item = w.image_list_widget.itemAt(pos)
        if not item: return
        menu = QMenu()
        
        if w.mode == 'location':
            action = menu.addAction("📸 Сделать обложкой локации")
            action.triggered.connect(lambda: self._set_cover_action(item))
            
        if w.mode == 'cleaning':
            action_trash_face = menu.addAction("🗑️ Переместить лицо в корзину")
            action_trash_face.triggered.connect(lambda: self._trash_item_face(item))
            
            action_trash_photo = menu.addAction("🗑️ Переместить все лица с фото в корзину")
            action_trash_photo.triggered.connect(lambda: self._trash_item_all_faces(item))
            
        if not menu.isEmpty():
            menu.exec(w.image_list_widget.mapToGlobal(pos))

    def _set_cover_action(self, item):
        w = self.window
        fname = item.data(Qt.ItemDataRole.UserRole)["filename"]
        cid = w.active_cluster_id
        if not cid: return
        w.data_manager.set_location_cover(cid, fname)
        w._refresh_left_panel()

    def _trash_item_face(self, item):
        w = self.window
        user_data = item.data(Qt.ItemDataRole.UserRole)
        fname = user_data["filename"]
        face_idx = user_data.get("face_index")
        
        if face_idx is None: return
        
        w.data_manager.move_images_to_cluster(
            dict(), "trash", "🗑️ КОРЗИНА", [fname], {fname:[face_idx]}
        )
        w._refresh_left_panel()
        w._render_gallery(w.active_cluster_id, preserve_state=True)

    def _trash_item_all_faces(self, item):
        w = self.window
        user_data = item.data(Qt.ItemDataRole.UserRole)
        fname = user_data["filename"]
        
        w.data_manager.move_images_to_cluster(
            dict(), "trash", "🗑️ КОРЗИНА", [fname], None
        )
        w._refresh_left_panel()
        w._render_gallery(w.active_cluster_id, preserve_state=True)

    def show_face_details_context_menu(self, pos):
        w = self.window
        if w.mode in['cleaning', 'location']:
            return

        item = w.face_details_widget.itemAt(pos)
        if not item: return

        current_photo_item = w.image_list_widget.currentItem()
        if not current_photo_item: return
        fname = current_photo_item.data(Qt.ItemDataRole.UserRole)["filename"]
        
        record = w.data_manager.records.get(fname)
        if not record: return
        
        face_idx = item.data(Qt.ItemDataRole.UserRole)
        if face_idx is None or face_idx >= len(record.faces): return
        
        face = record.faces[face_idx]
        
        target_id = None
        if face.cluster_label is not None:
            target_id = face.cluster_label
        elif face.extra_data.get('matched_portrait_cluster_label') is not None:
            target_id = face.extra_data.get('matched_portrait_cluster_label')

        menu = QMenu()
        action_open = menu.addAction("📂 Перейти к кластеру")
        
        if target_id is not None:
            target_id_str = str(target_id)
            action_open.triggered.connect(lambda: self._activate_cluster_by_id(target_id_str))
        else:
            action_open.setEnabled(False)
            action_open.setText("Кластер не определен")

        menu.exec(w.face_details_widget.mapToGlobal(pos))

    def _activate_cluster_by_id(self, cluster_id: str):
        w = self.window
        if w.search_bar.text():
            w.search_bar.clear()
            
        for i in range(w.cluster_list_widget.count()):
            item = w.cluster_list_widget.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            
            if str(data["id"]) == str(cluster_id):
                w.cluster_list_widget.setCurrentItem(item)
                w.cluster_list_widget.scrollToItem(item)
                w.cluster_list_widget.setFocus()
                return
        
        QMessageBox.information(w, "Поиск", f"Кластер с ID {cluster_id} не найден в текущем списке.")
