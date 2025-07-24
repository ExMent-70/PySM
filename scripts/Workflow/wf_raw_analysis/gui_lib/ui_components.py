# gui_lib/ui_components.py

import os
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QListWidget, QAbstractItemView, QFileDialog, QMessageBox
)

# Импортируем тексты из созданного нами файла
from .ui_texts import UI_TEXTS
from .custom_widgets import PathSelectorWidget


class BaseGroup(QGroupBox):
    """Базовый класс для всех групп виджетов для избежания дублирования."""
    def get_data(self) -> Dict[str, Any]:
        """Собирает данные из виджетов этой группы."""
        raise NotImplementedError

    def set_data(self, data: Dict[str, Any]):
        """Устанавливает данные в виджеты этой группы."""
        raise NotImplementedError

    def _add_item_to_list(self, list_widget: QListWidget, line_edit: QLineEdit, prefix: str = ""):
        """Общая утилита для добавления элемента в список."""
        item_text = line_edit.text().strip().lower()
        if prefix and item_text and not item_text.startswith(prefix):
            item_text = prefix + item_text.lstrip(prefix)
        if item_text and (not prefix or item_text != prefix):
            current_items = {list_widget.item(i).text() for i in range(list_widget.count())}
            if item_text not in current_items:
                list_widget.addItem(item_text)
                line_edit.clear()
            else:
                QMessageBox.information(self, UI_TEXTS["msg_info"], UI_TEXTS["msg_ext_already_exists"].format(item_text))
        else:
            QMessageBox.warning(self, UI_TEXTS["msg_error"], UI_TEXTS["msg_ext_invalid_prefix"].format(prefix))

    def _remove_selected_item_from_list(self, list_widget: QListWidget):
        """Общая утилита для удаления элемента из списка."""
        selected_items = list_widget.selectedItems()
        if selected_items:
            for item in reversed(selected_items):
                list_widget.takeItem(list_widget.row(item))
        else:
            QMessageBox.information(self, UI_TEXTS["msg_info"], UI_TEXTS["msg_ext_select_to_remove"])


class PathsGroup(BaseGroup):
    """Группа виджетов для настройки путей с вертикальным расположением."""
    def __init__(self):
        super().__init__(UI_TEXTS["paths_group_title"])
        self._create_widgets()

    def _create_widgets(self):
        main_layout = QVBoxLayout(self)
        
        # --- Виджеты создаются так же, но теперь все они относятся к 'paths' ---
        label_folder_path = QLabel(UI_TEXTS["folder_path_label"])
        self.folder_path_selector = PathSelectorWidget(UI_TEXTS["folder_path_dialog_title"])
        main_layout.addWidget(label_folder_path)
        main_layout.addWidget(self.folder_path_selector)

        label_output_path = QLabel(UI_TEXTS["output_path_label"])
        self.output_path_selector = PathSelectorWidget(UI_TEXTS["output_path_dialog_title"])
        main_layout.addWidget(label_output_path)
        main_layout.addWidget(self.output_path_selector)

        label_model_root = QLabel(UI_TEXTS["model_root_label"])
        self.model_root_selector = PathSelectorWidget(UI_TEXTS["model_root_dialog_title"])
        main_layout.addWidget(label_model_root)
        main_layout.addWidget(self.model_root_selector)

        # --- ИЗМЕНЕНИЕ: Путь к кэшу теперь логически здесь ---
        label_tensorrt_cache = QLabel(UI_TEXTS["provider_cache_label"])
        self.tensorRT_cache_path_selector = PathSelectorWidget(UI_TEXTS["provider_cache_dialog_title"])
        main_layout.addWidget(label_tensorrt_cache)
        main_layout.addWidget(self.tensorRT_cache_path_selector)
        
        main_layout.addStretch(1)

    # --- ИЗМЕНЕНИЕ: get_data теперь возвращает все пути в секции 'paths' ---
    def get_data(self) -> Dict[str, Any]:
        return {
            "paths": {
                "folder_path": self.folder_path_selector.text(),
                "output_path": self.output_path_selector.text(),
                "model_root": self.model_root_selector.text(),
                "tensorRT_cache_path": self.tensorRT_cache_path_selector.text()
            }
        }
    
    # --- ИЗМЕНЕНИЕ: set_data теперь ищет все пути в секции 'paths' ---
    def set_data(self, data: Dict[str, Any]):
        paths = data.get("paths", {})
        self.folder_path_selector.setText(paths.get("folder_path", ""))
        self.output_path_selector.setText(paths.get("output_path", ""))
        self.model_root_selector.setText(paths.get("model_root", ""))
        self.tensorRT_cache_path_selector.setText(paths.get("tensorRT_cache_path", ""))




class TasksGroup(BaseGroup):
    """Группа виджетов для выбора задач."""
    def __init__(self):
        super().__init__(UI_TEXTS["tasks_group_title"])
        self.checkboxes: Dict[str, QCheckBox] = {}
        self._create_widgets()

    def _create_widgets(self):
        main_layout = QVBoxLayout(self)
        grid_layout = QGridLayout()
        TASK_MAPPING = [
            ("run_image_analysis_and_clustering", "task_run_image_analysis_and_clustering"),
            ("analyze_gender", "task_analyze_gender"), ("analyze_emotion", "task_analyze_emotion"),
            ("analyze_age", "task_analyze_age"), ("analyze_beauty", "task_analyze_beauty"),
            ("analyze_eyeblink", "task_analyze_eyeblink"), ("keypoint_analysis", "task_keypoint_analysis"),
            ("create_xmp_file", "task_create_xmp_file"), ("move_files_to_claster", "task_move_files_to_claster"),
            ("generate_html", "task_generate_html"),
        ]
        rows_per_col = (len(TASK_MAPPING) + 1) // 2
        for i, (config_key, ui_text_key) in enumerate(TASK_MAPPING):
            checkbox = QCheckBox(UI_TEXTS.get(ui_text_key, config_key))
            self.checkboxes[config_key] = checkbox
            grid_layout.addWidget(checkbox, i % rows_per_col, i // rows_per_col)
        main_layout.addLayout(grid_layout)

    def get_main_task_checkbox(self) -> QCheckBox:
        return self.checkboxes["run_image_analysis_and_clustering"]

    def update_dependent_tasks_state(self, is_enabled: bool):
        dependent_keys = {"analyze_gender", "analyze_emotion", "analyze_age", "analyze_beauty", "analyze_eyeblink"}
        for key, checkbox in self.checkboxes.items():
            if key in dependent_keys: checkbox.setEnabled(is_enabled)

    def get_data(self) -> Dict[str, Any]:
        main_checked = self.checkboxes["run_image_analysis_and_clustering"].isChecked()
        dependent = {"analyze_gender", "analyze_emotion", "analyze_age", "analyze_beauty", "analyze_eyeblink"}
        return {"task": {k: (False if k in dependent and not main_checked else cb.isChecked()) for k, cb in self.checkboxes.items()}}

    def set_data(self, data: Dict[str, Any]):
        task_data = data.get("task", {})
        for key, checkbox in self.checkboxes.items():
            checkbox.setChecked(task_data.get(key, False))
        self.update_dependent_tasks_state(self.checkboxes["run_image_analysis_and_clustering"].isChecked())


class ProcessingGroup(BaseGroup):
    """Группа виджетов для настроек обработки."""
    def __init__(self):
        super().__init__(UI_TEXTS["processing_group_title"])
        self._create_widgets()

    def _create_widgets(self):
        layout = QFormLayout(self)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([UI_TEXTS["provider_combo_auto"], "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        layout.addRow(UI_TEXTS["provider_label"], self.provider_combo)
        
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(["RAW", "JPEG", "PSD"])
        layout.addRow(UI_TEXTS["proc_image_type_label"], self.image_type_combo)

        self.save_jpeg_check = QCheckBox(UI_TEXTS["proc_save_jpeg_check"])
        layout.addRow(self.save_jpeg_check)

        self.min_preview_spin = QSpinBox()
        self.min_preview_spin.setRange(256, 8192); self.min_preview_spin.setSingleStep(128); self.min_preview_spin.setValue(2048)
        layout.addRow(UI_TEXTS["proc_min_prev_label"], self.min_preview_spin)

        target_layout = QHBoxLayout()
        self.target_w_spin = QSpinBox(); self.target_w_spin.setRange(64, 2048); self.target_w_spin.setValue(640)
        self.target_h_spin = QSpinBox(); self.target_h_spin.setRange(64, 2048); self.target_h_spin.setValue(640)
        target_layout.addWidget(self.target_w_spin); target_layout.addWidget(self.target_h_spin)
        layout.addRow(UI_TEXTS["proc_target_size_label"], target_layout)
        
        self.max_workers_spin = QSpinBox(); self.max_workers_spin.setRange(0, os.cpu_count() or 64); self.max_workers_spin.setValue(0)
        self.max_workers_spin.setToolTip(UI_TEXTS["proc_max_workers_tooltip"])
        layout.addRow(UI_TEXTS["proc_max_workers_label"], self.max_workers_spin)

        self.block_size_spin = QSpinBox(); self.block_size_spin.setRange(0, 1024); self.block_size_spin.setValue(0)
        layout.addRow(UI_TEXTS["proc_block_size_label"], self.block_size_spin)
        
        self.max_limit_spin = QSpinBox(); self.max_limit_spin.setRange(1, 64); self.max_limit_spin.setValue(16)
        layout.addRow(UI_TEXTS["proc_max_limit_label"], self.max_limit_spin)

        self.xmp_tasks_spin = QSpinBox(); self.xmp_tasks_spin.setRange(1, 200); self.xmp_tasks_spin.setValue(50)
        layout.addRow(UI_TEXTS["proc_xmp_tasks_label"], self.xmp_tasks_spin)
        
        layout.addRow(QLabel(UI_TEXTS["proc_raw_ext_label"]))
        self.raw_ext_list = QListWidget(); self.raw_ext_list.setFixedHeight(60)
        layout.addRow(self.raw_ext_list)
        
        raw_ext_controls = QHBoxLayout()
        self.raw_ext_add_edit = QLineEdit(); self.raw_ext_add_edit.setPlaceholderText(UI_TEXTS["proc_raw_ext_placeholder"])
        raw_ext_add_btn = QPushButton(UI_TEXTS["proc_add_btn"]); raw_ext_add_btn.clicked.connect(self.add_raw_extension)
        raw_ext_del_btn = QPushButton(UI_TEXTS["proc_del_btn"]); raw_ext_del_btn.clicked.connect(self.remove_raw_extension)
        raw_ext_controls.addWidget(self.raw_ext_add_edit); raw_ext_controls.addWidget(raw_ext_add_btn); raw_ext_controls.addWidget(raw_ext_del_btn)
        layout.addRow(raw_ext_controls)

    @Slot()
    def add_raw_extension(self): self._add_item_to_list(self.raw_ext_list, self.raw_ext_add_edit, ".")
    @Slot()
    def remove_raw_extension(self): self._remove_selected_item_from_list(self.raw_ext_list)
        
    def get_data(self) -> Dict[str, Any]:
        max_workers = self.max_workers_spin.value()
        return {
            "provider": {"provider_name": None if self.provider_combo.currentText() == UI_TEXTS["provider_combo_auto"] else self.provider_combo.currentText()},
            "processing": {
                "select_image_type": self.image_type_combo.currentText(),
                "save_jpeg": self.save_jpeg_check.isChecked(),
                "min_preview_size": self.min_preview_spin.value(),
                "target_size": [self.target_w_spin.value(), self.target_h_spin.value()],
                "max_workers": None if max_workers == 0 else max_workers,
                "block_size": self.block_size_spin.value(),
                "max_workers_limit": self.max_limit_spin.value(),
                "max_concurrent_xmp_tasks": self.xmp_tasks_spin.value(),
                "raw_extensions": [self.raw_ext_list.item(i).text() for i in range(self.raw_ext_list.count())]
            }
        }

    def set_data(self, data: Dict[str, Any]):
        proc = data.get("processing", {})
        prov = data.get("provider", {})
        self.provider_combo.setCurrentText(prov.get("provider_name") or UI_TEXTS["provider_combo_auto"])
        self.image_type_combo.setCurrentText(proc.get("select_image_type", "RAW"))
        self.save_jpeg_check.setChecked(proc.get("save_jpeg", False))
        self.min_preview_spin.setValue(proc.get("min_preview_size", 2048))
        target_size = proc.get("target_size", [640, 640])
        if isinstance(target_size, list) and len(target_size) == 2:
            self.target_w_spin.setValue(target_size[0]); self.target_h_spin.setValue(target_size[1])
        self.max_workers_spin.setValue(proc.get("max_workers") or 0)
        self.block_size_spin.setValue(proc.get("block_size", 0))
        self.max_limit_spin.setValue(proc.get("max_workers_limit", 16))
        self.xmp_tasks_spin.setValue(proc.get("max_concurrent_xmp_tasks", 50))
        self.raw_ext_list.clear(); self.raw_ext_list.addItems(proc.get("raw_extensions", []))

class ReportGroup(BaseGroup):
    """Группа виджетов для настроек отчета."""
    def __init__(self):
        super().__init__(UI_TEXTS["report_group_title"])
        self._create_widgets()

    def _create_widgets(self):
        layout = QFormLayout(self)
        self.thumb_spin = QSpinBox(); self.thumb_spin.setRange(50, 500); self.thumb_spin.setSingleStep(10); self.thumb_spin.setValue(200)
        layout.addRow(UI_TEXTS["report_thumb_label"], self.thumb_spin)
        self.vis_combo = QComboBox(); self.vis_combo.addItems(["t-SNE", "PCA"])
        layout.addRow(UI_TEXTS["report_vis_method_label"], self.vis_combo)

    def get_data(self) -> Dict[str, Any]:
        return {"report": {"thumbnail_size": self.thumb_spin.value(), "visualization_method": self.vis_combo.currentText()}}
    
    def set_data(self, data: Dict[str, Any]):
        report = data.get("report", {})
        self.thumb_spin.setValue(report.get("thumbnail_size", 200))
        self.vis_combo.setCurrentText(report.get("visualization_method", "t-SNE"))

class ClusteringGroup(BaseGroup):
    """Группа виджетов для настроек кластеризации."""
    def __init__(self):
        super().__init__(UI_TEXTS["clustering_group_title"])
        self._create_widgets()
        
    def _create_widgets(self):
        layout = QVBoxLayout(self)
        
        # Portrait
        p_group = QGroupBox(UI_TEXTS["clus_portrait_group"])
        p_layout = QFormLayout(p_group)
        self.p_algo = QComboBox(); self.p_algo.addItems(["DBSCAN", "HDBSCAN"])
        self.p_eps = QDoubleSpinBox(); self.p_eps.setRange(0.01, 1.0); self.p_eps.setSingleStep(0.01); self.p_eps.setDecimals(3); self.p_eps.setValue(0.5)
        self.p_min = QSpinBox(); self.p_min.setRange(1, 100); self.p_min.setValue(5)
        p_layout.addRow(UI_TEXTS["clus_algo_label"], self.p_algo)
        p_layout.addRow(UI_TEXTS["clus_eps_label"], self.p_eps)
        p_layout.addRow(UI_TEXTS["clus_min_samples_label"], self.p_min)
        
        # Group
        g_group = QGroupBox(UI_TEXTS["clus_group_group"])
        g_layout = QFormLayout(g_group)
        self.g_algo = QComboBox(); self.g_algo.addItems(["DBSCAN", "HDBSCAN"])
        self.g_eps = QDoubleSpinBox(); self.g_eps.setRange(0.01, 1.0); self.g_eps.setSingleStep(0.01); self.g_eps.setDecimals(3); self.g_eps.setValue(0.5)
        self.g_min = QSpinBox(); self.g_min.setRange(1, 100); self.g_min.setValue(5)
        g_layout.addRow(UI_TEXTS["clus_algo_label"], self.g_algo)
        g_layout.addRow(UI_TEXTS["clus_group_eps_label"], self.g_eps)
        g_layout.addRow(UI_TEXTS["clus_group_min_samples_label"], self.g_min)

        layout.addWidget(p_group)
        layout.addWidget(g_group)

    def get_data(self) -> Dict[str, Any]:
        return {"clustering": {
            "portrait": {"algorithm": self.p_algo.currentText(), "eps": self.p_eps.value(), "min_samples": self.p_min.value()},
            "group": {"algorithm": self.g_algo.currentText(), "eps": self.g_eps.value(), "min_samples": self.g_min.value()}
        }}

    def set_data(self, data: Dict[str, Any]):
        clus = data.get("clustering", {})
        p = clus.get("portrait", {})
        g = clus.get("group", {})
        self.p_algo.setCurrentText(p.get("algorithm", "HDBSCAN"))
        self.p_eps.setValue(p.get("eps", 0.5))
        self.p_min.setValue(p.get("min_samples", 5))
        self.g_algo.setCurrentText(g.get("algorithm", "HDBSCAN"))
        self.g_eps.setValue(g.get("eps", 0.5))
        self.g_min.setValue(g.get("min_samples", 5))

class MovingGroup(BaseGroup):
    """Группа виджетов для настроек сортировки."""
    def __init__(self):
        super().__init__(UI_TEXTS["moving_group_title"])
        self._create_widgets()

    def _create_widgets(self):
        layout = QFormLayout(self)
        self.move_copy_check = QCheckBox(UI_TEXTS["moving_move_copy_check"])
        layout.addRow(self.move_copy_check)
        
        layout.addRow(QLabel(UI_TEXTS["moving_extensions_label"]))
        self.ext_list = QListWidget(); self.ext_list.setFixedHeight(60)
        layout.addRow(self.ext_list)
        
        ext_controls = QHBoxLayout()
        self.ext_add_edit = QLineEdit(); self.ext_add_edit.setPlaceholderText(UI_TEXTS["moving_ext_placeholder"])
        ext_add_btn = QPushButton(UI_TEXTS["moving_add_btn"]); ext_add_btn.clicked.connect(self.add_extension)
        ext_del_btn = QPushButton(UI_TEXTS["moving_del_btn"]); ext_del_btn.clicked.connect(self.remove_extension)
        ext_controls.addWidget(self.ext_add_edit); ext_controls.addWidget(ext_add_btn); ext_controls.addWidget(ext_del_btn)
        layout.addRow(ext_controls)

    @Slot()
    def add_extension(self): self._add_item_to_list(self.ext_list, self.ext_add_edit, ".")
    @Slot()
    def remove_extension(self): self._remove_selected_item_from_list(self.ext_list)

    def get_data(self) -> Dict[str, Any]:
        return {"moving": {
            "move_or_copy_files": self.move_copy_check.isChecked(),
            "file_extensions_to_action": [self.ext_list.item(i).text() for i in range(self.ext_list.count())]
        }}

    def set_data(self, data: Dict[str, Any]):
        moving = data.get("moving", {})
        self.move_copy_check.setChecked(moving.get("move_or_copy_files", False))
        self.ext_list.clear(); self.ext_list.addItems(moving.get("file_extensions_to_action", []))

class DebugGroup(BaseGroup):
    """Группа виджетов для настроек отладки."""
    def __init__(self):
        super().__init__(UI_TEXTS["debug_group_title"])
        self._create_widgets()
    
    def _create_widgets(self):
        layout = QVBoxLayout(self)
        self.save_kps_check = QCheckBox(UI_TEXTS["debug_save_kps_check"])
        layout.addWidget(self.save_kps_check)
        
    def get_data(self) -> Dict[str, Any]:
        return {"debug": {"save_analyzed_kps_images": self.save_kps_check.isChecked()}}

    def set_data(self, data: Dict[str, Any]):
        debug = data.get("debug", {})
        self.save_kps_check.setChecked(debug.get("save_analyzed_kps_images", False))