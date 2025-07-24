# gui_lib/ui_logic.py

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Type

import toml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMessageBox, QLineEdit, QCheckBox, QSpinBox,
    QDoubleSpinBox, QComboBox, QListWidget
)
from pydantic import BaseModel, ValidationError

# Импорты из основного проекта
from fc_lib.fc_config import (
    Config, TaskConfig, PathsConfig, ProcessingConfig, ReportConfig,
    MovingConfig, ProviderConfig, ClusteringConfig, DebugConfig
)
from fc_lib.fc_config import ConfigManager

# Импорт текстов
from .ui_texts import UI_TEXTS

logger = logging.getLogger(__name__)


class GuiLogicHandler:
    """
    Класс, инкапсулирующий логику работы с конфигурацией в GUI.
    """

    def load_config_data(self, filepath: str) -> Optional[dict]:
        """Загружает данные конфигурации из TOML файла."""
        try:
            # ConfigManager уже преобразует пути в абсолютные при загрузке
            config_mgr = ConfigManager(filepath)
            return config_mgr.config.copy()
        except FileNotFoundError:
            logger.error(f"Файл не найден: {filepath}")
            QMessageBox.warning(None, UI_TEXTS["msg_error"], UI_TEXTS["msg_file_not_found"].format(filepath))
            return None
        except ValidationError as val_err:
            logger.error(f"Ошибка валидации '{filepath}': {val_err}", exc_info=False)
            QMessageBox.critical(None, UI_TEXTS["msg_error"], UI_TEXTS["msg_load_validate_error"].format(val_err))
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки '{filepath}': {e}", exc_info=True)
            QMessageBox.critical(None, UI_TEXTS["msg_error"], UI_TEXTS["msg_load_error"].format(filepath, e))
            return None



    def update_gui_from_config_data(self, config_data: Optional[dict], widgets: Dict[str, Any]):
        """Обновляет виджеты GUI данными из загруженного словаря конфигурации."""
        if not config_data:
            return

        def set_widget_value(
            widget_key: str, section_key: str, field_key: str,
            pydantic_model_cls: Type[BaseModel], default_value: Any = None
        ):
            widget = widgets.get(widget_key)
            if not widget:
                logger.warning(f"Виджет с ключом '{widget_key}' не найден.")
                return

            section_data = config_data.get(section_key, {})
            config_value = section_data
            model_field_info = None
            default_value_from_model = ...
            keys = field_key.split(".")
            current_model_fields = pydantic_model_cls.model_fields
            valid_model_path = True

            try:
                for i, key_part in enumerate(keys):
                    if isinstance(config_value, dict):
                        config_value = config_value.get(key_part)
                    else:
                        config_value = None

                    if key_part in current_model_fields:
                        model_field_info = current_model_fields[key_part]
                        if i < len(keys) - 1:
                            next_level_annotation = getattr(model_field_info, "annotation", None)
                            if hasattr(next_level_annotation, "model_fields"):
                                current_model_fields = next_level_annotation.model_fields
                            else:
                                valid_model_path = False
                                model_field_info = None
                                break
                    else:
                        valid_model_path = False
                        model_field_info = None
                        break

                if model_field_info is not None:
                    default_value_from_model = model_field_info.get_default(call_default_factory=True)

            except Exception as e:
                logger.error(f"Ошибка доступа к model_fields для [{section_key}][{field_key}]: {e}")
                valid_model_path = False
                default_value_from_model = ...

            if config_value is not None:
                final_value = config_value
            elif valid_model_path and default_value_from_model is not ...:
                final_value = default_value_from_model
            else:
                final_value = default_value

            try:
                if isinstance(widget, QLineEdit):
                    widget.setText(str(final_value) if final_value is not None else "")
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(final_value) if final_value is not None else False)
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(final_value) if final_value is not None else 0)
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(final_value) if final_value is not None else 0.0)
                elif isinstance(widget, QComboBox):
                    current_text = str(final_value).upper() if final_value is not None else ""
                    idx = -1
                    if final_value is not None:
                        idx = widget.findText(str(final_value), Qt.MatchFlag.MatchFixedString | Qt.MatchFlag.MatchCaseSensitive)
                        if idx == -1:
                            idx = widget.findText(str(final_value), Qt.MatchFlag.MatchFixedString | Qt.MatchFlag.MatchCaseInsensitive)
                    if idx != -1:
                        widget.setCurrentIndex(idx)
                    else:
                        logger.warning(f"Значение '{final_value}' не найдено в QComboBox '{widget_key}'.")
                        # Попытка добавить значение, если его нет
                        if final_value is not None and str(final_value):
                            widget.addItem(str(final_value))
                            widget.setCurrentText(str(final_value))
                            logger.info(f"Значение '{final_value}' добавлено в QComboBox '{widget_key}'.")
                        elif widget.count() > 0:
                            widget.setCurrentIndex(0)
                elif isinstance(widget, QListWidget):
                    widget.clear()
                    if isinstance(final_value, list):
                        widget.addItems(map(str, final_value))
                else:
                    logger.warning(f"Неподдерживаемый тип виджета '{type(widget).__name__}' для ключа '{widget_key}'.")
            except (ValueError, TypeError) as set_err:
                logger.error(f"Ошибка установки значения '{final_value}' для виджета '{widget_key}': {set_err}")

        # Обновление виджетов
        # Paths
        set_widget_value("paths.folder_path", "paths", "folder_path", PathsConfig, "")
        set_widget_value("paths.output_path", "paths", "output_path", PathsConfig, "")
        set_widget_value("paths.model_root", "paths", "model_root", PathsConfig, "")
        # Provider
        set_widget_value("provider.tensorRT_cache_path", "provider", "tensorRT_cache_path", ProviderConfig, "")
        set_widget_value("provider.provider_name", "provider", "provider_name", ProviderConfig, UI_TEXTS["provider_combo_auto"])
        
        # Tasks
        tasks_config = config_data.get("task", {})
        task_checkboxes = widgets.get("task")
        if isinstance(task_checkboxes, dict):
            default_tasks = TaskConfig().model_dump()
            for key, checkbox in task_checkboxes.items():
                if isinstance(checkbox, QCheckBox):
                    checkbox.setChecked(tasks_config.get(key, default_tasks.get(key, False)))
        
        # Logging Level
        log_level = config_data.get("logging_level", "INFO").upper()
        set_widget_value("logging_level", "", "logging_level", Config, log_level)
        
        # Moving
        set_widget_value("moving.move_or_copy_files", "moving", "move_or_copy_files", MovingConfig, False)
        set_widget_value("moving.file_extensions_to_action", "moving", "file_extensions_to_action", MovingConfig, [])

        # Processing
        set_widget_value("processing.select_image_type", "processing", "select_image_type", ProcessingConfig, "RAW")
        set_widget_value("processing.save_jpeg", "processing", "save_jpeg", ProcessingConfig, True)
        set_widget_value("processing.min_preview_size", "processing", "min_preview_size", ProcessingConfig, 2048)
        
        target_w_spin = widgets.get("processing.target_size_w")
        target_h_spin = widgets.get("processing.target_size_h")
        if isinstance(target_w_spin, QSpinBox) and isinstance(target_h_spin, QSpinBox):
            target_size = config_data.get("processing", {}).get("target_size", [640, 640])
            if isinstance(target_size, list) and len(target_size) == 2:
                target_w_spin.setValue(int(target_size[0]))
                target_h_spin.setValue(int(target_size[1]))

        set_widget_value("processing.max_workers", "processing", "max_workers", ProcessingConfig, 0)
        set_widget_value("processing.block_size", "processing", "block_size", ProcessingConfig, 0)
        set_widget_value("processing.max_workers_limit", "processing", "max_workers_limit", ProcessingConfig, 16)
        set_widget_value("processing.max_concurrent_xmp_tasks", "processing", "max_concurrent_xmp_tasks", ProcessingConfig, 50)
        set_widget_value("processing.raw_extensions", "processing", "raw_extensions", ProcessingConfig, [])
        
        # Report
        set_widget_value("report.thumbnail_size", "report", "thumbnail_size", ReportConfig, 200)
        set_widget_value("report.visualization_method", "report", "visualization_method", ReportConfig, "t-SNE")
        
        # Clustering
        set_widget_value("clustering.portrait.algorithm", "clustering", "portrait.algorithm", ClusteringConfig, "HDBSCAN")
        set_widget_value("clustering.portrait.eps", "clustering", "portrait.eps", ClusteringConfig, 0.5)
        set_widget_value("clustering.portrait.min_samples", "clustering", "portrait.min_samples", ClusteringConfig, 5)
        set_widget_value("clustering.group.algorithm", "clustering", "group.algorithm", ClusteringConfig, "HDBSCAN")
        set_widget_value("clustering.group.eps", "clustering", "group.eps", ClusteringConfig, 0.5)
        set_widget_value("clustering.group.min_samples", "clustering", "group.min_samples", ClusteringConfig, 5)
        
        # Debug
        set_widget_value("debug.save_kps_images", "debug", "save_analyzed_kps_images", DebugConfig, False)

    def get_config_from_gui(self, widgets: Dict[str, Any]) -> dict:
        """Собирает конфигурацию из виджетов GUI."""
        def get_widget_value(widget_key: str) -> Any:
            widget = widgets.get(widget_key)
            if isinstance(widget, QLineEdit): return widget.text()
            if isinstance(widget, QCheckBox): return widget.isChecked()
            if isinstance(widget, QSpinBox): return widget.value()
            if isinstance(widget, QDoubleSpinBox): return widget.value()
            if isinstance(widget, QComboBox): return widget.currentText()
            if isinstance(widget, QListWidget): return [widget.item(i).text() for i in range(widget.count())]
            logger.warning(f"Не удалось получить значение для виджета '{widget_key}' типа {type(widget).__name__}")
            return None

        data = {}
        data["logging_level"] = get_widget_value("logging_level")
        data["paths"] = {
            "folder_path": get_widget_value("paths.folder_path"),
            "output_path": get_widget_value("paths.output_path"),
            "model_root": get_widget_value("paths.model_root"),
        }
        
        tasks_config = {}
        task_checkboxes = widgets.get("task")
        if isinstance(task_checkboxes, dict):
            main_task_key = "run_image_analysis_and_clustering"
            dependent_task_keys = {"analyze_gender", "analyze_emotion", "analyze_age", "analyze_beauty", "analyze_eyeblink"}
            main_cb = task_checkboxes.get(main_task_key)
            is_main_task_checked = main_cb.isChecked() if isinstance(main_cb, QCheckBox) else False
            tasks_config[main_task_key] = is_main_task_checked
            for key, checkbox in task_checkboxes.items():
                if key == main_task_key or not isinstance(checkbox, QCheckBox): continue
                is_checked = checkbox.isChecked()
                tasks_config[key] = False if key in dependent_task_keys and not is_main_task_checked else is_checked
        data["task"] = tasks_config
        
        data["moving"] = {
            "move_or_copy_files": get_widget_value("moving.move_or_copy_files"),
            "file_extensions_to_action": get_widget_value("moving.file_extensions_to_action"),
        }
        
        provider_selected = get_widget_value("provider.provider_name")
        data["provider"] = {
            "provider_name": None if provider_selected == UI_TEXTS["provider_combo_auto"] else provider_selected,
            "tensorRT_cache_path": get_widget_value("provider.tensorRT_cache_path"),
        }

        proc_max_workers_val = get_widget_value("processing.max_workers")
        data["processing"] = {
            "select_image_type": get_widget_value("processing.select_image_type"),
            "raw_extensions": get_widget_value("processing.raw_extensions"),
            "psd_extensions": ProcessingConfig().model_dump().get("psd_extensions", [".psd", ".psb"]),
            "save_jpeg": get_widget_value("processing.save_jpeg"),
            "min_preview_size": get_widget_value("processing.min_preview_size"),
            "target_size": [get_widget_value("processing.target_size_w"), get_widget_value("processing.target_size_h")],
            "max_workers": None if proc_max_workers_val == 0 else proc_max_workers_val,
            "block_size": get_widget_value("processing.block_size"),
            "max_workers_limit": get_widget_value("processing.max_workers_limit"),
            "max_concurrent_xmp_tasks": get_widget_value("processing.max_concurrent_xmp_tasks"),
        }
        
        data["report"] = {
            "thumbnail_size": get_widget_value("report.thumbnail_size"),
            "visualization_method": get_widget_value("report.visualization_method"),
        }
        
        data["clustering"] = {
            "portrait": {
                "algorithm": get_widget_value("clustering.portrait.algorithm"),
                "eps": get_widget_value("clustering.portrait.eps"),
                "min_samples": get_widget_value("clustering.portrait.min_samples"),
            },
            "group": {
                "algorithm": get_widget_value("clustering.group.algorithm"),
                "eps": get_widget_value("clustering.group.eps"),
                "min_samples": get_widget_value("clustering.group.min_samples"),
            },
        }
        
        data["debug"] = {
            "save_analyzed_kps_images": get_widget_value("debug.save_kps_images")
        }

        return data



    # --- НОВЫЙ ПРИВАТНЫЙ МЕТОД ---
    def _relativize_paths_in_section(self, config_section: Dict[str, Any], base_dir: Path):
        """Преобразует абсолютные пути в словаре в относительные."""
        if not isinstance(config_section, dict):
            return
            
        for key, value in config_section.items():
            if value and isinstance(value, str):
                try:
                    # Пытаемся создать объект Path. Может провалиться, если это не путь.
                    path_obj = Path(value)
                    if path_obj.is_absolute():
                        # Пытаемся вычислить относительный путь.
                        # Используем os.path.relpath, т.к. он лучше работает для путей
                        # "вверх" (../) на Windows, чем Path.relative_to.
                        relative_path = os.path.relpath(path_obj, base_dir)
                        # Заменяем разделители для консистентности
                        config_section[key] = str(Path(relative_path)) 
                        logger.debug(f"Путь '[paths].{key}' преобразован в относительный: '{value}' -> '{relative_path}'")
                except (ValueError, TypeError):
                    # Если возникает ошибка (например, путь на другом диске в Windows),
                    # или если value - не путь, просто оставляем как есть.
                    logger.debug(f"Не удалось сделать путь '[paths].{key}' относительным. Сохраняется как есть: '{value}'")
                    pass

    # --- ИЗМЕНЕННЫЙ МЕТОД save_config_to_file ---
    def save_config_to_file(self, filepath: str, config_data: dict):
        """
        Объединяет, преобразует пути в относительные и сохраняет 
        словарь конфигурации в TOML файл.
        """
        try:
            full_config_data = {}
            config_path_obj = Path(filepath)
            
            # 1. Загружаем существующий конфиг, чтобы не потерять не-GUI параметры
            if config_path_obj.exists() and config_path_obj.is_file():
                try:
                    with open(filepath, "r", encoding="utf-8") as f_read:
                        full_config_data = toml.load(f_read)
                except Exception as load_err:
                    logger.warning(f"Ошибка чтения {filepath}: {load_err}. Будет перезаписан.")

            # 2. Рекурсивно обновляем его данными из GUI
            def update_recursive(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d.get(k), dict):
                        d[k] = update_recursive(d[k], v)
                    else:
                        d[k] = v
                return d
            full_config_data = update_recursive(full_config_data, config_data)
            
            # --- ИЗМЕНЕНИЕ: Преобразуем пути в относительные ПЕРЕД сохранением ---
            save_base_dir = config_path_obj.parent
            logger.debug(f"Преобразование путей в относительные отн. {save_base_dir.resolve()}")
            if "paths" in full_config_data:
                self._relativize_paths_in_section(full_config_data["paths"], save_base_dir)
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            # 3. Валидируем и сохраняем
            try:
                # Для валидации нужно временно вернуть абсолютные пути
                temp_for_validation = toml.loads(toml.dumps(full_config_data))
                if "paths" in temp_for_validation:
                    # Создаем временный ConfigManager для доступа к его утилите
                    temp_mgr = ConfigManager(filepath)
                    temp_mgr._resolve_paths_in_section(temp_for_validation["paths"], save_base_dir)
                
                _ = Config(**temp_for_validation)
                data_to_save = full_config_data # Сохраняем данные с относительными путями
            except ValidationError as val_err:
                logger.warning(UI_TEXTS["msg_config_validation_warning"].format(val_err))
                data_to_save = full_config_data # При ошибке валидации все равно сохраняем как есть

            with open(filepath, "w", encoding="utf-8") as f:
                toml.dump(data_to_save, f)
            logger.info(f"Конфигурация сохранена: {filepath}")

        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации в '{filepath}': {e}", exc_info=True)
            raise IOError(UI_TEXTS["msg_save_error"].format(filepath, e))