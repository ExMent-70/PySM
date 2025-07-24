# run_gui.py
import sys
import logging
import os
from pathlib import Path
import threading
import traceback
import queue
import io
import toml
from typing import Optional, Dict, List, Union, Any, Tuple, Callable, Set, Type

# Импортируем необходимые классы из PySide6
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QFrame,
    QPlainTextEdit,
    QProgressBar,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QListWidget,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject
from pydantic import BaseModel, ValidationError


# Импорты из вашего проекта
try:
    from main import run_full_processing, setup_logging
    from fc_lib.fc_config import (
        ConfigManager,
        Config,
        TaskConfig,
        PathsConfig,  # Добавили PathsConfig
        ProcessingConfig,
        ReportConfig,
        MovingConfig,
        ProviderConfig,
        ClusteringConfig,
        ClusteringPortraitConfig,
        ClusteringGroupConfig,
        DebugConfig,  # Добавили DebugConfig
    )
    from fc_lib.fc_json_data_manager import JsonDataManager
except ImportError as import_err:
    print(
        f"Критическая ошибка: Не удалось импортировать необходимые модули: {import_err}"
    )
    print(
        "Убедитесь, что скрипт запускается из корневой папки проекта"
        " и все зависимости установлены."
    )
    sys.exit(1)


# --- Глобальные переменные ---
logger = logging.getLogger("GUI")
# DEFAULT_CONFIG_FILENAME = "face_config.toml"

# --- ИЗМЕНЕНИЕ: Определяем АБСОЛЮТНЫЙ путь к конфигу по умолчанию ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "face_config.toml"
# --- КОНЕЦ ИЗМЕНЕНИЯ ---


# === СЛОВАРЬ С ТЕКСТАМИ ИНТЕРФЕЙСА ===
UI_TEXTS = {
    "window_title": "Предварительный анализ фотографий",
    "config_label": "Текущая конфигурация:",
    "config_browse_btn": "Обзор...",
    "config_load_btn": "Обновить",
    "paths_group_title": "Настроить путь к рабочим папкам",
    "folder_path_label": "Папка с исходными фотографиями:",
    "folder_path_browse_btn": "Выбрать...",
    "folder_path_dialog_title": "Выбор папки с фотографиями",
    "output_path_label": "Папка для хранения результатов:",
    "output_path_browse_btn": "Выбрать...",
    "output_path_dialog_title": "Выбор папки для результатов",
    "model_root_label": "Папка для размещение моделей:",
    "model_root_browse_btn": "Выбрать...",
    "model_root_dialog_title": "Выбор папки моделей",
    "provider_cache_label": "Папка кэша TensorRT:",
    "provider_cache_browse_btn": "Выбрать...",
    "provider_cache_dialog_title": "Выбор папки кэша TensorRT",
    "tasks_group_title": "Доступные задачи",
    "task_run_image_analysis_and_clustering": "Выполнить анализ и кластеризацию",
    "task_analyze_gender": "Определить пол",
    "task_analyze_emotion": "Определить базовую эмоцию",
    "task_analyze_age": "Определить возраст",
    "task_analyze_beauty": "Определить привлекательность",
    "task_analyze_eyeblink": "Определить состояние глаз",
    "task_keypoint_analysis": "Анализировать ключевые точки",
    "task_create_xmp_file": "Создать XMP-файл",
    "task_move_files_to_claster": "Сортировать по папкам",
    "task_generate_html": "Создать HTML-отчет ",
    "moving_group_title": "Параметры сортировки",
    "moving_move_copy_check": "Перемещать файлы при сортировке (иначе копировать)",
    "moving_extensions_label": "Расширение сортируемых файлов:",
    "moving_ext_placeholder": ".jpg",
    "moving_add_btn": "+",
    "moving_del_btn": "-",
    "report_group_title": "Параметры генерации HTML-отчета",
    "report_thumb_label": "Размер миниатюр изображений (px):",
    "report_vis_method_label": "Метод визуализации кластеров:",
    "provider_label": "Провайдер:",
    "provider_combo_auto": "Авто",
    "processing_group_title": "Параметры базового анализа изображений",
    "proc_image_type_label": "Тип обрабатываемых файлов:",
    "proc_save_jpeg_check": "Сохранять JPEG извлеченный из RAW/PSD файла",
    "proc_min_prev_label": "Мин.размер JPEG (px):",
    "proc_target_size_label": "Target Size (W, H):",
    "proc_max_workers_label": "Max Workers (0=авто):",
    "proc_max_workers_tooltip": "0 = авто (CPU cores)",
    "proc_block_size_label": "Block Size (0=все):",
    "proc_max_limit_label": "Max Workers Limit:",
    "proc_xmp_tasks_label": "Max XMP Tasks:",
    "proc_raw_ext_label": "RAW расширения:",
    "proc_raw_ext_placeholder": ".nef",
    "proc_add_btn": "+",
    "proc_del_btn": "-",
    "clustering_group_title": "Параметры кластеризации фотографий",
    "clus_portrait_group": "Портретные фотографии",
    "clus_group_group": "Групповые фотографии",
    "clus_algo_label": "Алгоритм кластеризации:",
    "clus_eps_label": "Eps / Clus Eps:",
    "clus_min_samples_label": "Минимальный размер кластера:",
    "clus_group_eps_label": "Eps:",
    "clus_group_min_samples_label": "Минимальный размер кластера:",
    "log_level_label": "Уровень логирования:",
    "debug_group_title": "Параметры отладки",
    "debug_save_kps_check": "Сохранять отладочные изображения с KPS",
    "save_config_btn": "Сохранить файл конфигурации",
    "run_btn": "Запустить обработку",
    "exit_btn": "Выход",
    "log_area_label": "Лог выполнения:",
    "progress_label": "Прогресс:",
    "dialog_select_config_title": "Выбрать файл конфигурации",
    "dialog_select_config_filter": "TOML Files (*.toml);;All Files (*.*)",
    "dialog_save_config_title": "Сохранить конфигурацию как...",
    "dialog_save_config_filter": "TOML Files (*.toml);;All Files (*.*)",
    "dialog_select_folder_title": "Выбрать папку",
    "msg_error": "Ошибка",
    "msg_warning": "Внимание",
    "msg_info": "Информация",
    "msg_file_not_found": "Файл не найден:\n{}",
    "msg_load_error": "Ошибка загрузки:\n{}\n\n{}",
    "msg_save_error": "Не удалось сохранить конфигурацию:\n{}\n\n{}",
    "msg_load_validate_error": "Не удалось загрузить или провалидировать:\n{}",
    "msg_config_path_missing": "Путь к файлу конфигурации не указан.",
    "msg_processing_running": "Обработка уже запущена.",
    "msg_confirm_run_title": "Подтверждение запуска",
    "msg_confirm_run_text": 'Текущие настройки GUI будут сохранены в файл:\n"{}"\n\nЗапустить обработку?',
    "msg_save_cancelled": "Сохранение отменено пользователем. Запуск прерван.",
    "msg_config_not_found_run": "Файл конфигурации для запуска не найден или не указан:\n{}",
    "msg_processing_finished": "Обработка завершена!",
    "msg_processing_status": "Статус: {}",
    "msg_status_success": "Успешно",
    "msg_status_error": "С ошибками",
    "msg_critical_error": "Произошла критическая ошибка:\n{}\n\nСмотрите детали в логе.",
    "msg_config_saved": "Конфигурация сохранена.",
    "msg_ext_already_exists": "Элемент '{}' уже есть в списке.",
    "msg_ext_invalid_prefix": "Введите корректное значение (должно начинаться с '{}').",
    "msg_ext_select_to_remove": "Выберите элемент для удаления.",
    "msg_config_read_fail_warning": "Предупреждение: Не удалось прочитать начальный конфиг для уровня лога: {}",
    "msg_python_version_error": "Ошибка версии Python",
    "msg_python_version_required": "Требуется Python 3.8 или новее.",
    "msg_gui_error_critical": "Критическая ошибка GUI",
    "msg_gui_unhandled_error": "Произошла неперехваченная ошибка:\n{}\n\nСмотрите лог-файл.",
    "msg_gui_cannot_show_error": "Не удалось показать GUI сообщение об ошибке: {}",
    "msg_log_level_invalid": "Некорректный уровень логирования '{}' в конфиге. Установлен INFO.",
    "msg_provider_added_warning": "Провайдер '{}' из конфига добавлен в список GUI.",
    "msg_config_validation_warning": "Конфиг не прошел валидацию: {}. Сохраняем как есть.",
    "msg_gui_update_error": "Не удалось полностью обновить интерфейс из файла:\n{}\n\n{}",
}
# === КОНЕЦ СЛОВАРЯ ===


# === ГЛОБАЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def load_config_data(filepath: str) -> Optional[dict]:
    """Загружает данные конфигурации из TOML файла."""
    try:
        config_mgr = ConfigManager(filepath)
        return config_mgr.config
    except FileNotFoundError:
        logger.error(f"Файл не найден: {filepath}")
        QMessageBox.warning(
            None, UI_TEXTS["msg_error"], UI_TEXTS["msg_file_not_found"].format(filepath)
        )
        return None
    except ValidationError as val_err:
        logger.error(f"Ошибка валидации '{filepath}': {val_err}", exc_info=False)
        QMessageBox.critical(
            None,
            UI_TEXTS["msg_error"],
            UI_TEXTS["msg_load_validate_error"].format(val_err),
        )
        return None  # Возвращаем None при ошибке валидации
    except Exception as e:
        logger.error(f"Ошибка загрузки '{filepath}': {e}", exc_info=True)
        QMessageBox.critical(
            None, UI_TEXTS["msg_error"], UI_TEXTS["msg_load_error"].format(filepath, e)
        )
        return None


def update_gui_from_config_data(config_data: Optional[dict], widgets: Dict[str, Any]):
    """Обновляет виджеты GUI данными из загруженного словаря конфигурации."""
    if not config_data:
        return

    # Вспомогательная функция для установки значения виджета
    # Внутри функции update_gui_from_config_data в run_gui.py
    def set_widget_value(
        widget_key: str,
        section_key: str,
        field_key: str,
        # --- ИЗМЕНЕНИЕ: Принимаем саму модель, а не Type[BaseModel] ---
        # Это позволяет получить доступ к model_fields без инстанцирования
        pydantic_model_cls: Type[BaseModel],
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        default_value: Any = None,
    ):
        widget = widgets.get(widget_key)
        if not widget:
            logger.warning(f"Виджет с ключом '{widget_key}' не найден.")
            return

        # Получаем данные из загруженного конфига
        section_data = config_data.get(section_key, {})
        config_value = section_data
        # --- ИЗМЕНЕНИЕ: Получаем дефолты через model_fields ---
        model_field_info = None
        default_value_from_model = ...  # Используем Pydantic Undefined для проверки
        keys = field_key.split(".")
        current_model_fields = pydantic_model_cls.model_fields
        valid_model_path = True

        try:
            for i, key_part in enumerate(keys):
                if isinstance(config_value, dict):
                    config_value = config_value.get(key_part)
                else:
                    config_value = None  # Не можем идти глубже в конфиге

                if key_part in current_model_fields:
                    model_field_info = current_model_fields[key_part]
                    if i < len(keys) - 1:  # Если это не последний ключ
                        # Пытаемся получить тип аннотации для следующего уровня
                        next_level_annotation = getattr(
                            model_field_info, "annotation", None
                        )
                        if hasattr(next_level_annotation, "model_fields"):
                            current_model_fields = next_level_annotation.model_fields
                        else:
                            # Не можем идти глубже по модели
                            valid_model_path = False
                            model_field_info = None  # Сбрасываем info
                            break
                    # else: # Это последний ключ, model_field_info установлено верно
                else:
                    # Ключ не найден в модели
                    valid_model_path = False
                    model_field_info = None
                    break  # Прерываем поиск по модели

            # Получаем значение по умолчанию из model_field_info, если оно есть
            if model_field_info is not None:
                default_value_from_model = model_field_info.get_default(
                    call_default_factory=True  # Вызываем фабрику, если она есть
                )

        except Exception as e:
            logger.error(
                f"Ошибка доступа к model_fields для [{section_key}][{field_key}] модели {pydantic_model_cls.__name__}: {e}"
            )
            valid_model_path = False
            default_value_from_model = ...  # Сбрасываем в Undefined

        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # Определяем итоговое значение
        if config_value is not None:
            final_value = config_value
        # Pydantic Undefined используется как маркер отсутствия значения
        elif valid_model_path and default_value_from_model is not ...:
            final_value = default_value_from_model
            logger.debug(
                f"Значение для [{section_key}][{field_key}] не найдено в конфиге, используется default из модели: {final_value}"
            )
        else:
            final_value = default_value
            log_reason = (
                "путь в модели невалиден"
                if not valid_model_path
                else "в модели нет default"
            )
            logger.debug(
                f"Значение для [{section_key}][{field_key}] не найдено в конфиге ({log_reason}), используется переданный default: {final_value}"
            )

        # Устанавливаем значение в виджет (остальная часть без изменений)
        try:
            if isinstance(widget, QLineEdit):
                widget.setText(str(final_value) if final_value is not None else "")
            # ... (остальные типы виджетов как в предыдущей версии) ...
            elif isinstance(widget, QCheckBox):
                widget.setChecked(
                    bool(final_value) if final_value is not None else False
                )
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(final_value) if final_value is not None else 0)
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(final_value) if final_value is not None else 0.0)
            elif isinstance(widget, QComboBox):
                current_text = (
                    str(final_value).upper() if final_value is not None else ""
                )
                idx = -1
                if final_value is not None:
                    idx = widget.findText(
                        str(final_value),
                        Qt.MatchFlag.MatchFixedString | Qt.MatchFlag.MatchCaseSensitive,
                    )
                    if idx == -1:
                        idx = widget.findText(
                            str(final_value),
                            Qt.MatchFlag.MatchFixedString
                            | Qt.MatchFlag.MatchCaseInsensitive,
                        )
                if idx != -1:
                    widget.setCurrentIndex(idx)
                else:
                    logger.warning(
                        f"Значение '{final_value}' не найдено в QComboBox '{widget_key}'. Установлено первое значение."
                    )
                    if widget.count() > 0:
                        widget.setCurrentIndex(0)
            elif isinstance(widget, QListWidget):
                widget.clear()
                if isinstance(final_value, list):
                    widget.addItems(map(str, final_value))
            else:
                logger.warning(
                    f"Неподдерживаемый тип виджета '{type(widget).__name__}' для ключа '{widget_key}'."
                )
        except (ValueError, TypeError) as set_err:
            logger.error(
                f"Ошибка установки значения '{final_value}' (тип: {type(final_value)}) для виджета '{widget_key}' ({type(widget).__name__}): {set_err}"
            )
        except Exception as set_err:
            logger.error(
                f"Неожиданная ошибка установки значения для виджета '{widget_key}': {set_err}",
                exc_info=True,
            )

    # Обновление виджетов с использованием вложенных ключей, где необходимо
    # Paths
    set_widget_value("paths.folder_path", "paths", "folder_path", PathsConfig, "")
    set_widget_value("paths.output_path", "paths", "output_path", PathsConfig, "")
    set_widget_value("paths.model_root", "paths", "model_root", PathsConfig, "")
    # Provider
    set_widget_value(
        "provider.tensorRT_cache_path",
        "provider",
        "tensorRT_cache_path",
        ProviderConfig,
        "",
    )

    tasks_config = config_data.get("task", {})
    task_checkboxes = widgets.get("task")
    if isinstance(task_checkboxes, dict):
        default_tasks = TaskConfig().model_dump()
        for key, checkbox in task_checkboxes.items():
            if isinstance(checkbox, QCheckBox):
                checkbox.setChecked(
                    tasks_config.get(key, default_tasks.get(key, False))
                )

    log_level_combo = widgets.get("logging_level")
    if isinstance(log_level_combo, QComboBox):
        log_level = config_data.get("logging_level", "INFO").upper()
        log_levels = [
            log_level_combo.itemText(i) for i in range(log_level_combo.count())
        ]
        if log_level not in log_levels:
            log_level = "INFO"
            logger.warning(
                UI_TEXTS["msg_log_level_invalid"].format(
                    config_data.get("logging_level")
                )
            )
        idx = log_level_combo.findText(log_level)
        if idx != -1:
            log_level_combo.setCurrentIndex(idx)
        elif log_level_combo.count() > 0:
            log_level_combo.setCurrentIndex(0)

    set_widget_value(
        "moving.move_or_copy_files", "moving", "move_or_copy_files", MovingConfig, False
    )
    set_widget_value(
        "moving.file_extensions_to_action",
        "moving",
        "file_extensions_to_action",
        MovingConfig,
        [],
    )

    provider_name_combo = widgets.get("provider.provider_name")
    if isinstance(provider_name_combo, QComboBox):
        provider = config_data.get("provider", {})
        prov_name = provider.get("provider_name")
        current_providers = [
            provider_name_combo.itemText(i) for i in range(provider_name_combo.count())
        ]
        if "CPUExecutionProvider" not in current_providers:
            provider_name_combo.addItem("CPUExecutionProvider")
            current_providers.append("CPUExecutionProvider")
        if prov_name and prov_name in current_providers:
            provider_name_combo.setCurrentText(prov_name)
        elif prov_name:
            provider_name_combo.addItem(prov_name)
            provider_name_combo.setCurrentText(prov_name)
            logger.warning(UI_TEXTS["msg_provider_added_warning"].format(prov_name))
        else:
            auto_index = provider_name_combo.findText(UI_TEXTS["provider_combo_auto"])
            provider_name_combo.setCurrentIndex(auto_index if auto_index != -1 else 0)

    # Processing
    set_widget_value(
        "processing.select_image_type",
        "processing",
        "select_image_type",
        ProcessingConfig,
        "RAW",
    )
    set_widget_value(
        "processing.save_jpeg", "processing", "save_jpeg", ProcessingConfig, True
    )
    set_widget_value(
        "processing.min_preview_size",
        "processing",
        "min_preview_size",
        ProcessingConfig,
        2048,
    )

    target_w_spin = widgets.get("processing.target_size_w")
    target_h_spin = widgets.get("processing.target_size_h")
    if isinstance(target_w_spin, QSpinBox) and isinstance(target_h_spin, QSpinBox):
        target_size = config_data.get("processing", {}).get("target_size")
        default_target = ProcessingConfig().model_dump().get("target_size", [640, 640])
        if isinstance(target_size, list) and len(target_size) == 2:
            try:
                target_w_spin.setValue(int(target_size[0]))
                target_h_spin.setValue(int(target_size[1]))
            except (ValueError, TypeError):
                logger.warning(
                    f"Некорректные значения target_size в конфиге: {target_size}. Используются дефолтные."
                )
                target_w_spin.setValue(default_target[0])
                target_h_spin.setValue(default_target[1])
        else:
            target_w_spin.setValue(default_target[0])
            target_h_spin.setValue(default_target[1])

    set_widget_value(
        "processing.max_workers",
        "processing",
        "max_workers",
        ProcessingConfig,
        os.cpu_count() or 0,
    )
    set_widget_value(
        "processing.block_size", "processing", "block_size", ProcessingConfig, 0
    )
    set_widget_value(
        "processing.max_workers_limit",
        "processing",
        "max_workers_limit",
        ProcessingConfig,
        16,
    )
    set_widget_value(
        "processing.max_concurrent_xmp_tasks",
        "processing",
        "max_concurrent_xmp_tasks",
        ProcessingConfig,
        50,
    )
    set_widget_value(
        "processing.raw_extensions",
        "processing",
        "raw_extensions",
        ProcessingConfig,
        [],
    )
    # Report
    set_widget_value(
        "report.thumbnail_size", "report", "thumbnail_size", ReportConfig, 200
    )
    set_widget_value(
        "report.visualization_method",
        "report",
        "visualization_method",
        ReportConfig,
        "t-SNE",
    )
    # Clustering (Используем ClusteringConfig для доступа к вложенным моделям)
    set_widget_value(
        "clustering.portrait.algorithm",
        "clustering",
        "portrait.algorithm",
        ClusteringConfig,
        "HDBSCAN",
    )
    set_widget_value(
        "clustering.portrait.eps", "clustering", "portrait.eps", ClusteringConfig, 0.5
    )
    set_widget_value(
        "clustering.portrait.min_samples",
        "clustering",
        "portrait.min_samples",
        ClusteringConfig,
        5,
    )
    set_widget_value(
        "clustering.group.algorithm",
        "clustering",
        "group.algorithm",
        ClusteringConfig,
        "HDBSCAN",
    )
    set_widget_value(
        "clustering.group.eps", "clustering", "group.eps", ClusteringConfig, 0.5
    )
    set_widget_value(
        "clustering.group.min_samples",
        "clustering",
        "group.min_samples",
        ClusteringConfig,
        5,
    )
    # Debug
    set_widget_value(
        "debug.save_kps_images", "debug", "save_analyzed_kps_images", DebugConfig, False
    )


def get_config_from_gui_widgets(widgets: Dict[str, Any]) -> dict:
    """Собирает конфигурацию из виджетов GUI."""

    def get_widget_value(widget_key: str) -> Any:
        widget = widgets.get(widget_key)
        if isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QSpinBox):
            return widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            return widget.value()
        elif isinstance(widget, QComboBox):
            return widget.currentText()
        elif isinstance(widget, QListWidget):
            return [widget.item(i).text() for i in range(widget.count())]
        logger.warning(
            f"Не удалось получить значение для виджета '{widget_key}' типа {type(widget).__name__}"
        )
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
        dependent_task_keys = {
            "analyze_gender",
            "analyze_emotion",
            "analyze_age",
            "analyze_beauty",
            "analyze_eyeblink",
        }
        main_cb = task_checkboxes.get(main_task_key)
        is_main_task_checked = (
            main_cb.isChecked() if isinstance(main_cb, QCheckBox) else False
        )
        tasks_config[main_task_key] = is_main_task_checked
        for key, checkbox in task_checkboxes.items():
            if key == main_task_key or not isinstance(checkbox, QCheckBox):
                continue
            is_checked = checkbox.isChecked()
            tasks_config[key] = (
                False
                if key in dependent_task_keys and not is_main_task_checked
                else is_checked
            )
    data["task"] = tasks_config
    data["moving"] = {
        "move_or_copy_files": get_widget_value("moving.move_or_copy_files"),
        "file_extensions_to_action": get_widget_value(
            "moving.file_extensions_to_action"
        ),
    }
    provider_selected = get_widget_value("provider.provider_name")
    data["provider"] = {
        "provider_name": None
        if provider_selected == UI_TEXTS["provider_combo_auto"]
        else provider_selected,
        "tensorRT_cache_path": get_widget_value("provider.tensorRT_cache_path"),
    }
    default_proc_psd_ext = (
        ProcessingConfig().model_dump().get("psd_extensions", [".psd", ".psb"])
    )
    proc_max_workers_val = get_widget_value("processing.max_workers")
    data["processing"] = {
        "select_image_type": get_widget_value("processing.select_image_type"),
        "raw_extensions": get_widget_value("processing.raw_extensions"),
        "psd_extensions": default_proc_psd_ext,
        "save_jpeg": get_widget_value("processing.save_jpeg"),
        "min_preview_size": get_widget_value("processing.min_preview_size"),
        "target_size": [
            get_widget_value("processing.target_size_w"),
            get_widget_value("processing.target_size_h"),
        ],
        "max_workers": None if proc_max_workers_val == 0 else proc_max_workers_val,
        "block_size": get_widget_value("processing.block_size"),
        "max_workers_limit": get_widget_value("processing.max_workers_limit"),
        "max_concurrent_xmp_tasks": get_widget_value(
            "processing.max_concurrent_xmp_tasks"
        ),
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


def save_config_to_file(filepath: str, config_data: dict):
    """Сохраняет словарь конфигурации в TOML файл, объединяя с существующим."""
    try:
        full_config_data = {}
        config_path_obj = Path(filepath)
        if config_path_obj.exists() and config_path_obj.is_file():
            try:
                with open(filepath, "r", encoding="utf-8") as f_read:
                    full_config_data = toml.load(f_read)
            except Exception as load_err:
                logger.error(
                    f"Ошибка чтения {filepath}: {load_err}. Будет перезаписан."
                )
                full_config_data = {}

        # Рекурсивное обновление словарей
        def update_recursive(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_recursive(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        full_config_data = update_recursive(full_config_data, config_data)

        # Валидация перед сохранением
        try:
            _ = Config(**full_config_data)  # Попытка валидации
            data_to_save = full_config_data
            logger.debug("Конфиг валиден перед сохранением.")
        except ValidationError as val_err:
            logger.warning(UI_TEXTS["msg_config_validation_warning"].format(val_err))
            data_to_save = full_config_data  # Сохраняем как есть при ошибке валидации
        except Exception as val_err:  # Ловим другие возможные ошибки Pydantic
            logger.warning(
                f"Неожиданная ошибка валидации Pydantic: {val_err}. Сохраняем как есть."
            )
            data_to_save = full_config_data

        with open(filepath, "w", encoding="utf-8") as f:
            toml.dump(data_to_save, f)
        logger.info(f"Конфигурация сохранена: {filepath}")
    except Exception as e:
        logger.error(
            f"Ошибка сохранения конфигурации в '{filepath}': {e}", exc_info=True
        )
        raise IOError(UI_TEXTS["msg_save_error"].format(filepath, e))


# === КОНЕЦ ГЛОБАЛЬНЫХ ФУНКЦИЙ ===


# --- QtLogHandler, ProcessingWorker (без изменений) ---
class QtLogHandler(logging.Handler):
    def __init__(self, log_signal: Signal):
        super().__init__()
        self.log_signal = log_signal

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_signal.emit(msg)
        except Exception:
            self.handleError(record)


class ProcessingWorker(QObject):
    log_message = Signal(str)
    progress_updated = Signal(int, int)
    processing_finished = Signal(bool)
    error_occurred = Signal(str)

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self._is_running = False

    @Slot()
    def run(self):
        if self._is_running:
            self.log_message.emit(f"ERROR: {UI_TEXTS['msg_processing_running']}")
            return
        self._is_running = True
        logger.info("Запуск рабочего потока...")
        success = False
        try:

            def log_callback_impl(message: str):
                self.log_message.emit(message)

            def progress_callback_impl(current: int, total: int):
                self.progress_updated.emit(current, total)

            success = run_full_processing(
                self.config_path, log_callback_impl, progress_callback_impl
            )
        except Exception as e:
            error_msg = (
                f"Критическая ошибка в рабочем потоке: {e}\n{traceback.format_exc()}"
            )
            logger.critical(f"Критическая ошибка в рабочем потоке: {e}", exc_info=True)
            self.error_occurred.emit(error_msg)
            success = False
        finally:
            self._is_running = False
            self.processing_finished.emit(success)
            logger.info(f"Рабочий поток завершен (успех: {success}).")


# --- Основной класс окна GUI ---
class MainWindow(QWidget):
    """Главное окно приложения."""

    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI_TEXTS["window_title"])
        # Увеличил стандартную высоту окна
        self.setGeometry(100, 100, 750, 850)
        # --- ИЗМЕНЕНИЕ: Используем абсолютный путь ---
        self.current_config_path = str(DEFAULT_CONFIG_PATH)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        self.loaded_config_data: Optional[Dict] = None
        self.worker_thread: Optional[QThread] = None
        self.processing_worker: Optional[ProcessingWorker] = None
        self.qt_log_handler: Optional[QtLogHandler] = None
        self.widgets: Dict[str, Any] = {}  # Словарь для хранения виджетов

        self.init_ui()  # Инициализация UI (создает self.widgets)

        # Настройка логирования в GUI
        self.log_signal.connect(self._add_log_message)
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_datefmt = "%Y-%m-%d %H:%M:%S"
        gui_formatter = logging.Formatter(log_format, log_datefmt)
        self.qt_log_handler = QtLogHandler(self.log_signal)
        self.qt_log_handler.setFormatter(gui_formatter)
        # Добавляем хендлер к корневому логгеру
        logging.getLogger().addHandler(self.qt_log_handler)
        logger.info("QtLogHandler добавлен к корневому логгеру.")

        # Загрузка начальной конфигурации
        self.load_initial_config()  # Использует load_and_update_gui -> update_gui_from_config_data

        # Установка начального состояния зависимых задач
        task_checkboxes_dict = self.widgets.get("task")
        if isinstance(task_checkboxes_dict, dict):
            main_task_cb = task_checkboxes_dict.get("run_image_analysis_and_clustering")
            if main_task_cb and isinstance(main_task_cb, QCheckBox):
                self._update_dependent_tasks_state(main_task_cb.checkState().value)
            else:
                logger.error(
                    "Не найден виджет главного чекбокса в self.widgets['task']."
                )
        else:
            logger.error(
                "Не удалось получить словарь чекбоксов из self.widgets для инициализации."
            )

    @Slot(str)
    def _add_log_message(self, message):
        """Добавляет сообщение в виджет лога QPlainTextEdit."""
        if hasattr(self, "log_output") and self.log_output is not None:
            try:
                self.log_output.appendPlainText(message)
            except Exception as e:
                # Используем print, так как логирование в GUI могло вызвать ошибку
                print(f"Ошибка обновления лога GUI (_add_log_message): {e}")
        else:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: Попытка логирования до создания log_output: {message}"
            )

    @Slot(int)
    def _update_dependent_tasks_state(self, state: int):
        """Обновляет состояние (enabled/disabled) зависимых чекбоксов."""
        # logger.debug(f"_update_dependent_tasks_state called with state: {state}")
        is_main_task_enabled = state == Qt.CheckState.Checked.value
        dependent_task_keys = {
            "analyze_gender",
            "analyze_emotion",
            "analyze_age",
            "analyze_beauty",
            "analyze_eyeblink",
        }
        task_checkboxes_dict = self.widgets.get("task")
        if isinstance(task_checkboxes_dict, dict):
            for key in dependent_task_keys:
                checkbox = task_checkboxes_dict.get(key)
                if checkbox and isinstance(checkbox, QCheckBox):
                    checkbox.setEnabled(is_main_task_enabled)
                    # Можно добавить логику снятия галочки, если неактивно
                    # if not is_main_task_enabled:
                    #     checkbox.setChecked(False)
                # else: logger.warning(f"Чекбокс с ключом '{key}' не найден в словаре виджетов.")
        else:
            logger.error(
                "Не удалось получить словарь чекбоксов из self.widgets в _update_dependent_tasks_state."
            )
        # logger.debug("Обновление состояния зависимых задач завершено.")

    # Внутри класса MainWindow в run_gui.py

    def init_ui(self):
        """Инициализирует пользовательский интерфейс и заполняет self.widgets."""
        main_layout = QVBoxLayout(self)
        self.widgets: Dict[str, Any] = {}  # Очищаем или инициализируем словарь

        # 1. Верхняя часть: Файл конфигурации
        self._create_config_file_section(main_layout)

        # 2. Средняя часть: Основные настройки (в две колонки)
        settings_layout = QHBoxLayout()
        col1_layout = QVBoxLayout()  # Левая колонка
        col2_layout = QVBoxLayout()  # Правая колонка

        # --- Заполнение левой колонки (Колонка 1) ---
        self._create_tasks_section(col1_layout)
        self._create_processing_section(col1_layout)
        self._create_report_section(col1_layout)
        self._create_debug_section(col1_layout)
        col1_layout.addStretch(1)  # Добавляем растяжение в конце колонки 1

        # --- Заполнение правой колонки (Колонка 2) ---
        self._create_paths_section(col2_layout)
        self._create_clustering_section(col2_layout)
        self._create_moving_section(col2_layout)
        col2_layout.addStretch(1)  # Добавляем растяжение в конце колонки 2

        settings_layout.addLayout(col1_layout)
        settings_layout.addLayout(col2_layout)
        main_layout.addLayout(settings_layout)

        # 3. Нижняя часть: Уровень лога, Кнопки, Лог, Прогресс
        self._create_bottom_section(main_layout)

        self.setLayout(main_layout)
        logger.info(
            f"Словарь виджетов self.widgets инициализирован. Ключей: {len(self.widgets)}"
        )

    # --- Вспомогательные методы для создания секций UI ---

    def _create_config_file_section(self, parent_layout: QVBoxLayout):
        """Создает секцию для выбора файла конфигурации."""
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel(UI_TEXTS["config_label"]))

        # --- ИЗМЕНЕНИЕ: Используем абсолютный путь ---
        self.config_path_edit = QLineEdit(str(DEFAULT_CONFIG_PATH))
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        config_layout.addWidget(self.config_path_edit)
        # Не добавляем config_path_edit в self.widgets, т.к. он не часть данных TOML

        self.browse_config_btn = QPushButton(UI_TEXTS["config_browse_btn"])
        self.browse_config_btn.clicked.connect(self.browse_config_file)
        config_layout.addWidget(self.browse_config_btn)

        self.load_config_btn = QPushButton(UI_TEXTS["config_load_btn"])
        self.load_config_btn.clicked.connect(self.load_config_action)
        config_layout.addWidget(self.load_config_btn)

        parent_layout.addLayout(config_layout)

    def _create_tasks_section(self, parent_layout: QVBoxLayout):
        """Создает секцию выбора задач."""
        task_group = QGroupBox(UI_TEXTS["tasks_group_title"])
        task_main_layout = QVBoxLayout(task_group)
        task_grid_layout = QGridLayout()

        TASK_MAPPING = [
            (
                "run_image_analysis_and_clustering",
                "task_run_image_analysis_and_clustering",
            ),  # Используем ключ task_run...
            ("analyze_gender", "task_analyze_gender"),
            ("analyze_emotion", "task_analyze_emotion"),
            ("analyze_age", "task_analyze_age"),
            ("analyze_beauty", "task_analyze_beauty"),
            ("analyze_eyeblink", "task_analyze_eyeblink"),
            ("keypoint_analysis", "task_keypoint_analysis"),
            ("create_xmp_file", "task_create_xmp_file"),
            ("move_files_to_claster", "task_move_files_to_claster"),  # Исправил ключ UI
            ("generate_html", "task_generate_html"),
        ]

        self.task_checkboxes: Dict[str, QCheckBox] = {}
        num_tasks = len(TASK_MAPPING)
        rows_per_col = (num_tasks + 1) // 2
        current_row, current_col = 0, 0

        for config_key, ui_text_key in TASK_MAPPING:
            ui_text = UI_TEXTS.get(
                ui_text_key, config_key.replace("_", " ").capitalize()
            )
            checkbox = QCheckBox(ui_text)
            self.task_checkboxes[config_key] = checkbox
            task_grid_layout.addWidget(checkbox, current_row, current_col)

            current_row += 1
            if current_row >= rows_per_col and current_col == 0:
                current_row, current_col = 0, 1

        self.widgets["task"] = self.task_checkboxes  # Добавляем словарь в общий

        task_main_layout.addLayout(task_grid_layout)
        parent_layout.addWidget(task_group)

        # Подключение сигнала главного чекбокса
        main_task_cb = self.task_checkboxes.get("run_image_analysis_and_clustering")
        if main_task_cb:
            main_task_cb.stateChanged.connect(self._update_dependent_tasks_state)
        else:
            logger.error(
                "Не найден главный чекбокс 'run_image_analysis_and_clustering' при создании UI."
            )

    def _create_processing_section(self, parent_layout: QVBoxLayout):
        """Создает секцию настроек обработки изображений."""
        proc_group = QGroupBox(UI_TEXTS["processing_group_title"])
        proc_layout = QFormLayout(proc_group)

        # Провайдер
        self.provider_name_combo = QComboBox()
        self.provider_name_combo.addItems(
            [
                UI_TEXTS["provider_combo_auto"],
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        )
        proc_layout.addRow(UI_TEXTS["provider_label"], self.provider_name_combo)
        self.widgets["provider.provider_name"] = self.provider_name_combo

        # Тип файлов
        self.proc_image_type_combo = QComboBox()
        self.proc_image_type_combo.addItems(["RAW", "JPEG", "PSD"])
        proc_layout.addRow(
            UI_TEXTS["proc_image_type_label"], self.proc_image_type_combo
        )
        self.widgets["processing.select_image_type"] = self.proc_image_type_combo

        # Сохранение JPEG
        self.proc_save_jpeg_check = QCheckBox(UI_TEXTS["proc_save_jpeg_check"])
        proc_layout.addRow(self.proc_save_jpeg_check)
        self.widgets["processing.save_jpeg"] = self.proc_save_jpeg_check

        # Мин. размер превью
        self.proc_min_prev_spin = QSpinBox()
        self.proc_min_prev_spin.setRange(256, 8192)
        self.proc_min_prev_spin.setSingleStep(128)
        self.proc_min_prev_spin.setValue(
            2048
        )  # Устанавливаем разумное значение по умолчанию
        proc_layout.addRow(UI_TEXTS["proc_min_prev_label"], self.proc_min_prev_spin)
        self.widgets["processing.min_preview_size"] = self.proc_min_prev_spin

        # Target Size
        proc_target_layout = QHBoxLayout()
        self.proc_target_w_spin = QSpinBox()
        self.proc_target_w_spin.setRange(64, 2048)
        self.proc_target_w_spin.setValue(640)  # Default
        self.proc_target_h_spin = QSpinBox()
        self.proc_target_h_spin.setRange(64, 2048)
        self.proc_target_h_spin.setValue(640)  # Default
        proc_target_layout.addWidget(QLabel(UI_TEXTS["proc_target_size_label"]))
        proc_target_layout.addWidget(self.proc_target_w_spin)
        proc_target_layout.addWidget(self.proc_target_h_spin)
        proc_layout.addRow(proc_target_layout)
        self.widgets["processing.target_size_w"] = self.proc_target_w_spin
        self.widgets["processing.target_size_h"] = self.proc_target_h_spin

        # Max Workers
        self.proc_max_workers_spin = QSpinBox()
        self.proc_max_workers_spin.setRange(
            0, os.cpu_count() or 64
        )  # Верхний предел - кол-во ядер или 64
        self.proc_max_workers_spin.setValue(0)  # Default 0 (авто)
        self.proc_max_workers_spin.setToolTip(UI_TEXTS["proc_max_workers_tooltip"])
        proc_layout.addRow(
            UI_TEXTS["proc_max_workers_label"], self.proc_max_workers_spin
        )
        self.widgets["processing.max_workers"] = self.proc_max_workers_spin

        # Block Size
        self.proc_block_spin = QSpinBox()
        self.proc_block_spin.setRange(0, 1024)  # 0 = все файлы
        self.proc_block_spin.setValue(0)  # Default
        proc_layout.addRow(UI_TEXTS["proc_block_size_label"], self.proc_block_spin)
        self.widgets["processing.block_size"] = self.proc_block_spin

        # Max Workers Limit
        self.proc_max_limit_spin = QSpinBox()
        self.proc_max_limit_spin.setRange(1, 64)
        self.proc_max_limit_spin.setValue(16)  # Default
        proc_layout.addRow(UI_TEXTS["proc_max_limit_label"], self.proc_max_limit_spin)
        self.widgets["processing.max_workers_limit"] = self.proc_max_limit_spin

        # Max XMP Tasks
        self.proc_xmp_tasks_spin = QSpinBox()
        self.proc_xmp_tasks_spin.setRange(1, 200)
        self.proc_xmp_tasks_spin.setValue(50)  # Default
        proc_layout.addRow(UI_TEXTS["proc_xmp_tasks_label"], self.proc_xmp_tasks_spin)
        self.widgets["processing.max_concurrent_xmp_tasks"] = self.proc_xmp_tasks_spin

        # RAW Extensions
        proc_layout.addRow(QLabel(UI_TEXTS["proc_raw_ext_label"]))
        self.proc_raw_ext_list = QListWidget()
        self.proc_raw_ext_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.proc_raw_ext_list.setFixedHeight(60)  # Фиксированная высота для списка
        proc_layout.addRow(self.proc_raw_ext_list)
        self.widgets["processing.raw_extensions"] = self.proc_raw_ext_list

        raw_ext_controls_layout = QHBoxLayout()
        self.raw_ext_add_edit = QLineEdit()
        self.raw_ext_add_edit.setPlaceholderText(UI_TEXTS["proc_raw_ext_placeholder"])
        raw_ext_controls_layout.addWidget(self.raw_ext_add_edit)
        raw_ext_add_btn = QPushButton(UI_TEXTS["proc_add_btn"])
        raw_ext_add_btn.clicked.connect(self.add_raw_extension)
        raw_ext_controls_layout.addWidget(raw_ext_add_btn)
        raw_ext_del_btn = QPushButton(UI_TEXTS["proc_del_btn"])
        raw_ext_del_btn.clicked.connect(self.remove_raw_extension)
        raw_ext_controls_layout.addWidget(raw_ext_del_btn)
        proc_layout.addRow(raw_ext_controls_layout)

        parent_layout.addWidget(proc_group)

    def _create_report_section(self, parent_layout: QVBoxLayout):
        """Создает секцию настроек отчета."""
        report_group = QGroupBox(UI_TEXTS["report_group_title"])
        report_layout = QFormLayout(report_group)

        self.report_thumb_spin = QSpinBox()
        self.report_thumb_spin.setRange(50, 500)
        self.report_thumb_spin.setSingleStep(10)
        self.report_thumb_spin.setValue(200)  # Default
        report_layout.addRow(UI_TEXTS["report_thumb_label"], self.report_thumb_spin)
        self.widgets["report.thumbnail_size"] = self.report_thumb_spin

        self.report_vis_combo = QComboBox()
        self.report_vis_combo.addItems(["t-SNE", "PCA"])
        self.report_vis_combo.setCurrentText("t-SNE")  # Default
        report_layout.addRow(UI_TEXTS["report_vis_method_label"], self.report_vis_combo)
        self.widgets["report.visualization_method"] = self.report_vis_combo

        parent_layout.addWidget(report_group)

    def _create_debug_section(self, parent_layout: QVBoxLayout):
        """Создает секцию настроек отладки."""
        debug_group = QGroupBox(UI_TEXTS["debug_group_title"])
        debug_layout = QVBoxLayout(debug_group)  # Используем QVBoxLayout для простоты

        self.debug_save_kps_check = QCheckBox(UI_TEXTS["debug_save_kps_check"])
        debug_layout.addWidget(self.debug_save_kps_check)
        self.widgets["debug.save_kps_images"] = self.debug_save_kps_check

        parent_layout.addWidget(debug_group)

    def _create_paths_section(self, parent_layout: QVBoxLayout):
        """Создает секцию настроек путей."""
        path_group = QGroupBox(UI_TEXTS["paths_group_title"])
        # Используем QVBoxLayout для последовательного добавления строк
        path_main_layout = QVBoxLayout(path_group)

        # Вспомогательная функция для создания строки Path + Button
        def add_path_row(label_text: str, widget_key: str, dialog_title: str):
            path_main_layout.addWidget(QLabel(label_text))
            row_layout = QHBoxLayout()
            line_edit = QLineEdit()
            line_edit.setReadOnly(True)
            browse_btn = QPushButton(
                UI_TEXTS["config_browse_btn"]
            )  # Используем текст "Обзор..."
            browse_btn.clicked.connect(
                lambda: self.browse_directory(line_edit, dialog_title)
            )
            row_layout.addWidget(line_edit)
            row_layout.addWidget(browse_btn)
            path_main_layout.addLayout(row_layout)
            setattr(
                self, widget_key.split(".")[-1] + "_edit", line_edit
            )  # Сохраняем виджет в self для доступа
            self.widgets[widget_key] = line_edit  # Добавляем в общий словарь

        # Создаем строки для каждого пути
        add_path_row(
            UI_TEXTS["folder_path_label"],
            "paths.folder_path",
            UI_TEXTS["folder_path_dialog_title"],
        )
        add_path_row(
            UI_TEXTS["output_path_label"],
            "paths.output_path",
            UI_TEXTS["output_path_dialog_title"],
        )
        add_path_row(
            UI_TEXTS["model_root_label"],
            "paths.model_root",
            UI_TEXTS["model_root_dialog_title"],
        )
        add_path_row(
            UI_TEXTS["provider_cache_label"],
            "provider.tensorRT_cache_path",
            UI_TEXTS["provider_cache_dialog_title"],
        )

        parent_layout.addWidget(path_group)

    def _create_clustering_section(self, parent_layout: QVBoxLayout):
        """Создает секцию настроек кластеризации."""
        clus_group = QGroupBox(UI_TEXTS["clustering_group_title"])
        clus_layout = QVBoxLayout(clus_group)  # Основной layout для группы

        # --- Подсекция: Портреты ---
        clus_p_group = QGroupBox(UI_TEXTS["clus_portrait_group"])
        clus_p_layout = QFormLayout(clus_p_group)

        self.clus_p_algo_combo = QComboBox()
        self.clus_p_algo_combo.addItems(["DBSCAN", "HDBSCAN"])
        clus_p_layout.addRow(UI_TEXTS["clus_algo_label"], self.clus_p_algo_combo)
        self.widgets["clustering.portrait.algorithm"] = self.clus_p_algo_combo

        self.clus_p_eps_spin = QDoubleSpinBox()
        self.clus_p_eps_spin.setRange(0.01, 1.0)
        self.clus_p_eps_spin.setSingleStep(0.01)
        self.clus_p_eps_spin.setDecimals(3)
        self.clus_p_eps_spin.setValue(0.5)  # Default
        clus_p_layout.addRow(UI_TEXTS["clus_eps_label"], self.clus_p_eps_spin)
        self.widgets["clustering.portrait.eps"] = self.clus_p_eps_spin

        self.clus_p_min_spin = QSpinBox()
        self.clus_p_min_spin.setRange(1, 100)
        self.clus_p_min_spin.setValue(5)  # Default
        clus_p_layout.addRow(UI_TEXTS["clus_min_samples_label"], self.clus_p_min_spin)
        self.widgets["clustering.portrait.min_samples"] = self.clus_p_min_spin

        clus_layout.addWidget(clus_p_group)  # Добавляем подсекцию портретов

        # --- Подсекция: Группы ---
        clus_g_group = QGroupBox(UI_TEXTS["clus_group_group"])
        clus_g_layout = QFormLayout(clus_g_group)

        self.clus_g_algo_combo = QComboBox()
        self.clus_g_algo_combo.addItems(["DBSCAN", "HDBSCAN"])
        clus_g_layout.addRow(UI_TEXTS["clus_algo_label"], self.clus_g_algo_combo)
        self.widgets["clustering.group.algorithm"] = self.clus_g_algo_combo

        self.clus_g_eps_spin = QDoubleSpinBox()
        self.clus_g_eps_spin.setRange(0.01, 1.0)
        self.clus_g_eps_spin.setSingleStep(0.01)
        self.clus_g_eps_spin.setDecimals(3)
        self.clus_g_eps_spin.setValue(0.5)  # Default
        clus_g_layout.addRow(UI_TEXTS["clus_group_eps_label"], self.clus_g_eps_spin)
        self.widgets["clustering.group.eps"] = self.clus_g_eps_spin

        self.clus_g_min_spin = QSpinBox()
        self.clus_g_min_spin.setRange(1, 100)
        self.clus_g_min_spin.setValue(5)  # Default
        clus_g_layout.addRow(
            UI_TEXTS["clus_group_min_samples_label"], self.clus_g_min_spin
        )
        self.widgets["clustering.group.min_samples"] = self.clus_g_min_spin

        clus_layout.addWidget(clus_g_group)  # Добавляем подсекцию групп
        parent_layout.addWidget(clus_group)  # Добавляем основную группу кластеризации

    def _create_moving_section(self, parent_layout: QVBoxLayout):
        """Создает секцию настроек перемещения файлов."""
        moving_group = QGroupBox(UI_TEXTS["moving_group_title"])
        moving_layout = QFormLayout(moving_group)

        # Чекбокс перемещения/копирования
        self.moving_move_copy_check = QCheckBox(UI_TEXTS["moving_move_copy_check"])
        moving_layout.addRow(self.moving_move_copy_check)
        self.widgets["moving.move_or_copy_files"] = self.moving_move_copy_check

        # Список расширений
        moving_layout.addRow(QLabel(UI_TEXTS["moving_extensions_label"]))
        self.moving_ext_list = QListWidget()
        self.moving_ext_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.moving_ext_list.setFixedHeight(60)
        moving_layout.addRow(self.moving_ext_list)
        self.widgets["moving.file_extensions_to_action"] = self.moving_ext_list

        # Поле ввода и кнопки для списка расширений
        move_ext_controls_layout = QHBoxLayout()
        self.move_ext_add_edit = QLineEdit()
        self.move_ext_add_edit.setPlaceholderText(UI_TEXTS["moving_ext_placeholder"])
        move_ext_controls_layout.addWidget(self.move_ext_add_edit)
        move_ext_add_btn = QPushButton(UI_TEXTS["moving_add_btn"])
        move_ext_add_btn.clicked.connect(self.add_moving_extension)
        move_ext_controls_layout.addWidget(move_ext_add_btn)
        move_ext_del_btn = QPushButton(UI_TEXTS["moving_del_btn"])
        move_ext_del_btn.clicked.connect(self.remove_moving_extension)
        move_ext_controls_layout.addWidget(move_ext_del_btn)
        moving_layout.addRow(move_ext_controls_layout)

        parent_layout.addWidget(moving_group)

    def _create_bottom_section(self, parent_layout: QVBoxLayout):
        """Создает нижнюю часть UI: уровень лога, кнопки, лог, прогресс."""
        # Уровень логирования
        log_level_layout = QHBoxLayout()
        log_level_layout.addWidget(QLabel(UI_TEXTS["log_level_label"]))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText("INFO")
        self.log_level_combo.currentTextChanged.connect(self.change_log_level)
        log_level_layout.addWidget(self.log_level_combo)
        log_level_layout.addStretch(1)  # Растягиваем, чтобы прижать влево
        parent_layout.addLayout(log_level_layout)
        self.widgets["logging_level"] = self.log_level_combo

        # Кнопки управления
        button_layout = QHBoxLayout()
        self.save_config_btn = QPushButton(UI_TEXTS["save_config_btn"])
        self.save_config_btn.clicked.connect(self.save_config_action)
        button_layout.addWidget(self.save_config_btn)
        self.run_btn = QPushButton(UI_TEXTS["run_btn"])
        self.run_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_processing_action)
        button_layout.addWidget(self.run_btn)
        self.exit_btn = QPushButton(UI_TEXTS["exit_btn"])
        self.exit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.exit_btn)
        parent_layout.addLayout(button_layout)

        # Область вывода лога
        parent_layout.addWidget(QLabel(UI_TEXTS["log_area_label"]))
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(2000)
        # Политика размера для растягивания лога
        size_policy_log = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        size_policy_log.setVerticalStretch(1)  # Даем вертикальный вес
        self.log_output.setSizePolicy(size_policy_log)
        parent_layout.addWidget(self.log_output)

        # Прогресс бар
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel(UI_TEXTS["progress_label"]))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        parent_layout.addLayout(progress_layout)

    @Slot(int, int)
    def update_progress_bar(self, current_step, total_steps):
        """Обновляет прогресс бар."""
        value = int((current_step / total_steps) * 100) if total_steps > 0 else 0
        self.progress_bar.setValue(value)

    @Slot(bool)
    def on_processing_finished(self, success):
        """Обрабатывает завершение рабочего потока."""
        self.run_btn.setEnabled(True)
        self.run_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        # Не устанавливаем 100% если были ошибки, чтобы было видно последний %
        if success:
            self.progress_bar.setValue(100)
        QMessageBox.information(
            self,
            UI_TEXTS["msg_info"],
            UI_TEXTS["msg_processing_status"].format(
                UI_TEXTS["msg_status_success"]
                if success
                else UI_TEXTS["msg_status_error"]
            ),
        )
        # Очистка потока
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()  # Ждем завершения
        self.worker_thread = None
        self.processing_worker = None

    @Slot(str)
    def on_processing_error(self, error_message):
        """Обрабатывает критическую ошибку из рабочего потока."""
        QMessageBox.critical(
            self,
            UI_TEXTS["msg_error"],
            UI_TEXTS["msg_critical_error"].format(error_message),
        )
        # Разблокируем кнопку и сбрасываем поток
        self.run_btn.setEnabled(True)
        self.run_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker_thread = None
        self.processing_worker = None

    @Slot()
    def browse_config_file(self):
        """Открывает диалог выбора файла конфигурации."""
        start_dir = ""
        if self.current_config_path and Path(self.current_config_path).exists():
            start_dir = str(Path(self.current_config_path).parent)
        elif Path(DEFAULT_CONFIG_FILENAME).exists():
            start_dir = str(Path(DEFAULT_CONFIG_FILENAME).parent)
        else:
            start_dir = str(Path.cwd())  # Fallback на текущую директорию

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            UI_TEXTS["dialog_select_config_title"],
            start_dir,
            UI_TEXTS["dialog_select_config_filter"],
        )
        if filepath:
            self.config_path_edit.setText(filepath)
            self.load_config_action()  # Загружаем выбранный конфиг

    @Slot()
    def browse_directory(self, line_edit_widget: QLineEdit, title: str):
        """Открывает диалог выбора папки."""
        start_dir = line_edit_widget.text() or str(Path.home())
        # Проверяем существование start_dir
        if not Path(start_dir).is_dir():
            start_dir = str(Path.home())

        directory = QFileDialog.getExistingDirectory(
            self,
            title,
            start_dir,
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if directory:
            line_edit_widget.setText(str(Path(directory)))  # Нормализуем путь

    @Slot()
    def load_config_action(self):
        """Загружает конфигурацию из файла, указанного в поле ввода."""
        filepath = self.config_path_edit.text()
        if filepath:
            path_obj = Path(filepath)
            if path_obj.exists() and path_obj.is_file():
                loaded_data = load_config_data(filepath)
                if loaded_data:
                    self.load_and_update_gui(filepath, loaded_data)  # Обновляем GUI
            else:
                QMessageBox.warning(
                    self,
                    UI_TEXTS["msg_error"],
                    UI_TEXTS["msg_file_not_found"].format(filepath),
                )
        else:
            QMessageBox.warning(
                self, UI_TEXTS["msg_warning"], UI_TEXTS["msg_config_path_missing"]
            )

    @Slot()
    def save_config_action(self):  # PEP8
        filepath = self.config_path_edit.text()
        if not filepath:
            start_dir = (
                str(Path(self.current_config_path).parent)
                if self.current_config_path
                else str(Path.cwd())
            )
            default_name = (
                Path(self.current_config_path).name
                if self.current_config_path
                else Path(DEFAULT_CONFIG_PATH).name
            )
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                UI_TEXTS["dialog_save_config_title"],
                str(Path(start_dir) / default_name),
                UI_TEXTS["dialog_save_config_filter"],
            )
            if not filepath:
                return
        filepath_obj = Path(filepath)
        if filepath_obj.suffix.lower() != ".toml":
            filepath = str(filepath_obj.with_suffix(".toml"))
        config_to_save = self.get_config_from_gui()
        try:
            self.save_config_to_file(filepath, config_to_save)
            self.current_config_path = filepath
            self.config_path_edit.setText(filepath)
            reloaded_data = load_config_data(self.current_config_path)
            if reloaded_data:
                self.load_and_update_gui(self.current_config_path, reloaded_data)
            else:
                logger.error("Не удалось перезагрузить конфиг после сохранения.")
            QMessageBox.information(
                self, UI_TEXTS["msg_info"], UI_TEXTS["msg_config_saved"]
            )
        except IOError as e:
            logger.error(f"Обработанная ошибка сохранения файла в GUI: {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при сохранении: {e}", exc_info=True)
            QMessageBox.critical(self, UI_TEXTS["msg_error"], f"Ошибка:\n{e}")

    @Slot()
    def add_moving_extension(self):
        list_widget = self.widgets.get("moving.file_extensions_to_action")
        if isinstance(list_widget, QListWidget):
            self._add_item_to_list(list_widget, self.move_ext_add_edit, ".")

    @Slot()
    def remove_moving_extension(self):
        list_widget = self.widgets.get("moving.file_extensions_to_action")
        if isinstance(list_widget, QListWidget):
            self._remove_selected_item_from_list(list_widget)

    @Slot()
    def add_raw_extension(self):
        list_widget = self.widgets.get("processing.raw_extensions")
        if isinstance(list_widget, QListWidget):
            self._add_item_to_list(list_widget, self.raw_ext_add_edit, ".")

    @Slot()
    def remove_raw_extension(self):
        list_widget = self.widgets.get("processing.raw_extensions")
        if isinstance(list_widget, QListWidget):
            self._remove_selected_item_from_list(list_widget)

    def _add_item_to_list(
        self, list_widget: QListWidget, line_edit: QLineEdit, prefix: str = ""
    ):
        """Добавляет элемент в список, если его там еще нет."""
        item_text = line_edit.text().strip()
        if prefix and item_text and not item_text.startswith(prefix):
            item_text = prefix + item_text.lstrip(
                prefix
            )  # Добавляем префикс, если его нет

        item_text = item_text.lower()  # Приводим к нижнему регистру

        if item_text and (
            not prefix or item_text != prefix
        ):  # Проверяем, что строка не пустая и не равна просто префиксу
            current_items = {
                list_widget.item(i).text() for i in range(list_widget.count())
            }  # Используем set для быстрой проверки
            if item_text not in current_items:
                list_widget.addItem(item_text)
                line_edit.clear()
            else:
                QMessageBox.information(
                    self,
                    UI_TEXTS["msg_info"],
                    UI_TEXTS["msg_ext_already_exists"].format(item_text),
                )
        else:
            QMessageBox.warning(
                self,
                UI_TEXTS["msg_error"],
                UI_TEXTS["msg_ext_invalid_prefix"].format(prefix),
            )

    def _remove_selected_item_from_list(self, list_widget: QListWidget):
        """Удаляет выбранный элемент из списка."""
        selected_items = list_widget.selectedItems()
        if selected_items:
            # Удаляем с конца, чтобы индексы не сдвигались
            for item in reversed(selected_items):
                list_widget.takeItem(list_widget.row(item))
        else:
            QMessageBox.information(
                self, UI_TEXTS["msg_info"], UI_TEXTS["msg_ext_select_to_remove"]
            )

    @Slot()
    def run_processing_action(self):
        """Запускает процесс обработки в отдельном потоке после сохранения конфига."""
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(
                self, UI_TEXTS["msg_warning"], UI_TEXTS["msg_processing_running"]
            )
            return

        config_path_to_use = self.config_path_edit.text() or self.current_config_path
        # Если путь все еще не определен, предлагаем сохранить
        if not config_path_to_use:
            start_dir = str(Path.cwd())
            config_path_to_use, _ = QFileDialog.getSaveFileName(
                self,
                UI_TEXTS["dialog_save_config_title"],
                str(DEFAULT_CONFIG_PATH),
                UI_TEXTS["dialog_save_config_filter"],
            )
            if not config_path_to_use:
                logger.warning(UI_TEXTS["msg_save_cancelled"])
                return
            if not config_path_to_use.lower().endswith(".toml"):
                config_path_to_use += ".toml"
            # Обновляем путь в GUI
            self.current_config_path = config_path_to_use
            self.config_path_edit.setText(config_path_to_use)

        # Подтверждение запуска с сохранением
        confirmation_text = UI_TEXTS["msg_confirm_run_text"].format(config_path_to_use)
        reply = QMessageBox.question(
            self,
            UI_TEXTS["msg_confirm_run_title"],
            confirmation_text,
            buttons=QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            defaultButton=QMessageBox.StandardButton.Ok,
        )

        if reply == QMessageBox.StandardButton.Ok:
            logger.info(f"Сохранение настроек перед запуском в: {config_path_to_use}")
            config_saved_and_validated = False
            try:
                config_to_save = self.get_config_from_gui()  # Собираем данные
                self.save_config_to_file(
                    config_path_to_use, config_to_save
                )  # Сохраняем
                # Перезагружаем и обновляем GUI для проверки и консистентности
                reloaded_data = load_config_data(config_path_to_use)
                if reloaded_data:
                    self.load_and_update_gui(config_path_to_use, reloaded_data)
                    config_saved_and_validated = True
                else:
                    logger.error(
                        "Не удалось перезагрузить/провалидировать конфигурацию после сохранения."
                    )
                    # Ошибка уже должна была быть показана в load_config_data
            except IOError as io_err:
                # Ошибка уже показана в self.save_config_to_file
                logger.error(
                    f"Ошибка сохранения файла конфигурации перед запуском (обработана): {io_err}"
                )
            except Exception as other_err:
                logger.error(
                    f"Критическая ошибка при сохранении/перезагрузке конфига перед запуском: {other_err}",
                    exc_info=True,
                )
                QMessageBox.critical(
                    self,
                    UI_TEXTS["msg_error"],
                    f"Критическая ошибка при сохранении/загрузке настроек:\n{other_err}",
                )

            # Если сохранение или валидация не удались, прерываем запуск
            if not config_saved_and_validated:
                logger.error(
                    "Сохранение или перезагрузка конфигурации не удались. Запуск отменен."
                )
                return

            # Обновляем текущий путь (на случай, если пользователь выбрал новый файл)
            self.current_config_path = config_path_to_use

            # Финальная проверка существования файла перед запуском потока
            if (
                not self.current_config_path
                or not Path(self.current_config_path).exists()
            ):
                QMessageBox.critical(
                    self,
                    UI_TEXTS["msg_error"],
                    UI_TEXTS["msg_config_not_found_run"].format(
                        self.current_config_path or "N/A"
                    ),
                )
                return

            # Запускаем рабочий поток
            logger.info(f"Запуск обработки с конфигурацией: {self.current_config_path}")
            self.log_output.clear()
            self.progress_bar.setValue(0)
            self.run_btn.setEnabled(False)
            self.run_btn.setStyleSheet("background-color: orange; font-weight: bold;")

            self.worker_thread = QThread()
            self.processing_worker = ProcessingWorker(self.current_config_path)
            self.processing_worker.moveToThread(self.worker_thread)

            # Соединяем сигналы и слоты
            self.processing_worker.log_message.connect(self._add_log_message)
            self.processing_worker.progress_updated.connect(self.update_progress_bar)
            self.processing_worker.processing_finished.connect(
                self.on_processing_finished
            )
            self.processing_worker.error_occurred.connect(self.on_processing_error)
            self.worker_thread.started.connect(self.processing_worker.run)

            # Очистка после завершения потока
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            # Очистка объекта worker после завершения его работы
            self.processing_worker.processing_finished.connect(
                self.processing_worker.deleteLater
            )

            self.worker_thread.start()

        elif reply == QMessageBox.StandardButton.Cancel:
            logger.info("Запуск обработки отменен пользователем.")

    @Slot(str)
    def change_log_level(self, level_str: str):
        """Изменяет уровень логирования."""
        try:
            new_level_enum = getattr(logging, level_str.upper(), logging.INFO)
            # Устанавливаем уровень для корневого логгера
            logging.getLogger().setLevel(new_level_enum)
            # Устанавливаем уровень для GUI логгера (если он есть)
            if self.qt_log_handler:
                self.qt_log_handler.setLevel(new_level_enum)
            logger.info(f"Уровень логирования изменен на: {level_str}")
        except Exception as e:
            logger.error(
                f"Не удалось изменить уровень логирования на '{level_str}': {e}"
            )

    # --- ИЗМЕНЕНИЕ: load_initial_config использует DEFAULT_CONFIG_PATH ---
    def load_initial_config(self):
        """Загружает конфигурацию при запуске приложения."""
        config_to_load_path = Path(
            self.current_config_path
        )  # Берем текущий (инициализирован DEFAULT_CONFIG_PATH)
        if not config_to_load_path.is_file():
            # Если текущий (дефолтный) не найден
            logger.warning(
                f"Файл конфигурации по умолчанию {self.current_config_path} не найден."
            )
            self.loaded_config_data = (
                self.get_config_from_gui()
            )  # Используем дефолты GUI
            return

        logger.info(f"Загрузка начальной конфигурации из {config_to_load_path}...")
        loaded_data = load_config_data(str(config_to_load_path))
        if loaded_data:
            self.load_and_update_gui(str(config_to_load_path), loaded_data)
        else:
            logger.warning(
                f"Не удалось загрузить {config_to_load_path.name}, GUI исп. defaults."
            )
            self.loaded_config_data = self.get_config_from_gui()

    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    def load_and_update_gui(
        self, filepath: str, loaded_data: Optional[dict] = None
    ) -> None:
        """Загружает данные из файла и обновляет GUI, используя self.widgets."""
        if loaded_data is None:
            loaded_data = load_config_data(filepath)  # Загружаем, если не переданы

        if loaded_data:
            try:
                # Обновляем виджеты
                update_gui_from_config_data(loaded_data, self.widgets)
                self.current_config_path = filepath  # Обновляем путь
                self.config_path_edit.setText(filepath)  # Обновляем поле ввода пути
                self.loaded_config_data = (
                    loaded_data  # Сохраняем валидные загруженные данные
                )
                logger.info(f"GUI обновлен из файла: {filepath}")

                # Обновляем состояние зависимых задач
                task_checkboxes_dict = self.widgets.get("task")
                if isinstance(task_checkboxes_dict, dict):
                    main_task_cb = task_checkboxes_dict.get(
                        "run_image_analysis_and_clustering"
                    )
                    if main_task_cb and isinstance(main_task_cb, QCheckBox):
                        self._update_dependent_tasks_state(
                            main_task_cb.checkState().value
                        )
            except Exception as update_err:
                logger.error(
                    f"Ошибка обновления GUI из {filepath}: {update_err}", exc_info=True
                )
                QMessageBox.warning(
                    self,
                    UI_TEXTS["msg_error"],
                    UI_TEXTS["msg_gui_update_error"].format(filepath, update_err),
                )
                # Сохраняем данные, даже если GUI не обновился полностью
                self.loaded_config_data = loaded_data
        else:
            # Ошибка загрузки была показана ранее в load_config_data
            logger.warning(
                f"Не удалось загрузить данные из {filepath}, GUI не обновлен."
            )
            # Не меняем self.loaded_config_data

    def save_config_to_file(self, filepath: str, config_data: dict):
        """Вызывает глобальную функцию сохранения, обрабатывая ошибки GUI."""
        try:
            save_config_to_file(filepath, config_data)
        except IOError as e:
            # Показываем ошибку пользователю
            QMessageBox.critical(self, UI_TEXTS["msg_error"], str(e))
            raise  # Пробрасываем дальше, чтобы прервать, например, запуск
        except Exception as e:
            logger.error(f"Неожиданная ошибка при вызове save_config_to_file: {e}")
            QMessageBox.critical(
                self, UI_TEXTS["msg_error"], f"Неожиданная ошибка сохранения:\n{e}"
            )
            raise

    def get_config_from_gui(self) -> dict:
        """Вызывает глобальную функцию для сбора данных из виджетов."""
        return get_config_from_gui_widgets(self.widgets)

    def closeEvent(self, event):
        """Обрабатывает событие закрытия окна."""
        logger.info("Окно GUI закрывается.")
        # Удаляем GUI логгер
        if self.qt_log_handler:
            logger.info("Удаление QtLogHandler.")
            logging.getLogger().removeHandler(self.qt_log_handler)
            self.qt_log_handler.close()
            self.qt_log_handler = None
        # Останавливаем рабочий поток, если он запущен
        if self.worker_thread and self.worker_thread.isRunning():
            logger.warning("Обработка прерывается из-за закрытия окна.")
            self.worker_thread.quit()
            # Даем потоку немного времени на завершение
            if not self.worker_thread.wait(1000):  # 1 секунда
                logger.warning(
                    "Рабочий поток не завершился вовремя. Возможно принудительное завершение."
                )
                # Можно добавить self.worker_thread.terminate(), но это рискованно
        super().closeEvent(event)


# --- Запуск приложения ---
if __name__ == "__main__":
    # Проверка версии Python
    if sys.version_info < (3, 8):
        print(UI_TEXTS["msg_python_version_required"])
        # Показать сообщение об ошибке, если возможно
        try:
            app_temp = QApplication([])
            QMessageBox.critical(
                None,
                UI_TEXTS["msg_python_version_error"],
                UI_TEXTS["msg_python_version_required"],
            )
        except Exception:
            pass  # Если GUI не доступен, просто выходим
        sys.exit(1)

    # Начальная настройка логирования до создания основного окна
    initial_log_level = "INFO"
    initial_log_file = "face_processing_gui.log"
    config_for_log_setup = None

    # --- ИЗМЕНЕНИЕ: Используем DEFAULT_CONFIG_PATH ---
    if DEFAULT_CONFIG_PATH.is_file():
        try:
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f_cfg:
                config_for_log_setup = toml.load(f_cfg)
        except Exception as cfg_err:
            print(
                f"Предупреждение: Не удалось прочитать {DEFAULT_CONFIG_PATH.name} для логгера: {cfg_err}"
            )
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    if config_for_log_setup:
        log_level_from_config = config_for_log_setup.get("logging_level")
        if isinstance(log_level_from_config, str) and log_level_from_config:
            potential_level = log_level_from_config.upper()
            if potential_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                initial_log_level = potential_level
            else:
                print(
                    f"Предупреждение: Некорректный уровень логирования '{log_level_from_config}' в конфиге. Используется INFO."
                )
        elif log_level_from_config is not None:
            print(
                f"Предупреждение: Некорректный тип logging_level: {type(log_level_from_config)}. Используется INFO."
            )

    # Настраиваем логирование (без GUI хендлера пока)
    setup_logging(
        log_level_str=initial_log_level, log_file=initial_log_file, gui_log_handler=None
    )

    # Включаем перехват стандартных предупреждений Python
    logging.captureWarnings(True)
    logger_warnings = logging.getLogger("py.warnings")
    # Устанавливаем уровень для логгера предупреждений (можно INFO/DEBUG, чтобы видеть все)
    logger_warnings.setLevel(logging.WARNING)
    logger_warnings.propagate = True  # Убедимся, что сообщения идут в корневой логгер
    logging.info("Перехват стандартных предупреждений Python включен.")

    # Создаем и запускаем приложение
    app = QApplication(sys.argv)
    try:
        main_window = MainWindow()  # Теперь логгер GUI добавится внутри MainWindow
        main_window.show()
        sys.exit(app.exec())
    except Exception as main_err:
        # Логируем неперехваченную ошибку в главном потоке
        logging.exception("Неперехваченная ошибка в главном потоке GUI!")
        # Пытаемся показать сообщение об ошибке
        try:
            QMessageBox.critical(
                None,  # Нет родительского окна
                UI_TEXTS["msg_gui_error_critical"],
                UI_TEXTS["msg_gui_unhandled_error"].format(main_err),
            )
        except Exception as msg_err:
            # Если даже QMessageBox не работает, выводим в stderr
            print(
                f"КРИТИЧЕСКАЯ ОШИБКА GUI! Не удалось показать QMessageBox: {msg_err}",
                file=sys.stderr,
            )
            print(
                f"Исходная ошибка: {main_err}\n{traceback.format_exc()}",
                file=sys.stderr,
            )
        sys.exit(1)
