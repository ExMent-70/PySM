# pysm_lib/app_controller.py

import logging
import pathlib
import json
import os
from typing import List, Dict, Optional, Any

from PySide6.QtCore import QObject, Signal, QTimer, Slot, QThread
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog

# 1. Блок: Измененные импорты
# ==============================================================================
from .gui.dialogs import SettingsDialog # <--- Добавляем импорт
from .config_manager import ConfigManager, AppConfigModel
from .theme_manager import ThemeManager

from .script_scanner import ScriptScannerWorker
from .models import (
    ScriptInfoModel,
    CategoryNodeModel,
    ScanTreeNodeType,
    SetFolderNodeModel,
    ScriptSetNodeModel,
    SetHierarchyNodeType,
    ScriptSetEntryModel,
    ScriptSetEntryValueEnabled,
    ContextVariableModel,
    ScriptArgMetaDetailModel,
)
from . import pysm_context
from .set_manager import SetManager
from .locale_manager import LocaleManager
from .set_runner_orchestrator import SetRunnerOrchestrator
from .app_enums import AppState, ScriptRunStatus
from .app_constants import APPLICATION_ROOT_DIR

logger = logging.getLogger(f"PyScriptManager.{__name__}")


class AppController(QObject):
    available_scripts_updated = Signal(list)
    script_info_updated = Signal(ScriptInfoModel)
    current_collection_updated = Signal(list, object)
    script_instance_updated = Signal(str, ScriptSetEntryModel)
    active_set_node_changed = Signal(object)
    collection_dirty_state_changed = Signal(bool)
    log_message_to_console = Signal(str, str)
    script_instance_status_changed = Signal(str, object)
    set_run_started = Signal(str)
    set_run_completed = Signal(str, bool)
    set_run_stopped = Signal(str)
    status_message_updated = Signal(str)
    clear_console_request = Signal()
    script_progress_updated = Signal(str, int, int, object)
    app_busy_state_changed = Signal(bool)
    controller_state_updated = Signal()
    config_updated = Signal()
    scan_state_changed = Signal(bool)
    run_mode_restored = Signal(str)

    def __init__(
        self,
        config_manager_instance: Optional[ConfigManager] = None,
        locale_manager: Optional[LocaleManager] = None,
    ):
        super().__init__()
        self.locale_manager = locale_manager or LocaleManager()
        logger.info(self.locale_manager.get("app_controller.log_info.init_start"))

        self.config_manager = config_manager_instance or ConfigManager()


        # --- ИЗМЕНЕНИЕ: Инициализация ThemeManager ---
        self.theme_manager = ThemeManager()
        self.theme_manager.set_active_theme(self.config_manager.config.general.active_theme_name)
        
        
        self.set_manager = SetManager()
        self.available_scripts: List[ScanTreeNodeType] = []
        self._scripts_by_id_cache: Dict[str, ScriptInfoModel] = {}

        self.current_collection_file_path: Optional[pathlib.Path] = None
        self.selected_set_node_id: Optional[str] = None
        self.selected_set_node_model: Optional[ScriptSetNodeModel] = None

        self.scanner_thread: Optional[QThread] = None
        self.scanner_worker: Optional[ScriptScannerWorker] = None
        self._node_id_to_select_after_scan: Optional[str] = None

        self._copied_script_entry: Optional[ScriptSetEntryModel] = None
        self.current_orchestrator: Optional[SetRunnerOrchestrator] = None

        self._app_state: AppState = AppState.IDLE
        self.script_run_statuses: Dict[str, ScriptRunStatus] = {}

        self._apply_path_updates_to_current_process()
        QTimer.singleShot(0, self.load_initial_state)
        logger.info(self.locale_manager.get("app_controller.log_info.init_done"))




    # 3. Блок: Метод apply_application_theme (ПЕРЕРАБОТАН)
    # ==============================================================================
    def apply_application_theme(self):
        """
        Получает QSS-стиль из ThemeManager, применяет его к приложению,
        очищает консоль и обновляет все виджеты.
        """
        try:
            stylesheet = self.theme_manager.get_active_theme_qss()
            theme_name = self.theme_manager.get_active_theme_name()
            
            app_instance = QApplication.instance()
            if not app_instance:
                logger.error("Экземпляр QApplication не найден, тема не может быть применена.")
                return

            app_instance.setStyleSheet(stylesheet)
            logger.info(f"Тема '{theme_name}' успешно применена.")
            
            # Очищаем консоль и выводим приветственное сообщение в новом стиле
            self.clear_console_request.emit()
            self._log_welcome_message()
            
            # --- НАЧАЛО ИЗМЕНЕНИЙ ---
            # Запрашиваем полное обновление (перерисовку) дерева коллекции.
            # Это необходимо, чтобы обновились все цвета, иконки и, что важно,
            # всплывающие подсказки (tooltips), которые генерируются с учетом
            # стилей из theme.toml.
            self._request_collection_view_update()
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            
            # Отправляем сигнал, чтобы остальные виджеты (например, AvailableScripts)
            # также могли обновить свои не-QSS стили (в основном, подсказки).
            self.config_updated.emit()
            
        except Exception as e:
            logger.error(f"Не удалось применить тему оформления: {e}", exc_info=True)

    @Slot(dict)
    def apply_new_config(self, settings_data: Dict[str, Any]):
        if not settings_data:
            return
        
        initial_language = self.config_manager.language
        initial_theme = self.config_manager.config.general.active_theme_name
        
        new_config_model = AppConfigModel(**settings_data)
        self.config_manager.config = new_config_model
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        new_theme_name = self.config_manager.config.general.active_theme_name
        
        # Пытаемся установить новую тему. Метод вернет False, если произошел откат.
        success = self.theme_manager.set_active_theme(new_theme_name)
        
        # Применяем стили в любом случае (либо новой темы, либо 'default' после отката)
        self.apply_application_theme()

        # Если произошел откат, выводим сообщение и обновляем конфиг
        if not success:
            final_theme_name = self.theme_manager.get_active_theme_name()
            self.status_message_updated.emit(
                f"Ошибка загрузки темы '{new_theme_name}'. Выполнено переключение на '{final_theme_name}'."
            )
            # Обновляем модель конфигурации, чтобы при следующем запуске
            # открылась правильная (работающая) тема.
            self.config_manager.config.general.active_theme_name = final_theme_name
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        if self.config_manager.save_config():
            self.log_message_to_console.emit(
                "runner_info", self.locale_manager.get("user_actions.config_saved")
            )
            
            self._apply_path_updates_to_current_process()

            if initial_language != self.config_manager.language:
                self.status_message_updated.emit(
                    self.locale_manager.get("main_window.status.settings_saved_restart_required")
                )
            # Если сообщение об ошибке темы не было показано, показываем стандартное
            elif success:
                self.status_message_updated.emit(
                    self.locale_manager.get("main_window.status.settings_saved")
                )
        else:
            self.log_message_to_console.emit(
                "script_error_block",
                self.locale_manager.get("user_actions.config_save_error"),
            )
            self.status_message_updated.emit(
                self.locale_manager.get("app_controller.status_config_save_error")
            )

    def _apply_path_updates_to_current_process(self):
        paths_to_add = []
        path_logger = logging.getLogger("PyScriptManager.PathUpdate")
        path_logger.info(
            self.locale_manager.get("app_controller.path_update.log_info.updating_path")
        )

        try:
            py_interpreter_path = self.config_manager.python_interpreter
            if py_interpreter_path and py_interpreter_path.is_file():
                interpreter_dir = str(py_interpreter_path.parent)
                if interpreter_dir not in paths_to_add:
                    paths_to_add.append(interpreter_dir)
        except Exception as e:
            path_logger.error(
                self.locale_manager.get(
                    "app_controller.path_update.log_error.get_interpreter_path_failed",
                    error=e,
                )
            )

        try:
            additional_paths_str = self.config_manager.additional_env_paths
            for path_str in additional_paths_str:
                p_obj = pathlib.Path(path_str)
                if p_obj.is_dir():
                    abs_path_str = str(p_obj.resolve())
                    if abs_path_str not in paths_to_add:
                        paths_to_add.append(abs_path_str)
        except Exception as e:
            path_logger.error(
                self.locale_manager.get(
                    "app_controller.path_update.log_error.get_additional_paths_failed",
                    error=e,
                )
            )

        if not paths_to_add:
            return
        current_path = os.environ.get("PATH", "")
        new_path_elements = paths_to_add + ([current_path] if current_path else [])
        unique_path_elements = list(dict.fromkeys(filter(None, new_path_elements)))
        os.environ["PATH"] = os.pathsep.join(unique_path_elements)
        path_logger.info(
            self.locale_manager.get("app_controller.path_update.log_info.path_updated")
        )

    def load_initial_state(self):
        logger.info(
            self.locale_manager.get("app_controller.log_info.loading_initial_state")
        )
        last_collection_path_str = self.config_manager.last_used_sets_collection_file
        last_active_set_id = self.config_manager.last_used_script_set
        if (
            last_collection_path_str
            and pathlib.Path(last_collection_path_str).is_file()
        ):
            self.open_collection_requested_by_gui(
                pathlib.Path(last_collection_path_str), last_active_set_id
            )
        else:
            self.new_collection_requested_by_gui()
        logger.info(
            self.locale_manager.get("app_controller.log_info.initial_state_loaded")
        )

    def _log_welcome_message(self):
        self.log_message_to_console.emit(
            "set_header",
            "Python Script Manager - автоматизация выполнения Python-скриптов",
        )
        self.log_message_to_console.emit(
            "runner_info",
            'Telegram-канал "Рабочий блокнот школьного фотографа | Андрей Пугачев"',
        )

        self.log_message_to_console.emit(
            "runner_info", "https://t.me/pugachev_fotodeti03"
        )
        self.log_message_to_console.emit("EMPTY_LINE", "")

    def open_collection_requested_by_gui(
        self, file_path: pathlib.Path, node_id_to_select: Optional[str] = None
    ):
        if self.set_manager.load_collection_from_file(file_path):
            self.current_collection_file_path = file_path.resolve()
            self.clear_console_request.emit()
            self._log_welcome_message()
            collection_model = self.set_manager.current_collection_model
            collection_name = collection_model.collection_name
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get(
                    "user_actions.collection_opened", name=collection_name
                ),
            )
            self._log_collection_properties()

            # --- НАЧАЛО ИЗМЕНЕНИЙ ---
            # КОММЕНТАРИЙ: Режим запуска всегда берется из модели коллекции,
            # независимо от того, выбран какой-то набор или нет.
            self._internal_set_and_emit_run_mode(collection_model.execution_mode)
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

            self.config_manager.last_used_sets_collection_file = str(
                self.current_collection_file_path
            )
            self._node_id_to_select_after_scan = node_id_to_select
            self.config_manager.save_config()
            self.refresh_available_scripts_list()
            self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)
            self.status_message_updated.emit(
                self.locale_manager.get(
                    "app_controller.status_collection_loaded", name=file_path.name
                )
            )
        else:
            self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)
            self.status_message_updated.emit(
                self.locale_manager.get(
                    "app_controller.status_collection_load_error", name=file_path.name
                )
            )

    def _log_collection_properties(self):
        model = self.set_manager.current_collection_model

        self.log_message_to_console.emit(
            "script_info",
            "</br>",
        )
        self.log_message_to_console.emit("EMPTY_LINE", "")
        self.log_message_to_console.emit(
            "script_header_block",
            self.locale_manager.get(
                "app_controller.console_log.collection_roots_header"
            ),
        )
        if model.script_roots:
            for root in model.script_roots:
                self.log_message_to_console.emit(
                    "script_info",
                    self.locale_manager.get(
                        "general.bullet_point_prefix", text=root.path
                    ),
                )
        else:
            self.log_message_to_console.emit(
                "script_info",
                self.locale_manager.get(
                    "general.bullet_point_prefix",
                    text=self.locale_manager.get("general.not_available_short"),
                ),
            )
        self.log_message_to_console.emit("EMPTY_LINE", "")
        self.log_message_to_console.emit("EMPTY_LINE", "")

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Добавлен блок для вывода описания коллекции.
        # Если описание отсутствует или пустое, выводится "н/д"
        # --- 3. БЛОК: Логика вывода описания УПРОЩЕНА (ИЗМЕНЕН) ---
        description_raw = model.description or self.locale_manager.get(
            "general.not_available_short"
        )
        
        self.log_message_to_console.emit(
            "script_header_block",
            self.locale_manager.get(
                "app_controller.console_log.collection_description",
            ),
        )
        
        # Просто отправляем "сырой" HTML в консоль.
        # ConsoleWidget сам разберется с плейсхолдерами.
        self.log_message_to_console.emit(
            "html_block",
            description_raw
        )

        self.log_message_to_console.emit("EMPTY_LINE", "")

    @Slot(AppState)
    def _set_app_state(self, new_state: AppState):
        old_state = self._app_state
        if old_state == new_state:
            return

        logger.debug(
            self.locale_manager.get(
                "app_controller.log_debug.state_change",
                old=old_state.name,
                new=new_state.name,
            )
        )

        # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ ЛОГИКИ ---
        # "Занято" - это ЛЮБОЕ состояние, кроме IDLE.
        is_busy = new_state != AppState.IDLE
        was_busy = old_state != AppState.IDLE
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        self._app_state = new_state
        self.controller_state_updated.emit()

        if was_busy != is_busy:
            self.app_busy_state_changed.emit(is_busy)

    def refresh_available_scripts_list(self):
        if self.is_busy():
            return
        self._set_app_state(AppState.SCANNING_SCRIPTS)
        self.scan_state_changed.emit(True)
        script_roots = self.set_manager.current_collection_model.script_roots
        self.status_message_updated.emit(
            self.locale_manager.get(
                "app_controller.status_scanning", count=len(script_roots)
            )
        )
        self.scanner_thread = QThread()
        self.scanner_worker = ScriptScannerWorker()
        self.scanner_worker.moveToThread(self.scanner_thread)
        self.scanner_thread.started.connect(
            lambda: self.scanner_worker.run(script_roots)
        )
        self.scanner_worker.finished.connect(self._on_scan_finished)
        self.scanner_worker.error.connect(self._on_scan_error)
        self.scanner_worker.finished.connect(self.scanner_thread.quit)
        self.scanner_thread.finished.connect(self.scanner_thread.deleteLater)
        self.scanner_worker.finished.connect(self.scanner_worker.deleteLater)
        self.scanner_thread.start()

    @Slot(list)
    def _on_scan_finished(self, scanned_nodes: List[ScanTreeNodeType]):
        logger.debug(
            self.locale_manager.get("app_controller.log_debug.scan_finished_signal")
        )
        self.available_scripts = scanned_nodes
        self._rebuild_scripts_cache()
        self.available_scripts_updated.emit(self.available_scripts)
        self.status_message_updated.emit(
            self.locale_manager.get("app_controller.status_scan_complete")
        )
        self._request_collection_view_update(self._node_id_to_select_after_scan)
        self._node_id_to_select_after_scan = None
        self._set_app_state(AppState.IDLE)
        self.scan_state_changed.emit(False)

    @Slot(str)
    def _on_scan_error(self, error_message: str):
        logger.error(
            self.locale_manager.get(
                "app_controller.log_error.async_scan_error", error=error_message
            )
        )
        self.status_message_updated.emit(
            self.locale_manager.get("app_controller.status_scan_error")
        )
        self.available_scripts = []
        self._rebuild_scripts_cache()
        self.available_scripts_updated.emit(self.available_scripts)
        self._set_app_state(AppState.IDLE)
        self.scan_state_changed.emit(False)

    def _rebuild_scripts_cache(self):
        self._scripts_by_id_cache.clear()

        def _recursive_walk(nodes: List[ScanTreeNodeType]):
            for node in nodes:
                if isinstance(node, ScriptInfoModel):
                    self._scripts_by_id_cache[node.id] = node
                elif isinstance(node, CategoryNodeModel) and node.children:
                    _recursive_walk(node.children)

        _recursive_walk(self.available_scripts)
        logger.debug(
            self.locale_manager.get(
                "app_controller.log_debug.cache_rebuilt",
                count=len(self._scripts_by_id_cache),
            )
        )

    @Slot(str, ScriptInfoModel)
    def save_script_passport(self, script_id: str, updated_model: ScriptInfoModel):
        original_model = self.get_script_info_by_id(script_id)
        if not original_model:
            return
        folder_path = pathlib.Path(original_model.folder_abs_path)
        passport_file_path = folder_path / "script_passport.json"
        passport_data = updated_model.model_dump(
            mode="json",
            exclude={
                "id",
                "name",
                "folder_abs_path",
                "run_filename",
                "run_file_abs_path",
                "type",
                "passport_valid",
                "passport_error",
                "is_raw",
                "requirements",
            },
            exclude_none=True,
        )
        try:
            with open(passport_file_path, "w", encoding="utf-8") as f:
                json.dump(passport_data, f, indent=2, ensure_ascii=False)
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get(
                    "user_actions.passport_saved", name=original_model.name
                ),
            )
            original_model.description = updated_model.description
            original_model.author = updated_model.author
            original_model.version = updated_model.version
            original_model.command_line_args_meta = updated_model.command_line_args_meta
            original_model.specific_python_interpreter = (
                updated_model.specific_python_interpreter
            )
            original_model.script_specific_env_paths = (
                updated_model.script_specific_env_paths
            )
            original_model.passport_valid = True
            original_model.is_raw = False
            original_model.passport_error = None
            self.script_info_updated.emit(original_model)
        except Exception as e:
            logger.error(
                f"Ошибка при сохранении паспорта для скрипта '{script_id}': {e}",
                exc_info=True,
            )

    def new_collection_requested_by_gui(self):
        self.set_manager.create_new_empty_collection()
        self.current_collection_file_path = None
        self.log_message_to_console.emit(
            "runner_info", self.locale_manager.get("user_actions.collection_new")
        )
        default_mode = self.set_manager.current_collection_model.execution_mode
        self._internal_set_and_emit_run_mode(default_mode)
        self.config_manager.last_used_sets_collection_file = ""
        self.set_active_script_set_node(None)
        self.config_manager.save_config()
        self.refresh_available_scripts_list()
        self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)
        self.status_message_updated.emit(
            self.locale_manager.get("app_controller.status_new_collection")
        )

    def save_current_collection_requested_by_gui(
        self, target_file_path: Optional[pathlib.Path]
    ) -> bool:
        self.set_manager.current_collection_model.execution_mode = (
            self.set_manager.current_collection_model.execution_mode
        )

        if self.set_manager.save_collection_to_file(target_file_path):
            self.current_collection_file_path = (
                self.set_manager.current_collection_file_path
            )

            # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            # Выводим в консоль сообщение об успешном сохранении
            self.clear_console_request.emit()
            self._log_welcome_message()
            
            # Добавляем вызов метода для логирования актуальных свойств коллекции
            self._log_collection_properties()

            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get(
                    "user_actions.collection_saved",
                    path=self.current_collection_file_path,
                ),
            )            
            # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

            self.config_manager.last_used_sets_collection_file = str(
                self.current_collection_file_path
            )
            self.config_manager.save_config()
            self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)
            self._request_collection_view_update()
            self.status_message_updated.emit(
                self.locale_manager.get(
                    "app_controller.status_collection_saved",
                    name=self.current_collection_file_path.name,
                )
            )
            return True
        return False

    @Slot(str)
    def set_collection_run_mode(self, mode_id: str):
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Обновляем режим запуска в главной модели коллекции.
        if self.set_manager.current_collection_model.execution_mode != mode_id:
            self.set_manager.current_collection_model.execution_mode = mode_id
            self.set_manager._set_dirty(True)
            self.collection_dirty_state_changed.emit(True)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    def _internal_set_and_emit_run_mode(self, mode_id: str):
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Этот метод также должен обновлять свойство модели коллекции.
        self.set_manager.current_collection_model.execution_mode = mode_id
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        self.run_mode_restored.emit(mode_id)

    def discard_changes_in_current_collection(self):
        if self.current_collection_file_path:
            self.open_collection_requested_by_gui(self.current_collection_file_path)
        else:
            self.new_collection_requested_by_gui()

    def update_collection_properties(self, name: str, description: Optional[str]):
        old_name = self.set_manager.current_collection_model.collection_name
        if old_name != name:
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get(
                    "user_actions.collection_renamed", old_name=old_name, new_name=name
                ),
            )
        self.log_message_to_console.emit(
            "runner_info",
            self.locale_manager.get("user_actions.collection_props_updated"),
        )
        self.set_manager.update_collection_properties(name, description)
        self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)

    def update_collection_context(self, new_context: Dict[str, ContextVariableModel]):
        self.set_manager.update_collection_context(new_context)
        self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)

    def add_script_root_to_collection(self, path: str):
        if self.set_manager.add_script_root(path):
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get("user_actions.script_root_added", path=path),
            )
            self.refresh_available_scripts_list()
            self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)

    def remove_script_root_from_collection(self, root_id: str):
        if self.set_manager.remove_script_root(root_id):
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get("user_actions.script_root_removed"),
            )
            self.refresh_available_scripts_list()
            self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)

    def update_script_root_in_collection(self, root_id: str, new_path: str):
        if self.set_manager.update_script_root(root_id, new_path):
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get("user_actions.script_root_updated"),
            )
            self.refresh_available_scripts_list()
            self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)

    def _emit_collection_updated(self, node_id_to_select: Optional[str] = None):
        self.current_collection_updated.emit(
            self.set_manager.get_all_nodes_for_display(), node_id_to_select
        )

    def _request_collection_view_update(self, node_id_to_select: Optional[str] = None):
        QTimer.singleShot(0, lambda: self._emit_collection_updated(node_id_to_select))

    def create_folder_in_collection(self, name: str, parent_id: Optional[str]):
        new_node = self.set_manager.add_folder_node(
            name=name, parent_folder_id=parent_id
        )
        if new_node:
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get("user_actions.folder_created", name=name),
            )
            self._request_collection_view_update(new_node.id)
            self.collection_dirty_state_changed.emit(True)

    def create_set_in_collection(self, name: str, parent_id: Optional[str]):
        new_node = self.set_manager.add_set_node(name=name, parent_folder_id=parent_id)
        if new_node:
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get("user_actions.set_created", name=name),
            )
            self._request_collection_view_update(new_node.id)
            self.collection_dirty_state_changed.emit(True)

    def delete_node_from_collection(self, node_id: str):
        node = self.set_manager.get_node_by_id(node_id)
        if not node:
            return
        node_name = node.name
        if self.selected_set_node_id == node_id:
            self.set_active_script_set_node(None)
        if self.set_manager.delete_node(node_id):
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get("user_actions.node_deleted", name=node_name),
            )
            self._request_collection_view_update()
            self.collection_dirty_state_changed.emit(True)

    def rename_node_in_collection(self, node_id: str, new_name: str):
        node = self.set_manager.get_node_by_id(node_id)
        if not node:
            return
        old_name = node.name
        if self.set_manager.update_node_properties(node_id, new_name=new_name):
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get(
                    "user_actions.node_renamed", old_name=old_name, new_name=new_name
                ),
            )
            if self.selected_set_node_id == node_id and self.selected_set_node_model:
                self.selected_set_node_model.name = new_name
                self.active_set_node_changed.emit(self.selected_set_node_model)
            self._request_collection_view_update(node_id)
            self.collection_dirty_state_changed.emit(True)

    def move_node_in_collection(
        self, node_id: str, new_parent_id: Optional[str], target_index: int = -1
    ):
        node = self.set_manager.get_node_by_id(node_id)
        if not node:
            return
        node_name = node.name
        if self.set_manager.move_node(node_id, new_parent_id, target_index):
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get("user_actions.node_moved", name=node_name),
            )
            self._request_collection_view_update(node_id)
            self.collection_dirty_state_changed.emit(True)

    def set_active_script_set_node(self, node_id: Optional[str]):
        if self.is_busy():
            return
        new_active_set_id = ""
        node_model = None
        if node_id:
            node_model = self.set_manager.get_set_node_by_id(node_id)
            if node_model:
                new_active_set_id = node_model.id
        if self.selected_set_node_id == new_active_set_id:
            return
        self.selected_set_node_id = new_active_set_id if node_model else None
        self.selected_set_node_model = node_model
        if self.config_manager.last_used_script_set != (new_active_set_id or ""):
            self.config_manager.last_used_script_set = new_active_set_id or ""
        self.active_set_node_changed.emit(self.selected_set_node_model)

    def duplicate_script_instance(self, set_id: str, instance_id: str):
        entry_to_copy = self.set_manager.get_script_entry_by_instance_id(
            set_id, instance_id
        )
        if not entry_to_copy:
            return
        new_entry = entry_to_copy.create_copy()
        if self.set_manager.add_script_entry_model_to_set(set_id, new_entry):
            self._request_collection_view_update(new_entry.instance_id)
            self.collection_dirty_state_changed.emit(True)

    def copy_script_instance_to_buffer(self, set_id: str, instance_id: str):
        entry_to_copy = self.set_manager.get_script_entry_by_instance_id(
            set_id, instance_id
        )
        if entry_to_copy:
            self._copied_script_entry = entry_to_copy
            self.controller_state_updated.emit()

    def paste_script_instance_from_buffer(self, target_set_id: str):
        if not self._copied_script_entry:
            return
        new_entry = self._copied_script_entry.create_copy()
        if self.set_manager.add_script_entry_model_to_set(target_set_id, new_entry):
            self._request_collection_view_update(new_entry.instance_id)
            self.collection_dirty_state_changed.emit(True)

    def add_script_to_active_set_node(self, script_id: str):
        if not self.selected_set_node_id:
            return
        script_info = self.get_script_info_by_id(script_id)
        set_info = self.set_manager.get_set_node_by_id(self.selected_set_node_id)
        if not script_info or not script_info.passport_valid or not set_info:
            return
        initial_args: Dict[str, ScriptSetEntryValueEnabled] = {}
        if script_info.command_line_args_meta:
            for name, meta in script_info.command_line_args_meta.items():
                is_enabled_by_default = meta.required or meta.default is not None
                initial_args[name] = ScriptSetEntryValueEnabled(
                    value=meta.default, enabled=is_enabled_by_default
                )
        entry_model = ScriptSetEntryModel(
            id=script_id, name=script_info.name, command_line_args=initial_args
        )
        if self.set_manager.add_script_entry_model_to_set(
            self.selected_set_node_id, entry_model
        ):
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get(
                    "user_actions.script_added_to_set",
                    script_name=script_info.name,
                    set_name=set_info.name,
                ),
            )
            self._request_collection_view_update(entry_model.instance_id)
            self.collection_dirty_state_changed.emit(True)

    def remove_script_from_active_set_node(self, instance_id: str):
        if not self.selected_set_node_id:
            return
        set_info = self.set_manager.get_set_node_by_id(self.selected_set_node_id)
        entry_info = self.set_manager.get_script_entry_by_instance_id(
            self.selected_set_node_id, instance_id
        )
        script_info = self.get_script_info_by_id(entry_info.id) if entry_info else None
        parent_id = self.selected_set_node_id
        if self.set_manager.remove_script_entry_from_set(
            self.selected_set_node_id, instance_id
        ):
            script_name = script_info.name if script_info else instance_id
            set_name = set_info.name if set_info else "..."
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get(
                    "user_actions.script_removed_from_set",
                    script_name=script_name,
                    set_name=set_name,
                ),
            )
            self._request_collection_view_update(parent_id)
            self.collection_dirty_state_changed.emit(True)

    def reorder_scripts_in_active_set_node(self, new_ordered_ids: List[str]):
        if not self.selected_set_node_id:
            return
        if self.set_manager.reorder_script_entries_in_set(
            self.selected_set_node_id, new_ordered_ids
        ):
            set_name = (
                self.selected_set_node_model.name
                if self.selected_set_node_model
                else "..."
            )
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get(
                    "user_actions.script_reordered", set_name=set_name
                ),
            )
            self._request_collection_view_update(self.selected_set_node_id)
            self.collection_dirty_state_changed.emit(True)

    def update_script_instance_in_active_set_node(
        self, updated_entry: ScriptSetEntryModel
    ):
        if not self.selected_set_node_id:
            return
        if self.set_manager.update_script_entry(
            self.selected_set_node_id, updated_entry
        ):
            script_info = self.get_script_info_by_id(updated_entry.id)
            script_name = updated_entry.name or (
                script_info.name if script_info else "..."
            )
            set_name = (
                self.selected_set_node_model.name
                if self.selected_set_node_model
                else "..."
            )
            self.log_message_to_console.emit(
                "runner_info",
                self.locale_manager.get(
                    "user_actions.script_params_updated",
                    script_name=script_name,
                    set_name=set_name,
                ),
            )
            self.script_instance_updated.emit(self.selected_set_node_id, updated_entry)
            self.collection_dirty_state_changed.emit(self.set_manager.is_dirty)


    # 1. Блок: Новый метод show_settings_dialog
    # ==============================================================================
    @Slot()
    def show_settings_dialog(self):
        """
        Создает, настраивает и отображает модальный диалог настроек.
        """
        cfg = self.config_manager.config
        main_window = self.get_main_window() # Находим главное окно для родительства
        
        # Создаем диалог, передавая ему все необходимые зависимости
        dialog = SettingsDialog(
            config_model=cfg, 
            theme_manager=self.theme_manager,
            locale_manager=self.locale_manager, 
            parent=main_window
        )
        
        # Подключаемся к сигналу для live-preview смены темы
        dialog.theme_changed.connect(self._on_theme_changed_in_settings)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings_data = dialog.get_settings_data()
            if settings_data:
                self.apply_new_config(settings_data)

    # 2. Блок: Новый приватный слот для live-preview
    # ==============================================================================
    @Slot(str)
    def _on_theme_changed_in_settings(self, theme_name: str):
        """
        Слот для обработки смены темы в диалоге настроек в реальном времени.
        """
        if self.theme_manager.set_active_theme(theme_name):
            self.apply_application_theme()

    # 3. Блок: Новый вспомогательный метод для получения главного окна
    # ==============================================================================
    def get_main_window(self) -> Optional[QWidget]:
        """Находит и возвращает экземпляр MainWindow."""
        # QApplication.topLevelWidgets() возвращает все окна верхнего уровня
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMainWindow):
                return widget
        return None


    def run_script_set(
        self,
        set_node_id: str,
        run_mode: str,
        selected_instance_id: Optional[str] = None,
        continue_on_error: bool = False,
    ):
        if self.is_busy():
            return

        set_node = self.set_manager.get_set_node_by_id(set_node_id)
        if not set_node:
            return

        collection_path = self.set_manager.current_collection_file_path
        if not collection_path:
            self.status_message_updated.emit(
                self.locale_manager.get("collection_widget.run_tooltip_save_first")
            )
            return
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Вся логика подготовки контекста УДАЛЕНА отсюда.
        # Мы просто получаем путь к файлу контекста.
        try:
            context_file_path = self.set_manager._get_context_file_path(collection_path)
        except Exception as e:
            logger.error(f"Не удалось определить путь к файлу контекста: {e}", exc_info=True)
            return
        
        self.script_run_statuses.clear()

        # Создаем оркестратор, передавая ему все необходимые зависимости.
        # Теперь ОН будет отвечать за подготовку контекста.
        self.current_orchestrator = SetRunnerOrchestrator(
            set_node=set_node,
            run_mode=run_mode,
            continue_on_error=continue_on_error,
            get_script_info_func=self.get_script_info_by_id,
            config_manager=self.config_manager,
            theme_manager=self.theme_manager,  # <--- Передаем новый ThemeManager
            locale_manager=self.locale_manager,
            context_file_path=context_file_path,
            selected_instance_id=selected_instance_id,
        )

        # Подключение сигналов остается прежним
        self.current_orchestrator.log_message.connect(self.log_message_to_console)
        self.current_orchestrator.clear_console.connect(self.clear_console_request)
        self.current_orchestrator.run_started.connect(self.set_run_started)
        self.current_orchestrator.instance_status_changed.connect(
            self.script_instance_status_changed
        )
        self.current_orchestrator.progress_updated.connect(self.script_progress_updated)
        self.current_orchestrator.app_state_changed.connect(self._set_app_state)
        self.current_orchestrator.run_completed.connect(self._on_orchestrator_finished)
        self.current_orchestrator.run_stopped.connect(self._on_orchestrator_finished)

        # Запускаем оркестратор
        self.current_orchestrator.start()
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    @Slot(str, bool)
    def _on_orchestrator_finished(self, set_name: str, success: Optional[bool] = None):
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Перезагружаем контекст здесь, когда все операции завершены.
        if self.set_manager.current_collection_file_path:
            self.set_manager.reload_context_from_file()
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        if success is not None:
            self.set_run_completed.emit(set_name, success)
        else:
            self.set_run_stopped.emit(set_name)

        if self.current_orchestrator:
            self.current_orchestrator.deleteLater()
            self.current_orchestrator = None

        # ВАЖНО: Сбрасываем состояние в IDLE после завершения
        self._set_app_state(AppState.IDLE)

    def proceed_to_next_script_in_set_step(self):
        if self.current_orchestrator and self.is_waiting_for_next_step():
            self.current_orchestrator.proceed_to_next_step()

    def stop_current_set_run(self):
        if self.current_orchestrator:
            self.current_orchestrator.stop()

    def get_script_info_by_id(self, script_id: str) -> Optional[ScriptInfoModel]:
        return self._scripts_by_id_cache.get(script_id)

    def get_set_node_display_name_by_id(self, node_id: Optional[str]) -> str:
        if not node_id:
            return self.locale_manager.get("general.not_applicable")
        node = self.set_manager.get_node_by_id(node_id)
        return (
            node.name
            if node
            else self.locale_manager.get("general.id_format", id=node_id)
        )

    def get_known_args_with_details(
        self,
    ) -> Dict[str, List[tuple[str, ScriptArgMetaDetailModel]]]:
        unique_script_ids_in_collection = set()

        def find_script_entries(nodes: List[SetHierarchyNodeType]):
            for node in nodes:
                if isinstance(node, ScriptSetNodeModel):
                    for entry in node.script_entries:
                        unique_script_ids_in_collection.add(entry.id)
                elif isinstance(node, SetFolderNodeModel) and node.children:
                    find_script_entries(node.children)

        find_script_entries(self.set_manager.current_collection_model.root_nodes)
        args_meta_dict: Dict[str, List[tuple[str, ScriptArgMetaDetailModel]]] = {}
        for script_id in unique_script_ids_in_collection:
            script_info = self.get_script_info_by_id(script_id)
            if (
                script_info
                and script_info.passport_valid
                and script_info.command_line_args_meta
            ):
                for arg_name, meta in script_info.command_line_args_meta.items():
                    if arg_name.startswith("__"):
                        continue
                    if arg_name not in args_meta_dict:
                        args_meta_dict[arg_name] = []
                    args_meta_dict[arg_name].append((script_info.name, meta))
        return args_meta_dict

    def is_busy(self) -> bool:
        # Этот метод больше не нужен, т.к. мы управляем UI через смену состояний,
        # но оставим его для совместимости, если где-то используется.
        return self._app_state != AppState.IDLE

    def is_waiting_for_next_step(self) -> bool:
        return self._app_state == AppState.SET_RUNNING_STEP_WAIT
