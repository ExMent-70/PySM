# pysm_lib/set_runner_orchestrator.py

import logging
import pathlib
import json
import time
import shlex
from typing import List, Dict, Optional, Callable
from datetime import datetime

from PySide6.QtCore import QObject, Signal, Slot

from .models import ScriptSetNodeModel, ScriptSetEntryModel, ScriptInfoModel
from .script_runner import ScriptRunner
from .locale_manager import LocaleManager
from .app_constants import APPLICATION_ROOT_DIR
from .config_manager import ConfigManager

# --- 1. Блок: Новый импорт ThemeManager ---
from .theme_manager import ThemeManager

from .app_enums import SetRunMode, ScriptRunStatus, AppState

logger = logging.getLogger(f"PyScriptManager.{__name__}")


class SetRunnerOrchestrator(QObject):
    log_message = Signal(str, str)
    clear_console = Signal()
    run_started = Signal(str)
    run_completed = Signal(str, bool)
    run_stopped = Signal(str)
    instance_status_changed = Signal(str, object)
    progress_updated = Signal(str, int, int, object)
    app_state_changed = Signal(AppState)
    context_reloaded = Signal()

    def __init__(
        self,
        set_node: ScriptSetNodeModel,
        run_mode: str,
        continue_on_error: bool,
        get_script_info_func: Callable[[str], Optional[ScriptInfoModel]],
        config_manager: ConfigManager,
        # --- 2. Блок: Новый аргумент в конструкторе ---
        theme_manager: ThemeManager,
        locale_manager: LocaleManager,
        context_file_path: pathlib.Path,
        selected_instance_id: Optional[str] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.set_node = set_node
        self.run_mode = run_mode
        self.continue_on_error = continue_on_error
        self.get_script_info_by_id = get_script_info_func
        self.config_manager = config_manager
        # --- 3. Блок: Сохраняем экземпляр ThemeManager ---
        self.theme_manager = theme_manager
        self.locale_manager = locale_manager
        self.context_file_path = context_file_path
        self.selected_instance_id = selected_instance_id
        self.script_queue: List[ScriptSetEntryModel] = []
        self.instance_id_to_index_map: Dict[str, int] = {}
        self.active_runners: Dict[str, ScriptRunner] = {}
        self.current_script_idx: int = -1
        self.run_had_errors: bool = False
        self.start_time: float = 0
        self.script_start_time: float = 0
        self._stop_requested: bool = False


    def _prepare_context_file(self):
        try:
            context_data = {}
            if self.context_file_path.is_file():
                with open(self.context_file_path, "r", encoding="utf-8") as f:
                    context_data = json.load(f)

            active_theme_name = self.theme_manager.get_active_theme_name()
            context_data["pysm_active_theme_name"] = {
                "type": "string",
                "value": active_theme_name,
                "description": "System: Name of the active PyScriptManager theme.",
                "read_only": True,
            }

            current_log_level = logging.getLevelName(logger.getEffectiveLevel())
            context_data["sys_log_level"] = {
                "type": "string", "value": current_log_level, "read_only": True
            }

            all_instances_data = []
            for e in self.set_node.script_entries:
                script_info = self.get_script_info_by_id(e.id)
                name = e.name or (script_info.name if script_info else e.id)
                all_instances_data.append({"id": e.instance_id, "name": name})

            context_data["pysm_set_instance_ids"] = {
                "type": "json", "value": all_instances_data, "read_only": True
            }
            context_data.pop("pysm_next_script", None)

            with open(self.context_file_path, "w", encoding="utf-8") as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Не удалось подготовить файл контекста: {e}", exc_info=True)
            raise



    @Slot()
    def start(self):
        self.clear_console.emit()
        
        try:
            self._prepare_context_file()
        except Exception:
            self.log_message.emit("script_error_block", "КРИТИЧЕСКАЯ ОШИБКА: Не удалось создать файл контекста.")
            self._finalize_run(was_stopped=True)
            return

        if self.run_mode == SetRunMode.SINGLE_FROM_SET:
            if not self.selected_instance_id: return
            entry = next((e for e in self.set_node.script_entries if e.instance_id == self.selected_instance_id), None)
            if not entry: return
            self.script_queue = [entry]
        else:
            self.script_queue = [e for e in self.set_node.script_entries if self.get_script_info_by_id(e.id) and self.get_script_info_by_id(e.id).passport_valid]

        if not self.script_queue: return

        for entry in self.set_node.script_entries: self.instance_status_changed.emit(entry.instance_id, None)
        for entry in self.script_queue: self.instance_status_changed.emit(entry.instance_id, ScriptRunStatus.PENDING)

        self.instance_id_to_index_map = {entry.instance_id: i for i, entry in enumerate(self.script_queue)}
        self.start_time = time.time()
        self._log_set_start_info()
        self.run_had_errors = False
        self.current_script_idx = -1
        self.run_started.emit(self.set_node.name)
        self._process_next_script()


    def stop(self):
        self._stop_requested = True
        if self.active_runners:
            for runner in self.active_runners.values():
                runner.stop(runner.script_info_model.id)
        else:
            self._finalize_run(was_stopped=True)

    def proceed_to_next_step(self):
        self._process_next_script()

    def _determine_next_script_index(self) -> int:
        if self.run_mode not in [
            SetRunMode.CONDITIONAL_FULL,
            SetRunMode.CONDITIONAL_STEP,
        ]:
            return self.current_script_idx + 1

        next_instance_id = None
        if self.context_file_path.is_file():
            try:
                with open(self.context_file_path, "r+", encoding="utf-8") as f:
                    context_data = json.load(f)
                    pysm_next_script_data = context_data.pop("pysm_next_script", None)
                    if pysm_next_script_data and isinstance(
                        pysm_next_script_data, dict
                    ):
                        target_data = pysm_next_script_data.get("value")
                        if isinstance(target_data, dict):
                            next_instance_id = target_data.get("id")

                    f.seek(0)
                    f.truncate()
                    json.dump(context_data, f, indent=2, ensure_ascii=False)
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Не удалось прочитать или обновить контекст: {e}")

        if next_instance_id and isinstance(next_instance_id, str):
            if next_instance_id in self.instance_id_to_index_map:
                logger.info(f"Условный переход к скрипту: {next_instance_id}")
                return self.instance_id_to_index_map[next_instance_id]
            else:
                logger.warning(
                    f"Неверный ID '{next_instance_id}' для условного перехода. Выполнение будет продолжено последовательно."
                )

        return self.current_script_idx + 1

    def _process_next_script(self):
        if self._stop_requested:
            self._finalize_run(was_stopped=True)
            return

        self.app_state_changed.emit(AppState.SET_RUNNING_AUTO)

        next_idx = self._determine_next_script_index()
        if next_idx >= len(self.script_queue):
            self._finalize_run(was_stopped=False)
            return

        self.current_script_idx = next_idx
        entry_to_run = self.script_queue[self.current_script_idx]
        script_info = self.get_script_info_by_id(entry_to_run.id)
        if not script_info:
            self.log_message.emit(
                "script_error_block",
                self.locale_manager.get(
                    "app_controller.console_error.skipping_script_no_info",
                    id=entry_to_run.id,
                ),
            )
            self.instance_status_changed.emit(
                entry_to_run.instance_id, ScriptRunStatus.SKIPPED
            )
            self._process_next_script()
            return

        self.script_start_time = time.time()
        py_path = script_info.specific_python_interpreter or str(
            self.config_manager.python_interpreter
        )
        args_for_run = {
            k: v.value for k, v in entry_to_run.command_line_args.items() if v.enabled
        }

        runner = ScriptRunner(
            script_info=script_info,
            python_interpreter=py_path,
            additional_env_paths=self.config_manager.additional_env_paths,
            global_python_paths=self.config_manager.python_paths,
            global_env_vars=self.config_manager.environment_variables,
            on_start=self._handle_script_start,
            on_output=self._handle_script_output,
            on_complete=self._handle_script_complete,
            on_error=self._handle_script_error,
            on_progress=self.progress_updated.emit,
            custom_command_args_dict=args_for_run,
            context_file_path=str(self.context_file_path),
            app_root_dir=APPLICATION_ROOT_DIR,
        )
        self.active_runners[entry_to_run.instance_id] = runner
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Выводим информационную шапку, только если
        # "тихий режим" для этого экземпляра ВЫКЛЮЧЕН.
        if not entry_to_run.silent_mode:
            self._log_script_start_info(script_info, entry_to_run, runner)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        runner.run(entry_to_run.instance_id)

    def _handle_script_output(self, instance_id: str, stream: str, line: str):
        self.log_message.emit(f"script_{stream}", line)

    def _handle_script_start(self, instance_id: str):
        self.instance_status_changed.emit(instance_id, ScriptRunStatus.RUNNING)

    def _handle_script_complete(self, instance_id: str, return_code: int):
        is_success = return_code == 0
        status = (
            self.locale_manager.get(
                "app_controller.console_script_status_label_success"
            )
            if is_success
            else self.locale_manager.get(
                "app_controller.console_script_status_label_error"
            )
        )
        duration = (
            time.time() - self.script_start_time if self.script_start_time > 0 else 0
        )
        key = (
            "app_controller.console_script_status_success_text"
            if is_success
            else "app_controller.console_script_status_error_text"
        )
        status_text = self.locale_manager.get(
            key,
            status=status,
            return_code=return_code,
            duration=f"{duration:.2f}",
            unit=self.locale_manager.get("general.seconds_unit_short"),
        )
        new_status = ScriptRunStatus.SUCCESS if is_success else ScriptRunStatus.ERROR
        self.instance_status_changed.emit(instance_id, new_status)

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Здесь мы только устанавливаем флаг ошибки.
        # Решение об остановке будет принято в _common_script_finish_handler.
        if not is_success:
            self.run_had_errors = True
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Возвращаем вывод информационной плашки о статусе завершения скрипта.
        # Это самое подходящее место: после всех расчетов, но до принятия
        # решения о следующем шаге.
        style_type = "script_success_block" if is_success else "script_error_block"
        self.log_message.emit(style_type, status_text)
        self.log_message.emit("EMPTY_LINE", "")
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---        

        self._common_script_finish_handler(instance_id)

    def _handle_script_error(self, instance_id: str, error_message: str):
        self.log_message.emit(
            "script_error_block",
            self.locale_manager.get(
                "app_controller.console_script_critical_error_header"
            ),
        )
        self.log_message.emit(
            "script_stderr",
            self.locale_manager.get(
                "app_controller.console_script_details", error_message=error_message
            ),
        )
        self.instance_status_changed.emit(instance_id, ScriptRunStatus.ERROR)

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Аналогично, только устанавливаем флаг.
        self.run_had_errors = True
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        self._common_script_finish_handler(instance_id)

    def _common_script_finish_handler(self, instance_id: str):
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Это центральное место для принятия решений.
        # Сначала проверяем, не была ли запрошена ручная остановка.
        if self._stop_requested:
            self._finalize_run(was_stopped=True)
            return

        # Затем проверяем, не произошла ли ошибка, которую нельзя игнорировать.
        if self.run_had_errors and not self.continue_on_error:
            self.log_message.emit(
                "script_stderr",
                self.locale_manager.get("app_controller.console_log.stopping_on_error"),
            )
            self._finalize_run(was_stopped=True)
            return
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        if instance_id in self.active_runners:
            del self.active_runners[instance_id]

        # self.context_reloaded.emit()

        is_last_step = self.current_script_idx >= len(self.script_queue) - 1

        if self.run_mode in [
            SetRunMode.SEQUENTIAL_STEP,
            SetRunMode.CONDITIONAL_STEP,
        ]:
            if is_last_step:
                self._finalize_run(was_stopped=False)
            else:
                self.app_state_changed.emit(AppState.SET_RUNNING_STEP_WAIT)
        else:
            self._process_next_script()

    def _finalize_run(self, was_stopped: bool):
        if self.context_file_path.is_file():
            try:
                with open(self.context_file_path, "r+", encoding="utf-8") as f:
                    context_data = json.load(f)
                    cleaned = False
                    if "pysm_next_script" in context_data:
                        del context_data["pysm_next_script"]
                        cleaned = True
                    if "pysm_set_instance_ids" in context_data:
                        del context_data["pysm_set_instance_ids"]
                        cleaned = True
                    if cleaned:
                        f.seek(0)
                        f.truncate()
                        json.dump(context_data, f, indent=2, ensure_ascii=False)
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Не удалось очистить файл контекста: {e}")

        set_name = self.set_node.name
        success = not self.run_had_errors and not was_stopped
        status_text = (
            self.locale_manager.get(
                "app_controller.console_set_finalize_status_stopped"
            )
            if was_stopped
            else (
                self.locale_manager.get(
                    "app_controller.console_set_finalize_status_success"
                )
                if success
                else self.locale_manager.get(
                    "app_controller.console_set_finalize_status_errors"
                )
            )
        )
        self.log_message.emit("EMPTY_LINE", "")
        self.log_message.emit(
            "set_header",
            self.locale_manager.get(
                "app_controller.console_set_finalize_header", set_name=set_name
            ),
        )
        self.log_message.emit(
            "set_info",
            f"{self.locale_manager.get('app_controller.console_set_finalize_summary_label')} {status_text}",
        )
        if self.start_time > 0:
            total_duration = time.time() - self.start_time
            self.log_message.emit(
                "set_info",
                self.locale_manager.get(
                    "app_controller.console_set.total_time_format",
                    label=self.locale_manager.get(
                        "app_controller.console_set_finalize_total_time_label"
                    ),
                    duration=f"{total_duration:.2f}",
                    unit=self.locale_manager.get("general.seconds_unit_short"),
                ),
            )

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Перезагружаем финальное состояние контекста в основную
        # модель приложения ОДИН РАЗ, после завершения всего набора.
        self.context_reloaded.emit()
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        if was_stopped:
            self.run_stopped.emit(set_name)
        else:
            self.run_completed.emit(set_name, success)

        self.progress_updated.emit("", 0, 0, None)
        self.app_state_changed.emit(AppState.IDLE)

    def _log_set_start_info(self):
        mode_map = {
            SetRunMode.SEQUENTIAL_FULL: self.locale_manager.get(
                "collection_widget.run_mode_full"
            ),
            SetRunMode.SEQUENTIAL_STEP: self.locale_manager.get(
                "collection_widget.run_mode_step"
            ),
            SetRunMode.SINGLE_FROM_SET: self.locale_manager.get(
                "collection_widget.run_mode_single"
            ),
            SetRunMode.CONDITIONAL_FULL: self.locale_manager.get(
                "collection_widget.run_mode_conditional_full"
            ),
            SetRunMode.CONDITIONAL_STEP: self.locale_manager.get(
                "collection_widget.run_mode_conditional_step"
            ),
        }
        self.log_message.emit(
            "set_header",
            self.locale_manager.get(
                "app_controller.console_set_start_header", set_name=self.set_node.name
            ),
        )
        self.log_message.emit(
            "set_info",
            f"{self.locale_manager.get('app_controller.console_set_time_label')} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        mode_text = mode_map.get(
            self.run_mode, self.locale_manager.get("general.unknown")
        )
        self.log_message.emit(
            "set_info",
            f"{self.locale_manager.get('app_controller.console_set_mode_label')} {mode_text}",
        )
        self.log_message.emit(
            "set_info",
            f"{self.locale_manager.get('app_controller.console_set_queued_label')} {len(self.script_queue)}",
        )
        if self.context_file_path:
            self.log_message.emit(
                "set_info",
                f"{self.locale_manager.get('app_controller.console_set_context_file_label')} {self.context_file_path}",
            )
            """
            self.log_message.emit(
                "runner_info",
                self.locale_manager.get(
                    "app_controller.console_set_context_usage_info"
                ),
            )
            """
        self.log_message.emit("EMPTY_LINE", "")
        self.log_message.emit(
            "set_info",
            self.locale_manager.get("app_controller.console_set_queue_header"),
        )
        for i, entry in enumerate(self.script_queue):
            script_info = self.get_script_info_by_id(entry.id)
            script_name = entry.name or (
                script_info.name
                if script_info
                else self.locale_manager.get(
                    "app_controller.console_set_unknown_script"
                )
            )
            self.log_message.emit(
                "set_info",
                self.locale_manager.get(
                    "app_controller.console_set.queue_item_format",
                    current=i + 1,
                    total=len(self.script_queue),
                    name=script_name,
                    id=entry.instance_id,
                ),
            )
        self.log_message.emit("EMPTY_LINE", "")

    def _log_script_start_info(
        self,
        script_info: ScriptInfoModel,
        entry_info: ScriptSetEntryModel,
        runner: ScriptRunner,
    ):
        self.log_message.emit("EMPTY_LINE", "")
        self.log_message.emit("EMPTY_LINE", "")
        self.log_message.emit(
            "script_header_block",
            self.locale_manager.get(
                "app_controller.console_script_start_header",
                current=self.current_script_idx + 1,
                total=len(self.script_queue),
                script_name=entry_info.name or script_info.name,
            ),
        )

        if entry_info.description:
            formatted_description = entry_info.description.replace("\n", "<br>")
            self.log_message.emit("html_block", f"  {formatted_description}")

        self.log_message.emit("EMPTY_LINE", "")
        self.log_message.emit(
            "script_stdout",
            f"{self.locale_manager.get('app_controller.console_script_interpreter_label')} {runner.python_interpreter}",
        )
        self.log_message.emit(
            "script_stdout",
            f"{self.locale_manager.get('app_controller.console_script_cwd_label')} {script_info.folder_abs_path}",
        )
        self.log_message.emit("EMPTY_LINE", "")

        html_lines = []
        
        # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
        # Получаем словарь с динамическими стилями из ThemeManager, а не ConfigManager
        dynamic_styles = self.theme_manager.get_active_theme_dynamic_styles()
        info_style = dynamic_styles.get("script_info", "color: #555555;")
        arg_value_style = dynamic_styles.get("script_arg_value", "color: #000080;")
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        
        html_lines.append(f"<div style='{info_style}'>")
        args_dict = runner.custom_command_args_dict or {}
        if args_dict:
            html_lines.append(
                self.locale_manager.get("app_controller.console_script_params_label")
            )
            html_lines.append(
                "<table style='margin-left: 15px; border-collapse: collapse;'>"
            )
            for key, value in sorted(args_dict.items()):
                formatted_value = ""
                if isinstance(value, bool):
                    formatted_value = f"<i style='{arg_value_style}'>{value}</i>"
                elif isinstance(value, list):
                    items_str = "<br>".join([f"  {shlex.quote(str(v))}" for v in value])
                    formatted_value = (
                        f"<br><span style='{arg_value_style}'>{items_str}</span>"
                    )
                else:
                    quoted_value = (
                        shlex.quote(str(value)) if value is not None else "''"
                    )
                    formatted_value = (
                        f"<span style='{arg_value_style}'>{quoted_value}</span>"
                    )
                html_lines.append("<tr>")
                html_lines.append(
                    f"<td style='vertical-align: top; padding-right: 10px;'>--{key}:</td>"
                )
                html_lines.append(
                    f"<td style='vertical-align: top;'>{formatted_value}</td>"
                )
                html_lines.append("</tr>")
            html_lines.append("</table>")
        html_lines.append("</div>")
        self.log_message.emit("html_block", "".join(html_lines))
        self.log_message.emit("EMPTY_LINE", "")