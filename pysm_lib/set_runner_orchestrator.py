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
from .theme_manager import ThemeManager

from .app_enums import SetRunMode, ScriptRunStatus, AppState

logger = logging.getLogger(f"PyScriptManager.{__name__}")

# Максимальная глубина вложенности вызовов (защита от бесконечных циклов A -> B -> A)
MAX_STACK_DEPTH = 20

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
    context_ipc_update_received = Signal(dict)

    def __init__(
        self,
        set_node: ScriptSetNodeModel,
        run_mode: str,
        continue_on_error: bool,
        get_script_info_func: Callable[[str], Optional[ScriptInfoModel]],
        config_manager: ConfigManager,
        theme_manager: ThemeManager,
        locale_manager: LocaleManager,
        context_file_path: pathlib.Path,
        selected_instance_id: Optional[str] = None,
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Передаем функцию для глобального поиска в SetManager
        get_set_manager_func: Optional[Callable] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.set_node = set_node
        self.run_mode = run_mode
        self.continue_on_error = continue_on_error
        self.get_script_info_by_id = get_script_info_func
        self.config_manager = config_manager
        self.theme_manager = theme_manager
        self.locale_manager = locale_manager
        self.context_file_path = context_file_path
        self.selected_instance_id = selected_instance_id
        
        self.get_set_manager_func = get_set_manager_func

        # Стек вызовов вместо линейной очереди
        # Формат кадра: {"set_node": ScriptSetNodeModel, "queue": List[ScriptSetEntryModel], "idx": int, "run_mode": str}
        self.call_stack: List[Dict] = []
        
        # Хранит целевой ID, если скрипт запросил прыжок
        self._pending_jump_target: Optional[str] = None

        self.active_runners: Dict[str, ScriptRunner] = {}
        self.run_had_errors: bool = False
        self.start_time: float = 0
        self.script_start_time: float = 0
        self._stop_requested: bool = False

    def _prepare_context_file(self):
        """Сбрасывает актуальное состояние RAM (пользовательские переменные + системные) на жесткий диск."""
        try:
            context_data = {}
            
            # Берем самые свежие данные пользователя прямо из памяти PySM! (Без чтения с диска)
            if self.get_set_manager_func:
                sm = self.get_set_manager_func()
                if sm:
                    for k, v in sm.current_collection_model.context_data.items():
                        context_data[k] = v.model_dump(mode="json")

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

            sys_info_data = {
                "app_root_dir": str(APPLICATION_ROOT_DIR),
                "collection_file_path": str(self.config_manager.last_used_sets_collection_file),
                "active_theme_name": active_theme_name,
                "log_level": current_log_level,
                "python_interpreter": str(self.config_manager.python_interpreter),
            }

            context_data["pysm_sys_info"] = {
                "type": "json",
                "value": sys_info_data,
                "description": "System: Information about the execution environment.",
                "read_only": True,
            }

            # Молча перезаписываем файл на диске
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

        # Формируем первоначальный список
        initial_queue =[]
        if self.run_mode == SetRunMode.SINGLE_FROM_SET:
            if not self.selected_instance_id: return
            entry = next((e for e in self.set_node.script_entries if e.instance_id == self.selected_instance_id), None)
            if not entry: return
            initial_queue = [entry]
        else:
            initial_queue =[e for e in self.set_node.script_entries if self.get_script_info_by_id(e.id) and self.get_script_info_by_id(e.id).passport_valid]

        if not initial_queue: return

        # Инициализация Стека. Кадр хранит свой собственный run_mode
        self.call_stack =[{
            "set_node": self.set_node,
            "queue": initial_queue,
            "idx": -1,
            "run_mode": self.run_mode
        }]

        # Очищаем статусы в UI (только для начального набора, для остальных это будет делаться по ходу)
        for entry in self.set_node.script_entries: 
            self.instance_status_changed.emit(entry.instance_id, None)
        for entry in initial_queue: 
            self.instance_status_changed.emit(entry.instance_id, ScriptRunStatus.PENDING)

        self.start_time = time.time()
        
        # Передаем длину стартовой очереди для логов
        self._log_set_start_info(len(initial_queue))
        
        self.run_had_errors = False
        self._pending_jump_target = None
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

    # Новый метод обработки IPC-сигнала перехода ---
    def _handle_script_routing(self, instance_id: str, target_id: str):
        """Коллбэк, вызываемый ScriptRunner при перехвате команды PYSM_ROUTING_CMD."""
        logger.info(f"Получена команда маршрутизации: {instance_id} -> {target_id}")
        self._pending_jump_target = target_id

    # Добавляем новый метод-перехватчик
    def _handle_context_update(self, instance_id: str, update_data: dict):
        """Пробрасывает данные обновления контекста в AppController."""
        self.context_ipc_update_received.emit(update_data)

    def _determine_next_script(self) -> Optional[Dict]:
        """
        Главный движок маршрутизации.
        Поддерживает гибридную логику: GOTO (для локальных переходов) и GOSUB (для макросов).
        """
        if not self.call_stack:
            return None

        # УБРАНА ПРОВЕРКА на глобальный run_mode != SINGLE_FROM_SET.
        # Теперь переходы работают всегда и везде.
        if self._pending_jump_target:
            target_ids_raw = self._pending_jump_target
            self._pending_jump_target = None 
            
            # Парсим все запрошенные ID (через запятую)
            target_ids =[t.strip() for t in target_ids_raw.split(",") if t.strip()]
            
            if target_ids:
                current_frame = self.call_stack[-1]
                first_target = target_ids[0]
                
                # Ищем во всех скриптах узла (даже если мы в Single-режиме, узел помнит все свои скрипты)
                is_local = any(e.instance_id == first_target for e in current_frame["set_node"].script_entries)
                
                if len(target_ids) == 1 and is_local:
                    # ---> СИТУАЦИЯ 2: ЛОКАЛЬНЫЙ ПЕРЕХОД (GOTO) <---
                    
                    if current_frame["run_mode"] == SetRunMode.SINGLE_FROM_SET:
                        # УМНЫЙ GOTO для одиночного режима: подменяем очередь на лету
                        target_entry = next((e for e in current_frame["set_node"].script_entries if e.instance_id == first_target), None)
                        info = self.get_script_info_by_id(target_entry.id) if target_entry else None
                        
                        if target_entry and info and info.passport_valid:
                            current_frame["queue"] = [target_entry]
                            current_frame["idx"] = 0
                            script_name = target_entry.name or info.name
                            self.log_message.emit("runner_info", f"➦ Локальный переход (GOTO) к скрипту '{script_name}'")
                            self.instance_status_changed.emit(first_target, ScriptRunStatus.PENDING)
                            return current_frame
                        else:
                            self.log_message.emit("script_error_block", "ОШИБКА: Целевой скрипт для GOTO не найден или не валиден.")
                            
                    else:
                        # В Авто/Пошаговом режиме просто смещаем указатель
                        local_idx = -1
                        for i, entry in enumerate(current_frame["queue"]):
                            if entry.instance_id == first_target:
                                local_idx = i
                                break
                        
                        if local_idx != -1:
                            script_name = current_frame["queue"][local_idx].name or first_target
                            self.log_message.emit("runner_info", f"➦ Локальный переход (GOTO) к скрипту '{script_name}'")
                            self.instance_status_changed.emit(first_target, ScriptRunStatus.PENDING)
                            current_frame["idx"] = local_idx
                            return current_frame
                        else:
                            self.log_message.emit("script_error_block", "ОШИБКА: Целевой скрипт для GOTO не валиден.")
                
                # ---> СИТУАЦИЯ 1: ВНЕШНИЙ ПЕРЕХОД (GOSUB / Динамический макрос) <---
                new_queue =[]
                if self.get_set_manager_func:
                    set_manager = self.get_set_manager_func()
                    if set_manager:
                        for t_id in target_ids:
                            result = set_manager.find_entry_and_parent_set(t_id)
                            if result:
                                entry, parent_set = result
                                info = self.get_script_info_by_id(entry.id)
                                if info and info.passport_valid:
                                    new_queue.append(entry)
                
                if new_queue:
                    if len(self.call_stack) >= MAX_STACK_DEPTH:
                        self.log_message.emit("script_error_block", f"ОШИБКА: Превышена глубина вызовов ({MAX_STACK_DEPTH}). Бесконечный цикл?")
                        self.run_had_errors = True
                        return None

                    names =[e.name or self.get_script_info_by_id(e.id).name for e in new_queue]
                    
                    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
                    # Формируем читаемый многострочный нумерованный список
                    macro_msg_lines =[
                        "↪ Динамический макрос (GOSUB).",
                        "В очередь выполнения добавлены следующие скрипты:"
                    ]
                    for i, name in enumerate(names, 1):
                        macro_msg_lines.append(f"{i}. {name}")
                        
                    self.log_message.emit("runner_info", "\n".join(macro_msg_lines))
                    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
                    
                    for entry in new_queue:
                        self.instance_status_changed.emit(entry.instance_id, ScriptRunStatus.PENDING)
                        
                    virtual_set = ScriptSetNodeModel(name="Динамический Макрос", script_entries=new_queue)
                    new_frame = {
                        "set_node": virtual_set,
                        "queue": new_queue,
                        "idx": 0,
                        # МАКРОСЫ ВСЕГДА ВЫПОЛНЯЮТСЯ В АВТОМАТИЧЕСКОМ РЕЖИМЕ!
                        "run_mode": SetRunMode.CONDITIONAL_FULL
                    }
                    self.call_stack.append(new_frame)
                    return new_frame
                else:
                    self.log_message.emit("script_error_block", "ОШИБКА МАРШРУТИЗАЦИИ: Целевые скрипты не найдены.")

        # 2. Обычное линейное выполнение (или возврат)
        current_frame = self.call_stack[-1]
        
        # Сдвигаем указатель к следующему скрипту
        current_frame["idx"] += 1

        # Если скрипты в текущем кадре закончились
        if current_frame["idx"] >= len(current_frame["queue"]):
            # Удаляем завершенный кадр (выход из макроса или набора)
            finished_frame = self.call_stack.pop()
            
            # Если стек пуст - все скрипты выполнены
            if not self.call_stack:
                return None
                
            # Иначе это возврат к предыдущему кадру
            self.log_message.emit("runner_info", f"↩ Возврат (RETURN) из '{finished_frame['set_node'].name}' в '{self.call_stack[-1]['set_node'].name}'")
            
            # Рекурсивно вызываем метод, чтобы он сделал +1 в родительском кадре
            return self._determine_next_script()

        # Возвращаем кадр для выполнения
        return current_frame

    def _process_next_script(self):
        if self._stop_requested:
            self._finalize_run(was_stopped=True)
            return

        self.app_state_changed.emit(AppState.SET_RUNNING_AUTO)

        # Вычисляем следующий скрипт, используя Стек
        next_frame = self._determine_next_script()
        
        # Если Стек пуст - выполнение завершено
        if not next_frame:
            self._finalize_run(was_stopped=False)
            return

        # Извлекаем скрипт для запуска
        entry_to_run = next_frame["queue"][next_frame["idx"]]
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

        # --- СИНХРОНИЗАЦИЯ ПЕРЕД КАЖДЫМ СКРИПТОМ ---
        # Обновляем файл на диске из памяти, чтобы новый Python процесс получил свежие данные
        try:
            self._prepare_context_file()
        except Exception:
            self.log_message.emit("script_error_block", "КРИТИЧЕСКАЯ ОШИБКА: Не удалось обновить файл контекста перед запуском.")
            self._finalize_run(was_stopped=True)
            return

        # Создаем раннер и передаем ему коллбэк для маршрутизации
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
            on_routing=self._handle_script_routing,
            on_context_update=self._handle_context_update,
            custom_command_args_dict=args_for_run,
            context_file_path=str(self.context_file_path),
            app_root_dir=APPLICATION_ROOT_DIR,
        )
        
        self.active_runners[entry_to_run.instance_id] = runner
        
        if not entry_to_run.silent_mode:
            self._log_script_start_info(script_info, entry_to_run, runner, next_frame)
        else:    
            self._log_script_start_info_silent(entry_to_run)
            
        runner.run(entry_to_run.instance_id)

    def _handle_script_output(self, instance_id: str, stream: str, line: str):
        self.log_message.emit(f"script_{stream}", line)

    def _handle_script_start(self, instance_id: str):
        self.instance_status_changed.emit(instance_id, ScriptRunStatus.RUNNING)

    def _handle_script_complete(self, instance_id: str, return_code: int):
        is_success = return_code == 0
        status = (
            self.locale_manager.get("app_controller.console_script_status_label_success")
            if is_success
            else self.locale_manager.get("app_controller.console_script_status_label_error")
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
        
        # Обновляем статусы
        self.instance_status_changed.emit(instance_id, new_status)
        self._on_orchestrator_instance_status_changed(instance_id, new_status)

        if not is_success:
            self.run_had_errors = True

        # Определяем, находимся ли мы в сайлент-режиме
        # Ищем во всем стеке вызовов (сверху вниз), так как скрипт может быть где угодно
        is_silent = False
        for frame in reversed(self.call_stack):
            entry = next((e for e in frame["queue"] if e.instance_id == instance_id), None)
            if entry:
                is_silent = entry.silent_mode
                break

        if not is_silent or not is_success:
            style_type = "script_success_block" if is_success else "script_error_block"
            self.log_message.emit(style_type, status_text)
            self.log_message.emit("EMPTY_LINE", "")

        self._common_script_finish_handler(instance_id)

    @Slot(str, object)
    def _on_orchestrator_instance_status_changed(self, instance_id: str, status: Optional[ScriptRunStatus]):
        """Кэширует статус для UI через AppController"""
        if self.get_set_manager_func and hasattr(self.parent(), "script_run_statuses"):
            app_controller = self.parent()
            if status is None:
                app_controller.script_run_statuses.pop(instance_id, None)
            else:
                app_controller.script_run_statuses[instance_id] = status

    def _handle_script_error(self, instance_id: str, error_message: str):
        self.log_message.emit(
            "script_error_block",
            self.locale_manager.get("app_controller.console_script_critical_error_header"),
        )
        self.log_message.emit(
            "script_stderr",
            self.locale_manager.get(
                "app_controller.console_script_details", error_message=error_message
            ),
        )
        self.instance_status_changed.emit(instance_id, ScriptRunStatus.ERROR)
        self._on_orchestrator_instance_status_changed(instance_id, ScriptRunStatus.ERROR)
        
        self.run_had_errors = True
        self._common_script_finish_handler(instance_id)

    def _common_script_finish_handler(self, instance_id: str):
        if self._stop_requested:
            self._finalize_run(was_stopped=True)
            return

        if self.run_had_errors and not self.continue_on_error:
            self.log_message.emit(
                "script_stderr",
                self.locale_manager.get("app_controller.console_log.stopping_on_error"),
            )
            self._finalize_run(was_stopped=True)
            return

        if instance_id in self.active_runners:
            del self.active_runners[instance_id]

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Испускаем сигнал, чтобы обновить контекст в UI после КАЖДОГО шага
        self.context_reloaded.emit()

        # Проверяем режим выполнения ТЕКУЩЕГО кадра в стеке!
        current_frame = self.call_stack[-1] if self.call_stack else None
        current_run_mode = current_frame["run_mode"] if current_frame else self.run_mode

        if current_run_mode == SetRunMode.CONDITIONAL_STEP:
            self.app_state_changed.emit(AppState.SET_RUNNING_STEP_WAIT)
        else:
            self._process_next_script()

    def _finalize_run(self, was_stopped: bool):
        # Удаляем устаревшие поля, если они вдруг есть
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
            self.locale_manager.get("app_controller.console_set_finalize_status_stopped")
            if was_stopped
            else (
                self.locale_manager.get("app_controller.console_set_finalize_status_success")
                if success
                else self.locale_manager.get("app_controller.console_set_finalize_status_errors")
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
                    label=self.locale_manager.get("app_controller.console_set_finalize_total_time_label"),
                    duration=f"{total_duration:.2f}",
                    unit=self.locale_manager.get("general.seconds_unit_short"),
                ),
            )

        self.context_reloaded.emit()

        if was_stopped:
            self.run_stopped.emit(set_name)
        else:
            self.run_completed.emit(set_name, success)

        self.progress_updated.emit("", 0, 0, None)
        self.app_state_changed.emit(AppState.IDLE)
        
        # Очищаем стек вызовов
        self.call_stack.clear()

    def _log_set_start_info(self, queue_len: int):
        mode_map = {
            SetRunMode.SINGLE_FROM_SET: self.locale_manager.get("collection_widget.run_mode_single"),
            SetRunMode.CONDITIONAL_FULL: self.locale_manager.get("collection_widget.run_mode_conditional_full"),
            SetRunMode.CONDITIONAL_STEP: self.locale_manager.get("collection_widget.run_mode_conditional_step"),
        }
        self.log_message.emit(
            "set_header",
            self.locale_manager.get("app_controller.console_set_start_header", set_name=self.set_node.name),
        )
        self.log_message.emit(
            "set_info",
            f"{self.locale_manager.get('app_controller.console_set_time_label')} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        mode_text = mode_map.get(self.run_mode, self.locale_manager.get("general.unknown"))
        self.log_message.emit(
            "set_info",
            f"{self.locale_manager.get('app_controller.console_set_mode_label')} {mode_text}",
        )
        self.log_message.emit(
            "set_info",
            f"{self.locale_manager.get('app_controller.console_set_queued_label')} {queue_len}",
        )
        if self.context_file_path:
            self.log_message.emit(
                "set_info",
                f"{self.locale_manager.get('app_controller.console_set_context_file_label')} {self.context_file_path}",
            )
        self.log_message.emit("EMPTY_LINE", "")


    def _log_script_start_info_silent(self, entry_info: ScriptSetEntryModel):
        self.log_message.emit("EMPTY_LINE", " ")
        if entry_info.description:
            formatted_description = entry_info.description.replace("\n", "<br>")
            self.log_message.emit("html_block", f"{formatted_description}")
            self.log_message.emit("EMPTY_LINE", "")

    def _log_script_start_info(
        self,
        script_info: ScriptInfoModel,
        entry_info: ScriptSetEntryModel,
        runner: ScriptRunner,
        current_frame: Dict,
    ):
        self.log_message.emit("EMPTY_LINE", "")
        self.log_message.emit("EMPTY_LINE", "")
        
        # Вычисляем глубину стека для индикации в консоли (отступы)
        indent = "  " * (len(self.call_stack) - 1)
        set_name = current_frame["set_node"].name
        current_idx = current_frame["idx"] + 1
        total_in_set = len(current_frame["queue"])
        
        header_text = self.locale_manager.get(
            "app_controller.console_script_start_header",
            current=current_idx,
            total=total_in_set,
            script_name=entry_info.name or script_info.name,
        )
        
        # Добавляем информацию о наборе, если мы "провалились" в другой набор
        if len(self.call_stack) > 1:
             header_text = f"{indent}[Вложенный набор '{set_name}'] {header_text}"
        
        self.log_message.emit("script_header_block", header_text)

        if entry_info.description:
            formatted_description = entry_info.description.replace("\n", "<br>")
            self.log_message.emit("html_block", f"{indent}  {formatted_description}")

        self.log_message.emit("EMPTY_LINE", "")
        self.log_message.emit(
            "script_stdout",
            f"{indent}{self.locale_manager.get('app_controller.console_script_interpreter_label')} {runner.python_interpreter}",
        )
        self.log_message.emit(
            "script_stdout",
            f"{indent}{self.locale_manager.get('app_controller.console_script_cwd_label')} {script_info.folder_abs_path}",
        )
        self.log_message.emit("EMPTY_LINE", "")

        html_lines =[]
        dynamic_styles = self.theme_manager.get_active_theme_dynamic_styles()
        info_style = dynamic_styles.get("script_info", "color: #555555;")
        arg_value_style = dynamic_styles.get("script_arg_value", "color: #000080;")
        
        # Добавляем отступ для таблицы
        html_lines.append(f"<div style='{info_style}; margin-left: {len(self.call_stack)*10}px;'>")
        args_dict = runner.custom_command_args_dict or {}
        if args_dict:
            html_lines.append(self.locale_manager.get("app_controller.console_script_params_label"))
            html_lines.append("<table style='margin-left: 15px; border-collapse: collapse;'>")
            for key, value in sorted(args_dict.items()):
                formatted_value = ""
                if isinstance(value, bool):
                    formatted_value = f"<i style='{arg_value_style}'>{value}</i>"
                elif isinstance(value, list):
                    items_str = "<br>".join([f"  {shlex.quote(str(v))}" for v in value])
                    formatted_value = f"<br><span style='{arg_value_style}'>{items_str}</span>"
                else:
                    quoted_value = shlex.quote(str(value)) if value is not None else "''"
                    formatted_value = f"<span style='{arg_value_style}'>{quoted_value}</span>"
                html_lines.append("<tr>")
                html_lines.append(f"<td style='vertical-align: top; padding-right: 10px;'>--{key}:</td>")
                html_lines.append(f"<td style='vertical-align: top;'>{formatted_value}</td>")
                html_lines.append("</tr>")
            html_lines.append("</table>")
        html_lines.append("</div>")
        self.log_message.emit("html_block", "".join(html_lines))
        self.log_message.emit("EMPTY_LINE", "")