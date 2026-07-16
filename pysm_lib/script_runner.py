# pysm_lib/script_runner.py

import os
import pathlib
import json
import logging
from typing import (
    Callable,
    List,
    Optional,
    Dict,
    Any,
)

# --- НОВЫЕ ИМПОРТЫ ДЛЯ QPROCESS ---
from PySide6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QTimer,
    Slot,
    QByteArray
)

from .models import ScriptInfoModel
from .locale_manager import LocaleManager

locale_manager = LocaleManager()
logger = logging.getLogger(f"PyScriptManager.{__name__}")

CONSOLE_BLOCK_START_PHRASE = "PYSM_CONSOLE_BLOCK_START"
CONSOLE_BLOCK_END_PHRASE = "PYSM_CONSOLE_BLOCK_END"


class ScriptRunner(QObject):
    """
    Класс для асинхронного запуска и управления пользовательскими Python-скриптами.
    Использует нативный QProcess для безопасной интеграции с Event Loop приложения,
    избавляя от необходимости создавать дополнительные потоки (threading).
    """
    def __init__(
        self,
        script_info: ScriptInfoModel,
        python_interpreter: str,
        additional_env_paths: Optional[List[str]] = None,
        on_start: Optional[Callable[[str], None]] = None,
        on_output: Optional[Callable[[str, str, str], None]] = None,
        on_complete: Optional[Callable[[str, int], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
        on_progress: Optional[Callable[[str, int, int, Optional[str]], None]] = None,
        on_routing: Optional[Callable[[str, str], None]] = None, 
        on_context_update: Optional[Callable[[str, dict], None]] = None,
        custom_command_args_dict: Optional[Dict[str, Any]] = None,
        context_file_path: Optional[str] = None,
        context_shm_name: Optional[str] = None,
        context_mode: str = "file",
        app_root_dir: Optional[pathlib.Path] = None,
        global_python_paths: Optional[List[str]] = None,
        global_env_vars: Optional[Dict[str, str]] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.script_info_model: ScriptInfoModel = script_info
        self.python_interpreter: str = python_interpreter
        self.additional_env_paths: List[str] = additional_env_paths or[]
        self.context_file_path = context_file_path
        self.context_shm_name = context_shm_name
        self.context_mode = context_mode
        self.app_root_dir = app_root_dir
        self.global_python_paths = global_python_paths or[]
        self.global_env_vars = global_env_vars or {}
        
        # Коллбэки
        self.on_start = on_start
        self.on_output = on_output
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_progress = on_progress
        self.on_routing = on_routing
        self.on_context_update = on_context_update
        self.custom_command_args_dict = custom_command_args_dict
        
        # Внутреннее состояние
        self.process: Optional[QProcess] = None
        self.effective_id: str = ""
        self._console_output_blocked: bool = False
        self._stop_requested: bool = False


    def _prepare_environment(self) -> QProcessEnvironment:
        script_id_for_log = self.script_info_model.id
        logger.debug(
            locale_manager.get("script_runner.log_debug.preparing_env", id=script_id_for_log)
        )
        
        # Создаем окружение на базе системного
        env = QProcessEnvironment.systemEnvironment()
        
        if self.global_env_vars:
            logger.debug(
                locale_manager.get("script_runner.log_debug.applying_global_vars", vars=self.global_env_vars)
            )
            for key, val in self.global_env_vars.items():
                env.insert(key, val)
                
        env.insert("PY_SCRIPT_MANAGER_ACTIVE", "1")
        env.insert("PYSM_CONTEXT_MODE", self.context_mode)
        if self.context_file_path:
            env.insert("PYSM_CONTEXT_FILE", self.context_file_path)
        if self.context_shm_name:
            env.insert("PYSM_CONTEXT_SHM_NAME", self.context_shm_name)
        logger.debug(locale_manager.get("script_runner.log_debug.pysm_active_set"))
        
        new_path_parts: List[str] =[]

        interpreter_path_obj = pathlib.Path(self.python_interpreter)
        if interpreter_path_obj.is_file():
            interpreter_dir = str(interpreter_path_obj.parent.resolve())
            if interpreter_dir not in new_path_parts:
                new_path_parts.append(interpreter_dir)

        # Разрешение относительных путей для скрипта
        script_folder_path = pathlib.Path(self.script_info_model.folder_abs_path)
        script_specific_paths = self.script_info_model.script_specific_env_paths
        if script_specific_paths:
            for p_str in script_specific_paths:
                if not p_str:
                    continue
                path_obj = pathlib.Path(p_str)
                abs_path_obj = (
                    (script_folder_path / path_obj).resolve()
                    if not path_obj.is_absolute()
                    else path_obj.resolve(strict=False)
                )
                abs_path_str = str(abs_path_obj)
                if abs_path_str not in new_path_parts:
                    new_path_parts.append(abs_path_str)

        if self.additional_env_paths:
            for p_str in self.additional_env_paths:
                path_obj_str = str(pathlib.Path(p_str).resolve(strict=False))
                if path_obj_str not in new_path_parts:
                    new_path_parts.append(path_obj_str)

        original_path = env.value("PATH", "")
        if original_path:
            new_path_parts.append(original_path)
            
        unique_path_parts = list(dict.fromkeys(filter(None, new_path_parts)))
        env.insert("PATH", os.pathsep.join(unique_path_parts))
        
        python_path_parts =[]
        if self.app_root_dir and self.app_root_dir.is_dir():
            python_path_parts.append(str(self.app_root_dir))
        if self.global_python_paths:
            python_path_parts.extend(self.global_python_paths)
            
        existing_pythonpath = env.value("PYTHONPATH", "")
        if existing_pythonpath:
            python_path_parts.extend(existing_pythonpath.split(os.pathsep))
            
        if python_path_parts:
            unique_python_paths = list(dict.fromkeys(filter(None, python_path_parts)))
            env.insert("PYTHONPATH", os.pathsep.join(unique_python_paths))
            logger.debug(
                locale_manager.get("script_runner.log_debug.pythonpath_set", path=env.value("PYTHONPATH"))
            )
            
        env.insert("PYTHONIOENCODING", "utf-8")
        logger.debug(
            locale_manager.get("script_runner.log_debug.final_path", id=script_id_for_log, path=env.value("PATH"))
        )
        return env


    def _format_args_from_dict(self, args_dict: Optional[Dict[str, Any]]) -> List[str]:
        if not args_dict:
            return []

        formatted_args: List[str] =[]
        for key, value in args_dict.items():
            arg_name = key.lstrip("-")
            formatted_arg_name = f"--{arg_name}"

            if isinstance(value, bool):
                if value:
                    formatted_args.append(formatted_arg_name)
            elif isinstance(value, (list, tuple)):
                formatted_args.append(formatted_arg_name)
                for v_item in value:
                    formatted_args.append(str(v_item))
            else:
                formatted_args.append(formatted_arg_name)
                value_to_add = str(value) if value is not None else ""
                formatted_args.append(value_to_add)

        return formatted_args


    def build_command_list(self) -> List[str]:
        run_file_path_str = self.script_info_model.run_file_abs_path
        command: List[str] =[
            self.python_interpreter,
            "-u",
            "-m",
            "pysm_lib.context_loader",
        ]
        command.append(run_file_path_str)
        if self.context_shm_name:
            command.extend(["--pysm-context-shm-name", self.context_shm_name])
            command.extend(["--pysm-context-mode", self.context_mode])
        if self.context_file_path:
            command.extend(["--pysm-context-file", self.context_file_path])
            
        additional_args_list: List[str] =[]
        if isinstance(self.custom_command_args_dict, dict):
            additional_args_list = self._format_args_from_dict(
                self.custom_command_args_dict
            )
        command.extend(additional_args_list)
        return command


    def run(self, effective_id_for_callbacks: str):
        self.effective_id = effective_id_for_callbacks
        script_name_for_log = self.script_info_model.name
        
        logger.info(
            locale_manager.get(
                "script_runner.log_info.starting_script",
                name=script_name_for_log,
                id=self.effective_id,
            )
        )

        run_file_abs_path_str = self.script_info_model.run_file_abs_path
        if not run_file_abs_path_str:
            if self.on_error:
                self.on_error(self.effective_id, locale_manager.get("script_runner.error.run_file_path_missing"))
            return

        run_file_abs_path = pathlib.Path(run_file_abs_path_str)
        if not run_file_abs_path.is_file():
            if self.on_error:
                self.on_error(
                    self.effective_id,
                    locale_manager.get("script_runner.error.run_file_not_found", path=run_file_abs_path)
                )
            return

        script_folder_abs_path = pathlib.Path(self.script_info_model.folder_abs_path)
        command = self.build_command_list()

        try:
            self._stop_requested = False
            self._console_output_blocked = False
            
            self.process = QProcess(self)
            self.process.setProcessEnvironment(self._prepare_environment())
            self.process.setWorkingDirectory(str(script_folder_abs_path))
            
            # Подключаем сигналы QProcess
            self.process.started.connect(self._on_process_started)
            self.process.finished.connect(self._on_process_finished)
            self.process.errorOccurred.connect(self._on_process_error)
            self.process.readyReadStandardOutput.connect(self._on_stdout_ready)
            self.process.readyReadStandardError.connect(self._on_stderr_ready)

            # Запуск (команда и список аргументов)
            self.process.start(command[0], command[1:])
            
        except Exception as e:
            logger.error(
                locale_manager.get(
                    "script_runner.log_error.generic_run_error",
                    name=script_name_for_log,
                    error=e,
                ),
                exc_info=True,
            )
            if self.on_error:
                self.on_error(
                    self.effective_id,
                    locale_manager.get(
                        "script_runner.error.generic_run_runtime_error",
                        name=script_name_for_log,
                        error=e,
                    ),
                )
            if self.process:
                self.process.deleteLater()
                self.process = None


    @Slot()
    def _on_process_started(self):
        if self.on_start:
            self.on_start(self.effective_id)

    @Slot(int, QProcess.ExitStatus)
    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        # --- НАЧАЛО ИЗМЕНЕНИЙ: Финальная вычитка буферов ---
        # Вычитываем всё, что могло остаться в буфере без символа переноса строки (\n)
        if self.process:
            self.process.setReadChannel(QProcess.ProcessChannel.StandardOutput)
            while self.process.canReadLine() or self.process.bytesAvailable() > 0:
                line_bytes: QByteArray = self.process.readLine()
                if line_bytes:
                    line_str = bytes(line_bytes).decode('utf-8', errors='replace').rstrip('\r\n')
                    self._process_line("stdout", line_str)
                    
            self.process.setReadChannel(QProcess.ProcessChannel.StandardError)
            while self.process.canReadLine() or self.process.bytesAvailable() > 0:
                line_bytes: QByteArray = self.process.readLine()
                if line_bytes:
                    line_str = bytes(line_bytes).decode('utf-8', errors='replace').rstrip('\r\n')
                    self._process_line("stderr", line_str)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        logger.info(
            locale_manager.get(
                "script_runner.log_info.process_finished",
                id=self.effective_id,
                code=exit_code,
            )
        )
        
        # Если процесс упал (CrashExit), генерируем код ошибки, отличный от 0
        final_exit_code = exit_code
        if exit_status == QProcess.ExitStatus.CrashExit:
            final_exit_code = -1 
            
        if self.on_complete:
            self.on_complete(self.effective_id, final_exit_code)
            
        if self.process:
            self.process.deleteLater()
            self.process = None

    @Slot(QProcess.ProcessError)
    def _on_process_error(self, error: QProcess.ProcessError):
        # Ошибка FailedToStart возникает ДО того, как процесс запущен (нет файла, нет прав)
        # В этом случае finished() не сработает, поэтому мы должны выбросить on_error сами
        if error == QProcess.ProcessError.FailedToStart:
            msg = locale_manager.get(
                "script_runner.error.generic_run_runtime_error",
                name=self.script_info_model.name,
                error="QProcess::FailedToStart (Убедитесь, что интерпретатор существует и доступен)"
            )
            logger.error(msg)
            if self.on_error:
                self.on_error(self.effective_id, msg)
            
            # Очищаем память, процесс мертв
            self.process.deleteLater()
            self.process = None
        else:
            # Для остальных ошибок (Crashed, ReadError, WriteError) 
            # сигнал finished() всё равно будет вызван, поэтому просто логируем
            logger.error(f"QProcess Error occurred for {self.effective_id}: {error.name}")


    # =========================================================================
    # БЛОК ЧТЕНИЯ ПОТОКОВ ВВОДА-ВЫВОДА
    # =========================================================================

    @Slot()
    def _on_stdout_ready(self):
        """Слот срабатывает, когда в буфере стандартного вывода есть новые данные."""
        if not self.process: return
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Явно переключаем канал чтения на StdOut
        self.process.setReadChannel(QProcess.ProcessChannel.StandardOutput)
        
        while self.process.canReadLine():
            line_bytes: QByteArray = self.process.readLine()
            # Декодируем и убираем переносы строки
            line_str = bytes(line_bytes).decode('utf-8', errors='replace').rstrip('\r\n')
            self._process_line("stdout", line_str)

    @Slot()
    def _on_stderr_ready(self):
        """Слот срабатывает, когда в буфере стандартных ошибок есть новые данные."""
        if not self.process: return
        
        # Переключаем канал чтения на StdErr
        self.process.setReadChannel(QProcess.ProcessChannel.StandardError)
        
        while self.process.canReadLine():
            line_bytes: QByteArray = self.process.readLine()
            line_str = bytes(line_bytes).decode('utf-8', errors='replace').rstrip('\r\n')
            self._process_line("stderr", line_str)

    def _process_line(self, stream_name: str, line_str: str):
        """Централизованный метод обработки строк из любого потока."""
        if self._stop_requested:
            return

        # 1. Проверяем команды блокировки
        if line_str == CONSOLE_BLOCK_START_PHRASE:
            if not self._console_output_blocked:
                self._console_output_blocked = True
                if self.on_output:
                    self.on_output(self.effective_id, "runner_info", locale_manager.get("script_runner.console_message.output_blocked"))
            return

        if line_str == CONSOLE_BLOCK_END_PHRASE:
            if self._console_output_blocked:
                self._console_output_blocked = False
                if self.on_output:
                    self.on_output(self.effective_id, "runner_info", locale_manager.get("script_runner.console_message.output_unblocked"))
            return

        # 2. Проверяем "богатый" контент в потоке stderr
        if stream_name == 'stderr':
            # Проверка на HTML-блок
            if line_str.startswith("PYSM_HTML_BLOCK:"):
                html_content = line_str.split(":", 1)[1]
                if self.on_output:
                    self.on_output(self.effective_id, "html_block", html_content)
                return


            # Проверка на команду маршрутизации (Jump)
            if line_str.startswith("PYSM_ROUTING_CMD:"):
                json_str = line_str.split(":", 1)[1]
                try:
                    routing_data = json.loads(json_str)
                    target_id = routing_data.get("target_id")
                    if target_id and self.on_routing:
                        self.on_routing(self.effective_id, target_id)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing routing command from {self.effective_id}: {e}")
                return
            # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

            # Проверка на команду обновления контекста (IPC)
            if line_str.startswith("PYSM_CONTEXT_UPDATE:"):
                json_str = line_str.split(":", 1)[1]
                try:
                    update_data = json.loads(json_str)
                    if self.on_context_update:
                        self.on_context_update(self.effective_id, update_data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing context update command: {e}")
                return

            # Проверка на JSON-прогресс
            if line_str.startswith("{") and "progress" in line_str:
                try:
                    progress_data = json.loads(line_str)
                    if isinstance(progress_data, dict) and progress_data.get("type") == "progress":
                        current = progress_data.get("current", 0)
                        total = progress_data.get("total", 0)
                        text = progress_data.get("text")
                        if self.on_progress:
                            self.on_progress(self.effective_id, int(current), int(total), text)
                        return
                except (json.JSONDecodeError, ValueError):
                    pass # Это не JSON-прогресс, а обычная строка ошибки

        # 3. Выводим обычный текст, если вывод не заблокирован
        if self._console_output_blocked and stream_name != "runner_info":
            return
        
        if self.on_output:
            self.on_output(self.effective_id, stream_name, line_str)


    # =========================================================================
    # БЛОК ОСТАНОВКИ ПРОЦЕССА
    # =========================================================================

    def stop(self, script_id_cb: str):
        """Плавная остановка процесса с последующим жестким убиением (Kill) через таймаут."""
        logger.info(
            locale_manager.get("script_runner.log_info.stop_requested", id=script_id_cb)
        )
        self._stop_requested = True

        if self.is_running() and self.process:
            pid = self.process.processId()
            logger.info(
                locale_manager.get(
                    "script_runner.log_info.terminating_process",
                    pid=pid,
                    id=script_id_cb,
                )
            )
            # Запрашиваем мягкую остановку (SIGTERM)
            self.process.terminate()
            
            # Заводим таймер на 1 секунду. Если процесс завис, мы его убьем (SIGKILL)
            QTimer.singleShot(1000, self._force_kill_if_running)


    @Slot()
    def _force_kill_if_running(self):
        """Жесткое завершение процесса, если он проигнорировал terminate()."""
        if self.is_running() and self.process:
            logger.warning(
                locale_manager.get(
                    "script_runner.log_warning.terminate_failed_killing",
                    pid=self.process.processId(),
                    id=self.effective_id,
                )
            )
            self.process.kill()


    def is_running(self) -> bool:
        """Проверяет, запущен ли процесс в данный момент."""
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning
