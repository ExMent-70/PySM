# pysm_lib/script_runner.py

import subprocess
import threading
import os
import pathlib
import json
from typing import (
    Callable,
    List,
    Optional,
    Dict,
    Any,
)
import logging
from .models import ScriptInfoModel

# --- НОВЫЙ ИМПОРТ ---
from .locale_manager import LocaleManager

# --- ИЗМЕНЕНИЕ: Создаем локальный экземпляр ---
locale_manager = LocaleManager()
logger = logging.getLogger(f"PyScriptManager.{__name__}")

CONSOLE_BLOCK_START_PHRASE = "PYSM_CONSOLE_BLOCK_START"
CONSOLE_BLOCK_END_PHRASE = "PYSM_CONSOLE_BLOCK_END"


class ScriptRunner:
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
        custom_command_args_dict: Optional[Dict[str, Any]] = None,
        context_file_path: Optional[str] = None,
        app_root_dir: Optional[pathlib.Path] = None,
        global_python_paths: Optional[List[str]] = None,
        global_env_vars: Optional[Dict[str, str]] = None,
    ):
        self.script_info_model: ScriptInfoModel = script_info
        self.python_interpreter: str = python_interpreter
        self.additional_env_paths: List[str] = additional_env_paths or []
        self.context_file_path = context_file_path
        self.app_root_dir = app_root_dir
        self.global_python_paths = global_python_paths or []
        self.global_env_vars = global_env_vars or {}
        self.on_start = on_start
        self.on_output = on_output
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_progress = on_progress
        self.custom_command_args_dict = custom_command_args_dict
        self.process: Optional[subprocess.Popen] = None
        self.stdout_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._console_output_blocked: bool = False

    # 1. БЛОК: Метод _prepare_environment (ИЗМЕНЕН)
    def _prepare_environment(self) -> Dict[str, str]:
        script_id_for_log = self.script_info_model.id
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.debug(
            locale_manager.get(
                "script_runner.log_debug.preparing_env", id=script_id_for_log
            )
        )
        env = os.environ.copy()
        if self.global_env_vars:
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.debug(
                locale_manager.get(
                    "script_runner.log_debug.applying_global_vars",
                    vars=self.global_env_vars,
                )
            )
            env.update(self.global_env_vars)
        env["PY_SCRIPT_MANAGER_ACTIVE"] = "1"
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.debug(locale_manager.get("script_runner.log_debug.pysm_active_set"))
        new_path_parts: List[str] = []

        interpreter_path_obj = pathlib.Path(self.python_interpreter)
        if interpreter_path_obj.is_file():
            interpreter_dir = str(interpreter_path_obj.parent.resolve())
            if interpreter_dir not in new_path_parts:
                new_path_parts.append(interpreter_dir)

        # --- НОВАЯ ЛОГИКА: Разрешение относительных путей ---
        # Здесь мы берем пути "как есть" из модели и преобразуем их в абсолютные
        script_folder_path = pathlib.Path(self.script_info_model.folder_abs_path)
        script_specific_paths = self.script_info_model.script_specific_env_paths
        if script_specific_paths:
            for p_str in script_specific_paths:
                if not p_str:
                    continue
                path_obj = pathlib.Path(p_str)
                # Если путь относительный, он разрешается относительно папки скрипта
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

        original_path = env.get("PATH", "")
        if original_path:
            new_path_parts.append(original_path)
        unique_path_parts = list(dict.fromkeys(filter(None, new_path_parts)))
        env["PATH"] = os.pathsep.join(unique_path_parts)
        python_path_parts = []
        if self.app_root_dir and self.app_root_dir.is_dir():
            python_path_parts.append(str(self.app_root_dir))
        if self.global_python_paths:
            python_path_parts.extend(self.global_python_paths)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            python_path_parts.extend(existing_pythonpath.split(os.pathsep))
        if python_path_parts:
            unique_python_paths = list(dict.fromkeys(filter(None, python_path_parts)))
            env["PYTHONPATH"] = os.pathsep.join(unique_python_paths)
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.debug(
                locale_manager.get(
                    "script_runner.log_debug.pythonpath_set", path=env["PYTHONPATH"]
                )
            )
        env["PYTHONIOENCODING"] = "utf-8"
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.debug(
            locale_manager.get(
                "script_runner.log_debug.final_path",
                id=script_id_for_log,
                path=env["PATH"],
            )
        )
        return env


    def _read_stream(self, stream_pipe, stream_name: str, script_id_cb: str):
        """
        Читает вывод из потока (stdout или stderr) в реальном времени,
        анализирует строки на наличие специальных команд и передает их
        соответствующим обработчикам.
        """
        logger.debug(
            locale_manager.get(
                "script_runner.log_debug.stream_reader_started",
                stream=stream_name,
                id=script_id_cb,
            )
        )
        try:
            for line in iter(stream_pipe.readline, ""):
                if self._stop_event.is_set():
                    logger.warning(
                        locale_manager.get(
                            "script_runner.log_warning.stream_reader_stopped",
                            stream=stream_name,
                            id=script_id_cb,
                        )
                    )
                    break

                line_str_stripped = line.rstrip("\r\n")

                # --- НАЧАЛО ИЗМЕНЕНИЙ ---
                # КОММЕНТАРИЙ: Логика блокировки/разблокировки консоли теперь
                # работает для ОБОИХ потоков (stdout и stderr).

                # 1. Проверяем команды блокировки в первую очередь
                if line_str_stripped == "PYSM_CONSOLE_BLOCK_START":
                    if not self._console_output_blocked:
                        self._console_output_blocked = True
                        if self.on_output:
                            self.on_output(script_id_cb, "runner_info", locale_manager.get("script_runner.console_message.output_blocked"))
                    continue # Команда обработана, переходим к следующей строке

                if line_str_stripped == "PYSM_CONSOLE_BLOCK_END":
                    if self._console_output_blocked:
                        self._console_output_blocked = False
                        if self.on_output:
                            self.on_output(script_id_cb, "runner_info", locale_manager.get("script_runner.console_message.output_unblocked"))
                    continue # Команда обработана, переходим к следующей строке

                # 2. Проверяем "богатый" контент ТОЛЬКО в потоке stderr
                if stream_name == 'stderr':
                    # Проверка на HTML-блок
                    if line_str_stripped.startswith("PYSM_HTML_BLOCK:"):
                        html_content = line_str_stripped.split(":", 1)[1]
                        if self.on_output:
                            self.on_output(script_id_cb, "html_block", html_content)
                        continue

                    # Проверка на JSON-прогресс
                    try:
                        progress_data = json.loads(line_str_stripped)
                        if (isinstance(progress_data, dict) and progress_data.get("type") == "progress"):
                            current = progress_data.get("current", 0)
                            total = progress_data.get("total", 0)
                            text = progress_data.get("text")
                            if self.on_progress:
                                self.on_progress(script_id_cb, int(current), int(total), text)
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass # Это не JSON-прогресс, а обычная строка ошибки
                # --- КОНЕЦ ИЗМЕНЕНИЙ ---
                
                # 3. Выводим обычный текст, если вывод не заблокирован
                if self._console_output_blocked and stream_name != "runner_info":
                    continue
                
                if self.on_output:
                    self.on_output(script_id_cb, stream_name, line_str_stripped)

        except Exception as e:
            logger.error(
                locale_manager.get(
                    "script_runner.log_error.stream_read_error",
                    stream=stream_name,
                    id=script_id_cb,
                    error=e,
                ),
                exc_info=True,
            )
            if self.on_error and not self._stop_event.is_set():
                self.on_error(
                    script_id_cb,
                    locale_manager.get(
                        "script_runner.error.stream_read_runtime_error",
                        stream=stream_name,
                        error=e,
                    ),
                )
        finally:
            stream_pipe.close()
            logger.debug(
                locale_manager.get(
                    "script_runner.log_debug.stream_reader_finished",
                    stream=stream_name,
                    id=script_id_cb,
                )
            )

    # ... (остальной код класса) ...

    # 1. БЛОК: Метод _format_args_from_dict (ИЗМЕНЕН)
    def _format_args_from_dict(self, args_dict: Optional[Dict[str, Any]]) -> List[str]:
        if not args_dict:
            return []

        formatted_args: List[str] = []
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

            # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            #
            # Заменяем 'elif value is not None:' на 'else:', чтобы
            # обрабатывать все остальные случаи, включая value = None.
            #
            else:
                formatted_args.append(formatted_arg_name)
                # Если значение None, передаем пустую строку.
                # Иначе - строковое представление значения.
                # Это гарантирует, что обязательный аргумент всегда
                # будет присутствовать в командной строке.
                value_to_add = str(value) if value is not None else ""
                formatted_args.append(value_to_add)
            # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

        return formatted_args

    def build_command_list(self) -> List[str]:
        run_file_path_str = self.script_info_model.run_file_abs_path
        command: List[str] = [
            self.python_interpreter,
            "-u",
            "-m",
            "pysm_lib.context_loader",
        ]
        command.append(run_file_path_str)
        if self.context_file_path:
            command.extend(["--pysm-context-file", self.context_file_path])
        additional_args_list: List[str] = []
        if isinstance(self.custom_command_args_dict, dict):
            additional_args_list = self._format_args_from_dict(
                self.custom_command_args_dict
            )
        command.extend(additional_args_list)
        return command

    def run(self, effective_id_for_callbacks: str):
        script_name_for_log = self.script_info_model.name
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.info(
            locale_manager.get(
                "script_runner.log_info.starting_script",
                name=script_name_for_log,
                id=effective_id_for_callbacks,
            )
        )

        run_file_abs_path_str = self.script_info_model.run_file_abs_path
        if not run_file_abs_path_str:
            if self.on_error:
                # --- СТРОКА ИЗМЕНЕНА ---
                self.on_error(
                    effective_id_for_callbacks,
                    locale_manager.get("script_runner.error.run_file_path_missing"),
                )
            return

        run_file_abs_path = pathlib.Path(run_file_abs_path_str)
        if not run_file_abs_path.is_file():
            if self.on_error:
                # --- СТРОКА ИЗМЕНЕНА ---
                self.on_error(
                    effective_id_for_callbacks,
                    locale_manager.get(
                        "script_runner.error.run_file_not_found", path=run_file_abs_path
                    ),
                )
            return

        script_folder_abs_path = pathlib.Path(self.script_info_model.folder_abs_path)
        command = self.build_command_list()

        try:
            env = self._prepare_environment()
            self._stop_event.clear()

            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(script_folder_abs_path),
                env=env,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )

            if self.on_start:
                self.on_start(effective_id_for_callbacks)

            if self.process.stdout:
                self.stdout_thread = threading.Thread(
                    target=self._read_stream,
                    args=(self.process.stdout, "stdout", effective_id_for_callbacks),
                )
                self.stdout_thread.daemon = True
                self.stdout_thread.start()

            if self.process.stderr:
                self.stderr_thread = threading.Thread(
                    target=self._read_stream,
                    args=(self.process.stderr, "stderr", effective_id_for_callbacks),
                )
                self.stderr_thread.daemon = True
                self.stderr_thread.start()

            wait_thread = threading.Thread(
                target=self._wait_for_completion, args=(effective_id_for_callbacks,)
            )
            wait_thread.daemon = True
            wait_thread.start()

        except Exception as e:
            # --- СТРОКИ ИЗМЕНЕНЫ ---
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
                    effective_id_for_callbacks,
                    locale_manager.get(
                        "script_runner.error.generic_run_runtime_error",
                        name=script_name_for_log,
                        error=e,
                    ),
                )
            self.process = None

    def _wait_for_completion(self, script_id_cb: str):
        if not self.process:
            return

        return_code = self.process.wait()
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.info(
            locale_manager.get(
                "script_runner.log_info.process_finished",
                id=script_id_cb,
                code=return_code,
            )
        )

        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join()
        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join()
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.debug(
            locale_manager.get(
                "script_runner.log_debug.stream_threads_joined", id=script_id_cb
            )
        )

        if self.on_complete:
            self.on_complete(script_id_cb, return_code)

        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None

    def stop(self, script_id_cb: str):
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.info(
            locale_manager.get("script_runner.log_info.stop_requested", id=script_id_cb)
        )
        self._stop_event.set()

        if self.process and self.process.poll() is None:
            pid = self.process.pid
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.info(
                locale_manager.get(
                    "script_runner.log_info.terminating_process",
                    pid=pid,
                    id=script_id_cb,
                )
            )
            try:
                self.process.terminate()
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                # --- СТРОКА ИЗМЕНЕНА ---
                logger.warning(
                    locale_manager.get(
                        "script_runner.log_warning.terminate_failed_killing",
                        pid=pid,
                        id=script_id_cb,
                    )
                )
                self.process.kill()
                self.process.wait(timeout=1)
            except Exception as e:
                # --- СТРОКА ИЗМЕНЕНА ---
                logger.error(
                    locale_manager.get(
                        "script_runner.log_error.stop_process_error", pid=pid, error=e
                    ),
                    exc_info=True,
                )

        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=0.5)
        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join(timeout=0.5)

        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None
