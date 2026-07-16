# _common/onnx_manager.py
import logging
import threading
import time
import sys
import os
import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import onnxruntime as ort

from .status_icons import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
)

logger = logging.getLogger(__name__)

@contextmanager
def suppress_output():
    """
    Контекстный менеджер для подавления вывода stdout и stderr.
    Глушит как Python print(), так и C++ printf() (важно для TensorRT/ONNX).
    """
    # Открываем "черную дыру" для данных
    with open(os.devnull, "w") as devnull:
        # Сохраняем оригинальные файловые дескрипторы
        old_stdout_fd = sys.stdout.fileno()
        old_stderr_fd = sys.stderr.fileno()
        
        # Делаем дубликаты оригинальных дескрипторов, чтобы потом восстановить
        saved_stdout_fd = os.dup(old_stdout_fd)
        saved_stderr_fd = os.dup(old_stderr_fd)

        try:
            # Перенаправляем stdout/stderr в devnull
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), old_stdout_fd)
            os.dup2(devnull.fileno(), old_stderr_fd)
            yield
        finally:
            # Восстанавливаем потоки
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, old_stdout_fd)
            os.dup2(saved_stderr_fd, old_stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

def get_best_provider(provider_config: Dict[str, Any]) -> Tuple[str, List[Dict], Optional[Path]]:
    try:
        available_providers = ort.get_available_providers()
        logger.debug(f"Доступные провайдеры ONNX: {available_providers}")
    except Exception as e:
        logger.error(f"{icon_error} Не удалось получить список провайдеров ONNX: {e}. Используется CPU.", exc_info=True)
        return "CPUExecutionProvider", [{}], None

    preferred_provider = provider_config.get("provider_name")
    selected_provider = "CPUExecutionProvider"
    if preferred_provider and preferred_provider in available_providers:
        selected_provider = preferred_provider
        logger.info(f"Выбран предпочтительный провайдер: <b>{selected_provider}</b>")
    else:
        priority_order = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        selected_provider = next((p for p in priority_order if p in available_providers), "CPUExecutionProvider")
        logger.info(f"Автоматически выбран провайдер: <b>{selected_provider}</b>")

    provider_options = []
    trt_cache_path: Optional[Path] = None

    if selected_provider == "TensorrtExecutionProvider":
        cache_str = provider_config.get("tensorRT_cache_path", "TensorRT_cache")
        trt_cache_path = Path(cache_str).resolve()
        try:
            trt_cache_path.mkdir(parents=True, exist_ok=True)
            options = {
                "device_id": str(provider_config.get("device_id", 0)),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_cache_path),
            }
            provider_options.append(options)
            logger.debug(f"Опции TensorRT: {options}")
        except OSError as e:
            logger.error(f"{icon_error} Не удалось создать кэш TensorRT ({trt_cache_path}): {e}. Откат к CPU.")
            return "CPUExecutionProvider", [{}], None
    
    elif selected_provider == "CUDAExecutionProvider":
        options = {"device_id": str(provider_config.get("device_id", 0))}
        provider_options.append(options)

    return selected_provider, provider_options, trt_cache_path


class ONNXModelManager:
    def __init__(self, provider_config: Dict[str, Any]):
        self._first_session = True
        self._sessions: Dict[Path, ort.InferenceSession] = {}
        self._lock = threading.Lock()
        self.provider_name, self.provider_options, self.trt_cache_path = get_best_provider(provider_config)

    def check_trt_cache_status(self) -> str:
        """Проверяет, есть ли файлы в папке кеша."""
        if not self.trt_cache_path:
            return ""
        try:
            has_files = any(self.trt_cache_path.iterdir())
            if has_files:
                return "(Папка кеша содержит файлы)" 
            else:
                return f"{icon_warning} <b>Кеш не найден. Компиляция моделей займёт некоторое время...</b>"
        except Exception:
            return ""

    def get_session(self, model_path: Path) -> Optional[ort.InferenceSession]:
        if self._first_session:
            self._first_session = False

        resolved_path = model_path.resolve()
        if resolved_path in self._sessions: return self._sessions[resolved_path]

        with self._lock:
            if resolved_path in self._sessions: return self._sessions[resolved_path]
            if not resolved_path.is_file():
                logger.error(f"Файл модели не найден: {resolved_path}")
                return None
            
            is_trt = self.provider_name == "TensorrtExecutionProvider"
            if is_trt:
                logger.info(f"{icon_ok} модель <i>{resolved_path.name}</i> готова")
                sys.stdout.flush()
            else:
                logger.debug(f"Загрузка модели: <i>{resolved_path.name}</i> ({self.provider_name})")
            
            try:
                start_time = time.time()
                
                # Дополнительная настройка, чтобы сам ONNX Runtime молчал
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3  # 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal
                
                # Принудительно глушим ВЕСЬ вывод (включая C++ TensorRT) на время создания сессии
                with suppress_output():
                    session = ort.InferenceSession(
                        str(resolved_path), 
                        sess_options=sess_options,
                        providers=[self.provider_name], 
                        provider_options=self.provider_options
                    )

                self._sessions[resolved_path] = session
                elapsed = time.time() - start_time
                if is_trt:
                    logger.debug(f" -> Модель загружена за {elapsed:.2f} сек.")
                else:
                    logger.debug(f" -> Загружено за {elapsed:.4f} сек.")
                return session
            except Exception as e:
                logger.error(f"{icon_error} Ошибка создания ONNX-сессии для <i>{resolved_path}</i>: {e}", exc_info=True)
                return None

    def shutdown(self):
        logger.debug(f"Завершение работы ONNXModelManager. Освобождение {len(self._sessions)} сессий...")
        sess_len = len(self._sessions)
        with self._lock:
            for model_path, session in self._sessions.items():
                try: del session
                except Exception: pass
            self._sessions.clear()
            gc.collect()
        logger.info(f" - сессии ONNX освобождены (<b>{sess_len}</b>)")