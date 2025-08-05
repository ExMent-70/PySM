# analize/analyze_faces/face_lib/onnx_manager.py
"""
Модуль для управления жизненным циклом сессий ONNX Runtime.
Загружает модели по запросу и кэширует их для потокобезопасного
повторного использования.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_best_provider(provider_config: Dict[str, Any]) -> tuple[str, list[dict]]:
    """
    Определяет наилучший доступный провайдер для ONNX Runtime на основе конфигурации.
    """
    try:
        available_providers = ort.get_available_providers()
        logger.info(f"Доступные провайдеры ONNX:")
        logger.info(f"<i>{available_providers}</i><br>")
    except Exception as e:
        logger.error(f"Не удалось получить список провайдеров ONNX: {e}. Используется CPU.", exc_info=True)
        return "CPUExecutionProvider", [{}]

    preferred_provider = provider_config.get("provider_name")

    if preferred_provider and preferred_provider in available_providers:
        selected_provider = preferred_provider
        logger.debug(f"Выбран предпочтительный провайдер из конфигурации: {selected_provider}")
    else:
        # Порядок предпочтения
        priority_order = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        selected_provider = next((p for p in priority_order if p in available_providers), "CPUExecutionProvider")
        logger.debug(f"Автоматически выбран провайдер: {selected_provider}")
    
    provider_options = []
    if selected_provider == "TensorrtExecutionProvider":
        cache_path = Path(provider_config.get("tensorRT_cache_path", "TensorRT_cache"))
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            options = {
                "device_id": str(provider_config.get("device_id", 0)),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_path),
            }
            provider_options.append(options)
            logger.debug(f"Опции для TensorrtExecutionProvider: {options}")
        except OSError as e:
            logger.error(f"Не удалось создать кэш TensorRT ({cache_path}): {e}. Откат к CPU.")
            return "CPUExecutionProvider", [{}]
    
    elif selected_provider == "CUDAExecutionProvider":
        options = {"device_id": str(provider_config.get("device_id", 0))}
        provider_options.append(options)
        logger.debug(f"Опции для CUDAExecutionProvider: {options}")

    return selected_provider, provider_options


class ONNXModelManager:
    """
    Управляет жизненным циклом сессий ONNX Runtime.
    Этот класс является потокобезопасным (thread-safe).
    """

    def __init__(self, provider_config: Dict[str, Any]):
        """
        Инициализирует менеджер.

        Args:
            provider_config: Словарь с конфигурацией провайдера (секция [provider]).
        """
        self._first_session = True
        self._sessions: Dict[Path, ort.InferenceSession] = {}
        self._lock = threading.Lock()
        self.provider_name, self.provider_options = get_best_provider(provider_config)

    def get_session(self, model_path: Path) -> Optional[ort.InferenceSession]:
        """
        Возвращает сессию ONNX для указанной модели (потокобезопасно).
        Если сессия не закэширована, создает новую.

        Args:
            model_path: Абсолютный путь к файлу модели (.onnx).

        Returns:
            Объект InferenceSession или None в случае ошибки.
        """
        if self._first_session:
            logger.info(f"<br>Создание ONNX-сессии и начало анализа изображений...")
            self._first_session = False

        resolved_path = model_path.resolve()

        # Быстрая проверка без блокировки для уже существующих сессий
        if resolved_path in self._sessions:
            return self._sessions[resolved_path]

        # Блокировка для потокобезопасного создания новой сессии
        with self._lock:
            # Повторная проверка внутри блокировки (double-checked locking)
            if resolved_path in self._sessions:
                return self._sessions[resolved_path]

            if not resolved_path.is_file():
                logger.error(f"Файл модели не найден: {resolved_path}", sys)
                return None

            logger.debug(f"Создание новой ONNX-сессии для: {resolved_path.name} ({self.provider_name})")
            try:
                session = ort.InferenceSession(
                    str(resolved_path),
                    providers=[self.provider_name],
                    provider_options=self.provider_options,
                )
                self._sessions[resolved_path] = session
                logger.debug(f"Сессия для {resolved_path.name} успешно создана и закэширована.")
                return session
            except Exception as e:
                logger.error(f"Ошибка создания ONNX-сессии для {resolved_path}: {e}", exc_info=True)
                return None

    def shutdown(self):
        """
        Освобождает ресурсы, связанные с сессиями ONNX.
        """
        logger.debug(f"Завершение работы ONNXModelManager. Освобождение {len(self._sessions)} сессий...")
        sess_len = len(self._sessions)
        with self._lock:
            for model_path, session in self._sessions.items():
                try:
                    # В новых версиях onnxruntime этот метод помогает освободить ресурсы
                    if hasattr(session, 'end_profiling'):
                        session.end_profiling()
                except Exception as e:
                    logger.warning(f"Ошибка при завершении сессии для {model_path.name}: {e}")
            self._sessions.clear()
        logger.info(f" - сессии ONNX освобождены (<b>{sess_len}</b>)")