# fc_lib/fc_onnx_manager.py

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import threading

import onnxruntime as ort

from .fc_config import ConfigManager
from .fc_utils import get_best_provider

logger = logging.getLogger(__name__)


class ONNXModelManager:
    """
    Управляет жизненным циклом сессий ONNX Runtime.
    Загружает модели по запросу и кэширует их для повторного использования.
    Этот класс является потокобезопасным (thread-safe).
    """

    def __init__(self, config: ConfigManager):
        """
        Инициализирует менеджер. Определяет лучший доступный провайдер ONNX.
        """
        self.config = config
        self._sessions: Dict[Path, ort.InferenceSession] = {}
        # Создаем объект блокировки для потокобезопасного доступа к словарю сессий
        self._lock = threading.Lock()
        self.provider_name: str
        self.provider_options: list[dict]

        try:
            # Получаем всю секцию [provider] из конфига
            provider_config = self.config.get("provider", default={})
            
            # Получаем путь к кэшу из секции [paths]
            paths_config = self.config.get("paths", default={})
            cache_path = paths_config.get("tensorRT_cache_path", "TensorRT_cache")
            
            # Передаем оба аргумента в get_best_provider
            self.provider_name, self.provider_options = get_best_provider(
                provider_config, cache_path
            )
            
            logger.info(
                f"[ONNXManager] Инициализирован с провайдером: {self.provider_name}"
            )
        except Exception as e:
            logger.error(f"[ONNXManager] Ошибка определения провайдера ONNX: {e}", exc_info=True)
            logger.warning("Откат к CPUExecutionProvider.")
            self.provider_name = "CPUExecutionProvider"
            self.provider_options = [{}]

    def get_session(self, model_path: Path) -> Optional[ort.InferenceSession]:
        """
        Возвращает сессию ONNX для указанной модели (потокобезопасно).

        Если сессия уже была создана, возвращает ее из кэша.
        Если нет, создает новую сессию, кэширует и возвращает ее.

        Args:
            model_path: Абсолютный путь к файлу модели (.onnx).

        Returns:
            Объект InferenceSession или None в случае ошибки.
        """
        resolved_path = model_path.resolve()

        # Быстрая проверка без блокировки для уже существующих сессий
        if resolved_path in self._sessions:
            return self._sessions[resolved_path]

        # Блокировка, чтобы только один поток мог создавать новую сессию
        with self._lock:
            # Повторная проверка внутри блокировки на случай, если другой
            # поток уже создал сессию, пока мы ждали освобождения замка.
            if resolved_path in self._sessions:
                logger.debug(f"[ONNXManager] Возврат сессии из кэша (после блокировки) для: {resolved_path.name}")
                return self._sessions[resolved_path]

            if not resolved_path.is_file():
                logger.error(f"[ONNXManager] Файл модели не найден: {resolved_path}")
                return None

            logger.debug(f"[ONNXManager] Создание новой сессии для: {resolved_path.name} ({self.provider_name})")
            try:
                # Для CPU не нужно передавать provider_options
                current_provider_options = (
                    self.provider_options if self.provider_name != "CPUExecutionProvider" else []
                )
                
                session = ort.InferenceSession(
                    str(resolved_path),
                    providers=[self.provider_name],
                    provider_options=current_provider_options,
                )
                
                # Сохраняем сессию в кэш
                self._sessions[resolved_path] = session
                logger.debug(f"[ONNXManager] Сессия для {resolved_path.name} успешно создана и закэширована.")
                return session
                
            except Exception as e:
                logger.error(f"[ONNXManager] Ошибка создания сессии для {resolved_path}: {e}", exc_info=True)
                return None
                
    def shutdown(self):
        """
        Явно освобождает ресурсы, связанные с сессиями ONNX.
        Вызывает внутренний метод ONNX Runtime 'end_profiling' для каждой сессии,
        что является рекомендуемым способом для очистки.
        """
        logger.info(f"[ONNXManager] Завершение работы. Освобождение {len(self._sessions)} сессий...")
        for model_path, session in self._sessions.items():
            try:
                # В новых версиях onnxruntime рекомендуется вызывать end_profiling()
                # для корректного освобождения некоторых ресурсов.
                # Если этого метода нет, ничего страшного не произойдет.
                if hasattr(session, 'end_profiling'):
                    session.end_profiling()
                logger.debug(f"[ONNXManager] Сессия для {model_path.name} помечена для освобождения.")
            except Exception as e:
                logger.error(f"[ONNXManager] Ошибка при попытке завершить сессию для {model_path.name}: {e}")

        self._sessions.clear()
        logger.info("[ONNXManager] Все сессии ONNX освобождены.")                