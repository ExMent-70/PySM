# gui_lib/workers.py

import logging
import traceback
from PySide6.QtCore import QObject, Signal, Slot

# Импортируем тексты из нашего пакета
from .ui_texts import UI_TEXTS

# Импортируем основную функцию обработки
from main import run_full_processing

logger = logging.getLogger(__name__)


class QtLogHandler(logging.Handler):
    """Перенаправляет логи в GUI."""
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
    """Выполняет длительную обработку в отдельном потоке."""
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