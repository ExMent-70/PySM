"""Qt lifecycle controller for one cluster-editor export operation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .editor_workers import ExportWorker


class ExportController(QObject):
    """Own the export worker/thread pair and guarantee deterministic cleanup."""

    progress_updated = Signal(int)
    finished = Signal(str)
    stopped = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: ExportWorker | None = None

    @property
    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.isRunning())

    def start(
        self,
        tasks: List[Dict[str, Any]],
        num_workers: int,
        enhancement_factors: Dict[str, Any],
        target_size: Tuple[int, int],
        target_dpi: Tuple[int, int],
        quality: int,
        apply_watermarks: bool,
    ) -> None:
        if self.is_running:
            raise RuntimeError("Экспорт уже выполняется.")

        thread = QThread(self)
        worker = ExportWorker(
            tasks,
            num_workers,
            enhancement_factors,
            target_size,
            target_dpi,
            quality,
            apply_watermarks,
        )
        self._thread = thread
        self._worker = worker
        worker.moveToThread(thread)
        worker.progress_updated.connect(self.progress_updated)
        worker.finished.connect(self.finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_stopped)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)
        thread.start()

    def request_interruption(self) -> None:
        if self._worker is not None:
            self._worker.request_interruption()

    @Slot()
    def _on_thread_stopped(self) -> None:
        self._thread = None
        self._worker = None
        self.stopped.emit()
