"""Shared asynchronous image-cache lifecycle for photo_selection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from pysm_lib.pysm_image_cache import AsyncImageLoader, QtImageCache


class PhotoSelectionImagePipeline(QObject):
    """Own the shared cache and worker pool used by GUI image previews."""

    shutdownFinished = Signal()

    MEMORY_LIMIT_BYTES = 256 * 1024**2
    DISK_LIMIT_BYTES = 2 * 1024**3
    MAX_THREADS = 2

    def __init__(self, analysis_dir: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        cache_root = Path(analysis_dir) / ".thumbnails" / "photo_selection-v1"
        self.cache = QtImageCache(
            cache_root,
            memory_limit_bytes=self.MEMORY_LIMIT_BYTES,
            disk_limit_bytes=self.DISK_LIMIT_BYTES,
        )
        self.loader = AsyncImageLoader(
            self.cache,
            parent=self,
            max_threads=self.MAX_THREADS,
        )
        self._closed = False
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setInterval(25)
        self._shutdown_timer.timeout.connect(self._poll_shutdown)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def shutdown(self) -> bool:
        """Begin non-blocking shutdown and return whether it is already done."""

        if self._closed:
            return True
        self.loader.cancel_all()
        if self.loader.is_idle:
            self._closed = True
            return True
        self._shutdown_timer.start()
        return False

    @Slot()
    def _poll_shutdown(self) -> None:
        if not self.loader.is_idle:
            return
        self._shutdown_timer.stop()
        self._closed = True
        self.shutdownFinished.emit()
