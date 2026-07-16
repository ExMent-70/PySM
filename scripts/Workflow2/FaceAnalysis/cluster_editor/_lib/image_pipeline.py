"""Lifecycle management for the shared asynchronous image-cache pipeline."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from pysm_lib.pysm_image_cache import AsyncImageLoader, QtImageCache

try:
    import psutil
except ImportError:  # optional dependency
    psutil = None


def image_pipeline_limits(num_workers: int) -> tuple[int, int]:
    """Choose a bounded RAM budget and decode concurrency for this machine."""

    fallback_available = 4 * 1024**3
    available = (
        int(psutil.virtual_memory().available)
        if psutil is not None
        else fallback_available
    )
    memory_limit = max(
        256 * 1024**2,
        min(1024 * 1024**2, available // 10),
    )
    memory_thread_limit = max(2, memory_limit // (128 * 1024**2))
    thread_limit = max(1, min(8, num_workers, memory_thread_limit))
    return memory_limit, thread_limit


class ImagePipelineController(QObject):
    """Replace cache sessions without leaking or blocking the GUI thread."""

    image_ready = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.cache: QtImageCache | None = None
        self.loader: AsyncImageLoader | None = None
        self._retired: list[AsyncImageLoader] = []
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.setSingleShot(True)
        self._cleanup_timer.timeout.connect(self._cleanup_retired)

    def reset(
        self,
        cache_root: Path,
        num_workers: int,
    ) -> tuple[QtImageCache, AsyncImageLoader]:
        if self.loader is not None:
            self.loader.cancel_all()
            try:
                self.loader.imageReady.disconnect(self.image_ready)
            except (RuntimeError, TypeError):
                pass
            self._retired.append(self.loader)
            self._cleanup_timer.start(50)

        memory_limit, thread_limit = image_pipeline_limits(num_workers)
        self.cache = QtImageCache(
            cache_root,
            memory_limit_bytes=memory_limit,
            disk_limit_bytes=8 * 1024**3,
        )
        self.loader = AsyncImageLoader(
            self.cache,
            max_threads=thread_limit,
            parent=self,
        )
        self.loader.imageReady.connect(self.image_ready)
        return self.cache, self.loader

    @Slot()
    def _cleanup_retired(self) -> None:
        active = []
        for loader in self._retired:
            if loader.is_idle:
                loader.deleteLater()
            else:
                active.append(loader)
        self._retired = active
        if active:
            self._cleanup_timer.start(50)

    def shutdown(self) -> None:
        self._cleanup_timer.stop()
        loaders = [loader for loader in [self.loader, *self._retired] if loader]
        for loader in loaders:
            loader.cancel_all()
        for loader in loaders:
            loader.wait_for_done()
        self._retired.clear()
        self.loader = None
        self.cache = None
