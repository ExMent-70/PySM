"""Asynchronous Qt image requests with in-flight deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from .models import ImageCacheKey, ImageRequest
from .qt_cache import QtImageCache


@dataclass(frozen=True, slots=True)
class AsyncImageResult:
    """Image result delivered on the loader object's thread."""

    channel: Hashable
    request_id: int
    key: ImageCacheKey | None
    image: QImage
    error: str = ""


@dataclass(frozen=True, slots=True)
class AsyncImageLoaderStats:
    submitted: int
    coalesced: int
    completed: int
    stale_dropped: int


class _WorkerSignals(QObject):
    finished = Signal(object, int, object, str)


class _ImageLoadRunnable(QRunnable):
    def __init__(
        self,
        cache: QtImageCache,
        request: ImageRequest,
        task_key: tuple[object, ...],
        task_id: int,
        *,
        persist: bool,
        disk_format: str,
        quality: int,
    ) -> None:
        super().__init__()
        # ``tryTake`` is unsafe for auto-deleted runnables because the pool may
        # recycle the C++ address before cancellation reaches it (ABA race).
        # The loader keeps the runnable alive in ``_pending`` instead.
        self.setAutoDelete(False)
        self.cache = cache
        self.request = request
        self.task_key = task_key
        self.task_id = task_id
        self.persist = persist
        self.disk_format = disk_format
        self.quality = quality
        self.signals = _WorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Suppress useful work when a runnable has already started."""

        self._cancelled = True

    @Slot()
    def run(self) -> None:
        if self._cancelled:
            self.signals.finished.emit(
                self.task_key,
                self.task_id,
                QImage(),
                "cancelled",
            )
            return
        try:
            image = self.cache.load(
                self.request,
                persist=self.persist,
                disk_format=self.disk_format,
                quality=self.quality,
            )
            error = "" if not image.isNull() else "image decoding failed"
        except Exception as exc:  # worker boundary: report errors to the GUI
            image = QImage()
            error = str(exc)
        if self._cancelled:
            image = QImage()
            error = "cancelled"
        self.signals.finished.emit(self.task_key, self.task_id, image, error)


@dataclass(slots=True)
class _PendingTask:
    runnable: _ImageLoadRunnable
    task_id: int
    subscribers: list[tuple[Hashable, int, ImageCacheKey]]


class AsyncImageLoader(QObject):
    """Load images off the GUI thread and suppress stale channel results."""

    imageReady = Signal(object)

    def __init__(
        self,
        cache: QtImageCache,
        parent: QObject | None = None,
        *,
        max_threads: int = 4,
    ) -> None:
        super().__init__(parent)
        self.cache = cache
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(1, max_threads))
        self._next_request_id = 0
        self._latest_by_channel: dict[Hashable, int] = {}
        self._pending: dict[tuple[object, ...], _PendingTask] = {}
        # Running tasks removed from ``_pending`` stay alive until their worker
        # signal is delivered. This lets an identical replacement request use
        # the same cache key without attaching to a cancelled runnable.
        self._detached: dict[int, _ImageLoadRunnable] = {}
        self._submitted = 0
        self._coalesced = 0
        self._completed = 0
        self._stale_dropped = 0

    def request(
        self,
        request: ImageRequest,
        *,
        channel: Hashable,
        persist: bool = False,
        disk_format: str = "PNG",
        quality: int = -1,
    ) -> int:
        """Schedule an image and return the channel-local request identifier."""

        if QThread.currentThread() != self.thread():
            raise RuntimeError("AsyncImageLoader.request must be called on its owner thread")
        self._next_request_id += 1
        request_id = self._next_request_id
        self._latest_by_channel[channel] = request_id
        self._submitted += 1

        try:
            key = self.cache.key_for(request)
        except OSError as exc:
            QTimer.singleShot(
                0,
                lambda: self._deliver(
                    AsyncImageResult(channel, request_id, None, QImage(), str(exc))
                ),
            )
            return request_id

        if not persist:
            cached = self.cache.get_cached(request)
            if cached is not None:
                _, image = cached
                QTimer.singleShot(
                    0,
                    lambda: self._deliver(
                        AsyncImageResult(channel, request_id, key, image)
                    ),
                )
                return request_id

        task_key = (key, bool(persist), disk_format.upper(), int(quality))
        subscriber = (channel, request_id, key)
        pending = self._pending.get(task_key)
        if pending is not None:
            pending.subscribers.append(subscriber)
            self._coalesced += 1
            return request_id

        runnable = _ImageLoadRunnable(
            self.cache,
            request,
            task_key,
            request_id,
            persist=persist,
            disk_format=disk_format,
            quality=quality,
        )
        runnable.signals.finished.connect(self._on_finished)
        self._pending[task_key] = _PendingTask(
            runnable,
            request_id,
            [subscriber],
        )
        self.pool.start(runnable)
        return request_id

    def cancel(self, channel: Hashable) -> None:
        """Cancel queued work and suppress running work for ``channel``."""

        self._latest_by_channel.pop(channel, None)
        self._remove_subscribers(lambda subscriber: subscriber[0] == channel)

    def cancel_all(self) -> None:
        """Cancel every queued request owned by this loader."""

        self._latest_by_channel.clear()
        self._remove_subscribers(lambda _subscriber: True)

    def _remove_subscribers(
        self,
        predicate: Callable[[tuple[Hashable, int, ImageCacheKey]], bool],
    ) -> None:
        for task_key, pending in list(self._pending.items()):
            pending.subscribers = [
                subscriber
                for subscriber in pending.subscribers
                if not predicate(subscriber)
            ]
            if pending.subscribers:
                continue

            self._pending.pop(task_key, None)
            if not self.pool.tryTake(pending.runnable):
                pending.runnable.cancel()
                self._detached[pending.task_id] = pending.runnable

    @Slot(object, int, object, str)
    def _on_finished(
        self,
        task_key: tuple[object, ...],
        task_id: int,
        image: QImage,
        error: str,
    ) -> None:
        self._detached.pop(task_id, None)
        pending = self._pending.get(task_key)
        if pending is None or pending.task_id != task_id:
            return
        self._pending.pop(task_key, None)
        self._completed += 1
        for channel, request_id, key in pending.subscribers:
            self._deliver(AsyncImageResult(channel, request_id, key, QImage(image), error))

    def _deliver(self, result: AsyncImageResult) -> None:
        if self._latest_by_channel.get(result.channel) != result.request_id:
            self._stale_dropped += 1
            return
        self._latest_by_channel.pop(result.channel, None)
        self.imageReady.emit(result)

    def wait_for_done(self, timeout_ms: int = -1) -> bool:
        return self.pool.waitForDone(timeout_ms)

    @property
    def is_idle(self) -> bool:
        """Return whether no queued, running, or undelivered task remains."""

        return not self._pending and not self._detached

    @property
    def stats(self) -> AsyncImageLoaderStats:
        return AsyncImageLoaderStats(
            submitted=self._submitted,
            coalesced=self._coalesced,
            completed=self._completed,
            stale_dropped=self._stale_dropped,
        )
