"""Qt image decoding backed by the shared memory and disk caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QSize, Qt
from PySide6.QtGui import QImage, QImageReader, QImageWriter

from .models import ImageCacheKey, ImageRequest
from .service import ImageCache


_LOAD_LOCKS = tuple(RLock() for _ in range(64))


def _load_lock(key: ImageCacheKey) -> RLock:
    return _LOAD_LOCKS[int(key.digest[:8], 16) % len(_LOAD_LOCKS)]


@dataclass(frozen=True, slots=True)
class QtImageCacheStats:
    """Snapshot of image-loader activity."""

    requests: int
    memory_hits: int
    disk_hits: int
    source_decodes: int
    decode_failures: int
    disk_writes: int
    memory_items: int
    memory_bytes: int
    memory_evictions: int


class QtImageCache:
    """Decode and cache ``QImage`` derivatives without creating ``QPixmap``.

    The class is safe to call from worker threads. Callers are responsible for
    converting the returned ``QImage`` to ``QPixmap`` on the GUI thread.
    """

    def __init__(
        self,
        root: Path,
        memory_limit_bytes: int,
        *,
        disk_limit_bytes: int | None = None,
    ) -> None:
        self._cache: ImageCache[QImage] = ImageCache(
            root,
            memory_limit_bytes,
            disk_limit_bytes=disk_limit_bytes,
        )
        self._stats_lock = RLock()
        self._requests = 0
        self._memory_hits = 0
        self._disk_hits = 0
        self._source_decodes = 0
        self._decode_failures = 0
        self._disk_writes = 0

    @staticmethod
    def key_for(request: ImageRequest) -> ImageCacheKey:
        return ImageCacheKey.from_request(request)

    @staticmethod
    def source_size(source: Path, *, auto_transform: bool = True) -> tuple[int, int]:
        """Read image dimensions without decoding its pixel payload."""

        reader = QImageReader(str(source))
        reader.setAutoTransform(auto_transform)
        size = reader.size()
        if not size.isValid():
            return 0, 0
        transformation = int(reader.transformation().value)
        if auto_transform and transformation in {4, 5, 6, 7}:
            return size.height(), size.width()
        return size.width(), size.height()

    def get_cached(self, request: ImageRequest) -> tuple[ImageCacheKey, QImage] | None:
        """Return a RAM-cached derivative without performing I/O."""

        try:
            key = self.key_for(request)
        except OSError:
            return None
        image = self._cache.get_memory(key)
        if image is None or image.isNull():
            return None
        self._increment("requests")
        self._increment("memory_hits")
        return key, QImage(image)

    def load(
        self,
        request: ImageRequest,
        *,
        persist: bool = False,
        disk_format: str = "PNG",
        quality: int = -1,
    ) -> QImage:
        """Load a derivative synchronously, using RAM and disk when possible."""

        self._increment("requests")
        try:
            key = self.key_for(request)
        except OSError:
            self._increment("decode_failures")
            return QImage()

        extension = self._extension_for_format(disk_format)
        cached = self._cache.get_memory(key)
        if cached is not None and not cached.isNull():
            self._increment("memory_hits")
            if persist and self._cache.disk.get_path(key, extension) is None:
                self._persist_image(key, cached, disk_format, quality)
            return QImage(cached)

        with _load_lock(key):
            cached = self._cache.get_memory(key)
            if cached is not None and not cached.isNull():
                self._increment("memory_hits")
                if persist and self._cache.disk.get_path(key, extension) is None:
                    self._persist_image(key, cached, disk_format, quality)
                return QImage(cached)

            if persist:
                payload = self._cache.get_disk_bytes(key, extension)
                if payload is not None:
                    image = QImage.fromData(payload)
                    if not image.isNull():
                        self._cache_image(key, image)
                        self._increment("disk_hits")
                        return QImage(image)
                    self._cache.disk.discard(key, extension)

            image = self._decode_source(request)
            if image.isNull():
                self._increment("decode_failures")
                return image
            self._cache_image(key, image)
            if persist:
                self._persist_image(key, image, disk_format, quality)
            return QImage(image)

    def _decode_source(self, request: ImageRequest) -> QImage:
        reader = QImageReader(str(request.source))
        reader.setAutoTransform(request.auto_transform)
        source_size = reader.size()
        if not source_size.isValid():
            return QImage()

        target_size = QSize(*request.target_size)
        if request.crop is None:
            scaled_size = self._scaled_size(source_size, target_size, request)
            if scaled_size.isValid() and scaled_size != source_size:
                reader.setScaledSize(scaled_size)
            image = reader.read()
            if not image.isNull():
                self._increment("source_decodes")
        else:
            image = self._load_full_source(request, source_size)
        if image.isNull():
            return image

        if request.crop is not None:
            x, y, width, height = request.crop
            crop_x = min(image.width() - 1, round(image.width() * x))
            crop_y = min(image.height() - 1, round(image.height() * y))
            crop_width = max(1, round(image.width() * width))
            crop_height = max(1, round(image.height() * height))
            crop_width = max(1, min(crop_width, image.width() - crop_x))
            crop_height = max(1, min(crop_height, image.height() - crop_y))
            image = image.copy(crop_x, crop_y, crop_width, crop_height)
            scaled_size = self._scaled_size(image.size(), target_size, request)
            if scaled_size != image.size():
                aspect_mode = (
                    Qt.AspectRatioMode.IgnoreAspectRatio
                    if request.mode == "stretch"
                    else Qt.AspectRatioMode.KeepAspectRatioByExpanding
                    if request.mode == "fill"
                    else Qt.AspectRatioMode.KeepAspectRatio
                )
                image = image.scaled(
                    scaled_size,
                    aspect_mode,
                    Qt.TransformationMode.SmoothTransformation,
                )

        if request.mode == "fill" and image.size() != target_size:
            fill_target = (
                target_size
                if request.allow_upscale
                else QSize(
                    min(target_size.width(), image.width()),
                    min(target_size.height(), image.height()),
                )
            )
            expanded = image.scaled(
                fill_target,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = max(0, (expanded.width() - fill_target.width()) // 2)
            y = max(0, (expanded.height() - fill_target.height()) // 2)
            image = expanded.copy(x, y, fill_target.width(), fill_target.height())
        return image

    def _load_full_source(self, request: ImageRequest, source_size: QSize) -> QImage:
        """Reuse one decoded source for multiple face crops of the same photo."""

        source_request = ImageRequest(
            request.source,
            (source_size.width(), source_size.height()),
            mode="fit",
            auto_transform=request.auto_transform,
            allow_upscale=False,
            variant="pysm.image_cache.decoded_source.v1",
        )
        try:
            source_key = self.key_for(source_request)
        except OSError:
            return QImage()

        cached = self._cache.get_memory(source_key)
        if cached is not None and not cached.isNull():
            return QImage(cached)

        with _load_lock(source_key):
            cached = self._cache.get_memory(source_key)
            if cached is not None and not cached.isNull():
                return QImage(cached)

            reader = QImageReader(str(request.source))
            reader.setAutoTransform(request.auto_transform)
            image = reader.read()
            if image.isNull():
                return image
            self._increment("source_decodes")
            self._cache_image(source_key, image)
            return QImage(image)

    @staticmethod
    def _scaled_size(source: QSize, target: QSize, request: ImageRequest) -> QSize:
        if request.mode == "stretch":
            if request.allow_upscale:
                return target
            return QSize(min(source.width(), target.width()), min(source.height(), target.height()))

        expanding = request.mode == "fill"
        aspect_mode = (
            Qt.AspectRatioMode.KeepAspectRatioByExpanding
            if expanding
            else Qt.AspectRatioMode.KeepAspectRatio
        )
        scaled = source.scaled(target, aspect_mode)
        if request.allow_upscale:
            return scaled
        if scaled.width() >= source.width() and scaled.height() >= source.height():
            return source
        return scaled

    def _cache_image(self, key: ImageCacheKey, image: QImage) -> None:
        self._cache.put_memory(key, QImage(image), max(1, int(image.sizeInBytes())))

    def _persist_image(
        self,
        key: ImageCacheKey,
        image: QImage,
        disk_format: str,
        quality: int,
    ) -> None:
        extension = self._extension_for_format(disk_format)
        existed = self._cache.disk.get_path(key, extension) is not None
        try:
            payload = self._encode(image, disk_format, quality)
            self._cache.put_disk_bytes(
                key,
                payload,
                extension,
            )
        except (OSError, RuntimeError, ValueError):
            return
        if not existed:
            self._increment("disk_writes")

    @staticmethod
    def _encode(image: QImage, disk_format: str, quality: int) -> bytes:
        data = QByteArray()
        buffer = QBuffer(data)
        if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
            raise RuntimeError("failed to open image cache buffer")
        try:
            writer = QImageWriter(buffer, disk_format.upper().encode("ascii"))
            if quality >= 0:
                writer.setQuality(quality)
            if not writer.write(image):
                raise RuntimeError(
                    f"failed to encode image as {disk_format}: {writer.errorString()}"
                )
            return bytes(data)
        finally:
            buffer.close()

    @staticmethod
    def _extension_for_format(disk_format: str) -> str:
        normalized = str(disk_format or "").strip().upper()
        if normalized in {"JPG", "JPEG"}:
            return ".jpg"
        if normalized == "PNG":
            return ".png"
        if not normalized.isalnum() or len(normalized) > 10:
            raise ValueError(f"unsupported disk image format: {disk_format}")
        return f".{normalized.lower()}"

    def _increment(self, field: str) -> None:
        attribute = f"_{field}"
        with self._stats_lock:
            setattr(self, attribute, getattr(self, attribute) + 1)

    @property
    def stats(self) -> QtImageCacheStats:
        memory = self._cache.memory.stats
        with self._stats_lock:
            return QtImageCacheStats(
                requests=self._requests,
                memory_hits=self._memory_hits,
                disk_hits=self._disk_hits,
                source_decodes=self._source_decodes,
                decode_failures=self._decode_failures,
                disk_writes=self._disk_writes,
                memory_items=memory.item_count,
                memory_bytes=memory.current_bytes,
                memory_evictions=memory.evictions,
            )
