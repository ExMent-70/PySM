"""High-level facade combining source keys, memory LRU and disk storage."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Generic, TypeVar

from .disk_cache import DiskImageCache
from .memory_cache import ByteLRUCache
from .models import ImageCacheKey, ImageRequest


ValueT = TypeVar("ValueT")


class ImageCache(Generic[ValueT]):
    """Shared cache service without a Qt dependency.

    Consumers may store ``QImage`` objects in memory, but must create
    ``QPixmap`` objects only on the GUI thread. The disk layer stores encoded
    derivatives such as PNG or JPEG bytes.
    """

    def __init__(
        self,
        root: Path,
        memory_limit_bytes: int,
        *,
        disk_limit_bytes: int | None = None,
        durable_disk_writes: bool = False,
    ) -> None:
        self.memory: ByteLRUCache[ImageCacheKey, ValueT] = ByteLRUCache(
            memory_limit_bytes
        )
        self.disk = DiskImageCache(
            root,
            max_bytes=disk_limit_bytes,
            durable_writes=durable_disk_writes,
        )

    @staticmethod
    def key_for(request: ImageRequest) -> ImageCacheKey:
        """Fingerprint a request using the current source metadata."""

        return ImageCacheKey.from_request(request)

    def get_memory(self, key: ImageCacheKey) -> ValueT | None:
        return self.memory.get(key)

    def put_memory(
        self,
        key: ImageCacheKey,
        value: ValueT,
        size_bytes: int,
    ) -> bool:
        return self.memory.put(key, value, size_bytes)

    def get_disk_bytes(
        self,
        key: ImageCacheKey,
        extension: str = ".png",
    ) -> bytes | None:
        return self.disk.read_bytes(key, extension)

    def put_disk_bytes(
        self,
        key: ImageCacheKey,
        payload: bytes,
        extension: str = ".png",
    ) -> Path:
        return self.disk.write_bytes(key, payload, extension)

    def get_or_create_disk_bytes(
        self,
        key: ImageCacheKey,
        producer: Callable[[], bytes],
        extension: str = ".png",
    ) -> tuple[bytes, bool]:
        return self.disk.get_or_create_bytes(key, producer, extension)
