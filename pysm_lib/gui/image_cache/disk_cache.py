"""Persistent immutable derivative cache with atomic writes."""

from __future__ import annotations

import os
from pathlib import Path
import re
from threading import RLock
import tempfile
from typing import Callable

from .models import ImageCacheKey


# Fixed lock striping bounds memory use while coordinating all cache instances
# in the current process. Atomic replacement also prevents partial files when
# independent processes happen to write the same derivative.
_WRITE_LOCKS = tuple(RLock() for _ in range(64))
_PRUNE_LOCK = RLock()
_EXTENSION_PATTERN = re.compile(r"\.[A-Za-z0-9]{1,10}\Z")


def _write_lock(digest: str) -> RLock:
    return _WRITE_LOCKS[int(digest[:8], 16) % len(_WRITE_LOCKS)]


class DiskImageCache:
    """Store encoded image derivatives below a caller-owned cache root."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int | None = None,
        durable_writes: bool = False,
    ) -> None:
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive or None")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.durable_writes = durable_writes
        self._known_total_bytes: int | None = None

    def path_for(self, key: ImageCacheKey, extension: str = ".png") -> Path:
        """Return the deterministic, two-level-sharded derivative path."""

        normalized = extension if extension.startswith(".") else f".{extension}"
        if not _EXTENSION_PATTERN.fullmatch(normalized):
            raise ValueError(f"invalid cache file extension: {extension}")
        digest = key.digest
        return self.root / digest[:2] / f"{digest}{normalized.lower()}"

    def get_path(
        self,
        key: ImageCacheKey,
        extension: str = ".png",
    ) -> Path | None:
        """Return the derivative path only when a complete file exists."""

        path = self.path_for(key, extension)
        return path if path.is_file() else None

    def read_bytes(
        self,
        key: ImageCacheKey,
        extension: str = ".png",
    ) -> bytes | None:
        path = self.get_path(key, extension)
        if path is None:
            return None
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            return None
        if self.max_bytes is not None:
            try:
                path.touch(exist_ok=True)
            except OSError:
                pass
        return payload

    def discard(
        self,
        key: ImageCacheKey,
        extension: str = ".png",
    ) -> bool:
        """Remove one derivative, normally after failed validation."""

        path = self.path_for(key, extension)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def write_bytes(
        self,
        key: ImageCacheKey,
        payload: bytes,
        extension: str = ".png",
    ) -> Path:
        """Atomically publish an immutable derivative.

        The first completed writer in this process wins. Other writers reuse
        that complete file. Temporary files are always created beside the
        destination so ``os.replace`` stays atomic on the same filesystem.
        """

        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if not payload:
            raise ValueError("payload must not be empty")

        target = self.path_for(key, extension)
        lock = _write_lock(key.digest)
        with lock:
            if target.is_file():
                return target
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                )
                temporary_path = Path(temporary_name)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    if self.durable_writes:
                        os.fsync(stream.fileno())
                os.replace(temporary_path, target)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        self._record_write(len(payload))
        return target

    def get_or_create_bytes(
        self,
        key: ImageCacheKey,
        producer: Callable[[], bytes],
        extension: str = ".png",
    ) -> tuple[bytes, bool]:
        """Return bytes and whether ``producer`` created the derivative.

        Concurrent callers in the same process execute ``producer`` once for a
        missing key. This prevents duplicate decoding and image conversion.
        """

        lock = _write_lock(key.digest)
        with lock:
            cached = self.read_bytes(key, extension)
            if cached is not None:
                return cached, False
            payload = producer()
            if not isinstance(payload, bytes):
                raise TypeError("producer must return bytes")
            if not payload:
                raise ValueError("producer must not return empty bytes")
            self.write_bytes(key, payload, extension)
            return payload, True

    def _record_write(self, payload_size: int) -> None:
        if self.max_bytes is None:
            return
        with _PRUNE_LOCK:
            if self._known_total_bytes is None:
                self._known_total_bytes = self._measure_size()
            else:
                self._known_total_bytes += payload_size
            if self._known_total_bytes > self.max_bytes:
                self.prune()

    def _measure_size(self) -> int:
        total = 0
        if not self.root.exists():
            return total
        for path in self.root.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def prune(self, max_bytes: int | None = None) -> int:
        """Remove least-recently-used derivatives until the budget is met."""

        budget = self.max_bytes if max_bytes is None else max_bytes
        if budget is None:
            return 0
        if budget <= 0:
            raise ValueError("max_bytes must be positive")

        with _PRUNE_LOCK:
            entries: list[tuple[int, int, Path]] = []
            total = 0
            if not self.root.exists():
                return 0
            for path in self.root.rglob("*"):
                if not path.is_file() or path.name.startswith("."):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                total += stat.st_size
                entries.append((stat.st_mtime_ns, stat.st_size, path))

            removed = 0
            for _, size, path in sorted(entries):
                if total <= budget:
                    break
                try:
                    path.unlink()
                except OSError:
                    continue
                total -= size
                removed += 1
            self._known_total_bytes = total
            return removed
