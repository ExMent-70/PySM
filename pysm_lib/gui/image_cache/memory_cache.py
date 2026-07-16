"""Thread-safe byte-bounded LRU cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Generic, Hashable, TypeVar


KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Snapshot of memory-cache usage and counters."""

    item_count: int
    current_bytes: int
    max_bytes: int
    hits: int
    misses: int
    evictions: int


class ByteLRUCache(Generic[KeyT, ValueT]):
    """Store most recently used values within a strict byte budget.

    Value sizes are supplied by the caller because Python object size does not
    reflect the native memory occupied by ``QImage`` and similar objects.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._max_bytes = max_bytes
        self._items: OrderedDict[KeyT, tuple[ValueT, int]] = OrderedDict()
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    def get(self, key: KeyT) -> ValueT | None:
        """Return a value and promote it to the most-recent position."""

        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return entry[0]

    def put(self, key: KeyT, value: ValueT, size_bytes: int) -> bool:
        """Insert a value, returning ``False`` when it exceeds the budget."""

        if size_bytes < 0:
            raise ValueError("size_bytes must not be negative")

        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._current_bytes -= previous[1]

            if size_bytes > self._max_bytes:
                return False

            self._items[key] = (value, size_bytes)
            self._current_bytes += size_bytes
            while self._current_bytes > self._max_bytes:
                _, (_, evicted_size) = self._items.popitem(last=False)
                self._current_bytes -= evicted_size
                self._evictions += 1
            return True

    def discard(self, key: KeyT) -> bool:
        """Remove one entry if present."""

        with self._lock:
            entry = self._items.pop(key, None)
            if entry is None:
                return False
            self._current_bytes -= entry[1]
            return True

    def clear(self) -> None:
        """Remove all values while preserving diagnostic counters."""

        with self._lock:
            self._items.clear()
            self._current_bytes = 0

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                item_count=len(self._items),
                current_bytes=self._current_bytes,
                max_bytes=self._max_bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )
