"""Shared-memory backend for PySM runtime context.

The segment stores one UTF-8 JSON payload behind a small validated header. A
Windows named mutex serializes readers and writers across PySM and script
processes.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import struct
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Any, Dict, Iterator, Optional

from .context_store import ContextStoreError


class SharedMemoryContextError(ContextStoreError):
    """Raised when shared-memory context cannot be read or written."""


MAGIC = b"PYSMCTX1"
SCHEMA_VERSION = 1
STATE_CLEAN = 0
STATE_WRITING = 1
_HEADER = struct.Struct("<8sIIQQQII")
HEADER_SIZE = _HEADER.size
DEFAULT_MUTEX_TIMEOUT_MS = 30_000


def make_context_shm_name(prefix: str = "pysm_context") -> str:
    return f"{prefix}_{os.getpid()}_{uuid.uuid4().hex}"


def calculate_segment_size(
    data: Dict[str, Any],
    min_size_mb: int,
    max_size_mb: int,
) -> int:
    min_size = max(1, int(min_size_mb)) * 1024 * 1024
    max_size = max(min_size, int(max_size_mb) * 1024 * 1024)
    payload = _encode_payload(data)
    required = HEADER_SIZE + len(payload)
    size = max(min_size, required * 2)
    if size > max_size:
        raise SharedMemoryContextError(
            f"Initial context requires {required} bytes, max shared memory is {max_size} bytes."
        )
    return size


def encoded_payload_size(data: Dict[str, Any]) -> int:
    return len(_encode_payload(data))


def _encode_payload(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _mutex_name_for_shm(shm_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", shm_name)
    return f"Local\\PySM_Context_{safe}"


class _WindowsNamedMutex:
    def __init__(self, name: str, timeout_ms: int = DEFAULT_MUTEX_TIMEOUT_MS):
        self.name = name
        self.timeout_ms = timeout_ms
        self._handle: Optional[int] = None
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateMutexW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_wchar_p,
        )
        self._kernel32.CreateMutexW.restype = ctypes.c_void_p
        self._kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        self._kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        self._kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
        self._kernel32.ReleaseMutex.restype = ctypes.c_bool
        self._kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        self._kernel32.CloseHandle.restype = ctypes.c_bool

    def __enter__(self):
        kernel32 = self._kernel32
        handle = kernel32.CreateMutexW(None, False, self.name)
        if not handle:
            raise SharedMemoryContextError(f"CreateMutexW failed: {ctypes.get_last_error()}")

        result = kernel32.WaitForSingleObject(handle, self.timeout_ms)
        wait_object_0 = 0
        wait_abandoned = 0x00000080
        wait_timeout = 0x00000102
        if result == wait_timeout:
            kernel32.CloseHandle(handle)
            raise SharedMemoryContextError(f"Timed out waiting for mutex {self.name!r}.")
        if result not in (wait_object_0, wait_abandoned):
            err = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise SharedMemoryContextError(f"WaitForSingleObject failed: {err}")

        self._handle = handle
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle:
            self._kernel32.ReleaseMutex(self._handle)
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


_FALLBACK_LOCKS: Dict[str, threading.RLock] = {}


@contextmanager
def _named_mutex(name: str, timeout_ms: int = DEFAULT_MUTEX_TIMEOUT_MS) -> Iterator[None]:
    if os.name == "nt":
        with _WindowsNamedMutex(name, timeout_ms):
            yield
        return

    lock = _FALLBACK_LOCKS.setdefault(name, threading.RLock())
    acquired = lock.acquire(timeout=timeout_ms / 1000)
    if not acquired:
        raise SharedMemoryContextError(f"Timed out waiting for lock {name!r}.")
    try:
        yield
    finally:
        lock.release()


class SharedMemoryContextStore:
    backend_name = "shared_memory"

    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        *,
        owner: bool = False,
        mutex_timeout_ms: int = DEFAULT_MUTEX_TIMEOUT_MS,
    ):
        self._shm = shm
        self._owner = owner
        self._mutex_name = _mutex_name_for_shm(shm.name)
        self._mutex_timeout_ms = mutex_timeout_ms

    @classmethod
    def create(
        cls,
        name: str,
        size: int,
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> "SharedMemoryContextStore":
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        store = cls(shm, owner=True)
        store.save(initial_data or {})
        return store

    @classmethod
    def open(cls, name: str) -> "SharedMemoryContextStore":
        shm = shared_memory.SharedMemory(name=name, create=False)
        return cls(shm, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def capacity(self) -> int:
        return len(self._shm.buf) - HEADER_SIZE

    @property
    def payload_size(self) -> int:
        try:
            return int(self._read_header()["payload_length"])
        except SharedMemoryContextError:
            return 0

    @property
    def usage_ratio(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return self.payload_size / self.capacity

    @property
    def generation(self) -> int:
        try:
            header = self._read_header()
            return int(header["generation"])
        except SharedMemoryContextError:
            return 0

    def load(self) -> Dict[str, Any]:
        with _named_mutex(self._mutex_name, self._mutex_timeout_ms):
            for _ in range(3):
                header = self._read_header()
                if header["state"] == STATE_CLEAN:
                    break
                time.sleep(0.01)
            else:
                raise SharedMemoryContextError("Context segment is still marked as writing.")

            payload_length = header["payload_length"]
            payload = bytes(self._shm.buf[HEADER_SIZE : HEADER_SIZE + payload_length])
            checksum = zlib.crc32(payload) & 0xFFFFFFFF
            if checksum != header["checksum"]:
                raise SharedMemoryContextError("Context payload checksum mismatch.")

            if not payload:
                return {}
            data = json.loads(payload.decode("utf-8"))
            if not isinstance(data, dict):
                raise SharedMemoryContextError("Context payload must be a JSON object.")
            return data

    def save(self, data: Dict[str, Any]) -> None:
        payload = _encode_payload(data)
        if len(payload) > self.capacity:
            raise SharedMemoryContextError(
                f"Context payload is {len(payload)} bytes, shared memory capacity is {self.capacity} bytes."
            )

        with _named_mutex(self._mutex_name, self._mutex_timeout_ms):
            previous_generation = 0
            try:
                previous_generation = int(self._read_header()["generation"])
            except SharedMemoryContextError:
                previous_generation = 0

            writing_header = self._pack_header(
                payload_length=0,
                generation=previous_generation,
                state=STATE_WRITING,
                checksum=0,
            )
            self._shm.buf[:HEADER_SIZE] = writing_header
            self._shm.buf[HEADER_SIZE : HEADER_SIZE + len(payload)] = payload

            clean_header = self._pack_header(
                payload_length=len(payload),
                generation=previous_generation + 1,
                state=STATE_CLEAN,
                checksum=zlib.crc32(payload) & 0xFFFFFFFF,
            )
            self._shm.buf[:HEADER_SIZE] = clean_header

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if not self._owner:
            return
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass

    def _pack_header(
        self,
        *,
        payload_length: int,
        generation: int,
        state: int,
        checksum: int,
    ) -> bytes:
        capacity = self.capacity
        return _HEADER.pack(
            MAGIC,
            HEADER_SIZE,
            SCHEMA_VERSION,
            capacity,
            payload_length,
            generation,
            state,
            checksum,
        )

    def _read_header(self) -> Dict[str, int]:
        raw = bytes(self._shm.buf[:HEADER_SIZE])
        magic, header_size, version, capacity, payload_length, generation, state, checksum = _HEADER.unpack(raw)
        if magic != MAGIC or header_size != HEADER_SIZE or version != SCHEMA_VERSION:
            raise SharedMemoryContextError("Invalid shared-memory context header.")
        if capacity != self.capacity:
            raise SharedMemoryContextError("Shared-memory context capacity mismatch.")
        if payload_length > capacity:
            raise SharedMemoryContextError("Shared-memory context payload length is invalid.")
        return {
            "capacity": capacity,
            "payload_length": payload_length,
            "generation": generation,
            "state": state,
            "checksum": checksum,
        }
