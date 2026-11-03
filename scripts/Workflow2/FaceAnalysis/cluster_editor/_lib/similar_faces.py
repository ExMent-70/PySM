"""Read-only cosine search against fixed, manually confirmed face samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np

from _common._shared import EmbeddingLoader


@dataclass(frozen=True)
class FaceAddress:
    """Keep the current list position separate from the persisted vector slot."""

    filename: str
    position: int
    slot: int


@dataclass(frozen=True)
class SimilarFace:
    """A candidate's original address and its distance to the nearest sample."""

    address: FaceAddress
    distance: float


class SearchCancelled(Exception):
    """The user cancelled a read-only search."""


def check_cancelled(cancel: Event) -> None:
    """Stop between I/O and bounded comparison blocks when cancellation is set."""

    if cancel.is_set():
        raise SearchCancelled()


def face_address(filename: str, position: int, face) -> FaceAddress:
    """Resolve the original embedding slot without confusing it with UI order."""

    slot = face.face_index
    if slot is None and face.embedding_key:
        prefix = filename + "::"
        if face.embedding_key.startswith(prefix):
            slot = int(face.embedding_key[len(prefix):])
        elif face.embedding_key == filename:
            slot = 0
        else:
            raise ValueError(f"Некорректный ключ эмбеддинга: {face.embedding_key}")
    if slot is None:
        slot = position
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        raise ValueError(f"Некорректный индекс лица: {filename}::{slot}")
    return FaceAddress(filename, position, slot)


class FaceEmbeddingIndex:
    """Cache validated working-session vectors; never write embedding files."""

    def __init__(self) -> None:
        self._signature = None
        self._vectors = None
        self._index = None

    def load(self, directory: Path, cancel: Event) -> None:
        check_cancelled(cancel)
        paths = [directory / "faces_embeddings.npy", directory / "faces_index.json"]
        try:
            metadata = []
            for path in paths:
                stat = path.stat()
                metadata.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
            signature = tuple(metadata)
        except OSError as exc:
            raise ValueError("Не найдены файлы эмбеддингов рабочей фотосессии.") from exc
        if signature == self._signature:
            return
        vectors, index = EmbeddingLoader(directory).load("faces")
        check_cancelled(cancel)
        if (
            not isinstance(vectors, np.ndarray)
            or vectors.ndim != 2
            or vectors.shape[1] == 0
            or not np.issubdtype(vectors.dtype, np.number)
            or np.iscomplexobj(vectors)
            or not isinstance(index, dict)
        ):
            raise ValueError("Некорректные эмбеддинги или индекс рабочей фотосессии.")
        self._vectors, self._index = vectors, index
        self._signature = signature

    def vector(self, address: FaceAddress) -> np.ndarray | None:
        """Skip missing, nonfinite and zero vectors instead of guessing indices."""

        indices = self._index.get(address.filename)
        if not isinstance(indices, list) or not 0 <= address.slot < len(indices):
            return None
        row = indices[address.slot]
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(self._vectors):
            return None
        vector = np.asarray(self._vectors[row], dtype=np.float64)
        if not np.isfinite(vector).all():
            return None
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            return None
        return vector / norm

    def search(
        self,
        seeds: tuple[FaceAddress, ...],
        candidates: tuple[FaceAddress, ...],
        threshold: float,
        cancel: Event,
        progress: Callable[[int], None] = lambda value: None,
    ) -> tuple[list[SimilarFace], int]:
        """Search fixed seeds in bounded blocks; candidates never become seeds."""

        if not np.isfinite(threshold) or not 0 <= threshold <= 2:
            raise ValueError("Косинусное расстояние должно быть от 0 до 2.")
        samples = []
        for address in seeds:
            check_cancelled(cancel)
            vector = self.vector(address)
            if vector is None:
                raise ValueError(f"Не найден корректный эмбеддинг выбранного лица: {address.filename}")
            samples.append(vector)
        if not samples:
            return [], 0
        seed_matrix = np.asarray(samples)
        found = {}
        skipped = 0
        for offset in range(0, len(candidates), 512):
            check_cancelled(cancel)
            addresses, vectors = [], []
            for address in candidates[offset:offset + 512]:
                vector = self.vector(address)
                if vector is None:
                    skipped += 1
                    continue
                addresses.append(address)
                vectors.append(vector)
            if vectors:
                # One seed at a time bounds memory even for large manual batches.
                matrix = np.asarray(vectors)
                distances = np.full(len(vectors), 2.0)
                for sample in seed_matrix:
                    check_cancelled(cancel)
                    distances = np.minimum(
                        distances, np.clip(1.0 - matrix @ sample, 0.0, 2.0)
                    )
                for address, distance in zip(addresses, distances):
                    if distance <= threshold:
                        found[address] = SimilarFace(address, float(distance))
            progress(round(100 * min(offset + 512, len(candidates)) / len(candidates)))
        check_cancelled(cancel)
        ordered = sorted(
            found.values(),
            key=lambda hit: (hit.distance, hit.address.filename, hit.address.position),
        )
        return ordered, skipped
