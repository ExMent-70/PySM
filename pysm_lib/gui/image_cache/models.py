"""Immutable request and cache-key models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Literal


ResizeMode = Literal["fit", "fill", "stretch"]


@dataclass(frozen=True, slots=True)
class ImageRequest:
    """Describe one deterministic derivative of a source image.

    ``variant`` and ``algorithm_version`` let consumers invalidate cached
    derivatives when UI-specific processing or the shared algorithm changes.
    """

    source: Path
    target_size: tuple[int, int]
    mode: ResizeMode = "fit"
    crop: tuple[float, float, float, float] | None = None
    auto_transform: bool = True
    allow_upscale: bool = False
    variant: str = "default"
    algorithm_version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        target_size = tuple(self.target_size)
        object.__setattr__(self, "target_size", target_size)
        width, height = target_size
        if width <= 0 or height <= 0:
            raise ValueError("target_size dimensions must be positive")
        if self.mode not in {"fit", "fill", "stretch"}:
            raise ValueError(f"unsupported resize mode: {self.mode}")
        if self.crop is not None:
            crop = tuple(self.crop)
            if len(crop) != 4:
                raise ValueError("crop must contain four coordinates")
            x, y, width, height = crop
            if not all(math.isfinite(value) for value in crop):
                raise ValueError("crop coordinates must be finite")
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError("crop must be a positive normalized rectangle")
            if x + width > 1 or y + height > 1:
                raise ValueError("crop rectangle must fit inside normalized image bounds")
            object.__setattr__(self, "crop", crop)
        if not self.variant:
            raise ValueError("variant must not be empty")
        if not self.algorithm_version:
            raise ValueError("algorithm_version must not be empty")


@dataclass(frozen=True, slots=True)
class ImageCacheKey:
    """Content-independent key that invalidates on source metadata changes.

    Reading the whole source merely to hash it would remove much of the cache's
    benefit. PySM therefore uses the resolved path, nanosecond modification
    time and file size as a fast source fingerprint.
    """

    source_path: str
    source_mtime_ns: int
    source_ctime_ns: int
    source_size: int
    target_size: tuple[int, int]
    mode: ResizeMode
    crop: tuple[float, float, float, float] | None
    auto_transform: bool
    allow_upscale: bool
    variant: str
    algorithm_version: str

    @classmethod
    def from_request(cls, request: ImageRequest) -> "ImageCacheKey":
        source = request.source.resolve(strict=True)
        stat = source.stat()
        return cls(
            source_path=os.path.normcase(str(source)),
            source_mtime_ns=stat.st_mtime_ns,
            source_ctime_ns=stat.st_ctime_ns,
            source_size=stat.st_size,
            target_size=request.target_size,
            mode=request.mode,
            crop=request.crop,
            auto_transform=request.auto_transform,
            allow_upscale=request.allow_upscale,
            variant=request.variant,
            algorithm_version=request.algorithm_version,
        )

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 identifier suitable for a file name."""

        payload = {
            "algorithm_version": self.algorithm_version,
            "allow_upscale": self.allow_upscale,
            "auto_transform": self.auto_transform,
            "crop": self.crop,
            "mode": self.mode,
            "source_ctime_ns": self.source_ctime_ns,
            "source_mtime_ns": self.source_mtime_ns,
            "source_path": self.source_path,
            "source_size": self.source_size,
            "target_size": self.target_size,
            "variant": self.variant,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
