"""Canonical model output used by all background-removal adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SegmentInstance:
    """One optional detected instance associated with the merged mask."""

    label: str
    score: float
    mask: np.ndarray

    def __post_init__(self) -> None:
        _validate_mask(self.mask, field_name="instance mask")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Instance score must be within [0, 1].")


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Normalized float32 foreground mask and model execution metadata."""

    mask: np.ndarray
    source_size: tuple[int, int]
    model_id: str
    instances: tuple[SegmentInstance, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_mask(self.mask, field_name="mask")
        width, height = self.source_size
        if width <= 0 or height <= 0:
            raise ValueError("source_size must contain positive width and height.")
        if self.mask.shape != (height, width):
            raise ValueError(
                "Mask shape must match source_size: "
                f"mask={self.mask.shape}, source_size={self.source_size}."
            )
        if not self.model_id:
            raise ValueError("model_id must not be empty.")


def _validate_mask(mask: np.ndarray, *, field_name: str) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{field_name} must be a numpy.ndarray.")
    if mask.ndim != 2:
        raise ValueError(f"{field_name} must be a two-dimensional array.")
    if mask.dtype != np.float32:
        raise TypeError(f"{field_name} must use float32 dtype.")
    if not np.isfinite(mask).all():
        raise ValueError(f"{field_name} contains NaN or infinity.")
    if mask.size and (float(mask.min()) < 0.0 or float(mask.max()) > 1.0):
        raise ValueError(f"{field_name} values must be within [0, 1].")
