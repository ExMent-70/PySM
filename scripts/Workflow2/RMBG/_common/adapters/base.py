"""Stable lifecycle and inference interface for all RMBG model families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config_schema import PrecisionName
from ..results import SegmentationResult


class AdapterState(str, Enum):
    CREATED = "created"
    LOADED = "loaded"
    UNLOADED = "unloaded"


@dataclass(frozen=True, slots=True)
class AdapterLoadContext:
    device: str
    precision: PrecisionName
    model_cache_dir: Path
    process_resolution: int = 1024
    local_files_only: bool = False


class ModelAdapter(ABC):
    """Base adapter that prevents inference before successful model loading."""

    model_id: str

    def __init__(self) -> None:
        self._state = AdapterState.CREATED

    @property
    def state(self) -> AdapterState:
        return self._state

    def load(self, context: AdapterLoadContext) -> None:
        if self._state == AdapterState.LOADED:
            return
        self._load(context)
        self._state = AdapterState.LOADED

    def infer(self, image_rgb: np.ndarray) -> SegmentationResult:
        if self._state != AdapterState.LOADED:
            raise RuntimeError(f"Adapter '{self.model_id}' is not loaded.")
        self._validate_image(image_rgb)
        return self._infer(image_rgb)

    def infer_batch(
        self,
        images_rgb: Sequence[np.ndarray],
    ) -> tuple[SegmentationResult, ...]:
        """Infer one bounded batch, with a sequential fallback for adapters."""

        if self._state != AdapterState.LOADED:
            raise RuntimeError(f"Adapter '{self.model_id}' is not loaded.")
        images = tuple(images_rgb)
        if not images:
            return ()
        for image_rgb in images:
            self._validate_image(image_rgb)
        return self._infer_batch(images)

    @staticmethod
    def _validate_image(image_rgb: np.ndarray) -> None:
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError("image_rgb must be a numpy.ndarray.")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have HWC shape with three RGB channels.")
        if image_rgb.dtype != np.uint8:
            raise TypeError("image_rgb must use uint8 dtype.")

    def unload(self) -> None:
        if self._state == AdapterState.LOADED:
            self._unload()
        self._state = AdapterState.UNLOADED

    @abstractmethod
    def _load(self, context: AdapterLoadContext) -> None:
        """Load weights and initialize inference resources."""

    @abstractmethod
    def _infer(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Return one normalized foreground mask for one RGB image."""

    def _infer_batch(
        self,
        images_rgb: tuple[np.ndarray, ...],
    ) -> tuple[SegmentationResult, ...]:
        """Default batch contract for adapters without native batching."""

        return tuple(self._infer(image_rgb) for image_rgb in images_rgb)

    @abstractmethod
    def _unload(self) -> None:
        """Release CPU and GPU resources owned by this adapter."""
