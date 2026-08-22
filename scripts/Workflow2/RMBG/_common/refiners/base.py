"""Lifecycle contract for optional mask refiners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..adapters.base import AdapterLoadContext, AdapterState


class MaskRefiner(ABC):
    """Refine an existing normalized mask using the corresponding RGB image."""

    refiner_id: str

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

    def refine(
        self,
        image_rgb: np.ndarray,
        initial_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._state != AdapterState.LOADED:
            raise RuntimeError(f"Refiner '{self.refiner_id}' is not loaded.")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("SDMatte ожидает RGB-изображение HWC.")
        if initial_mask.ndim != 2 or initial_mask.shape != image_rgb.shape[:2]:
            raise ValueError("Размер исходной маски не совпадает с изображением.")
        return self._refine(image_rgb, initial_mask)

    def unload(self) -> None:
        if self._state == AdapterState.LOADED:
            self._unload()
        self._state = AdapterState.UNLOADED

    @abstractmethod
    def _load(self, context: AdapterLoadContext) -> None:
        pass

    @abstractmethod
    def _refine(
        self,
        image_rgb: np.ndarray,
        initial_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        pass

    @abstractmethod
    def _unload(self) -> None:
        pass


class RefinerDependencyError(RuntimeError):
    """Raised when an optional ML runtime required by a refiner is missing."""
