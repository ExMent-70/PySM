"""Apply each mask parameter exactly once in a deterministic order."""

from __future__ import annotations

import cv2
import numpy as np

from .config_schema import MaskConfig, RefinementMode


def postprocess_mask(
    mask: np.ndarray,
    config: MaskConfig,
    *,
    refinement: RefinementMode | None = None,
    sdmatte_applied: bool = False,
) -> np.ndarray:
    """Return a float32 alpha mask after cleanup, morphology and feathering."""

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Маска должна быть двумерным numpy-массивом.")
    result = np.array(mask, dtype=np.float32, order="C", copy=True)
    np.nan_to_num(result, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(result, 0.0, 1.0, out=result)

    effective = refinement or config.refinement
    if effective == RefinementMode.AUTO:
        effective = RefinementMode.FAST
    if effective == RefinementMode.FAST:
        result = fast_refine_mask(result)
    elif effective == RefinementMode.SDMATTE and not sdmatte_applied:
        raise RuntimeError(
            "SDMatte выбран, но его результат не был передан в postprocess_mask."
        )
    elif effective not in {RefinementMode.NONE, RefinementMode.SDMATTE}:
        raise ValueError(f"Неизвестный режим refinement: {effective!r}")

    if config.sensitivity < 1.0:
        result *= 2.0 - config.sensitivity
        np.clip(result, 0.0, 1.0, out=result)

    if config.blur > 0:
        result = cv2.GaussianBlur(
            result,
            (0, 0),
            sigmaX=float(config.blur),
            sigmaY=float(config.blur),
            borderType=cv2.BORDER_REPLICATE,
        )

    if config.offset:
        radius = abs(config.offset)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        operation = cv2.MORPH_DILATE if config.offset > 0 else cv2.MORPH_ERODE
        result = cv2.morphologyEx(result, operation, kernel)

    if config.remove_small_regions and config.min_region_area > 0:
        result = _remove_small_foreground(result, config.min_region_area)
    if config.fill_holes:
        result = _fill_enclosed_holes(result, config.max_hole_area)

    if config.feather > 0:
        result = cv2.GaussianBlur(
            result,
            (0, 0),
            sigmaX=float(config.feather),
            sigmaY=float(config.feather),
            borderType=cv2.BORDER_REPLICATE,
        )

    if config.invert:
        result = 1.0 - result
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def fast_refine_mask(mask: np.ndarray) -> np.ndarray:
    """Sharpen confident areas while preserving soft transition pixels."""

    source = np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)
    binary = (source > 0.45).astype(np.float32)
    edge_blur = cv2.GaussianBlur(
        binary,
        (3, 3),
        0,
        borderType=cv2.BORDER_REPLICATE,
    )
    transition = (source > 0.05) & (source < 0.95)
    refined = np.where(transition, 0.85 * source + 0.15 * edge_blur, binary)
    edge = (source > 0.2) & (source < 0.8)
    refined = np.where(edge, refined * 0.98, refined)
    return np.ascontiguousarray(np.clip(refined, 0.0, 1.0), dtype=np.float32)


def _remove_small_foreground(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask >= 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = mask.copy()
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_area:
            output[labels == label] = 0.0
    return output


def _fill_enclosed_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    background = (mask < 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    output = mask.copy()
    height, width = mask.shape
    border_labels = set(labels[0, :]) | set(labels[-1, :])
    border_labels |= set(labels[:, 0]) | set(labels[:, -1])
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if label not in border_labels and (max_area == 0 or area <= max_area):
            output[labels == label] = 1.0
    return output
