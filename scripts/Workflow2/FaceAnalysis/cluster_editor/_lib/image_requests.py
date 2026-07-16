"""Request builders for cluster-editor image derivatives."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pysm_lib.pysm_image_cache import ImageRequest, QtImageCache


def normalized_face_crop(
    source_size: tuple[int, int],
    bbox: Sequence[float],
    *,
    padding: float,
) -> tuple[float, float, float, float] | None:
    """Build a square normalized crop with a stable face-to-frame ratio.

    The square is shifted back inside the source instead of being clipped at
    image edges. Therefore equal padding produces equal apparent face sizes
    for regular clusters, Noise and rotated/edge cases.
    """

    if len(bbox) != 4:
        return None
    source_width, source_height = map(int, source_size)
    if source_width <= 0 or source_height <= 0:
        return None

    x1, y1, x2, y2 = map(int, bbox)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0, min(source_width, x1))
    x2 = max(0, min(source_width, x2))
    y1 = max(0, min(source_height, y1))
    y2 = max(0, min(source_height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    face_width = x2 - x1
    face_height = y2 - y1
    padding = max(0.0, float(padding))
    requested_side = max(face_width, face_height) * (1.0 + 2.0 * padding)
    crop_side = min(
        source_width,
        source_height,
        max(1, round(requested_side)),
    )
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = round(center_x - crop_side / 2.0)
    crop_y1 = round(center_y - crop_side / 2.0)
    crop_x1 = max(0, min(source_width - crop_side, crop_x1))
    crop_y1 = max(0, min(source_height - crop_side, crop_y1))
    if crop_side <= 0:
        return None
    return (
        crop_x1 / source_width,
        crop_y1 / source_height,
        crop_side / source_width,
        crop_side / source_height,
    )


def face_thumbnail_request(
    cache: QtImageCache,
    source: Path,
    bbox: Sequence[float],
    target_size: tuple[int, int],
    *,
    padding: float,
    variant: str,
    source_size: tuple[int, int] | None = None,
) -> ImageRequest | None:
    """Build a shared-cache request for a cropped face thumbnail."""

    size = source_size or cache.source_size(source)
    crop = normalized_face_crop(size, bbox, padding=padding)
    if crop is None:
        return None
    return ImageRequest(
        source,
        target_size,
        mode="fit",
        crop=crop,
        allow_upscale=True,
        variant=variant,
    )
