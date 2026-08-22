"""Build and atomically save RMBG output artifacts."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageColor, ImageOps

from .config_schema import (
    BackgroundFitMode,
    BackgroundMode,
    BackgroundPosition,
    ImageFormat,
    OutputConfig,
)


class ArtifactConflictError(RuntimeError):
    """Raised when an existing partial result set makes resuming ambiguous."""


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Optional paths enabled for one source image."""

    cutout: Path | None
    mask: Path | None
    composite: Path | None

    def enabled(self) -> tuple[Path, ...]:
        return tuple(path for path in (self.cutout, self.mask, self.composite) if path)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "cutout": str(self.cutout) if self.cutout else None,
            "mask": str(self.mask) if self.mask else None,
            "composite": str(self.composite) if self.composite else None,
        }


def build_artifact_paths(
    source_path: Path,
    input_root: Path,
    output_root: Path,
    config: OutputConfig,
) -> ArtifactPaths:
    """Map one input path to stable output folders while preserving subfolders."""

    relative = source_path.resolve().relative_to(input_root.resolve())
    relative_parent = relative.parent
    image_extension = f".{config.image_format.value}"
    return ArtifactPaths(
        cutout=(
            output_root / "Cutout" / relative_parent
            / f"{relative.stem}{config.image_suffix}{image_extension}"
            if config.save_cutout
            else None
        ),
        mask=(
            output_root / "Masks" / relative_parent
            / f"{relative.stem}{config.mask_suffix}.png"
            if config.save_mask
            else None
        ),
        composite=(
            output_root / "Composite" / relative_parent
            / f"{relative.stem}{config.composite_suffix}{image_extension}"
            if config.save_composite
            else None
        ),
    )


def existing_artifact_state(paths: ArtifactPaths, *, overwrite: bool) -> str:
    """Return ``write`` or ``skip`` and reject unsafe partial result sets."""

    enabled = paths.enabled()
    existing = tuple(path for path in enabled if path.exists())
    if overwrite or not existing:
        return "write"
    if len(existing) == len(enabled):
        return "skip"
    existing_text = "\n- ".join(str(path) for path in existing)
    missing_text = "\n- ".join(str(path) for path in enabled if not path.exists())
    raise ArtifactConflictError(
        "Обнаружен неполный набор результатов. Во избежание смешивания запусков "
        "используйте --overwrite или удалите этот набор вручную."
        f"\nСуществуют:\n- {existing_text}\nОтсутствуют:\n- {missing_text}"
    )


def save_artifacts(
    source: Image.Image,
    mask: np.ndarray,
    paths: ArtifactPaths,
    config: OutputConfig,
    *,
    background: Image.Image | None = None,
) -> None:
    """Render enabled outputs and replace each destination atomically."""

    if mask.shape != (source.height, source.width):
        raise ValueError("Размер маски не совпадает с размером исходного изображения.")
    alpha = Image.fromarray(_mask_to_uint8(mask), mode="L")
    metadata = _source_metadata(source)
    staged: list[tuple[Path, Path]] = []
    try:
        if paths.mask:
            mask_image = Image.fromarray(_mask_to_uint16(mask))
            staged.append(
                _stage_image(
                    mask_image,
                    paths.mask,
                    {"dpi": metadata.get("dpi")},
                    png_compress_level=config.png_compress_level,
                    jpeg_quality=config.jpeg_quality,
                )
            )
        if paths.cutout:
            cutout = source.convert("RGBA")
            cutout.putalpha(alpha)
            if config.image_format == ImageFormat.JPEG:
                cutout = _flatten_on_color(cutout, config.background_color)
            staged.append(
                _stage_image(
                    cutout,
                    paths.cutout,
                    metadata,
                    png_compress_level=config.png_compress_level,
                    jpeg_quality=config.jpeg_quality,
                )
            )
        if paths.composite:
            composite = _build_composite(source, alpha, config, background)
            staged.append(
                _stage_image(
                    composite,
                    paths.composite,
                    metadata,
                    png_compress_level=config.png_compress_level,
                    jpeg_quality=config.jpeg_quality,
                )
            )
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def assert_unique_artifact_paths(items: Iterable[ArtifactPaths]) -> None:
    """Reject same-stem inputs that would overwrite each other's outputs."""

    seen: dict[Path, int] = {}
    duplicates: set[Path] = set()
    for paths in items:
        for path in paths.enabled():
            normalized = Path(str(path).casefold())
            seen[normalized] = seen.get(normalized, 0) + 1
            if seen[normalized] > 1:
                duplicates.add(path)
    if duplicates:
        formatted = "\n- ".join(str(path) for path in sorted(duplicates))
        raise ArtifactConflictError(
            "Несколько исходных файлов формируют одинаковые пути результатов:"
            f"\n- {formatted}"
        )


def _build_composite(
    source: Image.Image,
    alpha: Image.Image,
    config: OutputConfig,
    background: Image.Image | None,
) -> Image.Image:
    foreground = source.convert("RGBA")
    foreground.putalpha(alpha)
    if config.background_mode == BackgroundMode.SOLID:
        rgb = ImageColor.getrgb(config.background_color)
        backdrop = Image.new("RGBA", source.size, (*rgb, 255))
    elif config.background_mode == BackgroundMode.IMAGE:
        if background is None:
            raise ValueError("Для composite не передано фоновое изображение.")
        backdrop = _prepare_background(background, source.size, config)
    else:
        raise ValueError(
            f"Режим {config.background_mode.value!r} не формирует composite."
        )
    return Image.alpha_composite(backdrop, foreground).convert("RGB")


def _prepare_background(
    background: Image.Image,
    target_size: tuple[int, int],
    config: OutputConfig,
) -> Image.Image:
    """Resize the selected background according to the configured placement."""

    prepared = ImageOps.exif_transpose(background).convert("RGBA")
    if config.background_fit == BackgroundFitMode.COVER:
        centering = {
            BackgroundPosition.CENTER: (0.5, 0.5),
            BackgroundPosition.TOP: (0.5, 0.0),
            BackgroundPosition.BOTTOM: (0.5, 1.0),
            BackgroundPosition.LEFT: (0.0, 0.5),
            BackgroundPosition.RIGHT: (1.0, 0.5),
        }[config.background_position]
        return ImageOps.fit(
            prepared,
            target_size,
            method=Image.Resampling.LANCZOS,
            centering=centering,
        )
    if config.background_fit == BackgroundFitMode.CONTAIN:
        fitted = ImageOps.contain(
            prepared,
            target_size,
            method=Image.Resampling.LANCZOS,
        )
        rgb = ImageColor.getrgb(config.background_color)
        canvas = Image.new("RGBA", target_size, (*rgb, 255))
        offset = (
            (target_size[0] - fitted.width) // 2,
            (target_size[1] - fitted.height) // 2,
        )
        canvas.alpha_composite(fitted, dest=offset)
        return canvas
    if config.background_fit == BackgroundFitMode.STRETCH:
        return prepared.resize(target_size, Image.Resampling.LANCZOS)
    raise ValueError(
        f"Неизвестный режим размещения фона: {config.background_fit.value!r}."
    )


def _stage_image(
    image: Image.Image,
    destination: Path,
    metadata: dict[str, object],
    *,
    png_compress_level: int,
    jpeg_quality: int,
) -> tuple[Path, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    save_options = {key: value for key, value in metadata.items() if value is not None}
    if destination.suffix.casefold() == ".webp":
        save_options.update(lossless=True, quality=100, method=6)
        save_options.pop("dpi", None)
    elif destination.suffix.casefold() == ".png":
        save_options["compress_level"] = png_compress_level
    elif destination.suffix.casefold() in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
        save_options.update(quality=jpeg_quality, subsampling=0)
    try:
        image.save(temporary, **save_options)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary, destination


def _flatten_on_color(image: Image.Image, color: str) -> Image.Image:
    """Return an opaque RGB image using ``color`` below its alpha channel."""

    background = Image.new("RGBA", image.size, (*ImageColor.getrgb(color), 255))
    return Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")


def _source_metadata(source: Image.Image) -> dict[str, object]:
    return {
        "icc_profile": source.info.get("icc_profile"),
        "dpi": source.info.get("dpi"),
    }


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    normalized = np.clip(mask, 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


def _mask_to_uint16(mask: np.ndarray) -> np.ndarray:
    normalized = np.clip(mask, 0.0, 1.0)
    return np.rint(normalized * 65535.0).astype(np.uint16)
