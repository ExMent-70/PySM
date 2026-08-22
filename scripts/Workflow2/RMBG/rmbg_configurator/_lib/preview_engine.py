"""Generate cached RMBG test-mask sets from the current configurator settings."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from _common.adapters.base import AdapterLoadContext
from _common.adapters.local_birefnet import LocalBiRefNetAdapter
from _common.config_schema import RefinementMode, RmbgSettings
from _common.manifests import load_models_lock
from _common.mask_ops import fast_refine_mask, postprocess_mask
from _common.model_downloader import ensure_model_files, ensure_sdmatte_files
from _common.model_registry import create_model_registry
from _common.path_contract import resolve_model_dir_value
from _common.refiners.sdmatte import SDMatteRefiner, validate_sdmatte_runtime

from .preview_store import PreviewStore, TestMaskSet


ProgressCallback = Callable[[str, int, int], None]
FAST_REFINEMENT_CACHE_VERSION = 1


class PreviewGenerationCancelled(RuntimeError):
    """Raised after a cooperative cancellation request from the GUI."""


class _DownloadProgress:
    def __init__(
        self,
        *,
        total: int,
        desc: str,
        callback: ProgressCallback,
        cancel_event: threading.Event,
        **_kwargs: Any,
    ) -> None:
        self.total = max(1, int(total))
        self.value = 0
        self.desc = desc
        self.callback = callback
        self.cancel_event = cancel_event
        self.callback(self.desc, self.value, self.total)

    def update(self, amount: int = 1) -> None:
        if self.cancel_event.is_set():
            raise PreviewGenerationCancelled("Генерация тестового набора отменена.")
        self.value = min(self.total, self.value + int(amount))
        self.callback(self.desc, self.value, self.total)

    def set_description(self, desc: str, refresh: bool = False) -> None:
        del refresh
        self.desc = desc
        self.callback(self.desc, self.value, self.total)

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def close(self) -> None:
        return


def generate_mask_set(
    *,
    store: PreviewStore,
    session_id: str,
    set_id: str,
    settings: RmbgSettings,
    progress: ProgressCallback,
    cancel_event: threading.Event,
) -> TestMaskSet:
    """Create one immutable final set while reusing base/refined disk caches."""

    session = store.get_session(session_id)
    mask_set = store.get_set(session_id, set_id)
    registry = create_model_registry()
    model_store = resolve_model_dir_value(settings.model.model_dir)
    model_id = settings.resolved_model_name()
    descriptor = registry.get(model_id)
    resolution = settings.model.process_resolution or descriptor.default_resolution
    models_lock = load_models_lock()
    model_manifest = models_lock["models"][model_id.value]
    model_identity = {
        "model": model_id.value,
        "revision": model_manifest.get("revision"),
        "files": model_manifest.get("files"),
        "resolution": resolution,
        "device": settings.model.device.value,
        "precision": settings.model.precision.value,
    }
    progress_factory = lambda **kwargs: _DownloadProgress(
        callback=progress,
        cancel_event=cancel_event,
        **kwargs,
    )

    sources = tuple(
        (source, session.path / source.relative_path)
        for source in session.sources
    )
    base_entries = [
        (
            source,
            path,
            _cache_key({"source_sha256": source.sha256, **model_identity}),
        )
        for source, path in sources
    ]
    missing_base = [
        entry for entry in base_entries
        if not store.cache_path(session_id, "Base", entry[2]).is_file()
    ]

    adapter = None
    refiner = None
    try:
        if missing_base:
            _check_cancel(cancel_event)
            ensured = ensure_model_files(
                model_id,
                descriptor,
                model_store,
                progress_factory=progress_factory,
            )
            adapter = LocalBiRefNetAdapter(model_id, ensured.files)
            progress(f"Загрузка {descriptor.display_name}", 0, 1)
            adapter.load(
                AdapterLoadContext(
                    device=settings.model.device.value,
                    precision=settings.model.precision,
                    model_cache_dir=model_store,
                    process_resolution=resolution,
                    local_files_only=True,
                )
            )
            progress(f"Загрузка {descriptor.display_name}", 1, 1)
            for index, (_source, path, key) in enumerate(missing_base, start=1):
                _check_cancel(cancel_event)
                progress(f"Основная маска: {path.name}", index - 1, len(missing_base))
                image_rgb = _load_rgb(path)
                result = adapter.infer(image_rgb)
                _save_array_atomic(
                    store.cache_path(session_id, "Base", key),
                    result.mask,
                )
                progress(f"Основная маска: {path.name}", index, len(missing_base))

        effective_refinement = settings.resolved_refinement()
        refined_manifest = models_lock.get("refiners", {}).get("sdmatte", {})
        refined_entries: list[tuple[Any, Path, str, str | None]] = []
        for source, path, base_key in base_entries:
            refined_key = _refined_key(
                base_key,
                settings,
                refined_manifest,
            )
            refined_entries.append((source, path, base_key, refined_key))

        if effective_refinement == RefinementMode.SDMATTE:
            missing_refined = [
                entry for entry in refined_entries
                if entry[3] is not None
                and not store.cache_path(session_id, "Refined", entry[3]).is_file()
            ]
            if missing_refined:
                _check_cancel(cancel_event)
                validate_sdmatte_runtime(
                    requested_device=settings.model.device.value,
                    requested_precision=settings.model.precision,
                )
                ensured_refiner = ensure_sdmatte_files(
                    settings.mask.sdmatte_variant,
                    model_store,
                    progress_factory=progress_factory,
                )
                refiner = SDMatteRefiner(
                    model_root=ensured_refiner.files.model_dir,
                    weights=ensured_refiner.files.weights,
                    transparent_object=settings.mask.sdmatte_transparent_object,
                    constraint=settings.mask.sdmatte_constraint,
                )
                progress("Загрузка SDMatte", 0, 1)
                refiner.load(
                    AdapterLoadContext(
                        device=settings.model.device.value,
                        precision=settings.model.precision,
                        model_cache_dir=model_store,
                        process_resolution=settings.mask.sdmatte_resolution,
                        local_files_only=True,
                    )
                )
                progress("Загрузка SDMatte", 1, 1)
                for index, (_source, path, base_key, refined_key) in enumerate(
                    missing_refined,
                    start=1,
                ):
                    _check_cancel(cancel_event)
                    progress(f"SDMatte: {path.name}", index - 1, len(missing_refined))
                    image_rgb = _load_rgb(path)
                    base_mask = _load_array(
                        store.cache_path(session_id, "Base", base_key)
                    )
                    refined_mask, _metadata = refiner.refine(image_rgb, base_mask)
                    assert refined_key is not None
                    _save_array_atomic(
                        store.cache_path(session_id, "Refined", refined_key),
                        refined_mask,
                    )
                    progress(f"SDMatte: {path.name}", index, len(missing_refined))
        elif effective_refinement == RefinementMode.FAST:
            for _source, _path, base_key, refined_key in refined_entries:
                assert refined_key is not None
                destination = store.cache_path(session_id, "Refined", refined_key)
                if destination.is_file():
                    continue
                _check_cancel(cancel_event)
                base_mask = _load_array(store.cache_path(session_id, "Base", base_key))
                _save_array_atomic(destination, fast_refine_mask(base_mask))

        source_masks: list[dict[str, Any]] = []
        base_keys: list[str] = []
        refined_keys: list[str] = []
        for index, (source, path, base_key, refined_key) in enumerate(
            refined_entries,
            start=1,
        ):
            _check_cancel(cancel_event)
            progress(f"Постобработка: {path.name}", index - 1, len(refined_entries))
            cache_path = (
                store.cache_path(session_id, "Refined", refined_key)
                if refined_key is not None
                else store.cache_path(session_id, "Base", base_key)
            )
            prepared = _load_array(cache_path)
            final_mask = postprocess_mask(
                prepared,
                settings.mask,
                refinement=RefinementMode.NONE,
            )
            filename = (
                f"{Path(source.filename).stem}{settings.output.mask_suffix}_"
                f"{mask_set.number:03d}.png"
            )
            destination = mask_set.path / "Masks" / filename
            _save_mask_atomic(destination, final_mask, settings.output.png_compress_level)
            source_masks.append(
                {
                    "source_id": source.source_id,
                    "source": source.relative_path,
                    "mask": destination.relative_to(mask_set.path).as_posix(),
                    "base_cache_key": base_key,
                    "refined_cache_key": refined_key,
                    "foreground_fraction": round(float(final_mask.mean()), 6),
                }
            )
            base_keys.append(base_key)
            if refined_key is not None:
                refined_keys.append(refined_key)
            progress(f"Постобработка: {path.name}", index, len(refined_entries))

        return store.complete_set(
            session_id,
            set_id,
            source_masks=source_masks,
            base_cache_keys=base_keys,
            refined_cache_keys=refined_keys,
        )
    except Exception as exc:
        store.fail_set(session_id, set_id, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if refiner is not None:
            refiner.unload()
        if adapter is not None:
            adapter.unload()


def _refined_key(
    base_key: str,
    settings: RmbgSettings,
    refiner_manifest: dict[str, Any],
) -> str | None:
    effective = settings.resolved_refinement()
    if effective == RefinementMode.NONE:
        return None
    if effective == RefinementMode.FAST:
        return _cache_key(
            {
                "base": base_key,
                "refinement": "fast",
                "version": FAST_REFINEMENT_CACHE_VERSION,
            }
        )
    return _cache_key(
        {
            "base": base_key,
            "refinement": "sdmatte",
            "variant": settings.mask.sdmatte_variant.value,
            "resolution": settings.mask.sdmatte_resolution,
            "transparent_object": settings.mask.sdmatte_transparent_object,
            "constraint": settings.mask.sdmatte_constraint,
            "revision": refiner_manifest.get("revision"),
            "weights": refiner_manifest.get("weights"),
        }
    )


def _cache_key(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.array(image, dtype=np.uint8, order="C", copy=True)


def _load_array(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2:
        raise ValueError(f"Кэш маски повреждён: {path}")
    return np.ascontiguousarray(np.clip(value, 0.0, 1.0), dtype=np.float32)


def _save_array_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.save(stream, np.ascontiguousarray(value, dtype=np.float32))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_mask_atomic(path: Path, mask: np.ndarray, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    try:
        pixels = np.rint(np.clip(mask, 0.0, 1.0) * 65535.0).astype(np.uint16)
        Image.fromarray(pixels).save(
            temporary,
            format="PNG",
            compress_level=compress_level,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_cancel(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise PreviewGenerationCancelled("Генерация тестового набора отменена.")
