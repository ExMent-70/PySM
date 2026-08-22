"""Sequential, manifest-driven RMBG batch processing engine."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .adapters.base import AdapterLoadContext, ModelAdapter
from .artifacts import (
    ArtifactPaths,
    assert_unique_artifact_paths,
    build_artifact_paths,
    existing_artifact_state,
    save_artifacts,
)
from .config_schema import BackgroundMode, RefinementMode, RmbgSettings
from .image_io import resolve_background_image
from .manifests import load_models_lock
from .mask_ops import postprocess_mask
from .pipeline import PipelinePlan
from .progress import ProgressFactory
from .refiners.base import MaskRefiner


def run_batch(
    *,
    settings: RmbgSettings,
    plan: PipelinePlan,
    adapter: ModelAdapter,
    load_context: AdapterLoadContext,
    refiner: MaskRefiner | None = None,
    refiner_load_context: AdapterLoadContext | None = None,
    images: tuple[Path, ...],
    input_root: Path,
    output_root: Path,
    overwrite: bool,
    fail_fast: bool,
    background_dir: Path | None,
    upstream: dict[str, Any],
    progress_factory: ProgressFactory | None = None,
    downloaded_model_files: tuple[str, ...] = (),
    downloaded_refiner_files: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run deterministic single-image inference and persist a resumable manifest."""

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    artifact_paths = tuple(
        build_artifact_paths(path, input_root, output_root, settings.output)
        for path in images
    )
    assert_unique_artifact_paths(artifact_paths)
    states = tuple(
        existing_artifact_state(paths, overwrite=overwrite)
        for paths in artifact_paths
    )
    backgrounds = _resolve_backgrounds(
        images,
        background_dir=background_dir,
        background_image=settings.output.background_image,
        required=(
            settings.output.save_composite
            and settings.output.background_mode == BackgroundMode.IMAGE
        ),
    )

    started_at = _utc_now()
    started = time.perf_counter()
    models_lock = load_models_lock()
    model_lock = models_lock["models"][plan.model_id]
    download_lock = model_lock.get("download")
    refiner_lock = models_lock.get("refiners", {}).get("sdmatte", {})
    refinement_report: dict[str, Any] = {
        "configured": settings.mask.refinement.value,
        "effective": settings.resolved_refinement().value,
        "id": refiner.refiner_id if refiner is not None else None,
        "downloaded_this_run": list(downloaded_refiner_files),
    }
    if settings.resolved_refinement() == RefinementMode.SDMATTE:
        variant = settings.mask.sdmatte_variant.value
        weight_name = refiner_lock.get("variants", {}).get(variant)
        refinement_report.update(
            {
                "variant": variant,
                "resolution": settings.mask.sdmatte_resolution,
                "transparent_object": settings.mask.sdmatte_transparent_object,
                "constraint": settings.mask.sdmatte_constraint,
                "repository": refiner_lock.get("repository"),
                "revision": refiner_lock.get("revision"),
                "upstream_commit": refiner_lock.get("upstream_commit"),
                "status": refiner_lock.get("status"),
                "checkpoint": weight_name,
                "checkpoint_manifest": (
                    refiner_lock.get("weights", {}).get(weight_name)
                    if weight_name is not None
                    else None
                ),
            }
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "elapsed_seconds": None,
        "config_hash": settings.stable_hash(),
        "pipeline": plan.to_dict(),
        "model": {
            "id": plan.model_id,
            "repository": model_lock["repository"],
            "revision": model_lock["revision"],
            "status": model_lock["status"],
            "accepted_local_files": model_lock["files"],
            "download_manifest": download_lock,
            "downloaded_this_run": list(downloaded_model_files),
        },
        "refinement": refinement_report,
        "upstream": {
            "release_version": upstream["release_version"],
            "commit": upstream["commit"],
            "tree": upstream["tree"],
        },
        "runtime": {
            "input_dir": str(input_root),
            "output_dir": str(output_root),
            "background_dir": str(background_dir.resolve()) if background_dir else None,
            "background_image": settings.output.background_image or None,
            "overwrite": overwrite,
            "fail_fast": fail_fast,
            "sequential_inference": True,
            "gpu_batch_size": 1,
            "requested_batch_size": settings.performance.batch_size,
            "io_workers": settings.performance.io_workers,
            "manifest_checkpoint_seconds": 2.0,
            "model_unload_policy": "always_after_run",
        },
        "summary": {
            "discovered": len(images),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
        },
        "items": [],
    }
    manifest_path = output_root / "Reports" / "manifest.json"
    _write_json_atomic(manifest_path, report)
    has_pending_writes = any(state == "write" for state in states)
    initial_progress_text = (
        f"RMBG: загрузка {plan.model_display_name}"
        if has_pending_writes
        else "RMBG: проверка готовых результатов"
    )
    progress = (
        progress_factory(
            total=len(images),
            desc=initial_progress_text,
            unit="изобр.",
        )
        if progress_factory is not None and images
        else None
    )

    load_started = time.perf_counter()
    processing_started: float | None = None
    loaded = False
    refiner_loaded = False
    executor: ThreadPoolExecutor | None = None
    read_executor: ThreadPoolExecutor | None = None
    load_futures: dict[int, Future[tuple[Image.Image, np.ndarray]]] = {}
    write_index_iter = iter(())
    pending: deque[tuple[int, dict[str, Any], float, Future[dict[str, Any]]]] = deque()
    ordered_items: list[dict[str, Any] | None] = [None] * len(images)
    last_checkpoint = time.perf_counter()

    def checkpoint(*, force: bool = False) -> None:
        nonlocal last_checkpoint
        now = time.perf_counter()
        if not force and now - last_checkpoint < 2.0:
            return
        report["items"] = [item for item in ordered_items if item is not None]
        _write_json_atomic(manifest_path, report)
        last_checkpoint = now

    def finish_item(
        index: int,
        item: dict[str, Any],
        item_started: float,
        *,
        status: str,
        model_runtime: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        item["status"] = status
        if model_runtime is not None:
            item["model_runtime"] = model_runtime
        if error is not None:
            item["error"] = f"{type(error).__name__}: {error}"
        item["elapsed_seconds"] = round(time.perf_counter() - item_started, 3)
        ordered_items[index] = item
        if status == "processed":
            report["summary"]["processed"] += 1
        elif status == "skipped_existing":
            report["summary"]["skipped"] += 1
        else:
            report["summary"]["failed"] += 1
        if progress is not None:
            progress.set_postfix(
                {
                    "готово": int(report["summary"]["processed"]),
                    "пропущено": int(report["summary"]["skipped"]),
                    "ошибок": int(report["summary"]["failed"]),
                },
                refresh=False,
            )
            progress.update(1)
        checkpoint(force=error is not None)

    def drain_one() -> bool:
        index, item, item_started, future = pending.popleft()
        try:
            runtime = future.result()
        except Exception as exc:
            finish_item(index, item, item_started, status="failed", error=exc)
            return False
        finish_item(
            index,
            item,
            item_started,
            status="processed",
            model_runtime=runtime,
        )
        return True

    def submit_next_read() -> None:
        if read_executor is None:
            return
        try:
            index = next(write_index_iter)
        except StopIteration:
            return
        load_futures[index] = read_executor.submit(_load_source, images[index])

    try:
        if has_pending_writes:
            adapter.load(load_context)
            loaded = True
            if refiner is not None:
                if refiner_load_context is None:
                    raise RuntimeError("Для SDMatte не передан load context.")
                if progress is not None:
                    progress.set_description(
                        "RMBG: загрузка SDMatte",
                        refresh=False,
                    )
                refiner.load(refiner_load_context)
                refiner_loaded = True
        report["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
        processing_started = time.perf_counter()
        io_workers = 1 if fail_fast else settings.performance.io_workers
        report["runtime"]["effective_io_workers"] = io_workers
        if io_workers > 1:
            executor = ThreadPoolExecutor(
                max_workers=io_workers,
                thread_name_prefix="rmbg-io",
            )
            read_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="rmbg-read",
            )
            write_index_iter = iter(
                index for index, state in enumerate(states) if state == "write"
            )
            for _ in range(2):
                submit_next_read()
        max_in_flight = max(1, io_workers * 2)
        for index, (source_path, paths, state, background_path) in enumerate(zip(
            images, artifact_paths, states, backgrounds,
        )):
            while len(pending) >= max_in_flight:
                if not drain_one() and fail_fast:
                    break
            if fail_fast and report["summary"]["failed"]:
                break
            if progress is not None:
                progress.set_description(f"RMBG: {source_path.name}", refresh=False)
            item = _base_item(source_path, input_root, paths, background_path)
            item_started = time.perf_counter()
            if state == "skip":
                finish_item(
                    index,
                    item,
                    item_started,
                    status="skipped_existing",
                )
                continue
            source: Image.Image | None = None
            try:
                if read_executor is None:
                    source, image_rgb = _load_source(source_path)
                else:
                    load_future = load_futures.pop(index)
                    try:
                        source, image_rgb = load_future.result()
                    finally:
                        submit_next_read()
                result = adapter.infer(image_rgb)
                raw_mask = result.mask
                model_metadata = dict(result.metadata)
                if refiner is not None:
                    raw_mask, refinement_metadata = refiner.refine(
                        image_rgb,
                        raw_mask,
                    )
                    model_metadata["refinement"] = refinement_metadata
                if executor is None:
                    runtime = _finalize_result(
                        source,
                        raw_mask,
                        model_metadata,
                        settings=settings,
                        paths=paths,
                        background_path=background_path,
                        sdmatte_applied=refiner is not None,
                    )
                    source = None
                    finish_item(
                        index,
                        item,
                        item_started,
                        status="processed",
                        model_runtime=runtime,
                    )
                else:
                    future = executor.submit(
                        _finalize_result,
                        source,
                        raw_mask,
                        model_metadata,
                        settings=settings,
                        paths=paths,
                        background_path=background_path,
                        sdmatte_applied=refiner is not None,
                    )
                    source = None
                    pending.append((index, item, item_started, future))
            except Exception as exc:
                if source is not None:
                    source.close()
                finish_item(index, item, item_started, status="failed", error=exc)
                if fail_fast:
                    break
        while pending:
            if not drain_one() and fail_fast:
                break
        report["items"] = [item for item in ordered_items if item is not None]
        processing_seconds = time.perf_counter() - processing_started
        report["processing_seconds"] = round(processing_seconds, 3)
        processed_count = int(report["summary"]["processed"])
        report["throughput_images_per_second"] = (
            round(processed_count / processing_seconds, 3)
            if processed_count and processing_seconds > 0
            else 0.0
        )
    except Exception as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "failed"
        raise
    finally:
        if read_executor is not None:
            read_executor.shutdown(wait=True, cancel_futures=False)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if refiner_loaded and refiner is not None:
            try:
                if progress is not None:
                    progress.set_description(
                        "RMBG: освобождение SDMatte",
                        refresh=False,
                    )
                refiner.unload()
            except Exception as exc:
                report["fatal_error"] = f"Ошибка освобождения SDMatte: {exc}"
        if loaded:
            try:
                if progress is not None:
                    progress.set_description(
                        "RMBG: освобождение модели",
                        refresh=False,
                    )
                adapter.unload()
            except Exception as exc:
                report["fatal_error"] = f"Ошибка освобождения модели: {exc}"
        report["finished_at"] = _utc_now()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if processing_started is not None and "processing_seconds" not in report:
            processing_seconds = time.perf_counter() - processing_started
            report["processing_seconds"] = round(processing_seconds, 3)
            processed_count = int(report["summary"]["processed"])
            report["throughput_images_per_second"] = (
                round(processed_count / processing_seconds, 3)
                if processed_count and processing_seconds > 0
                else 0.0
            )
        failed = int(report["summary"]["failed"])
        completed = len(report["items"])
        if report.get("fatal_error"):
            report["status"] = "failed"
        elif failed == 0 and completed == len(images):
            report["status"] = "success"
        elif failed < completed:
            report["status"] = "partial"
        else:
            report["status"] = "failed"
        try:
            _write_json_atomic(manifest_path, report)
        finally:
            if progress is not None:
                progress.close()
    return report


def _load_source(source_path: Path) -> tuple[Image.Image, np.ndarray]:
    """Decode one source while detaching it from the underlying file handle."""

    with Image.open(source_path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")
        source.info.update(opened.info)
        image_rgb = np.array(source, dtype=np.uint8, order="C", copy=True)
    return source, image_rgb


def _finalize_result(
    source: Image.Image,
    raw_mask: np.ndarray,
    model_metadata: dict[str, Any],
    *,
    settings: RmbgSettings,
    paths: ArtifactPaths,
    background_path: Path | None,
    sdmatte_applied: bool = False,
) -> dict[str, Any]:
    """Run CPU postprocessing and I/O after the GPU has produced a mask."""

    mask = postprocess_mask(
        raw_mask,
        settings.mask,
        refinement=settings.resolved_refinement(),
        sdmatte_applied=sdmatte_applied,
    )
    background = None
    try:
        if background_path:
            background = Image.open(background_path)
        save_artifacts(
            source,
            mask,
            paths,
            settings.output,
            background=background,
        )
    finally:
        if background is not None:
            background.close()
        source.close()
    return dict(model_metadata)


def _resolve_backgrounds(
    images: tuple[Path, ...],
    *,
    background_dir: Path | None,
    background_image: str,
    required: bool,
) -> tuple[Path | None, ...]:
    if not required:
        return (None,) * len(images)
    selected = resolve_background_image(background_dir, background_image)
    return (selected,) * len(images)


def _base_item(
    source: Path,
    input_root: Path,
    paths: ArtifactPaths,
    background: Path | None,
) -> dict[str, Any]:
    return {
        "source": str(source),
        "relative_source": source.resolve().relative_to(input_root).as_posix(),
        "background": str(background) if background else None,
        "outputs": paths.to_dict(),
        "status": "pending",
        "elapsed_seconds": None,
        "error": None,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
