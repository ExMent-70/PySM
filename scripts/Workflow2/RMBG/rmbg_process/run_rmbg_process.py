#!/usr/bin/env python3
"""Validate runtime inputs and execute the configured RMBG processing pipeline."""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUBSYSTEM_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for import_path in (PROJECT_ROOT, SUBSYSTEM_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm as pysm_tqdm

    IS_MANAGED_RUN = getattr(pysm_context, "_context_file_path", None) is not None
except ImportError:
    pysm_context = None
    ConfigResolver = None
    pysm_tqdm = None
    IS_MANAGED_RUN = False

try:
    from pysm_lib.pysm_report_api import (
        DashboardBuilder,
        ResourceNode,
        StandardTreeBuilder,
    )
except ImportError:
    DashboardBuilder = None
    ResourceNode = None
    StandardTreeBuilder = None

from _common.context_config import DEFAULT_CONFIG_VAR, load_context_settings
from _common.adapters.base import AdapterLoadContext
from _common.adapters.local_birefnet import LocalBiRefNetAdapter
from _common.config_schema import ModelName, RefinementMode, RmbgSettings
from _common.image_io import discover_images, resolve_background_image
from _common.manifests import load_models_lock, load_upstream_lock
from _common.model_registry import create_model_registry
from _common.model_downloader import ensure_model_files, ensure_sdmatte_files
from _common.model_store import (
    resolve_model_files,
)
from _common.pipeline import build_pipeline_plan
from _common.path_contract import resolve_model_dir_value
from _common.processing import run_batch
from _common.refiners.sdmatte import SDMatteRefiner, validate_sdmatte_runtime


def get_config() -> argparse.Namespace:
    """Define arguments and return the fully resolved script configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Обрабатывает изображения по RMBG-профилю из контекста PySM. "
            "Поддерживает безопасный dry-run и проверенные production-модели."
        )
    )
    parser.add_argument(
        "--config_var",
        default=DEFAULT_CONFIG_VAR,
        help="Имя JSON-переменной контекста с настройками RMBG.",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Папка с исходными изображениями. Поддерживает шаблоны {var}.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Папка будущих результатов. Поддерживает шаблоны {var}.",
    )
    parser.add_argument(
        "--background_dir",
        default="",
        help=(
            "Папка, содержащая выбранное в Configurator фоновое изображение "
            "для background_mode=image."
        ),
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Искать изображения во вложенных папках.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Разрешить перезапись существующих результатов.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Только проверить конфигурацию и показать effective plan.",
    )
    parser.add_argument(
        "--fail_fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Останавливать пакет после первой ошибки изображения.",
    )

    if ConfigResolver is not None:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


def _build_summary(
    config: argparse.Namespace,
    settings: RmbgSettings | None = None,
) -> dict[str, object]:
    if not IS_MANAGED_RUN or pysm_context is None:
        raise RuntimeError(
            "RMBG Process необходимо запускать внутри PySM: "
            "конфигурация должна быть получена из контекста."
        )

    if settings is None:
        settings = load_context_settings(pysm_context, config.config_var)
    model_store = resolve_model_dir_value(settings.model.model_dir)
    registry = create_model_registry()
    model_id = settings.resolved_model_name()
    model_lock = load_models_lock()["models"][model_id.value]
    if model_lock["status"] == "verified":
        descriptor = registry.get(model_id)
        registry.register_factory(
            model_id,
            lambda: LocalBiRefNetAdapter(
                model_id,
                resolve_model_files(model_store, descriptor),
            ),
        )
    plan = build_pipeline_plan(settings, registry)

    if "background_dir" in plan.required_cli_inputs and not config.background_dir:
        raise ValueError(
            "Профиль использует background_mode=image, "
            "но параметр --background_dir не задан."
        )
    selected_background = None
    if "background_dir" in plan.required_cli_inputs:
        selected_background = resolve_background_image(
            config.background_dir,
            settings.output.background_image,
        )

    images = discover_images(config.input_dir, recursive=bool(config.recursive))
    if not images:
        raise ValueError("В input_dir не найдено поддерживаемых изображений.")
    input_dir = Path(config.input_dir).resolve()
    output_dir = Path(config.output_dir).resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError(
            "output_dir не должен совпадать с input_dir или находиться внутри него: "
            "иначе повторный recursive-запуск может принять результаты за исходники."
        )
    upstream = load_upstream_lock()
    refiners_lock = load_models_lock().get("refiners", {})
    sdmatte_lock = refiners_lock.get("sdmatte", {})
    return {
        "mode": "dry_run" if config.dry_run else "process",
        "config_var": config.config_var,
        "pipeline": plan.to_dict(),
        "runtime": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "background_dir": config.background_dir or None,
            "background_image": (
                str(selected_background) if selected_background else None
            ),
            "model_dir": settings.model.model_dir,
            "recursive": bool(config.recursive),
            "overwrite": bool(config.overwrite),
            "fail_fast": bool(config.fail_fast),
            "image_count": len(images),
        },
        "model": {
            "id": model_id.value,
            "status": model_lock["status"],
            "production_enabled": model_lock["status"] == "verified",
            "download_available": isinstance(model_lock.get("download"), dict),
        },
        "refinement": {
            "configured": settings.mask.refinement.value,
            "effective": settings.resolved_refinement().value,
            "sdmatte_variant": settings.mask.sdmatte_variant.value,
            "sdmatte_status": sdmatte_lock.get("status"),
            "sdmatte_download_bytes": (
                sdmatte_lock.get("weights", {})
                .get(
                    {
                        "sdmatte": "SDMatte.safetensors",
                        "sdmatte_plus": "SDMatte_plus.safetensors",
                    }[settings.mask.sdmatte_variant.value],
                    {},
                )
                .get("size")
            ),
        },
        "upstream": {
            "release_version": upstream["release_version"],
            "commit": upstream["commit"],
            "tree": upstream["tree"],
            "runtime_status": upstream["runtime_status"],
        },
    }


def _format_limit(enabled: bool, value: int) -> str:
    if not enabled:
        return "выключено"
    return "включено, без ограничения" if value == 0 else f"включено, до {value} px"


def _emit_html(html: str) -> bool:
    """Send an HTML block to PySM and report whether rendering succeeded."""

    if not IS_MANAGED_RUN or pysm_context is None:
        return False
    try:
        pysm_context.log_html(html)
        return True
    except Exception as exc:
        print(f"Не удалось сформировать HTML-отчёт: {exc}", file=sys.stderr)
        return False


def _settings_rows(
    settings: RmbgSettings,
    summary: dict[str, object],
) -> list[tuple[str, str]]:
    """Build operator-facing mask settings without presentation markup."""

    pipeline = summary["pipeline"]
    runtime = summary["runtime"]
    assert isinstance(pipeline, dict)
    assert isinstance(runtime, dict)
    mask = settings.mask
    rows = [
        ("Профиль", str(settings.profile_name)),
        (
            "Модель",
            f"{pipeline['model_display_name']} | "
            f"разрешение: {pipeline['process_resolution']} | "
            f"устройство: {pipeline['device']} | "
            f"точность: {pipeline['precision']}",
        ),
        ("Папка моделей", settings.model.model_dir),
        ("Уточнение края", str(pipeline["refinement"])),
    ]
    if settings.resolved_refinement() == RefinementMode.SDMATTE:
        rows.append(
            (
                "SDMatte",
                f"{mask.sdmatte_variant.value}, "
                f"разрешение {mask.sdmatte_resolution}, "
                f"строгость {mask.sdmatte_constraint:g}, "
                "прозрачный объект: "
                f"{'да' if mask.sdmatte_transparent_object else 'нет'}",
            )
        )
    rows.extend(
        [
            (
                "Чувствительность",
                f"{mask.sensitivity:g}",
            ),
            ("Размытие", str(mask.blur)),
            ("Смещение края", str(mask.offset)),
            ("Растушёвка", str(mask.feather)),
            (
                "Заполнение отверстий",
                _format_limit(mask.fill_holes, mask.max_hole_area),
            ),
            (
                "Удаление мелких областей",
                _format_limit(mask.remove_small_regions, mask.min_region_area),
            ),
            ("Инверсия маски", "да" if mask.invert else "нет"),
            ("Найдено исходных изображений", str(runtime["image_count"])),
        ]
    )
    return rows


def _rows_html(
    rows: list[tuple[str, str]],
    *,
    margin_top: int = 0,
    margin_bottom: int = 12,
) -> str:
    """Render compact key/value rows with bold labels."""

    body = "".join(
        f"<div><b>{escape(label)}:</b> {escape(value)}</div>"
        for label, value in rows
    )
    return (
        "<div style='font-family:sans-serif;font-size:13px;line-height:1.45;"
        f"margin-top:{margin_top}px;margin-bottom:{margin_bottom}px;'>{body}</div>"
    )


def _print_mask_settings(
    settings: RmbgSettings,
    summary: dict[str, object],
) -> None:
    """Show a compact preflight summary before model loading."""

    rows = _settings_rows(settings, summary)
    if DashboardBuilder is not None:
        builder = DashboardBuilder(icon_size=20)
        builder.add_header_simple("Параметры создания масок:")
        builder.parts.append(_rows_html(rows))
        settings_html = (
            "<div style='font-family:sans-serif;'>"
            f"{builder.get_html()}</div>"
        )
        if _emit_html(settings_html):
            return

    print("\nОсновные настройки создания масок")
    for label, value in rows:
        print(f"{label}: {value}")
    print(flush=True)


def _print_completion_summary(report: dict[str, object]) -> None:
    """Show only operator-facing totals, never the full manifest payload."""
    summary = report["summary"]
    assert isinstance(summary, dict)
    status_labels = {
        "success": "успешно",
        "partial": "завершено частично",
        "failed": "ошибка",
    }
    rows = [
        ("Статус", str(status_labels.get(str(report["status"]), report["status"]))),
        ("Найдено", str(summary["discovered"])),
        ("Обработано", str(summary["processed"])),
        ("Пропущено", str(summary["skipped"])),
        ("Ошибок", str(summary["failed"])),
        ("Общее время", f"{report['elapsed_seconds']} сек."),
    ]
    if _emit_html(_rows_html(rows, margin_top=12, margin_bottom=4)):
        return
    print()
    for label, value in rows:
        print(f"{label}: {value}")
    print(flush=True)


def _report_result_links(
    input_dir: Path,
    output_dir: Path,
    settings: RmbgSettings,
) -> None:
    """Show links for source and every enabled output artifact directory."""

    source_root = input_dir.resolve()
    result_root = output_dir.resolve()
    manifest_path = result_root / "Reports" / "manifest.json"
    if (
        IS_MANAGED_RUN
        and pysm_context is not None
        and ResourceNode is not None
        and StandardTreeBuilder is not None
    ):
        try:
            nodes = [
                ResourceNode(
                    "Исходные изображения",
                    source_root,
                    "folder",
                    "Папка исходных файлов",
                )
            ]
            output_links = (
                (
                    settings.output.save_cutout,
                    "Итоговые изображения",
                    "Cutout",
                    "Папка итоговых изображений",
                ),
                (
                    settings.output.save_mask,
                    "Полутоновые маски",
                    "Masks",
                    "Папка созданных масок",
                ),
                (
                    settings.output.save_composite,
                    "Изображения с новым фоном",
                    "Composite",
                    "Папка composite-изображений",
                ),
            )
            nodes.extend(
                ResourceNode(name, result_root / folder, "folder", description)
                for enabled, name, folder, description in output_links
                if enabled
            )
            nodes.append(
                ResourceNode(
                    "manifest.json",
                    manifest_path,
                    "code",
                    "Подробный отчёт обработки",
                )
            )
            builder = StandardTreeBuilder(icon_size=24)
            builder.add_section(
                "Папки и отчёт RMBG",
                nodes,
            )
            pysm_context.log_html(builder.get_html())
            return
        except Exception as exc:
            print(
                f"Не удалось сформировать кликабельные ссылки: {exc}",
                file=sys.stderr,
            )
    print(f"Исходные изображения: {source_root}")
    if settings.output.save_cutout:
        print(f"Итоговые изображения: {result_root / 'Cutout'}")
    if settings.output.save_mask:
        print(f"Полутоновые маски: {result_root / 'Masks'}")
    if settings.output.save_composite:
        print(f"Изображения с новым фоном: {result_root / 'Composite'}")
    print(f"Отчёт manifest.json: {manifest_path}", flush=True)


def main() -> int:
    config = get_config()
    if not IS_MANAGED_RUN or pysm_context is None:
        print(
            "RMBG Process необходимо запускать внутри PySM: "
            "конфигурация должна быть получена из контекста.",
            file=sys.stderr,
        )
        return 2
    try:
        settings = load_context_settings(pysm_context, config.config_var)
        summary = _build_summary(config, settings)
    except Exception as exc:
        print(f"Проверка RMBG не пройдена: {exc}", file=sys.stderr)
        return 2

    pipeline = summary["pipeline"]
    assert isinstance(pipeline, dict)
    _print_mask_settings(settings, summary)

    if config.dry_run:
        print("Dry-run завершён: конфигурация и рабочие пути проверены.")
        return 0
    model = summary["model"]
    assert isinstance(model, dict)
    if not model["production_enabled"]:
        print(
            f"Модель '{model['id']}' имеет статус '{model['status']}' и ещё не "
            "разрешена для production-обработки. Доступен --dry_run; статус "
            "verified присваивается после QA-проверки адаптера и результатов.",
            file=sys.stderr,
        )
        return 3
    refinement_summary = summary["refinement"]
    assert isinstance(refinement_summary, dict)
    if (
        refinement_summary["effective"] == RefinementMode.SDMATTE.value
        and refinement_summary["sdmatte_status"] != "verified"
    ):
        print(
            "SDMatte ещё не имеет production-статус verified для текущего "
            "runtime. Выберите быстрый refinement или обновите подсистему.",
            file=sys.stderr,
        )
        return 3

    try:
        model_id = ModelName(str(model["id"]))
        registry = create_model_registry()
        descriptor = registry.get(model_id)
        model_dir = resolve_model_dir_value(settings.model.model_dir)
        if settings.resolved_refinement() == RefinementMode.SDMATTE:
            # Check optional packages and CUDA before downloading any weights.
            validate_sdmatte_runtime(
                requested_device=settings.model.device.value,
                requested_precision=settings.model.precision,
            )
        ensured = ensure_model_files(
            model_id,
            descriptor,
            model_dir,
            progress_factory=pysm_tqdm,
        )
        files = ensured.files
        refiner = None
        refiner_context = None
        downloaded_refiner_files: tuple[str, ...] = ()
        if settings.resolved_refinement() == RefinementMode.SDMATTE:
            ensured_refiner = ensure_sdmatte_files(
                settings.mask.sdmatte_variant,
                model_dir,
                progress_factory=pysm_tqdm,
            )
            downloaded_refiner_files = ensured_refiner.downloaded
            refiner = SDMatteRefiner(
                model_root=ensured_refiner.files.model_dir,
                weights=ensured_refiner.files.weights,
                transparent_object=settings.mask.sdmatte_transparent_object,
                constraint=settings.mask.sdmatte_constraint,
            )
            refiner_context = AdapterLoadContext(
                device=settings.model.device.value,
                precision=settings.model.precision,
                model_cache_dir=model_dir,
                process_resolution=settings.mask.sdmatte_resolution,
                local_files_only=True,
            )
        registry.register_factory(
            model_id,
            lambda: LocalBiRefNetAdapter(model_id, files),
        )
        plan = build_pipeline_plan(settings, registry)
        adapter = registry.create(model_id)
        images = discover_images(config.input_dir, recursive=bool(config.recursive))
        report = run_batch(
            settings=settings,
            plan=plan,
            adapter=adapter,
            load_context=AdapterLoadContext(
                device=settings.model.device.value,
                precision=settings.model.precision,
                model_cache_dir=model_dir,
                process_resolution=plan.process_resolution,
                local_files_only=True,
            ),
            refiner=refiner,
            refiner_load_context=refiner_context,
            images=images,
            input_root=Path(config.input_dir),
            output_root=Path(config.output_dir),
            overwrite=bool(config.overwrite),
            fail_fast=bool(config.fail_fast),
            background_dir=(Path(config.background_dir) if config.background_dir else None),
            upstream=load_upstream_lock(),
            progress_factory=pysm_tqdm,
            downloaded_model_files=ensured.downloaded,
            downloaded_refiner_files=downloaded_refiner_files,
        )
    except Exception as exc:
        print(f"Обработка RMBG не выполнена: {exc}", file=sys.stderr)
        return 4

    _report_result_links(Path(config.input_dir), Path(config.output_dir), settings)
    _print_completion_summary(report)
    return 0 if report["status"] == "success" else 4


if __name__ == "__main__":
    raise SystemExit(main())
