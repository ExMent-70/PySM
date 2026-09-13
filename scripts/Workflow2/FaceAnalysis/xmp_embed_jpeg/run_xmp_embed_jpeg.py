#!/usr/bin/env python3
"""Embed sidecar XMP metadata into JPEG files without re-encoding pixels."""

from __future__ import annotations

import argparse
import logging
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for import_path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm

    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
    IS_MANAGED_RUN = False
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda iterable, **_kwargs: iterable

from _lib.jpeg_xmp import (
    JpegXmpError,
    build_jpeg_with_xmp,
    write_jpeg_atomically,
)


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
SCRIPT_LOG_TITLE = (
    "ВНЕДРЕНИЕ ВНЕШНИХ XMP-МЕТАДАННЫХ В ФАЙЛЫ JPG"
)
JPEG_SUFFIXES = {".jpg", ".jpeg"}
BACKUP_DIR_NAME = "backup"
XMP_SOURCE_CHOICES = ("auto", "alongside", "subfolder")


@dataclass(frozen=True)
class FileSnapshot:
    """File identity used to catch changes between preflight and writing."""

    size: int
    modified_ns: int


@dataclass(frozen=True)
class PlannedPair:
    """A validated image/sidecar pair ready for the apply pass."""

    image_path: Path
    xmp_path: Path
    image_snapshot: FileSnapshot
    xmp_snapshot: FileSnapshot
    changed: bool
    merged_xmp_size: int


def get_config() -> Namespace:
    """Определяет аргументы и возвращает полностью обработанную конфигурацию."""

    parser = argparse.ArgumentParser(
        description=(
            "Внедряет внешние XMP-файлы в JPG/JPEG без повторного сжатия "
            "изображений."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Папка с JPG/JPEG и соответствующими XMP-файлами.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Искать изображения во вложенных папках.",
    )
    parser.add_argument(
        "--xmp_source",
        choices=XMP_SOURCE_CHOICES,
        default="auto",
        help=(
            "Расположение XMP: auto, alongside (рядом с JPG) или "
            "subfolder (в подпапке XMP)."
        ),
    )
    parser.add_argument(
        "--backup_originals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Сохранять исходную копию как backup/image.jpg.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Выполнить полную проверку без изменения файлов.",
    )

    if ConfigResolver is not None:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


def discover_images(image_dir: Path, recursive: bool) -> list[Path]:
    """Return JPEG sources outside managed backup and XMP directories."""

    candidates = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file()
            and path.suffix.casefold() in JPEG_SUFFIXES
            and not _is_in_managed_directory(path, image_dir)
        ),
        key=lambda path: str(path).casefold(),
    )


def _is_in_managed_directory(path: Path, image_dir: Path) -> bool:
    """Return whether a file is below a managed backup or XMP directory."""

    relative_parts = path.relative_to(image_dir).parts[:-1]
    managed_names = {BACKUP_DIR_NAME, "xmp"}
    return any(part.casefold() in managed_names for part in relative_parts)


def locate_sidecar(image_path: Path, source_mode: str) -> Path | None:
    """Resolve one unambiguous same-stem sidecar according to the selected mode."""

    alongside = image_path.with_suffix(".xmp")
    subfolder = image_path.parent / "XMP" / f"{image_path.stem}.xmp"
    if source_mode == "alongside":
        return alongside if alongside.is_file() else None
    if source_mode == "subfolder":
        return subfolder if subfolder.is_file() else None

    existing = [path for path in (alongside, subfolder) if path.is_file()]
    if len(existing) > 1:
        raise ValueError(
            "Найдены два XMP для одного JPEG: "
            + ", ".join(str(path) for path in existing)
        )
    return existing[0] if existing else None


def snapshot(path: Path) -> FileSnapshot:
    """Capture the minimum file state needed for a pre-apply race check."""

    stat_result = path.stat()
    return FileSnapshot(stat_result.st_size, stat_result.st_mtime_ns)


def prepare_pair(image_path: Path, xmp_path: Path) -> PlannedPair:
    """Read and validate a pair without changing either source file."""

    image_snapshot = snapshot(image_path)
    xmp_snapshot = snapshot(xmp_path)
    plan = build_jpeg_with_xmp(image_path.read_bytes(), xmp_path.read_bytes())
    return PlannedPair(
        image_path=image_path,
        xmp_path=xmp_path,
        image_snapshot=image_snapshot,
        xmp_snapshot=xmp_snapshot,
        changed=plan.changed,
        merged_xmp_size=len(plan.embedded_xmp),
    )


def run(config: Namespace) -> int:
    """Preflight every pair, then apply only when the whole batch is valid."""

    LOGGER.info(f"<b>{SCRIPT_LOG_TITLE}</b><br>")
    image_dir = Path(config.image_dir).resolve()
    if not image_dir.is_dir():
        LOGGER.error(f"[Ошибка] Папка изображений не найдена: {image_dir}")
        return 1

    images = discover_images(image_dir, bool(config.recursive))
    LOGGER.info(f"Найдено JPG/JPEG: {len(images)}")
    if not images:
        LOGGER.warning("[Внимание] Подходящие изображения не найдены.")
        return 0

    planned: list[PlannedPair] = []
    without_sidecar = 0
    preflight_errors: list[str] = []
    for image_path in tqdm(images, desc="Проверка XMP"):
        try:
            xmp_path = locate_sidecar(image_path, config.xmp_source)
            if xmp_path is None:
                without_sidecar += 1
                continue
            planned.append(prepare_pair(image_path, xmp_path))
        except (JpegXmpError, OSError, ValueError) as error:
            preflight_errors.append(f"{image_path}: {error}")

    LOGGER.info(f"Найдено пар JPG + XMP: {len(planned)}")
    LOGGER.info(f"Без XMP: {without_sidecar}")
    LOGGER.info(f"Требуют обновления: {sum(item.changed for item in planned)}")
    LOGGER.info(f"Уже актуальны: {sum(not item.changed for item in planned)}")
    if planned:
        LOGGER.info(
            "Максимальный итоговый XMP: "
            f"{max(item.merged_xmp_size for item in planned)} байт"
        )
    if preflight_errors:
        LOGGER.error(
            "[Ошибка] Предварительная проверка не пройдена; JPG не изменялись:"
        )
        for message in preflight_errors:
            LOGGER.error(f"  • {message}")
        return 1

    if not planned:
        LOGGER.warning(
            "[Внимание] Внешние XMP для найденных изображений не найдены."
        )
        return 0

    if bool(config.dry_run):
        LOGGER.info("[OK] Проверка завершена. Режим dry-run: файлы не изменялись.")
        return 0

    changed_pairs = [item for item in planned if item.changed]
    if not changed_pairs:
        LOGGER.info("[OK] Все найденные XMP уже встроены в JPEG.")
        return 0

    # Nothing may change between the validation pass and the first write.
    for item in changed_pairs:
        if snapshot(item.image_path) != item.image_snapshot:
            LOGGER.error(f"[Ошибка] После проверки изменился JPEG: {item.image_path}")
            return 1
        if snapshot(item.xmp_path) != item.xmp_snapshot:
            LOGGER.error(f"[Ошибка] После проверки изменился XMP: {item.xmp_path}")
            return 1

    updated = 0
    write_errors: list[str] = []
    for item in tqdm(changed_pairs, desc="Внедрение XMP"):
        try:
            source_data = item.image_path.read_bytes()
            plan = build_jpeg_with_xmp(
                source_data,
                item.xmp_path.read_bytes(),
            )
            result = write_jpeg_atomically(
                item.image_path,
                plan.output_data,
                backup_original=bool(config.backup_originals),
                expected_source=source_data,
            )
            updated += int(result.changed)
        except (JpegXmpError, OSError) as error:
            write_errors.append(f"{item.image_path}: {error}")

    if write_errors:
        LOGGER.error(f"[Ошибка] Ошибок записи: {len(write_errors)}")
        for message in write_errors:
            LOGGER.error(f"  • {message}")
        LOGGER.error(f"До ошибки/ошибок успешно обновлено: {updated}")
        return 1

    LOGGER.info(f"[OK] XMP успешно внедрён в JPEG: {updated}")
    if bool(config.backup_originals):
        LOGGER.info("Резервные копии сохранены в подпапке backup.")
    return 0


def main() -> int:
    """CLI entry point."""

    return run(get_config())


if __name__ == "__main__":
    raise SystemExit(main())
