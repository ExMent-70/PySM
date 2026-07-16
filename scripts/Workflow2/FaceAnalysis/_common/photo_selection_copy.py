"""Safe materialization of selected physical files into location folders."""

from __future__ import annotations

from dataclasses import dataclass
import filecmp
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Protocol

from .photo_selection_core import BuildResult, Issue

__all__ = ["CopySummary", "ProgressFactory", "ProgressReporter", "copy_selected_files"]


class ProgressReporter(Protocol):
    """Minimal progress reporter contract used by the copy operation."""

    def set_description(self, value: str) -> None: ...

    def update(self, value: int) -> None: ...

    def close(self) -> None: ...


ProgressFactory = Callable[..., ProgressReporter]


@dataclass(frozen=True)
class CopySummary:
    copied: int
    skipped: int
    issues: tuple[Issue, ...]


def copy_selected_files(
    result: BuildResult,
    source_dir: Path,
    dest_dir: Path,
    *,
    progress_factory: ProgressFactory | None = None,
) -> CopySummary:
    """Copy student and photographer exports into metadata-defined locations.

    The first RAW pass copies student-selected originals. During the second
    pass, ``source_dir`` contains Capture One exports, including ``PH_`` files
    selected by the photographer. If Capture One has already created the
    location folder, that first relative component is not duplicated.
    """
    copied = skipped = 0
    issues: list[Issue] = []
    source_root = source_dir.resolve()
    destination_root = dest_dir.resolve()
    total = sum(
        len(record.source_files)
        for record in result.records.values()
        if record.selected_student_ids or record.photographer_selected
    )
    progress = (
        progress_factory(
            total=total,
            desc="Копирование выбранных файлов",
            unit="file",
        )
        if progress_factory is not None and total > 0
        else None
    )
    try:
        for record in result.records.values():
            if not record.selected_student_ids and not record.photographer_selected:
                continue
            location_dir = (destination_root / (record.location or "unknown")).resolve()
            try:
                location_dir.relative_to(destination_root)
            except ValueError:
                issues.append(Issue(
                    "error",
                    "unsafe_location_path",
                    f"Локация выходит за пределы целевой папки: {record.location!r}",
                    record.number,
                ))
                continue
            for source in record.source_files:
                target = location_dir / source.name
                try:
                    source.resolve().relative_to(source_root)
                    relative = source.relative_to(source_dir)
                    relative_parts = relative.parts
                    if (
                        len(relative_parts) > 1
                        and relative_parts[0].casefold()
                        == (record.location or "unknown").casefold()
                    ):
                        relative = Path(*relative_parts[1:])
                    target = location_dir / relative
                    if progress is not None:
                        progress.set_description(f"Копирование: {source.name}")

                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if source.resolve() == target.resolve():
                            skipped += 1
                            continue
                        if (
                            target.stat().st_size == source.stat().st_size
                            and filecmp.cmp(source, target, shallow=False)
                        ):
                            skipped += 1
                            continue
                        issues.append(Issue(
                            "error",
                            "copy_conflict",
                            f"Файл назначения отличается: {target}",
                            record.number,
                        ))
                        continue

                    fd, temp_name = tempfile.mkstemp(
                        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
                    )
                    os.close(fd)
                    temp_path = Path(temp_name)
                    try:
                        shutil.copy2(source, temp_path)
                        # On Windows ``rename`` fails if another process created
                        # the target after our conflict check; it never silently
                        # replaces that independently published file.
                        os.rename(temp_path, target)
                        copied += 1
                    finally:
                        temp_path.unlink(missing_ok=True)
                except ValueError:
                    issues.append(Issue(
                        "error",
                        "source_outside_root",
                        f"Исходный файл находится вне source_dir: {source}",
                        record.number,
                    ))
                except Exception as exc:
                    issues.append(Issue(
                        "error",
                        "copy_failed",
                        f"{source} -> {target}: {exc}",
                        record.number,
                    ))
                finally:
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return CopySummary(copied, skipped, tuple(issues))
