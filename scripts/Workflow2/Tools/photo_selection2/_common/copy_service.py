"""Safe materialization of selected physical files into location folders."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Protocol

from .assignment_core import BuildResult, Issue

__all__ = ["CopySummary", "ProgressFactory", "ProgressReporter", "copy_selected_files"]


class ProgressReporter(Protocol):
    """Minimal progress reporter contract used by the copy operation."""

    def set_description(self, value: str) -> None: ...

    def update(self, value: int) -> None: ...

    def close(self) -> None: ...


ProgressFactory = Callable[..., ProgressReporter]


@dataclass(frozen=True)
class CopySummary:
    """Published copies and completed moves; copied includes moved files."""

    copied: int
    skipped: int
    issues: tuple[Issue, ...]
    moved: int = 0


def _source_version(path: Path) -> tuple[int, ...]:
    """Detect replacement or editing of an original before move cleanup."""
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _publish_copy(temp_path: Path, target: Path, on_conflict: str) -> bool:
    """Publish a complete copy, handling names claimed during copying too.

    Windows rename refuses to replace an existing path. Only the explicit
    overwrite policy uses replace; failure leaves the old destination intact.
    """
    if on_conflict == "overwrite":
        os.replace(temp_path, target)
        return True
    candidate = target
    suffix = 0
    while True:
        if not candidate.exists():
            try:
                os.rename(temp_path, candidate)
                return True
            except FileExistsError:
                # Another process published this name after the existence check.
                pass
        if on_conflict == "skip":
            return False
        suffix += 1
        candidate = target.with_name(f"{target.stem} ({suffix}){target.suffix}")


def copy_selected_files(
    result: BuildResult,
    source_dir: Path,
    dest_dir: Path,
    *,
    mode: str = "copy",
    on_conflict: str = "skip",
    progress_factory: ProgressFactory | None = None,
) -> CopySummary:
    """Copy student and photographer exports into metadata-defined locations.

    The first RAW pass copies student-selected originals. During the second
    pass, ``source_dir`` contains Capture One exports, including ``PH_`` files
    selected by the photographer. If Capture One has already created the
    location folder, that first relative component is not duplicated.

    Existing names follow on_conflict without content comparison. Copies are
    staged before publication; the same physical source and target is skipped.
    Move removes the original only after publication, including across volumes.
    """
    if mode not in {"copy", "move"}:
        raise ValueError(f"Неизвестный режим mode: {mode!r}")
    if on_conflict not in {"skip", "overwrite", "rename"}:
        raise ValueError(f"Неизвестный режим on_conflict: {on_conflict!r}")
    copied = skipped = moved = 0
    operation = "Перемещение" if mode == "move" else "Копирование"
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
            desc=f"{operation} выбранных файлов",
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
                        progress.set_description(f"{operation}: {source.name}")

                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if source.samefile(target):
                            skipped += 1
                            continue
                        if on_conflict == "skip":
                            skipped += 1
                            continue

                    fd, temp_name = tempfile.mkstemp(
                        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
                    )
                    os.close(fd)
                    temp_path = Path(temp_name)
                    try:
                        source_version = _source_version(source) if mode == "move" else None
                        shutil.copy2(source, temp_path)
                        if _publish_copy(temp_path, target, on_conflict):
                            copied += 1
                            if mode == "move":
                                try:
                                    if _source_version(source) != source_version:
                                        raise OSError("Исходный файл изменён во время перемещения.")
                                    source.unlink()
                                    moved += 1
                                except OSError as exc:
                                    issues.append(Issue(
                                        "error", "move_source_remove_failed",
                                        f"Копия сохранена, но оригинал не удалён: {source}. {exc}",
                                        record.number,
                                    ))
                        else:
                            skipped += 1
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
                        "move_failed" if mode == "move" else "copy_failed",
                        f"{source} -> {target}: {exc}",
                        record.number,
                    ))
                finally:
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return CopySummary(copied, skipped, tuple(issues), moved)
