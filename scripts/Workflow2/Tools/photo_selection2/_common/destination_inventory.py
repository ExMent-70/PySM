"""Read-only destination snapshots used to publish a current layout plan."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import stat
from typing import Iterable

from .assignment_core import (
    Issue, _photo_file_stem, extract_photo_numbers, normalize_exclude_dirs,
)


def file_version(path: Path) -> tuple[int, ...]:
    """Detect replacement or editing without reading large PSD contents."""
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def analysis_version(path: Path) -> str:
    """Fingerprint analysis contents even when filesystem timestamps are reused."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_path(root: Path, path: Path) -> bool:
    """Reject escapes and all symlink/reparse ancestors, including junctions."""
    try:
        path.absolute().relative_to(root.absolute())
        for component in (path.absolute(), *path.absolute().parents):
            if component.is_symlink():
                return False
            try:
                value = component.lstat()
            except FileNotFoundError:
                continue
            if getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                return False
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class DestinationInventory:
    """Flat files and analysis version inspected before atomic publication."""
    root: Path
    analysis_path: Path
    analysis_signature: str
    exclude_dirs: tuple[str, ...]
    inventory: tuple[tuple[Path, tuple[int, ...]], ...]
    issues: tuple[Issue, ...]


def scan_destination(
    root: Path, analysis_path: Path, exclude_dirs: Iterable[str] | str = (),
) -> DestinationInventory:
    """Inspect only direct files; subfolders never define a logical location."""
    root = root.absolute()
    signature = analysis_version(analysis_path)
    excludes = tuple(normalize_exclude_dirs(exclude_dirs))
    excluded = {name.casefold() for name in excludes}
    issues: list[Issue] = []
    inventory: list[tuple[Path, tuple[int, ...]]] = []
    try:
        if not root.is_dir() or not _safe_path(root, root):
            raise ValueError(f"Целевая папка не найдена или содержит ссылку/junction: {root}")
        for path in sorted(root.iterdir()):
            if path.name.startswith('.'):
                continue
            if path.name.casefold() in excluded and path.is_dir():
                continue
            if not _safe_path(root, path):
                raise ValueError(f"Небезопасный путь (ссылка/junction): {path}")
            if not path.is_file():
                continue
            inventory.append((path, file_version(path)))
            if len(extract_photo_numbers(_photo_file_stem(path))) > 1:
                issues.append(Issue(
                    "error", "ambiguous_filename", f"Несколько номеров в имени: {path.name}",
                ))
    except (OSError, ValueError) as exc:
        issues.append(Issue("error", "destination_scan_error", str(exc)))
    if analysis_version(analysis_path) != signature:
        issues.append(Issue(
            "error", "analysis_changed", "Анализ изменён во время проверки. Обновите список.",
        ))
    return DestinationInventory(
        root, analysis_path, signature, excludes,
        tuple(inventory), tuple(issues),
    )


def assert_inventory_current(inventory: DestinationInventory) -> None:
    """Reject changed inputs or file versions before publishing the JSON."""
    if scan_destination(inventory.root, inventory.analysis_path, inventory.exclude_dirs) != inventory:
        raise ValueError("Данные изменились после проверки. Обновите список и повторите действие.")
