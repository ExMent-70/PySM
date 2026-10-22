"""Audit destination locations and relocate files without replacing retouched work."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import stat
from typing import Callable

from .assignment_core import (
    Issue, _build_records, _load_object, _photo_file_stem,
    extract_photo_numbers, normalize_exclude_dirs,
)


def file_version(path: Path) -> tuple[int, ...]:
    """Detect replacement or editing without reading large PSD contents."""
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def analysis_version(path: Path) -> str:
    """Fingerprint analysis contents even when filesystem timestamps are reused."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(path: Path) -> str:
    return str(path.absolute()).casefold()


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
class LocationMove:
    """One proposed move with the source version and any destination conflict."""

    source: Path
    target: Path
    number: str
    version: tuple[int, ...]
    conflict: str = ""


@dataclass(frozen=True)
class LocationPlan:
    """Immutable snapshot displayed to the user before any changes."""
    root: Path
    analysis_path: Path
    analysis_signature: str
    exclude_dirs: tuple[str, ...]
    inventory: tuple[tuple[Path, tuple[int, ...]], ...]
    moves: tuple[LocationMove, ...]
    issues: tuple[Issue, ...]

    @property
    def executable(self) -> tuple[LocationMove, ...]:
        # Structural/metadata errors cannot be repaired by moving a file.
        if any(i.severity == "error" and i.code != "location_mismatch" for i in self.issues):
            return ()
        return tuple(move for move in self.moves if not move.conflict)


@dataclass(frozen=True)
class LocationRepair:
    """Actual successful moves and failures; no speculative rollback is reported."""

    moved: tuple[LocationMove, ...] = ()
    issues: tuple[Issue, ...] = ()


def plan_locations(root: Path, analysis_path: Path, exclude_dirs=()) -> LocationPlan:
    """Scan real metadata and every destination file; never synthesize locations."""
    root = root.absolute()
    signature = analysis_version(analysis_path)
    issues: list[Issue] = []
    records = _build_records(_load_object(analysis_path, "info_faces.json"), issues)
    excludes = tuple(normalize_exclude_dirs(exclude_dirs))
    excluded = {name.casefold() for name in excludes}
    inventory = []
    moves = []

    def error(message: str) -> None:
        issues.append(Issue("error", "location_scan_error", message))

    if not root.is_dir():
        error(f"Целевая папка не найдена: {root}")
    elif not _safe_path(root, root):
        error(f"Проверка локаций не поддерживает ссылки и junction: {root}")
    else:
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda exc: error(str(exc))):
            parent = Path(directory)
            allowed = []
            for name in sorted(dirs):
                if name.casefold() in excluded:
                    continue
                if not _safe_path(root, parent / name):
                    error(f"Небезопасная папка (ссылка/junction): {parent / name}")
                else:
                    allowed.append(name)
            dirs[:] = allowed
            for name in sorted(files):
                source = parent / name
                if not _safe_path(root, source):
                    error(f"Небезопасный файл (ссылка): {source}")
                    continue
                try:
                    version = file_version(source)
                except OSError as exc:
                    error(f"Не удалось проверить {source}: {exc}")
                    continue
                inventory.append((source, version))
                numbers = extract_photo_numbers(_photo_file_stem(source))
                if len(numbers) > 1:
                    error(f"Несколько номеров в имени, расположение не определено: {source}")
                    continue
                if not numbers:
                    continue
                number = numbers[0]
                if number not in records:
                    issues.append(Issue("warning", "location_unknown_number",
                                        f"Номер {number} отсутствует в info_faces.json: {source}", number))
                    continue
                relative = source.relative_to(root)
                location = records[number].location
                if len(relative.parts) > 1 and relative.parts[0].casefold() == location.casefold():
                    continue
                tail = relative.parts[1:] if len(relative.parts) > 1 else relative.parts
                target = root.joinpath(location, *tail)
                moves.append(LocationMove(source, target, number, version))

    counts = Counter(_key(move.target) for move in moves)
    checked = []
    for move in moves:
        conflict = ""
        if counts[_key(move.target)] > 1:
            conflict = "Несколько файлов претендуют на один путь"
        elif not _safe_path(root, move.target):
            conflict = "Небезопасный путь назначения"
        elif os.path.lexists(move.target):
            conflict = "Путь назначения уже занят"
        elif any(p.exists() and not p.is_dir() for p in move.target.parents if p != root):
            conflict = "Родительский путь занят файлом"
        elif any(part.casefold() in excluded for part in move.target.relative_to(root).parts[:-1]):
            conflict = "Целевая папка исключена из обработки"
        move = replace(move, conflict=conflict)
        checked.append(move)
        message = f"{move.source.relative_to(root)} → {move.target.relative_to(root)}"
        if conflict:
            message += f". Конфликт: {conflict}."
        issues.append(Issue("error", "location_mismatch", message, move.number))
    if analysis_version(analysis_path) != signature:
        error("info_faces.json изменился во время проверки. Обновите список.")
    return LocationPlan(root, analysis_path, signature, excludes, tuple(inventory), tuple(checked), tuple(issues))


def assert_plan_current(plan: LocationPlan) -> None:
    """Require the entire displayed snapshot to still match the filesystem."""
    if plan_locations(plan.root, plan.analysis_path, plan.exclude_dirs) != plan:
        raise ValueError("Данные изменились после проверки. Обновите список и повторите действие.")


def repair_locations(
    plan: LocationPlan,
    *,
    inputs_current: Callable[[], bool] = lambda: True,
    progress: Callable[[int, int], None] | None = None,
) -> LocationRepair:
    """Apply only free moves on Windows; preserve occupied targets and partial results."""
    moved = []
    issues = []
    try:
        assert_plan_current(plan)
        if not inputs_current():
            raise ValueError("Выбор изменился. Обновите список.")
        if os.name != "nt":
            raise OSError("Безопасное перемещение без перезаписи реализовано для Windows.")
        moves = plan.executable
        for index, move in enumerate(moves, 1):
            if not inputs_current() or analysis_version(plan.analysis_path) != plan.analysis_signature:
                raise ValueError("Исходные данные изменились. Оставшиеся файлы не перемещены.")
            try:
                if not _safe_path(plan.root, move.source) or not _safe_path(plan.root, move.target):
                    raise ValueError("Путь изменился или содержит ссылку/junction")
                if file_version(move.source) != move.version:
                    raise ValueError("Файл изменился после проверки")
                if os.path.lexists(move.target):
                    raise FileExistsError(f"Путь назначения занят: {move.target}")
                move.target.parent.mkdir(parents=True, exist_ok=True)
                if not _safe_path(plan.root, move.target):
                    raise ValueError("Путь назначения изменился")
                if (not inputs_current()
                        or analysis_version(plan.analysis_path) != plan.analysis_signature):
                    raise ValueError("Исходные данные изменились перед перемещением")
                if file_version(move.source) != move.version:
                    raise ValueError("Файл изменился перед перемещением")
                # Unlike shutil.move, Windows rename never replaces an occupied target.
                os.rename(move.source, move.target)
                moved.append(move)
            except (OSError, ValueError) as exc:
                issues.append(Issue("error", "location_move_failed", f"{move.source}: {exc}", move.number))
            if progress:
                progress(index, len(moves))
    except (OSError, ValueError) as exc:
        issues.append(Issue("error", "location_repair_stopped", str(exc)))
    return LocationRepair(tuple(moved), tuple(issues))
