"""Prepare copying from selection numbers and locations, without a roster."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

from .assignment_core import (
    SCHEMA_VERSION,
    BuildResult,
    Issue,
    PhotoRecord,
    _build_records,
    _load_object,
    _scan_files,
)


def _merge_copy_selection(
    selection: dict[str, Any],
    records: dict[str, PhotoRecord],
    issues: list[Issue],
) -> None:
    """Collect every selected number; student identity is irrelevant to copying."""
    if selection.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"photo_selection.json: поддерживается schema_version={SCHEMA_VERSION}."
        )
    students = selection.get("students", {})
    if not isinstance(students, dict):
        raise ValueError("photo_selection.json: students должен быть объектом.")
    for student_id, item in students.items():
        if not isinstance(item, dict):
            issues.append(Issue(
                "error", "invalid_selection_student",
                f"Некорректная запись выбора: {student_id}.",
            ))
            continue
        numbers = item.get("selected_numbers", [])
        if not isinstance(numbers, list):
            issues.append(Issue(
                "error", "invalid_selection_numbers",
                f"selected_numbers {student_id} должен быть массивом.",
            ))
            continue
        for raw_number in numbers:
            number = str(raw_number)
            if not re.fullmatch(r"\d{6}", number):
                issues.append(Issue(
                    "error", "invalid_photo_number",
                    f"Некорректный номер {number!r} у {student_id}.",
                ))
                continue
            if number not in records:
                issues.append(Issue(
                    "warning", "selection_not_analyzed",
                    f"Выбранный номер {number} отсутствует в info_faces.json.", number,
                ))
                records[number] = PhotoRecord(number, f"IMG_{number}.jpg", "unknown")
            # Keep the original copy-service record contract without validating
            # people or deriving assignments from recognition data.
            records[number].selected_student_ids.add(student_id)


def build_copy_selection(
    *,
    analysis_dir: Path,
    source_dir: Path,
    dest_dir: Path,
    exclude_dirs: Iterable[str] = (),
) -> BuildResult:
    """Reuse original file discovery and location checks for the copying stage.

    Only photo_selection.json and info_faces.json are read. Assignment-specific
    data, including the roster and optional matches file, is not required.
    """
    issues: list[Issue] = []
    records = _build_records(
        _load_object(analysis_dir / "info_faces.json", "info_faces.json"), issues
    )
    selection = _load_object(analysis_dir / "photo_selection.json", "photo_selection.json")
    _merge_copy_selection(selection, records, issues)
    _scan_files(source_dir, records, issues, destination=False, exclude_dirs=exclude_dirs)
    _scan_files(dest_dir, records, issues, destination=True, exclude_dirs=exclude_dirs)
    for record in records.values():
        if record.selected_student_ids and not record.source_files and not record.destination_files:
            issues.append(Issue(
                "warning", "selected_file_missing",
                f"Для выбранного номера {record.number} не найден ни один физический файл.",
                record.number,
            ))
    return BuildResult(str(selection.get("list_id") or ""), records, {}, issues)
