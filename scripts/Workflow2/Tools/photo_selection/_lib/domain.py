"""Domain model for per-student photo-number selections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


SCHEMA_VERSION = 1


def unique_numbers(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        number = str(value).strip()
        if number and number not in seen:
            seen.add(number)
            result.append(number)
    return result


@dataclass
class StudentSelection:
    selected_numbers: list[str] = field(default_factory=list)
    responded: bool = True
    source: str = "manual"

    def to_dict(self) -> dict:
        return {
            "selected_numbers": unique_numbers(self.selected_numbers),
            "responded": bool(self.responded),
            "source": self.source,
        }


@dataclass
class SelectionDocument:
    list_id: str
    session_name: str
    photo_session: str
    students: dict[str, StudentSelection] = field(default_factory=dict)

    def apply(
        self,
        student_id: str,
        numbers: Iterable[str],
        *,
        source: str,
        mode: str = "replace",
        responded: bool = True,
    ) -> None:
        incoming = unique_numbers(numbers)
        if mode not in {"replace", "merge"}:
            raise ValueError("Режим импорта должен быть replace или merge.")
        if mode == "merge" and student_id in self.students:
            incoming = unique_numbers([
                *self.students[student_id].selected_numbers,
                *incoming,
            ])
        self.students[student_id] = StudentSelection(incoming, responded, source)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "list_id": self.list_id,
            "session_name": self.session_name,
            "photo_session": self.photo_session,
            "students": {
                student_id: selection.to_dict()
                for student_id, selection in self.students.items()
            },
        }


@dataclass(frozen=True)
class ImportEntry:
    student_id: str
    selected_numbers: tuple[str, ...]
    source_person: str = ""
    responded: bool = True


def coalesce_import_entries(entries: Iterable[ImportEntry]) -> list[ImportEntry]:
    """Merge repeated student rows without losing earlier photo numbers."""

    merged: dict[str, ImportEntry] = {}
    for entry in entries:
        previous = merged.get(entry.student_id)
        numbers = unique_numbers(
            (*previous.selected_numbers, *entry.selected_numbers)
            if previous is not None
            else entry.selected_numbers
        )
        merged[entry.student_id] = ImportEntry(
            entry.student_id,
            tuple(numbers),
            entry.source_person or (previous.source_person if previous else ""),
            entry.responded,
        )
    return list(merged.values())
