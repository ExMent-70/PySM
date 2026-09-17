"""Flexible CSV reading and conversion to selection import entries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .domain import ImportEntry
from .number_parser import extract_numbers
from .roster import StudentRoster, normalize_student_id


@dataclass(frozen=True)
class CsvTable:
    path: Path
    headers: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    encoding: str
    delimiter: str


def _read_text(path: Path) -> tuple[str, str]:
    """Read a CSV-like text file without letting a permissive codec hide its BOM."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeError:
            continue
    raise ValueError(f"Не удалось определить кодировку CSV: {path}")


def read_csv_table(path: Path, *, delimiter: str | None = None) -> CsvTable:
    text, encoding_used = _read_text(path)
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=";,\t").delimiter
        except csv.Error:
            delimiter = ";"
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError("CSV не содержит строку заголовков.")
    headers = tuple(str(header or "").strip() for header in reader.fieldnames)
    rows = tuple(
        {str(key or "").strip(): str(value or "").strip() for key, value in row.items()}
        for row in reader
    )
    return CsvTable(path, headers, rows, encoding_used, delimiter)


def suggest_columns(headers: Iterable[str]) -> tuple[str | None, list[str]]:
    headers = list(headers)
    identity_words = ("фио", "имя", "ученик", "student", "name")
    number_words = ("фото", "номер", "кадр", "photo", "image")
    identity = next(
        (header for header in headers if any(word in header.casefold() for word in identity_words)),
        None,
    )
    numbers = [
        header for header in headers
        if any(word in header.casefold() for word in number_words) and header != identity
    ]
    return identity, numbers


def import_table(
    table: CsvTable,
    roster: StudentRoster,
    identity_column: str,
    number_columns: Iterable[str],
    *,
    min_digits: int,
    max_digits: int,
    pad_to_digits: int = 0,
) -> tuple[list[ImportEntry], list[dict[str, object]]]:
    number_columns = tuple(number_columns)
    if identity_column not in table.headers or not number_columns:
        raise ValueError("Не выбраны столбец идентификации и столбцы номеров.")
    entries: list[ImportEntry] = []
    unresolved: list[dict[str, object]] = []
    for row_number, row in enumerate(table.rows, start=2):
        source_person = row.get(identity_column, "").strip()
        if not source_person:
            continue
        numbers = extract_numbers(
            (row.get(column, "") for column in number_columns),
            min_digits=min_digits,
            max_digits=max_digits,
            pad_to_digits=pad_to_digits,
        )
        student_id = None
        if source_person.upper() in roster.by_id:
            student_id = normalize_student_id(source_person.upper(), roster.list_id)
        else:
            student_id = roster.resolve_name(source_person)
        if student_id is None:
            unresolved.append({
                "source_person": source_person,
                "selected_numbers": numbers,
                "reason": "ФИО не найдено или неоднозначно",
                "row": row_number,
            })
            continue
        entries.append(ImportEntry(student_id, tuple(numbers), source_person, True))
    return entries, unresolved


def import_personal_file(
    path: Path,
    roster: StudentRoster,
    *,
    min_digits: int,
    max_digits: int,
    pad_to_digits: int = 0,
) -> ImportEntry | None:
    student_id = roster.resolve_name(path.stem)
    if student_id is None:
        return None
    numbers = read_personal_numbers(
        path,
        min_digits=min_digits,
        max_digits=max_digits,
        pad_to_digits=pad_to_digits,
    )
    return ImportEntry(student_id, tuple(numbers), path.stem, True)


def read_personal_numbers(
    path: Path,
    *,
    min_digits: int,
    max_digits: int,
    pad_to_digits: int = 0,
) -> list[str]:
    text, _encoding = _read_text(path)
    return extract_numbers(
        text.splitlines(),
        min_digits=min_digits,
        max_digits=max_digits,
        pad_to_digits=pad_to_digits,
    )
