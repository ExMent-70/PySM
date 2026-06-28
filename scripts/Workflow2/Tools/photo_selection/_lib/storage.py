"""Strict and atomic persistence for photo-selection documents."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .domain import SCHEMA_VERSION, SelectionDocument, StudentSelection
from .roster import StudentRoster, normalize_student_id


def load_document(
    path: Path,
    roster: StudentRoster,
    session_name: str,
    photo_session: str,
) -> SelectionDocument:
    document = SelectionDocument(roster.list_id, session_name, photo_session)
    if not path.exists():
        return document
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать selection JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Неподдерживаемая версия selection JSON.")
    if data.get("list_id") != roster.list_id:
        raise ValueError("Selection JSON относится к другому *.list.")
    if data.get("session_name") != session_name or data.get("photo_session") != photo_session:
        raise ValueError("Selection JSON относится к другой сессии или фотосессии.")
    raw_students = data.get("students", {})
    if not isinstance(raw_students, dict):
        raise ValueError("Поле students selection JSON должно быть объектом.")
    for raw_id, raw in raw_students.items():
        student_id = normalize_student_id(raw_id, roster.list_id)
        if student_id not in roster.by_id or not isinstance(raw, dict):
            raise ValueError(f"Некорректная запись selection для {student_id}.")
        numbers = raw.get("selected_numbers", [])
        if not isinstance(numbers, list):
            raise ValueError(f"selected_numbers {student_id} должен быть массивом.")
        document.students[student_id] = StudentSelection(
            [str(number) for number in numbers],
            bool(raw.get("responded", True)),
            str(raw.get("source") or "manual"),
        )
    return document


def save_document(path: Path, document: SelectionDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            json.dump(document.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise
