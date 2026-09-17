"""Strict loading and name resolution for the shared ``*.list`` file."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any


LIST_ID_PATTERN = re.compile(r"^[A-HJ-NP-Z2-9]{4}$")
STUDENT_ID_PATTERN = re.compile(r"^(?P<list_id>[A-HJ-NP-Z2-9]{4})-S\d{3}$")


def _name_key(value: str) -> str:
    return "".join(str(value).casefold().replace("ё", "е").split())


@dataclass(frozen=True)
class StudentRecord:
    student_id: str
    surname: str
    name: str
    patronymic: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def display_name(self) -> str:
        return " ".join(part for part in (self.surname, self.name, self.patronymic) if part)


class StudentRoster:
    def __init__(self, path: Path, list_id: str, students: list[StudentRecord]):
        self.path = path
        self.list_id = list_id
        self.students = tuple(students)
        self.by_id = {student.student_id: student for student in students}
        self._names: dict[str, list[str]] = {}
        for student in students:
            variants = (
                f"{student.surname} {student.name}",
                f"{student.name} {student.surname}",
                student.display_name,
            )
            for variant in variants:
                self._names.setdefault(_name_key(variant), []).append(student.student_id)

    def resolve_name(self, value: str) -> str | None:
        candidates = tuple(dict.fromkeys(self._names.get(_name_key(value), ())))
        return candidates[0] if len(candidates) == 1 else None


def normalize_student_id(value: Any, list_id: str) -> str:
    raw = str(value or "").strip()
    student_id = raw.upper()
    match = STUDENT_ID_PATTERN.fullmatch(student_id)
    if not match or raw != student_id:
        raise ValueError(f"student_id {raw or '<пусто>'} должен иметь формат A7K3-S001.")
    if match.group("list_id") != list_id:
        raise ValueError(f"student_id {student_id} относится к другому списку.")
    return student_id


def load_roster(path: Path) -> StudentRoster:
    if not path.is_file():
        raise FileNotFoundError(f"Файл списка не найден: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Корень *.list должен быть JSON-объектом.")
    list_id = str(data.get("list_id") or "").strip()
    if not LIST_ID_PATTERN.fullmatch(list_id):
        raise ValueError("Некорректный list_id в *.list.")
    raw_students = data.get("students")
    if not isinstance(raw_students, list) or not raw_students:
        raise ValueError("Поле students должно быть непустым массивом.")
    students: list[StudentRecord] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_students):
        if not isinstance(raw, dict):
            raise ValueError(f"students[{index}] должен быть объектом.")
        student_id = normalize_student_id(raw.get("student_id"), list_id)
        if student_id in seen:
            raise ValueError(f"student_id {student_id} повторяется.")
        surname = str(raw.get("surname") or "").strip()
        name = str(raw.get("name") or "").strip()
        if not surname or not name:
            raise ValueError(f"У записи {student_id} отсутствует фамилия или имя.")
        seen.add(student_id)
        students.append(StudentRecord(
            student_id,
            surname,
            name,
            str(raw.get("patronymic") or "").strip(),
            dict(raw),
        ))
    return StudentRoster(path.resolve(), list_id, students)
