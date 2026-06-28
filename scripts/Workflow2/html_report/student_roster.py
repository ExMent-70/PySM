"""Загрузка и проверка реестра учеников из файла ``*.list``."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


LIST_ID_PATTERN = re.compile(r"^[A-HJ-NP-Z2-9]{4}$")
STUDENT_ID_PATTERN = re.compile(
    r"^(?P<list_id>[A-HJ-NP-Z2-9]{4})-S(?P<number>\d{3})$"
)


@dataclass(frozen=True)
class StudentRecord:
    student_id: str
    surname: str
    name: str

    @property
    def display_name(self) -> str:
        return f"{self.surname} {self.name}".strip()


class StudentRoster:
    """Неизменяемое отображение ``student_id -> ученик``."""

    def __init__(self, path: Path, list_id: str, students: list[StudentRecord]):
        self.path = path
        self.list_id = list_id
        self._students = {student.student_id: student for student in students}

    @property
    def students(self) -> tuple[StudentRecord, ...]:
        return tuple(self._students.values())

    def name_for(self, student_id: str) -> str:
        student = self._students.get(student_id)
        if student is None:
            raise ValueError(
                f"student_id {student_id!r} отсутствует в {self.path.name}."
            )
        return student.display_name


def normalize_student_id(value: Any, expected_list_id: str) -> str:
    """Проверяет формат ID и его принадлежность текущему списку."""

    raw_value = str(value or "").strip()
    student_id = raw_value.upper()
    match = STUDENT_ID_PATTERN.fullmatch(student_id)
    if not match:
        raise ValueError(
            f"student_id {student_id or '<пусто>'} должен иметь формат A7K3-S001."
        )
    if raw_value != student_id:
        raise ValueError(f"student_id {raw_value!r} должен быть записан как {student_id}.")
    if match.group("list_id") != expected_list_id:
        raise ValueError(
            f"student_id {student_id} относится к списку {match.group('list_id')}, "
            f"ожидался {expected_list_id}."
        )
    if not 1 <= int(match.group("number")) <= 999:
        raise ValueError(f"student_id {student_id} выходит за диапазон S001-S999.")
    return student_id


def load_student_roster(path: Path) -> StudentRoster:
    """Загружает ``*.list`` и проверяет обязательные поля идентичности."""

    if not path.is_file():
        raise FileNotFoundError(f"Файл списка учеников не найден: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать список учеников {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Корень файла *.list должен быть JSON-объектом.")

    list_id = str(data.get("list_id") or "").strip().upper()
    if not LIST_ID_PATTERN.fullmatch(list_id):
        raise ValueError(
            "Поле list_id должно содержать четыре допустимых символа, например A7K3."
        )
    raw_students = data.get("students")
    if not isinstance(raw_students, list) or not raw_students:
        raise ValueError("Файл *.list не содержит непустой массив students.")

    students: list[StudentRecord] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_students):
        if not isinstance(raw, dict):
            raise ValueError(f"Запись students[{index}] должна быть объектом.")
        student_id = normalize_student_id(raw.get("student_id"), list_id)
        if student_id in seen:
            raise ValueError(f"student_id {student_id} повторяется в файле *.list.")
        surname = str(raw.get("surname") or "").strip()
        name = str(raw.get("name") or "").strip()
        if not surname or not name:
            raise ValueError(
                f"У записи {student_id} должны быть заполнены surname и name."
            )
        seen.add(student_id)
        students.append(StudentRecord(student_id, surname, name))

    return StudentRoster(path.resolve(), list_id, students)
