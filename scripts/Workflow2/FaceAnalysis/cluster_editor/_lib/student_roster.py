"""Загрузка единственного источника ФИО для редактора кластеров."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


LIST_ID_PATTERN = re.compile(r"^[A-HJ-NP-Z2-9]{4}$")
STUDENT_ID_PATTERN = re.compile(
    r"^(?P<list_id>[A-HJ-NP-Z2-9]{4})-S(?P<number>\d{3})$"
)


@dataclass(frozen=True)
class StudentRecord:
    """Минимальные данные ученика, необходимые редактору."""

    student_id: str
    surname: str
    name: str

    @property
    def display_name(self) -> str:
        return f"{self.surname} {self.name}".strip()

    @property
    def display_label(self) -> str:
        return f"{self.display_name} [{self.student_id}]"


class StudentRoster:
    """Проверенный реестр записей одного файла ``*.list``."""

    def __init__(self, path: Path, list_id: str, students: Iterable[StudentRecord]):
        self.path = path
        self.list_id = list_id
        self._students = {student.student_id: student for student in students}

    @property
    def students(self) -> tuple[StudentRecord, ...]:
        return tuple(self._students.values())

    def contains(self, student_id: str | None) -> bool:
        return bool(student_id and student_id in self._students)

    def get(self, student_id: str | None) -> StudentRecord | None:
        return self._students.get(student_id or "")

    def name_for(self, student_id: str | None) -> str:
        if not student_id:
            return ""
        student = self.get(student_id)
        return student.display_name if student else f"Неизвестный ID [{student_id}]"

    def label_for(self, student_id: str | None) -> str:
        if not student_id:
            return ""
        student = self.get(student_id)
        return student.display_label if student else f"Неизвестный ID [{student_id}]"

    def available(self, assigned_ids: Iterable[str]) -> tuple[StudentRecord, ...]:
        assigned = set(assigned_ids)
        return tuple(s for s in self.students if s.student_id not in assigned)


def _normalize_list_id(value: Any) -> str:
    list_id = str(value or "").strip().upper()
    if not LIST_ID_PATTERN.fullmatch(list_id):
        raise ValueError(
            "Поле list_id должно содержать четыре допустимых символа, "
            "например A7K3."
        )
    return list_id


def _normalize_student_id(value: Any, list_id: str) -> str:
    student_id = str(value or "").strip().upper()
    match = STUDENT_ID_PATTERN.fullmatch(student_id)
    if not match:
        raise ValueError(f"student_id {student_id or '<пусто>'} имеет неверный формат.")
    if match.group("list_id") != list_id:
        raise ValueError(
            f"student_id {student_id} относится к списку {match.group('list_id')}, "
            f"ожидался {list_id}."
        )
    number = int(match.group("number"))
    if not 1 <= number <= 999:
        raise ValueError(f"student_id {student_id} выходит за диапазон S001-S999.")
    return student_id


def load_student_roster(path: Path) -> StudentRoster:
    """Загружает и строго проверяет файл списка учеников."""

    if not path.is_file():
        raise FileNotFoundError(f"Файл списка учеников не найден: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать список учеников {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Корень файла *.list должен быть JSON-объектом.")

    list_id = _normalize_list_id(data.get("list_id"))
    raw_students = data.get("students")
    if not isinstance(raw_students, list) or not raw_students:
        raise ValueError("Файл *.list не содержит непустой массив students.")

    students: list[StudentRecord] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_students, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Запись students[{index - 1}] должна быть объектом.")
        student_id = _normalize_student_id(raw.get("student_id"), list_id)
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
