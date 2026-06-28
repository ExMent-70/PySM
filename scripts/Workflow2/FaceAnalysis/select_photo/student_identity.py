"""Идентичность учеников для поиска, отображения и именования фотографий."""

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
WINDOWS_FORBIDDEN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class StudentRecord:
    student_id: str
    surname: str
    name: str

    @property
    def display_name(self) -> str:
        return f"{self.surname} {self.name}".strip()


class StudentRoster:
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


@dataclass(frozen=True)
class PhotoIdentity:
    student_ids: tuple[str, ...]
    student_names: tuple[str, ...]

    @property
    def display_students(self) -> str:
        return ", ".join(self.student_names)

    @property
    def rename_person_name(self) -> str:
        return self.student_names[0] if len(self.student_names) == 1 else ""


def format_students_for_table(student_names: Iterable[str]) -> tuple[str, str]:
    """Возвращает компактную ячейку и многострочную подсказку со всеми ФИО."""

    names = tuple(str(name).strip() for name in student_names if str(name).strip())
    if not names:
        return "", ""

    tooltip = "Распознанные ученики:\n" + "\n".join(f"• {name}" for name in names)
    if len(names) == 1:
        return names[0], tooltip
    return f"Список учеников ({len(names)})", tooltip


def normalize_student_id(value: Any, expected_list_id: str) -> str:
    student_id = str(value or "").strip().upper()
    match = STUDENT_ID_PATTERN.fullmatch(student_id)
    if not match:
        raise ValueError(
            f"student_id {student_id or '<пусто>'} должен иметь формат A7K3-S001."
        )
    if match.group("list_id") != expected_list_id:
        raise ValueError(
            f"student_id {student_id} относится к списку {match.group('list_id')}, "
            f"ожидался {expected_list_id}."
        )
    number = int(match.group("number"))
    if not 1 <= number <= 999:
        raise ValueError(f"student_id {student_id} выходит за диапазон S001-S999.")
    return student_id


def load_student_roster(path: Path) -> StudentRoster:
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


def collect_photo_identity(
    faces: Iterable[dict[str, Any]], roster: StudentRoster
) -> PhotoIdentity:
    """Собирает уникальные ID в порядке лиц и разрешает их ФИО."""

    student_ids: list[str] = []
    seen: set[str] = set()
    for face in faces:
        if not isinstance(face, dict):
            continue
        raw_student_id = face.get("student_id")
        if raw_student_id is None or not str(raw_student_id).strip():
            continue
        student_id = normalize_student_id(raw_student_id, roster.list_id)
        roster.name_for(student_id)
        if str(raw_student_id).strip() != student_id:
            raise ValueError(
                f"student_id {raw_student_id!r} должен быть записан как {student_id}."
            )
        if student_id not in seen:
            seen.add(student_id)
            student_ids.append(student_id)

    names = tuple(roster.name_for(student_id) for student_id in student_ids)
    return PhotoIdentity(tuple(student_ids), names)


def safe_windows_component(value: str) -> str:
    """Удаляет символы, запрещённые в компоненте имени файла Windows."""

    cleaned = WINDOWS_FORBIDDEN_CHARS.sub("_", str(value)).strip().rstrip(". ")
    return cleaned or "student"


def build_rename_stem(identity: PhotoIdentity, number: str, original_stem: str) -> str:
    """Переименовывает только фотографию ровно одного ученика."""

    if not identity.rename_person_name:
        return ""
    name = safe_windows_component(identity.rename_person_name)
    return f"{name}-{number}"


def _non_noise_label(face: dict[str, Any], key: str) -> int | None:
    value = face.get(key)
    if value is None:
        return None
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Поле {key} содержит неверное значение {value!r}.") from exc
    return label if label >= 0 else None


def validate_identity_contract(
    metadata: dict[str, Any], matches_data: dict[str, Any], roster: StudentRoster
) -> None:
    """Проверяет согласованность `.list`, info_faces и matches."""

    matches_by_label: dict[str, str] = {}
    student_labels: dict[str, str] = {}
    for cluster_key, record in matches_data.items():
        if not isinstance(record, dict):
            raise ValueError(f"matches[{cluster_key!r}] должен быть объектом.")
        raw_student_id = record.get("student_id")
        if raw_student_id is None or not str(raw_student_id).strip():
            raise ValueError(f"matches[{cluster_key!r}] не содержит student_id.")
        student_id = normalize_student_id(raw_student_id, roster.list_id)
        roster.name_for(student_id)
        if str(raw_student_id).strip() != student_id:
            raise ValueError(
                f"matches[{cluster_key!r}] содержит ненормализованный "
                f"student_id {raw_student_id!r}."
            )
        normalized_label = str(int(cluster_key)) if str(cluster_key).lstrip("-").isdigit() else None
        if normalized_label is None:
            raise ValueError(f"Ключ matches {cluster_key!r} не является номером кластера.")
        previous_label = student_labels.setdefault(student_id, normalized_label)
        if previous_label != normalized_label:
            raise ValueError(
                f"student_id {student_id} назначен кластерам "
                f"{previous_label} и {normalized_label}."
            )
        existing_match_id = matches_by_label.get(normalized_label)
        if existing_match_id is not None and existing_match_id != student_id:
            raise ValueError(
                f"Кластер matches {normalized_label} содержит student_id "
                f"{existing_match_id} и {student_id}."
            )
        matches_by_label[normalized_label] = student_id

    portrait_ids: dict[int, str] = {}
    portrait_labels_by_student: dict[str, int] = {}
    for filename, file_data in metadata.items():
        if not isinstance(file_data, dict):
            raise ValueError(f"{filename}: запись фотографии должна быть объектом.")
        faces = file_data.get("faces", [])
        if not isinstance(faces, list):
            raise ValueError(f"{filename}: поле faces должно быть массивом.")
        for face_index, face in enumerate(faces):
            if not isinstance(face, dict):
                raise ValueError(f"{filename}: faces[{face_index}] должен быть объектом.")
            raw_student_id = face.get("student_id")
            student_id: str | None = None
            if raw_student_id is not None and str(raw_student_id).strip():
                try:
                    student_id = normalize_student_id(raw_student_id, roster.list_id)
                    roster.name_for(student_id)
                    if str(raw_student_id).strip() != student_id:
                        raise ValueError(
                            f"student_id {raw_student_id!r} должен быть записан как {student_id}."
                        )
                except ValueError as exc:
                    raise ValueError(f"{filename}, лицо {face_index}: {exc}") from exc

            try:
                portrait_label = _non_noise_label(face, "cluster_label")
                matched_label = _non_noise_label(
                    face, "matched_portrait_cluster_label"
                )
            except ValueError as exc:
                raise ValueError(f"{filename}, лицо {face_index}: {exc}") from exc

            if portrait_label is not None and student_id is None:
                raise ValueError(
                    f"{filename}, лицо {face_index}: портретный кластер "
                    f"{portrait_label} не содержит student_id."
                )
            if portrait_label is not None and student_id is not None:
                previous_id = portrait_ids.setdefault(portrait_label, student_id)
                if previous_id != student_id:
                    raise ValueError(
                        f"Портретный кластер {portrait_label} содержит "
                        f"student_id {previous_id} и {student_id}."
                    )
                previous_label = portrait_labels_by_student.setdefault(
                    student_id, portrait_label
                )
                if previous_label != portrait_label:
                    raise ValueError(
                        f"student_id {student_id} назначен портретным кластерам "
                        f"{previous_label} и {portrait_label}."
                    )
                expected_match_id = matches_by_label.get(str(portrait_label))
                if expected_match_id is not None and expected_match_id != student_id:
                    raise ValueError(
                        f"Портретный кластер {portrait_label}: student_id "
                        f"{student_id} не совпадает с matches {expected_match_id}."
                    )
            if matched_label is not None:
                expected_id = matches_by_label.get(str(matched_label))
                if expected_id is None:
                    raise ValueError(
                        f"{filename}, лицо {face_index}: кластер matches "
                        f"{matched_label} не найден."
                    )
                if student_id != expected_id:
                    raise ValueError(
                        f"{filename}, лицо {face_index}: student_id "
                        f"{student_id!r} не совпадает с {expected_id}."
                    )
