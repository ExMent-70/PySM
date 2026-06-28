"""Build per-student photo assignments from selections and photographer exports."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
import re
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = 1
PHOTO_NUMBER_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
STUDENT_ID_RE = re.compile(r"^(?P<list_id>[A-HJ-NP-Z2-9]{4})-S\d{3}$")
PHOTOGRAPHER_PREFIX = "PH_"
LAYOUT_READY_SUFFIXES = {".jpg", ".jpeg", ".psd"}
KNOWN_FILE_ICON_SUFFIXES = {
    ".arw", ".cr2", ".cr3", ".dng", ".jpeg", ".jpg",
    ".nef", ".psd", ".raf", ".xmp",
}


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str
    photo_number: str = ""


@dataclass
class PhotoRecord:
    number: str
    analysis_filename: str
    location: str
    recognized_student_ids: set[str] = field(default_factory=set)
    selected_student_ids: set[str] = field(default_factory=set)
    photographer_selected: bool = False
    source_files: list[Path] = field(default_factory=list)
    destination_files: list[Path] = field(default_factory=list)

    @property
    def assigned_student_ids(self) -> set[str]:
        result = set(self.selected_student_ids)
        if self.photographer_selected:
            result.update(self.recognized_student_ids)
        return result


@dataclass
class BuildResult:
    list_id: str
    records: dict[str, PhotoRecord]
    assignments: dict[str, list[str]]
    issues: list[Issue]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def assignment_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "list_id": self.list_id,
            "assignments": self.assignments,
        }


def extract_photo_numbers(value: str) -> list[str]:
    return list(dict.fromkeys(PHOTO_NUMBER_RE.findall(str(value))))


def normalize_exclude_dirs(values: Iterable[str] | str) -> list[str]:
    """Accept PySM lists, serialized lists and space-separated legacy values."""
    raw_values = [values] if isinstance(values, str) else values
    result: list[str] = []
    for value in raw_values:
        for part in re.split(r"[\s,;]+", str(value).strip()):
            cleaned = part.strip(" \t\r\n'\"[]")
            if cleaned:
                result.append(PureWindowsPath(cleaned).name or Path(cleaned).name or cleaned)
    return list(dict.fromkeys(result))


def is_excluded_relative_path(relative: Path, exclude_dirs: Iterable[str] | str) -> bool:
    """Return True when any parent directory is listed in exclude_dirs."""
    excluded = {
        value.casefold() for value in normalize_exclude_dirs(exclude_dirs)
    }
    return any(part.casefold() in excluded for part in relative.parts[:-1])


def extract_one_photo_number(value: str, known: set[str] | None = None) -> str | None:
    numbers = extract_photo_numbers(value)
    if known is not None:
        numbers = [number for number in numbers if number in known]
    return numbers[0] if len(numbers) == 1 else None


def _load_object(path: Path, title: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден {title}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать {title} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{title} должен содержать JSON-объект.")
    return data


def load_roster(path: Path) -> tuple[str, set[str]]:
    data = _load_object(path, "*.list")
    list_id = str(data.get("list_id") or "").strip()
    students = data.get("students")
    if not re.fullmatch(r"[A-HJ-NP-Z2-9]{4}", list_id) or not isinstance(students, list):
        raise ValueError("Некорректные list_id или students в *.list.")
    ids: set[str] = set()
    for index, item in enumerate(students):
        student_id = str(item.get("student_id") if isinstance(item, dict) else "").strip()
        match = STUDENT_ID_RE.fullmatch(student_id)
        if not match or match.group("list_id") != list_id or student_id in ids:
            raise ValueError(f"Некорректный student_id в students[{index}]: {student_id!r}.")
        ids.add(student_id)
    return list_id, ids


def _build_records(
    info_faces: dict[str, Any], issues: list[Issue]
) -> dict[str, PhotoRecord]:
    records: dict[str, PhotoRecord] = {}
    for key, raw in info_faces.items():
        if not isinstance(raw, dict):
            continue
        filename = str(raw.get("filename") or key)
        numbers = extract_photo_numbers(filename)
        if len(numbers) != 1:
            issues.append(Issue(
                "error",
                "invalid_analysis_number",
                f"Не найден один шестизначный номер: {filename}",
            ))
            continue
        number = numbers[0]
        if number in records and records[number].analysis_filename != filename:
            issues.append(Issue(
                "error",
                "duplicate_number",
                f"Номер {number} относится к нескольким кадрам.",
                number,
            ))
            continue
        record = PhotoRecord(
            number=number,
            analysis_filename=filename,
            location=str(raw.get("location_name") or "unknown"),
        )
        faces = raw.get("faces", [])
        if isinstance(faces, list):
            for face in faces:
                if not isinstance(face, dict):
                    continue
                student_id = str(face.get("student_id") or "").strip()
                if student_id:
                    record.recognized_student_ids.add(student_id)
        records[number] = record
    return records


def _merge_matches(
    matches: dict[str, Any],
    records: dict[str, PhotoRecord],
    roster_ids: set[str],
    issues: list[Issue],
) -> None:
    for key, item in matches.items():
        if not isinstance(item, dict):
            issues.append(Issue(
                "error", "invalid_match", f"matches[{key}] должен быть объектом."
            ))
            continue
        student_id = str(item.get("student_id") or "").strip()
        if student_id not in roster_ids:
            issues.append(Issue(
                "error",
                "unknown_student",
                f"matches[{key}]: неизвестный {student_id!r}.",
            ))
            continue
        photos = item.get("group_photos", [])
        if not isinstance(photos, list):
            issues.append(Issue(
                "error",
                "invalid_group_photos",
                f"matches[{key}].group_photos должен быть массивом.",
            ))
            continue
        for photo in photos:
            filename = str(photo.get("filename") if isinstance(photo, dict) else "")
            number = extract_one_photo_number(filename)
            if number and number in records:
                records[number].recognized_student_ids.add(student_id)


def _merge_selection(
    selection: dict[str, Any],
    list_id: str,
    roster_ids: set[str],
    records: dict[str, PhotoRecord],
    issues: list[Issue],
) -> None:
    if selection.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"photo_selection.json: поддерживается schema_version={SCHEMA_VERSION}."
        )
    if selection.get("list_id") != list_id:
        raise ValueError("photo_selection.json относится к другому *.list.")
    if not str(selection.get("session_name") or "").strip():
        raise ValueError("photo_selection.json: отсутствует session_name.")
    if not str(selection.get("photo_session") or "").strip():
        raise ValueError("photo_selection.json: отсутствует photo_session.")
    students = selection.get("students", {})
    if not isinstance(students, dict):
        raise ValueError("photo_selection.json: students должен быть объектом.")
    for student_id, item in students.items():
        if student_id not in roster_ids or not isinstance(item, dict):
            issues.append(Issue(
                "error",
                "invalid_selection_student",
                f"Некорректная запись выбора: {student_id}.",
            ))
            continue
        numbers = item.get("selected_numbers", [])
        if not isinstance(numbers, list):
            issues.append(Issue(
                "error",
                "invalid_selection_numbers",
                f"selected_numbers {student_id} должен быть массивом.",
            ))
            continue
        for raw_number in numbers:
            number = str(raw_number)
            if not re.fullmatch(r"\d{6}", number):
                issues.append(Issue(
                    "error",
                    "invalid_photo_number",
                    f"Некорректный номер {number!r} у {student_id}.",
                ))
                continue
            record = records.get(number)
            if record is None:
                issues.append(Issue(
                    "warning",
                    "selection_not_analyzed",
                    f"Выбранный номер {number} отсутствует в info_faces.json.",
                    number,
                ))
                record = records.setdefault(
                    number, PhotoRecord(number, f"IMG_{number}.jpg", "unknown")
                )
            record.selected_student_ids.add(student_id)
            # A portrait may have no recognition evidence in matches. Warn only
            # when the frame does contain identified faces but not this student.
            if (
                record.recognized_student_ids
                and student_id not in record.recognized_student_ids
            ):
                issues.append(Issue(
                    "warning",
                    "student_not_recognized",
                    f"На выбранном кадре {number} не подтверждён {student_id}.",
                    number,
                ))


def _scan_files(
    root: Path,
    records: dict[str, PhotoRecord],
    issues: list[Issue],
    *,
    destination: bool,
    exclude_dirs: Iterable[str] = (),
) -> None:
    if not root.is_dir():
        issues.append(Issue("warning", "directory_missing", f"Папка не найдена: {root}"))
        return
    known = set(records)
    records_by_stem = {
        Path(record.analysis_filename).stem.casefold(): record
        for record in records.values()
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if is_excluded_relative_path(relative, exclude_dirs):
            continue

        record = None
        raw_numbers = extract_photo_numbers(path.stem)
        numbers = [number for number in raw_numbers if number in known]
        if len(numbers) > 1:
            issues.append(Issue(
                "error",
                "ambiguous_filename",
                f"Несколько известных номеров в имени: {path.name}",
            ))
            continue
        if len(numbers) == 1:
            record = records[numbers[0]]
        else:
            record = records_by_stem.get(path.stem.casefold())

        # Capture One exports selected-by-photographer files into source_dir
        # during the second RAW pass. The same prefix is also recognized in an
        # already populated destination so repeated runs remain idempotent.
        photographer_file = path.name.casefold().startswith(
            PHOTOGRAPHER_PREFIX.casefold()
        )
        if photographer_file and record is None and raw_numbers:
            issues.append(Issue(
                "warning",
                "photographer_file_not_analyzed",
                f"Файл фотографа отсутствует в info_faces.json: {path.name}",
                raw_numbers[0] if len(raw_numbers) == 1 else "",
            ))
        if record is None:
            continue

        if photographer_file:
            record.photographer_selected = True
        if destination:
            record.destination_files.append(path)
        else:
            record.source_files.append(path)


def build_assignments(
    *,
    student_list_file: Path,
    analysis_dir: Path,
    source_dir: Path,
    dest_dir: Path,
    exclude_dirs: Iterable[str] = (),
) -> BuildResult:
    issues: list[Issue] = []
    list_id, roster_ids = load_roster(student_list_file)
    records = _build_records(
        _load_object(analysis_dir / "info_faces.json", "info_faces.json"), issues
    )
    for record in records.values():
        unknown_ids = record.recognized_student_ids - roster_ids
        for student_id in sorted(unknown_ids):
            issues.append(Issue(
                "error",
                "unknown_recognized_student",
                f"Кадр {record.number} содержит неизвестный student_id {student_id!r}.",
                record.number,
            ))
        record.recognized_student_ids.intersection_update(roster_ids)
    matches_path = analysis_dir / "matches_portrait_to_group.json"
    if matches_path.is_file():
        _merge_matches(
            _load_object(matches_path, "matches_portrait_to_group.json"),
            records,
            roster_ids,
            issues,
        )
    selection = _load_object(analysis_dir / "photo_selection.json", "photo_selection.json")
    _merge_selection(selection, list_id, roster_ids, records, issues)
    _scan_files(source_dir, records, issues, destination=False, exclude_dirs=exclude_dirs)
    _scan_files(dest_dir, records, issues, destination=True, exclude_dirs=exclude_dirs)

    for record in records.values():
        if record.selected_student_ids and not record.source_files and not record.destination_files:
            issues.append(Issue(
                "warning",
                "selected_file_missing",
                f"Для выбранного номера {record.number} не найден ни один физический файл.",
                record.number,
            ))
        if (
            record.destination_files
            and not record.selected_student_ids
            and not record.photographer_selected
        ):
            issues.append(Issue(
                "warning",
                "unknown_selection_source",
                f"У кадра {record.number} нет префикса фотографа и персонального выбора.",
                record.number,
            ))
        for path in record.destination_files:
            try:
                relative = path.relative_to(dest_dir)
            except ValueError:
                continue
            actual_location = relative.parts[0] if len(relative.parts) > 1 else ""
            if (
                actual_location
                and actual_location.casefold() != record.location.casefold()
            ):
                issues.append(Issue(
                    "warning",
                    "location_folder_mismatch",
                    f"Кадр {record.number}: папка {actual_location!r} не совпадает с "
                    f"location_name {record.location!r}.",
                    record.number,
                ))
        if record.photographer_selected and not record.recognized_student_ids:
            issues.append(Issue(
                "warning",
                "photographer_photo_without_students",
                f"На выбранном фотографом кадре {record.number} нет распознанных учеников.",
                record.number,
            ))

    assignments: dict[str, list[str]] = {}
    selected_order: dict[str, list[str]] = {}
    for student_id, item in selection.get("students", {}).items():
        if student_id in roster_ids and isinstance(item, dict):
            selected_order[student_id] = [
                str(number)
                for number in item.get("selected_numbers", [])
                if str(number) in records
            ]
    for student_id in sorted(roster_ids):
        numbers = list(dict.fromkeys(selected_order.get(student_id, [])))
        numbers.extend(
            number for number, record in sorted(records.items())
            if student_id in record.assigned_student_ids and number not in numbers
        )
        if numbers:
            assignments[student_id] = numbers
    return BuildResult(list_id, records, assignments, issues)


def save_assignments(path: Path, result: BuildResult) -> None:
    publish_errors = publishability_issues(result)
    if result.has_errors:
        raise ValueError(
            "Назначения не записаны: присутствуют блокирующие ошибки."
        )
    if publish_errors:
        raise ValueError(
            "Назначения не записаны: часть назначенных фотографий отсутствует "
            "в целевой папке."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            json.dump(result.assignment_payload(), stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def has_layout_ready_destination_file(record: PhotoRecord) -> bool:
    """Return True when a destination contains an album-layout image file."""
    return any(
        path.suffix.casefold() in LAYOUT_READY_SUFFIXES
        for path in record.destination_files
    )


def publishability_issues(result: BuildResult) -> list[Issue]:
    """Return errors that make photo_assignments.json unsafe to publish."""
    issues: list[Issue] = []
    checked_numbers = {
        number
        for numbers in result.assignments.values()
        for number in numbers
    }
    for number in sorted(checked_numbers):
        record = result.records.get(number)
        if record is not None and has_layout_ready_destination_file(record):
            continue
        issues.append(Issue(
            "error",
            "assignment_layout_file_missing",
            f"Кадр {number} назначен ученикам, но в целевой папке нет JPG/JPEG/PSD файла для верстки.",
            number,
        ))
    return issues
