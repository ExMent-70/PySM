"""Проверка JSON-контракта и подготовка данных HTML-отчёта."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

from student_roster import StudentRoster, normalize_student_id


def load_required_json(path: Path) -> dict[str, Any]:
    """Строго загружает обязательный JSON-объект."""

    if not path.is_file():
        raise FileNotFoundError(f"Обязательный файл не найден: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Корень {path.name} должен быть JSON-объектом.")
    return data


def load_optional_json(path: Path) -> dict[str, Any]:
    """Возвращает пустой объект только для отсутствующего необязательного JSON."""

    if not path.exists():
        return {}
    return load_required_json(path)


def normalize_cluster_label(value: Any, field_name: str) -> int:
    if value is None:
        return -1
    if isinstance(value, bool):
        raise ValueError(f"{field_name} содержит неверное значение {value!r}.")
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} содержит неверное значение {value!r}.") from exc
    if str(value).strip() != str(label):
        raise ValueError(f"{field_name} содержит неверное значение {value!r}.")
    return label


def _get_faces(filename: str, photo: Any) -> list[dict[str, Any]]:
    if not isinstance(photo, dict):
        raise ValueError(f"{filename}: запись фотографии должна быть объектом.")
    faces = photo.get("faces", [])
    if not isinstance(faces, list):
        raise ValueError(f"{filename}: поле faces должно быть массивом.")
    for index, face in enumerate(faces):
        if not isinstance(face, dict):
            raise ValueError(f"{filename}: faces[{index}] должен быть объектом.")
    return faces


def prepare_portrait_clusters(
    ref_data: dict[str, Any],
    roster: StudentRoster,
    face_info_factory: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Группирует портреты и проверяет однозначность cluster/student ID."""

    clusters: dict[str, dict[str, Any]] = {}
    labels_by_student: dict[str, int] = {}
    used_students: set[str] = set()

    for filename, photo in ref_data.items():
        faces = _get_faces(filename, photo)
        if photo.get("face_count") != 1:
            continue
        if len(faces) != 1:
            raise ValueError(
                f"{filename}: для портретной фотографии ожидалось ровно одно лицо."
            )
        face = faces[0]
        label = normalize_cluster_label(face.get("cluster_label"), "cluster_label")
        key = str(label)

        if label < 0:
            cluster = clusters.setdefault(key, {
                "cluster_label": label,
                "student_id": None,
                "display_name": "Шум (Ref)",
                "files": [],
            })
        else:
            try:
                student_id = normalize_student_id(face.get("student_id"), roster.list_id)
                display_name = roster.name_for(student_id)
            except ValueError as exc:
                raise ValueError(
                    f"{filename}: портретный кластер {label}: {exc} "
                    "Завершите идентификацию в cluster_editor."
                ) from exc

            previous_label = labels_by_student.setdefault(student_id, label)
            if previous_label != label:
                raise ValueError(
                    f"student_id {student_id} назначен портретным кластерам "
                    f"{previous_label} и {label}."
                )
            cluster = clusters.setdefault(key, {
                "cluster_label": label,
                "student_id": student_id,
                "display_name": display_name,
                "files": [],
            })
            if cluster["student_id"] != student_id:
                raise ValueError(
                    f"Портретный кластер {label} содержит student_id "
                    f"{cluster['student_id']} и {student_id}."
                )
            used_students.add(student_id)

        cluster["files"].append(face_info_factory(filename, face))

    for cluster in clusters.values():
        cluster["files"].sort(key=lambda item: item["filename"])
    return dict(sorted(clusters.items(), key=lambda item: int(item[0]))), used_students


def prepare_matches(
    matches_data: dict[str, Any],
    target_data: dict[str, Any],
    portrait_clusters: dict[str, dict[str, Any]],
    roster: StudentRoster,
    path_factory: Callable[[str], str],
) -> dict[str, dict[str, Any]]:
    """Проверяет matches и создаёт независимый контекст шаблона."""

    matches_by_label: dict[int, str] = {}
    labels_by_student: dict[str, int] = {}
    prepared: dict[str, dict[str, Any]] = {}

    for raw_label, raw_match in matches_data.items():
        label = normalize_cluster_label(raw_label, "Ключ matches")
        if label < 0 or str(label) != str(raw_label):
            raise ValueError(f"Ключ matches {raw_label!r} должен быть номером кластера.")
        if not isinstance(raw_match, dict):
            raise ValueError(f"matches[{raw_label!r}] должен быть объектом.")
        portrait = portrait_clusters.get(str(label))
        if portrait is None or label < 0:
            raise ValueError(f"matches[{label}] не имеет эталонного портретного кластера.")
        try:
            student_id = normalize_student_id(raw_match.get("student_id"), roster.list_id)
            display_name = roster.name_for(student_id)
        except ValueError as exc:
            raise ValueError(f"matches[{label}]: {exc}") from exc
        if portrait["student_id"] != student_id:
            raise ValueError(
                f"matches[{label}] содержит {student_id}, а портретный кластер — "
                f"{portrait['student_id']}."
            )
        previous_label = labels_by_student.setdefault(student_id, label)
        if previous_label != label:
            raise ValueError(
                f"student_id {student_id} назначен matches-кластерам "
                f"{previous_label} и {label}."
            )
        matches_by_label[label] = student_id

        photos = raw_match.get("group_photos", [])
        if not isinstance(photos, list):
            raise ValueError(f"matches[{label}].group_photos должен быть массивом.")
        processed_photos: list[dict[str, Any]] = []
        for index, photo in enumerate(photos):
            if not isinstance(photo, dict):
                raise ValueError(
                    f"matches[{label}].group_photos[{index}] должен быть объектом."
                )
            filename = str(photo.get("filename") or "").strip()
            if not filename:
                raise ValueError(
                    f"matches[{label}].group_photos[{index}] не содержит filename."
                )
            if filename not in target_data:
                raise ValueError(
                    f"matches[{label}] ссылается на отсутствующую в info_faces.json "
                    f"фотографию {filename}."
                )
            photo_entry = deepcopy(photo)
            photo_entry["rel_path"] = path_factory(filename)
            photo_entry["confidence"] = photo.get("min_distance")
            processed_photos.append(photo_entry)

        if processed_photos:
            prepared[str(label)] = {
                "cluster_label": label,
                "student_id": student_id,
                "display_name": display_name,
                "group_photos": processed_photos,
            }

    _validate_target_faces(target_data, matches_by_label, roster)
    return dict(sorted(prepared.items(), key=lambda item: int(item[0])))


def _validate_target_faces(
    target_data: dict[str, Any],
    matches_by_label: dict[int, str],
    roster: StudentRoster,
) -> None:
    for filename, photo in target_data.items():
        for face_index, face in enumerate(_get_faces(filename, photo)):
            raw_label = face.get("matched_portrait_cluster_label")
            if raw_label is None:
                continue
            label = normalize_cluster_label(raw_label, "matched_portrait_cluster_label")
            if label < 0:
                continue
            expected_id = matches_by_label.get(label)
            if expected_id is None:
                raise ValueError(
                    f"{filename}, лицо {face_index}: matches для кластера {label} не найден."
                )
            try:
                student_id = normalize_student_id(face.get("student_id"), roster.list_id)
                roster.name_for(student_id)
            except ValueError as exc:
                raise ValueError(f"{filename}, лицо {face_index}: {exc}") from exc
            if student_id != expected_id:
                raise ValueError(
                    f"{filename}, лицо {face_index}: student_id {student_id} не совпадает "
                    f"с matches {expected_id}."
                )
