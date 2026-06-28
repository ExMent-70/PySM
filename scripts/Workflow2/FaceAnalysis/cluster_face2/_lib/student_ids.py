"""Контракт идентификаторов учеников для кластеризации лиц."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, MutableMapping, Sequence

import numpy as np


STUDENT_ID_PATTERN = re.compile(
    r"^(?P<list_id>[A-HJ-NP-Z2-9]{4})-S(?P<number>\d{3})$"
)


@dataclass(frozen=True)
class StudentIdList:
    """Проверенный список ID одной фотосессии в порядке съёмки."""

    path: Path
    list_id: str
    student_ids: tuple[str, ...]


def remove_legacy_name_fields(face: MutableMapping[str, Any]) -> None:
    """Удаляет устаревшие поля ФИО, не затрагивая student_id и temp-кластеры."""

    face.pop("child_name", None)
    face.pop("matched_child_name", None)


def parse_student_id(value: str) -> tuple[str, int]:
    """Проверяет полный ID и возвращает префикс списка и номер записи."""

    normalized = str(value).strip().upper()
    match = STUDENT_ID_PATTERN.fullmatch(normalized)
    if not match:
        raise ValueError("student_id должен иметь формат A7K3-S001")

    number = int(match.group("number"))
    if not 1 <= number <= 999:
        raise ValueError("номер student_id должен быть в диапазоне S001–S999")
    return match.group("list_id"), number


def load_student_ids(path: Path) -> StudentIdList:
    """Загружает TXT, сохраняя порядок и строго проверяя каждую строку."""

    if not path.is_file():
        raise FileNotFoundError(f"Файл идентификаторов фотосессии не найден: {path}")

    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OSError(f"Не удалось прочитать файл идентификаторов {path}: {exc}") from exc

    student_ids: list[str] = []
    seen: set[str] = set()
    list_id: str | None = None
    for line_number, raw_line in enumerate(raw_lines, start=1):
        value = raw_line.strip().upper()
        if not value:
            continue
        try:
            current_list_id, _ = parse_student_id(value)
        except ValueError as exc:
            raise ValueError(f"Строка {line_number} файла {path.name}: {exc}.") from exc

        if list_id is None:
            list_id = current_list_id
        elif current_list_id != list_id:
            raise ValueError(
                f"Строка {line_number} файла {path.name}: student_id {value} "
                f"относится к списку {current_list_id}, ожидался {list_id}."
            )
        if value in seen:
            raise ValueError(
                f"Строка {line_number} файла {path.name}: student_id {value} повторяется."
            )
        seen.add(value)
        student_ids.append(value)

    if not student_ids or list_id is None:
        raise ValueError(f"Файл идентификаторов пуст: {path}")
    return StudentIdList(path=path, list_id=list_id, student_ids=tuple(student_ids))


def find_student_ids_file(
    target_dir: Path,
    photo_session: str,
    children_file_name: str,
) -> Path:
    """Находит файл конкретной фотосессии без fallback на children.txt."""

    photo_session = str(photo_session).strip()
    children_file_name = str(children_file_name).strip()
    if not photo_session or not children_file_name:
        raise ValueError(
            "Не заданы wf_photo_session или wf_children_file_name; "
            "невозможно определить файл идентификаторов фотосессии."
        )

    filename = f"{photo_session}_{children_file_name}"
    if not filename.lower().endswith(".txt"):
        filename += ".txt"

    candidates = (target_dir.parent / filename, target_dir.parent.parent / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Файл идентификаторов текущей фотосессии не найден. "
        f"Проверены пути:\n{checked}"
    )


def build_cluster_student_map(
    labels: np.ndarray,
    filenames: Sequence[str],
    student_ids: Sequence[str],
) -> dict[int, str]:
    """Строит полную карту кластеров или возвращает пустую при расхождении."""

    if len(labels) != len(filenames):
        raise ValueError(
            "Количество меток кластеризации не совпадает с количеством файлов."
        )

    clusters: dict[int, list[str]] = defaultdict(list)
    for index, label in enumerate(labels):
        label_int = int(label)
        if label_int != -1:
            clusters[label_int].append(filenames[index])

    ordered_cluster_ids = sorted(
        clusters,
        key=lambda cluster_id: min(
            _filename_sort_key(filename) for filename in clusters[cluster_id]
        ),
    )
    if len(ordered_cluster_ids) != len(student_ids):
        # Частичное сопоставление по порядку небезопасно: один лишний или
        # объединённый кластер сдвинет все последующие student_id. Кластеры
        # сохраняются без ID и затем идентифицируются в cluster_editor.
        return {}

    return dict(zip(ordered_cluster_ids, student_ids))


def validate_single_list(student_ids: Iterable[str]) -> str:
    """Проверяет формат и общий list_id набора эталонных профилей."""

    list_id: str | None = None
    for student_id in student_ids:
        current_list_id, _ = parse_student_id(student_id)
        if list_id is None:
            list_id = current_list_id
        elif current_list_id != list_id:
            raise ValueError(
                f"В эталонах смешаны student_id списков {list_id} и {current_list_id}."
            )
    if list_id is None:
        raise ValueError("В эталонных кластерах отсутствуют student_id.")
    return list_id


def _filename_sort_key(filename: str) -> tuple[int, int, str]:
    """Стабильно сортирует фото по первому номеру, затем по имени файла."""

    match = re.search(r"(\d+)", filename)
    if match:
        return 0, int(match.group(1)), filename.casefold()
    return 1, 0, filename.casefold()
