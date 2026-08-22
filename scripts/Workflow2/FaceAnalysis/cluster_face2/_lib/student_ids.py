"""Контракт идентификаторов учеников для кластеризации лиц."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any, Iterable, MutableMapping, Sequence

import numpy as np


STUDENT_ID_PATTERN = re.compile(
    r"^(?P<list_id>[A-HJ-NP-Z2-9]{4})-S(?P<number>\d{3})$"
)
STUDENT_IDS_ORDER_CONTEXT_BASE = "wf_student_ids_order"
STUDENT_IDS_ORDER_CONTEXT_SUFFIX = "_ids_order"


@dataclass(frozen=True)
class StudentIdList:
    """Проверенный список ID одной фотосессии в порядке съёмки."""

    context_key: str
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


def build_student_ids_order_context_key(photo_session: Any) -> str:
    """Строит dot-notation путь порядка съёмки для текущей фотосессии."""

    normalized = str(photo_session or "").strip()
    if not normalized:
        raise ValueError("Не задана переменная контекста wf_photo_session.")
    if "." in normalized:
        raise ValueError("wf_photo_session не должна содержать точку.")
    return (
        f"{STUDENT_IDS_ORDER_CONTEXT_BASE}."
        f"{normalized}{STUDENT_IDS_ORDER_CONTEXT_SUFFIX}"
    )


def load_student_ids_order(raw_value: Any, context_key: str) -> StudentIdList:
    """Проверяет JSON-массив student_id, сохраняя порядок съёмки."""

    if not isinstance(raw_value, list):
        raise ValueError(
            f"Переменная {context_key} должна содержать JSON-массив student_id."
        )

    student_ids: list[str] = []
    seen: set[str] = set()
    list_id: str | None = None
    for item_number, raw_item in enumerate(raw_value, start=1):
        if not isinstance(raw_item, str):
            raise ValueError(
                f"Элемент {item_number} переменной {context_key} должен быть строкой."
            )
        value = raw_item.strip().upper()
        if not value:
            continue
        try:
            current_list_id, _ = parse_student_id(value)
        except ValueError as exc:
            raise ValueError(
                f"Элемент {item_number} переменной {context_key}: {exc}."
            ) from exc

        if list_id is None:
            list_id = current_list_id
        elif current_list_id != list_id:
            raise ValueError(
                f"Элемент {item_number} переменной {context_key}: student_id {value} "
                f"относится к списку {current_list_id}, ожидался {list_id}."
            )
        if value in seen:
            raise ValueError(
                f"Элемент {item_number} переменной {context_key}: "
                f"student_id {value} повторяется."
            )
        seen.add(value)
        student_ids.append(value)

    if not student_ids or list_id is None:
        raise ValueError(f"Переменная {context_key} не содержит student_id.")
    return StudentIdList(
        context_key=context_key,
        list_id=list_id,
        student_ids=tuple(student_ids),
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
