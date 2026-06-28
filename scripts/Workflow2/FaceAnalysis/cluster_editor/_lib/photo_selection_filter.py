"""Чтение общего списка выбранных номеров фотографий."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


PHOTO_NUMBER_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def extract_photo_numbers(filename: str) -> set[str]:
    """Возвращает отдельные шестизначные номера из имени файла."""

    return set(PHOTO_NUMBER_RE.findall(Path(filename).stem))


def load_selected_photo_numbers(path: Path) -> set[str] | None:
    """Загружает объединение ``selected_numbers`` или ``None``, если файла нет."""

    if not path.is_file():
        return None
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать {path.name}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: корень файла должен быть JSON-объектом.")
    students = data.get("students")
    if not isinstance(students, dict):
        raise ValueError(f"{path.name}: поле students должно быть объектом.")

    selected: set[str] = set()
    for student_id, student_data in students.items():
        if not isinstance(student_data, dict):
            raise ValueError(f"{path.name}: запись {student_id} должна быть объектом.")
        numbers = student_data.get("selected_numbers", [])
        if not isinstance(numbers, list):
            raise ValueError(
                f"{path.name}: selected_numbers для {student_id} должен быть массивом."
            )
        for value in numbers:
            number = str(value).strip()
            if not PHOTO_NUMBER_RE.fullmatch(number):
                raise ValueError(
                    f"{path.name}: номер {number or '<пусто>'} у {student_id} "
                    "должен содержать ровно шесть цифр."
                )
            selected.add(number)

    return selected
