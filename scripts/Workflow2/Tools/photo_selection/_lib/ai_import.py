"""Prompt generation and strict validation of manually returned AI JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import ImportEntry
from .number_parser import normalize_number
from .roster import StudentRoster, normalize_student_id


def build_prompt(template: str, roster: StudentRoster, raw_text: str) -> str:
    reference = [
        {
            "student_id": student.student_id,
            "surname": student.surname,
            "name": student.name,
            "patronymic": student.patronymic,
        }
        for student in roster.students
    ]
    return (
        template.replace(
            "{{STUDENT_LIST_JSON}}",
            json.dumps(reference, ensure_ascii=False, indent=2),
        )
        .replace("{{RAW_TEXT}}", raw_text)
    )


def extract_json_object(text: str) -> Any:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("В ответе AI не найден JSON-объект.")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON-ответ AI: {exc}") from exc


def validate_ai_response(
    payload: Any,
    roster: StudentRoster,
    *,
    min_digits: int,
    max_digits: int,
    pad_to_digits: int = 0,
) -> tuple[list[ImportEntry], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("Ответ AI должен быть JSON-объектом.")
    matched, unresolved = payload.get("matched", []), payload.get("unresolved", [])
    if not isinstance(matched, list) or not isinstance(unresolved, list):
        raise ValueError("matched и unresolved должны быть массивами.")
    entries: list[ImportEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(matched):
        if not isinstance(item, dict):
            raise ValueError(f"matched[{index}] должен быть объектом.")
        student_id = normalize_student_id(item.get("student_id"), roster.list_id)
        if student_id not in roster.by_id:
            raise ValueError(f"matched[{index}] содержит неизвестный {student_id}.")
        if student_id in seen:
            raise ValueError(f"student_id {student_id} повторяется в matched.")
        raw_numbers = item.get("selected_numbers")
        if not isinstance(raw_numbers, list):
            raise ValueError(f"matched[{index}].selected_numbers должен быть массивом.")
        numbers = [
            normalize_number(
                number,
                min_digits=min_digits,
                max_digits=max_digits,
                pad_to_digits=pad_to_digits,
            )
            for number in raw_numbers
        ]
        seen.add(student_id)
        entries.append(ImportEntry(
            student_id,
            tuple(dict.fromkeys(numbers)),
            str(item.get("source_person") or ""),
            True,
        ))
    normalized_unresolved: list[dict[str, Any]] = []
    for index, item in enumerate(unresolved):
        if not isinstance(item, dict):
            raise ValueError("Все элементы unresolved должны быть объектами.")
        raw_numbers = item.get("selected_numbers", [])
        if not isinstance(raw_numbers, list):
            raise ValueError(
                f"unresolved[{index}].selected_numbers должен быть массивом."
            )
        normalized = dict(item)
        normalized["selected_numbers"] = [
            normalize_number(
                number,
                min_digits=min_digits,
                max_digits=max_digits,
                pad_to_digits=pad_to_digits,
            )
            for number in raw_numbers
        ]
        normalized_unresolved.append(normalized)
    return entries, normalized_unresolved


def load_prompt_template(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Шаблон AI-промпта не найден: {path}")
    return path.read_text(encoding="utf-8")
