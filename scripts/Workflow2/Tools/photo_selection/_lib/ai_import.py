"""Domain adapter for the shared manual AI JSON workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pysm_lib.ai import (
    AiJsonRequest,
    extract_json_object as _extract_json_object,
    get_json_array,
    load_prompt_template,
    render_prompt_template,
    require_json_object,
    require_json_object_item,
)

from .constants import PHOTO_NUMBER_DIGITS
from .domain import ImportEntry
from .number_parser import normalize_number
from .roster import StudentRoster, normalize_student_id


def build_ai_student_reference(roster: StudentRoster) -> list[dict[str, str]]:
    """Build the minimal student reference needed for AI name matching."""

    return [
        {
            "student_id": student.student_id,
            "surname": student.surname,
            "name": student.name,
            "patronymic": student.patronymic,
        }
        for student in roster.students
    ]


def build_prompt(template: str, roster: StudentRoster, raw_text: str) -> str:
    """Compatibility wrapper around the shared prompt renderer."""

    return render_prompt_template(
        template,
        {
            "STUDENT_LIST_JSON": build_ai_student_reference(roster),
            "RAW_TEXT": raw_text,
        },
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Compatibility wrapper around the shared robust JSON extractor."""

    return _extract_json_object(text)


def validate_ai_response(
    payload: Any,
    roster: StudentRoster,
    *,
    min_digits: int,
    max_digits: int,
    pad_to_digits: int = 0,
) -> tuple[list[ImportEntry], list[dict[str, Any]]]:
    """Validate photo-selection rules after the shared JSON parsing step."""

    response = require_json_object(payload)
    matched = get_json_array(response, "matched")
    unresolved = get_json_array(response, "unresolved")
    entries: list[ImportEntry] = []
    seen: set[str] = set()
    for index, raw_item in enumerate(matched):
        item = require_json_object_item(
            raw_item,
            field_name="matched",
            index=index,
        )
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
        entries.append(
            ImportEntry(
                student_id,
                tuple(dict.fromkeys(numbers)),
                str(item.get("source_person") or ""),
                True,
            )
        )

    normalized_unresolved: list[dict[str, Any]] = []
    for index, raw_item in enumerate(unresolved):
        item = require_json_object_item(
            raw_item,
            field_name="unresolved",
            index=index,
        )
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


def create_ai_import_request(
    roster: StudentRoster,
) -> AiJsonRequest[tuple[list[ImportEntry], list[dict[str, Any]]]]:
    """Configure the generic dialog with photo-selection-specific validation."""

    template = load_prompt_template(
        Path(__file__).parent / "resources" / "ai_prompt_template.txt"
    )
    return AiJsonRequest(
        title="AI-импорт выбора",
        prompt_template=template,
        prompt_values={"STUDENT_LIST_JSON": build_ai_student_reference(roster)},
        raw_text_label="Неструктурированный исходный текст:",
        response_validator=lambda payload: validate_ai_response(
            payload,
            roster,
            min_digits=PHOTO_NUMBER_DIGITS,
            max_digits=PHOTO_NUMBER_DIGITS,
            pad_to_digits=0,
        ),
        show_success_message=False,
    )
