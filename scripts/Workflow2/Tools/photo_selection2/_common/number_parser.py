"""Extraction and normalization of six-digit photo numbers."""

from __future__ import annotations

import re
from typing import Any, Iterable


NUMBER_PATTERN = re.compile(r"(?<!\d)(\d+)(?!\d)")
MANUAL_SEPARATOR_PATTERN = re.compile(r"^[\d\s,;]*$")


def normalize_number(
    value: Any,
    *,
    min_digits: int,
    max_digits: int,
    pad_to_digits: int = 0,
) -> str:
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f"Некорректный номер фотографии: {value!r}.")
    if pad_to_digits and len(text) < pad_to_digits:
        text = text.zfill(pad_to_digits)
    if not min_digits <= len(text) <= max_digits:
        raise ValueError(
            f"Номер {text} должен содержать от {min_digits} до {max_digits} цифр."
        )
    return text


def extract_numbers(
    values: Iterable[Any],
    *,
    min_digits: int,
    max_digits: int,
    pad_to_digits: int = 0,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for match in NUMBER_PATTERN.finditer(str(value or "")):
            raw = match.group(1)
            try:
                number = normalize_number(
                    raw,
                    min_digits=min_digits,
                    max_digits=max_digits,
                    pad_to_digits=pad_to_digits,
                )
            except ValueError:
                continue
            if number not in seen:
                seen.add(number)
                result.append(number)
    return result


def parse_manual_numbers(
    value: str,
    *,
    min_digits: int,
    max_digits: int,
    pad_to_digits: int = 0,
) -> list[str]:
    """Parse a manually entered list without silently dropping invalid text."""
    text = str(value or "").strip()
    if not text:
        return []
    if not MANUAL_SEPARATOR_PATTERN.fullmatch(text):
        raise ValueError("Допустимы только номера, пробелы, запятые и точки с запятой.")
    raw_numbers = [part for part in re.split(r"[\s,;]+", text) if part]
    result: list[str] = []
    seen: set[str] = set()
    for raw in raw_numbers:
        number = normalize_number(
            raw,
            min_digits=min_digits,
            max_digits=max_digits,
            pad_to_digits=pad_to_digits,
        )
        if number not in seen:
            seen.add(number)
            result.append(number)
    return result
