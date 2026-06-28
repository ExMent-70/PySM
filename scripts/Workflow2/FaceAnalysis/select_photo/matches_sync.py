"""Синхронизация matches по `student_id` без зависимости от GUI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


def sync_matches_data(
    matches_data: dict[str, Any],
    search_results: list[dict[str, Any]],
    selected_numbers: set[str],
    min_digits: int,
    max_digits: int,
) -> dict[str, Any]:
    """Возвращает обновлённую копию matches, сопоставляя фото по ID."""

    min_d, max_d = sorted((min_digits, max_digits))
    number_regex = re.compile(rf"(?<!\d)(\d{{{min_d},{max_d}}})(?!\d)")
    found_map: dict[str, dict[str, Any]] = {}
    for item in search_results:
        if item.get("status") != "Найден":
            continue
        number = str(item.get("number") or "")
        if number not in selected_numbers:
            continue
        found_map[number] = {
            "student_ids": set(item.get("student_ids") or []),
            "filename": item.get("original_filename"),
        }

    result = json.loads(json.dumps(matches_data, ensure_ascii=False))
    for cluster_key, record in result.items():
        student_id = record.get("student_id")
        if not student_id:
            raise ValueError(f"matches[{cluster_key!r}] не содержит student_id.")

        current_photos = record.get("group_photos", [])
        new_group_photos: list[dict[str, Any]] = []
        existing_numbers: set[str] = set()
        for photo_info in current_photos:
            match = number_regex.search(str(photo_info.get("filename") or ""))
            number = match.group(1) if match else None
            found = found_map.get(number or "")
            if (
                number in selected_numbers
                and found is not None
                and student_id in found["student_ids"]
            ):
                new_group_photos.append(photo_info)
                existing_numbers.add(number)

        for number, info in found_map.items():
            if student_id not in info["student_ids"] or number in existing_numbers:
                continue
            filename = info.get("filename")
            if not filename:
                continue
            new_group_photos.append({
                "filename": filename,
                "min_distance": 0.0,
                "num_faces": 1,
            })

        new_group_photos.sort(key=lambda item: str(item.get("filename") or ""))
        record["group_photos"] = new_group_photos
        record.pop("child_name", None)
    return result


def atomic_write_json(path: Path, data: Any) -> None:
    """Записывает JSON через временный файл в целевом каталоге."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise
