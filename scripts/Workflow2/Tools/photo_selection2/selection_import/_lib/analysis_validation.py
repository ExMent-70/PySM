"""Non-blocking checks of imported numbers against face analysis."""
from __future__ import annotations

import json
from pathlib import Path

from ..._common.assignment_core import extract_one_photo_number


class SelectionAnalysis:
    """Index all analyzed variants of each frame and their recognized people."""

    def __init__(self, path: Path):
        self.people: dict[str, set[str]] = {}
        self.error = ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or any(not isinstance(item, dict) for item in data.values()):
                raise ValueError("ожидался объект с записями фотографий")
            for key, item in data.items():
                number = extract_one_photo_number(str(item.get("filename") or key))
                if number is None:
                    continue
                people = self.people.setdefault(number, set())
                faces = item.get("faces", [])
                if isinstance(faces, list):
                    people.update(str(face.get("student_id") or "").strip()
                                  for face in faces if isinstance(face, dict) and face.get("student_id"))
        except (OSError, UnicodeError, ValueError) as exc:
            self.people.clear()
            self.error = f"Проверка по info_faces.json недоступна: {exc}"

    def warnings(self, student_id: str, numbers, student_name: str) -> list[str]:
        if self.error:
            return []
        result = []
        for number in numbers:
            if number not in self.people:
                result.append(f"{number}: номер отсутствует в info_faces.json.")
            elif student_id not in self.people[number]:
                result.append(f"{number}: ученик {student_name} не найден на фотографии.")
        return result
