"""JPG preview selection helpers for the photo selection window."""

from __future__ import annotations

from pathlib import Path

from .assignment_core import PhotoRecord
from .constants import ITEM_NUMBER_ROLE, ITEM_STUDENT_ROLE, ITEM_LOCATION_ROLE
from PySide6.QtWidgets import QTreeWidgetItem


class PreviewMixin:
    def _show_import_student_preview(self, row: int) -> None:
        """Show the best available preview for a student selected on import tab."""
        if not 0 <= row < len(self.roster.students):
            self.preview.show_message("Выберите ученика для предпросмотра JPG")
            return
        student = self.roster.students[row]
        if not self.state.build_result:
            self.preview.show_message("Список назначений ещё не рассчитан")
            return
        preview_number = self._preferred_preview_number_for_student(student.student_id)
        if preview_number:
            self._show_preview_for_number(preview_number)
        else:
            self.preview.show_message("Для выбранного ученика нет найденных фотографий")

    def _find_preview_jpg(self, number: str) -> Path | None:
        result = self.state.build_result
        if not result or number not in result.records:
            return None
        record = result.records[number]
        expected_stem = Path(record.analysis_filename).stem.casefold()
        return self._preview_by_stem.get(expected_stem)

    def _show_preview_for_number(self, number: str) -> None:
        preview_path = self._find_preview_jpg(number)
        if preview_path is None:
            self.preview.show_message(f"JPG для номера {number} не найден в папке анализа")
            return
        self.preview.show_image(preview_path)

    def _on_assignment_item_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            return
        number = str(current.data(0, ITEM_NUMBER_ROLE) or "")
        if number:
            result = self.state.build_result
            if result and number in result.records:
                self._show_photo_report(result.records[number])
            self._show_preview_for_number(number)
            return
        student_id = str(current.data(0, ITEM_STUDENT_ROLE) or "")
        if student_id:
            location = str(current.data(0, ITEM_LOCATION_ROLE) or "") or None
            self._show_assignment_student_report(student_id, location=location)
            preview_number = self._preferred_preview_number_for_student(
                student_id,
                location,
            )
            if preview_number:
                self._show_preview_for_number(preview_number)
            else:
                self.preview.show_message("Для выбранной строки нет фотографии")

    def _preferred_preview_number_for_student(
        self,
        student_id: str,
        location: str | None = None,
    ) -> str | None:
        """Return the best preview candidate, preferring portrait for student rows."""
        result = self.state.build_result
        if not result:
            return None
        candidates: list[tuple[str, PhotoRecord]] = []
        for number in result.assignments.get(student_id, []):
            record = result.records.get(number)
            if record and (location is None or record.location == location):
                candidates.append((number, record))
        if location is None:
            for number, record in candidates:
                if (
                    record.location.casefold() == "portrait"
                    and self._find_preview_jpg(number) is not None
                ):
                    return number
        for number, _record in candidates:
            if self._find_preview_jpg(number) is not None:
                return number
        return candidates[0][0] if candidates else None
