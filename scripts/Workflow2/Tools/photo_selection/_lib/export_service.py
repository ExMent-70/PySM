"""HTML and CSV export helpers for the student-location view."""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from .assignment_core import LAYOUT_READY_SUFFIXES, is_excluded_relative_path

class ExportMixin:
    def _layout_file_matrix(
        self,
    ) -> tuple[list[str], list[tuple[str, dict[str, list[str]]]]]:
        result = self.state.build_result
        if result is None:
            return [], []
        locations = self._known_locations(result)
        destination = Path(self.config.dest_dir).resolve()
        csv_directory = destination
        exclude_dirs = tuple(self._exclude_dirs())
        students = []
        for student in sorted(self.roster.students, key=lambda item: item.display_name.casefold()):
            files_by_location: dict[str, list[str]] = {}
            for location in locations:
                values: set[str] = set()
                for record in result.records.values():
                    if (
                        record.location != location
                        or student.student_id not in record.assigned_student_ids
                    ):
                        continue
                    layout_paths: set[str] = set()
                    for path in record.destination_files:
                        if (
                            path.suffix.casefold() not in LAYOUT_READY_SUFFIXES
                            or not path.is_file()
                        ):
                            continue
                        try:
                            resolved = path.resolve()
                            resolved.relative_to(destination)
                            relative = resolved.relative_to(csv_directory)
                        except ValueError:
                            continue
                        if is_excluded_relative_path(relative, exclude_dirs):
                            continue
                        layout_paths.add(str(relative))
                    if layout_paths:
                        values.update(layout_paths)
                    else:
                        values.add(record.number)
                files_by_location[location] = sorted(values, key=str.casefold)
            students.append((student.display_name, files_by_location))
        return locations, students

    def _student_location_html(self) -> str:
        locations, students = self._layout_file_matrix()
        headers = "".join(f"<th>{escape(location)}</th>" for location in locations)
        rows = []
        for full_name, files_by_location in students:
            cells = []
            for location in locations:
                files = files_by_location[location]
                if files:
                    value = "<br>".join(escape(value) for value in files)
                    cells.append(f"<td>{value}</td>")
                else:
                    cells.append('<td class="missing">—</td>')
            rows.append(
                f"<tr><th class=\"student\">{escape(full_name)}</th>"
                f"{''.join(cells)}</tr>"
            )
        return (
            "<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\">"
            "<title>Выбор фотографий по ученикам и локациям</title>"
            "<style>"
            "body{font-family:Segoe UI,Arial,sans-serif;margin:12px;color:#202124;font-size:12px}"
            "h1{font-size:18px;margin:0 0 10px}table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #ccc;padding:3px 5px;text-align:left;vertical-align:top}"
            "thead th{background:#e9ecef;white-space:nowrap}.student{white-space:nowrap}"
            ".missing{color:#a06000;text-align:center}"
            "</style></head><body>"
            "<h1>Выбор фотографий по ученикам и локациям</h1>"
            f"<table><thead><tr><th>ФИО</th>{headers}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></body></html>"
        )

    def _save_student_location_html(self) -> None:
        default_path = Path(self.config.analysis_dir) / "photo_selection_by_student.html"
        filename, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить выбор по ученикам и локациям",
            str(default_path),
            "HTML (*.html *.htm)",
        )
        if not filename:
            return
        output_path = Path(filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".html")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(self._student_location_html(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Сохранение HTML", f"Не удалось сохранить файл:\n{exc}")
            return
        QMessageBox.information(self, "Сохранение HTML", f"Отчёт сохранён:\n{output_path}")

    def _save_student_location_csv(self) -> None:
        destination = Path(self.config.dest_dir)
        default_path = destination / "photo.csv"
        filename, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить выбор по ученикам и локациям",
            str(default_path),
            "CSV (*.csv)",
        )
        if not filename:
            return
        output_path = Path(filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".csv")
        locations, students = self._layout_file_matrix()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream, delimiter=";")
                writer.writerow(["ФИО", *locations])
                for full_name, files_by_location in students:
                    writer.writerow(
                        [
                            full_name,
                            *("\n".join(files_by_location[location]) for location in locations),
                        ]
                    )
        except OSError as exc:
            QMessageBox.critical(self, "Сохранение CSV", f"Не удалось сохранить файл:\n{exc}")
            return
        QMessageBox.information(self, "Сохранение CSV", f"Таблица сохранена:\n{output_path}")
