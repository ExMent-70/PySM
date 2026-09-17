"""HTML report builders for the photo selection window."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

from .assignment_core import (
    BuildResult,
    PHOTOGRAPHER_PREFIX,
    PhotoRecord,
    has_layout_ready_destination_file,
    index_records_by_student_location,
)
from pysm_lib.pysm_icons import icons as pysm_icons
from pysm_lib.pysm_report_api import DashboardBuilder, ResourceNode


class ReportMixin:
    """Read-only assignment details and readiness reports."""

    def _issues_html(self, result: BuildResult) -> str:
        """Show all diagnostics, including warnings that do not block copying."""
        if not result.issues:
            return ""
        builder = DashboardBuilder(icon_size=18)
        builder.add_header_boxed("Ошибки и предупреждения")
        for issue in result.issues:
            color = builder.theme.error if issue.severity == "error" else builder.theme.text_sub
            builder.parts.append(f"<p style='color:{color}'>{escape(issue.message)}</p>")
        return builder.get_html()

    def _message_header_html(self) -> str:
        return self.user_message_html if self.user_message_html else ""

    def _completion_report_html(
        self,
        result: BuildResult,
        *,
        include_user_message: bool = True,
    ) -> str:
        readiness = self._album_readiness(result)
        selected_numbers = self._selected_assignment_numbers(result)
        ready_numbers = self._layout_ready_assignment_numbers(result)
        builder = DashboardBuilder(icon_size=22)
        if include_user_message and self.user_message_html:
            builder.parts.append(self.user_message_html)
        builder.add_list_zebra([
            ResourceNode(
                "Папка с результатами АИ-анализа фотографий",
                Path(self.config.analysis_dir),
                "folder",
            ),
            ResourceNode("Исходная папка с изображениями", Path(self.config.source_dir), "folder"),
            ResourceNode("Целевая папка", Path(self.config.dest_dir), "folder"),
        ])
        if readiness["ready"]:
            builder.add_list_zebra([
                ResourceNode("Файл photo_assignments", self.assignment_path, "code")
            ])
        status_color = builder.theme.ok if readiness["ready"] else builder.theme.error
        status_text = "ГОТОВО К ВЕРСТКЕ" if readiness["ready"] else "НЕ ГОТОВО К ВЕРСТКЕ"
        builder.parts.append(
            f"<div style='font-family:sans-serif;font-size:16px;font-weight:bold;"
            f"color:{status_color};margin-top:10px;'>Статус: {status_text}</div>"
        )
        builder.parts.append(
            f"<p style='color:{builder.theme.text_main}; font-family: sans-serif;'>"
            f"<b>Префикс фотографа:</b> {escape(PHOTOGRAPHER_PREFIX)}<br>"
            f"<b>Назначено учеников:</b> {len(result.assignments)}<br>"
            f"<b>Отобрано фотографий для верстки:</b> {len(selected_numbers)}<br>"
            f"<b>Фотографий в целевой папке:</b> {len(ready_numbers)}</p>"
        )
        if self.state.copy_summary is not None:
            builder.parts.append(
                f"<p style='color:{builder.theme.text_main}; font-family: sans-serif;'>"
                f"<b>Скопировано файлов:</b> {self.state.copy_summary.copied}<br>"
                f"<b>Пропущено файлов:</b> {self.state.copy_summary.skipped}</p>"
            )
        assignment_status = "создан" if self.assignment_path.is_file() else "не создавался"
        builder.parts.append(
            f"<p style='color:{builder.theme.text_main}; font-family: sans-serif;'>"
            f"<b>Файл photo_assignments:</b> {assignment_status}</p>"
        )
        if self.state.assignments_dirty:
            builder.parts.append(
                f"<p style='color:#a06000; font-family: sans-serif;'>"
                "<b>Выбор изменён:</b> photo_assignments.json нужно пересоздать.</p>"
            )
        builder.parts.append(
            f"<p style='color:{builder.theme.text_main}; font-family: sans-serif;'>"
            "<b>Проверка готовности:</b></p>"
        )
        for passed, text in readiness["checks"]:
            icon = self._readiness_icon_html(passed, 18)
            color = builder.theme.ok if passed else builder.theme.error
            builder.parts.append(
                f"<div style='font-family:sans-serif;color:{color};"
                "margin:3px 0;display:flex;align-items:center;'>"
                f"{icon}<span style='margin-left:6px;'>{escape(text)}</span></div>"
            )
        if readiness["missing_numbers"]:
            builder.parts.append(
                f"<p style='color:{builder.theme.error}; font-family: sans-serif;'>"
                "<b>Нет JPG/JPEG/PSD в целевой папке для номеров:</b><br>"
                f"{escape(', '.join(readiness['missing_numbers']))}</p>"
            )
        builder.parts.append(self._issues_html(result))
        return builder.get_html()

    def _completion_log_html(self) -> str:
        """Render the final PySM console report without the launch instruction."""
        result = self.state.build_result
        if result is not None:
            return self._completion_report_html(result, include_user_message=False)
        builder = DashboardBuilder(icon_size=22)
        builder.add_list_zebra([
            ResourceNode(
                "Папка с результатами АИ-анализа фотографий",
                Path(self.config.analysis_dir),
                "folder",
            ),
            ResourceNode("Исходная папка с изображениями", Path(self.config.source_dir), "folder"),
            ResourceNode("Целевая папка", Path(self.config.dest_dir), "folder"),
        ])
        builder.parts.append(
            f"<div style='font-family:sans-serif;font-size:16px;font-weight:bold;"
            f"color:{builder.theme.error};margin-top:10px;'>"
            "Статус: НЕ ГОТОВО К ВЕРСТКЕ</div>"
        )
        builder.parts.append(
            f"<p style='color:{builder.theme.text_main}; font-family: sans-serif;'>"
            "<b>Список назначений:</b> ещё не рассчитан<br>"
            f"<b>Файл photo_assignments:</b> {'создан' if self.assignment_path.is_file() else 'не создавался'}</p>"
        )
        return builder.get_html()

    def _album_readiness(self, result: BuildResult) -> dict:
        selected_numbers = self._selected_assignment_numbers(result)
        ready_numbers = self._layout_ready_assignment_numbers(result)
        missing_numbers = sorted(selected_numbers - ready_numbers)
        errors = [issue for issue in result.issues if issue.severity == "error"]
        assignment_exists = self.assignment_path.is_file()
        assignment_current = self._assignment_file_matches_result(result)
        checks = [
            (not errors, "Блокирующих ошибок нет"),
            (assignment_exists, "photo_assignments.json создан"),
            (
                assignment_current,
                "photo_assignments.json актуален относительно текущих данных",
            ),
            (
                not missing_numbers,
                "Для всех назначенных номеров есть JPG/JPEG/PSD в целевой папке",
            ),
            (
                len(selected_numbers) == len(ready_numbers),
                "Количество отобранных фотографий совпадает с количеством готовых фотографий в целевой папке",
            ),
        ]
        return {
            "ready": all(passed for passed, _text in checks),
            "checks": checks,
            "missing_numbers": missing_numbers,
        }

    def _assignment_file_matches_result(self, result: BuildResult) -> bool:
        if not self.assignment_path.is_file():
            return False
        try:
            payload = json.loads(self.assignment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload == result.assignment_payload()

    @staticmethod
    def _selected_assignment_numbers(result: BuildResult) -> set[str]:
        return {
            number
            for numbers in result.assignments.values()
            for number in numbers
        }

    def _layout_ready_assignment_numbers(self, result: BuildResult) -> set[str]:
        return {
            number
            for number in self._selected_assignment_numbers(result)
            if (
                (record := result.records.get(number)) is not None
                and has_layout_ready_destination_file(record)
            )
        }

    @staticmethod
    def _readiness_icon_html(passed: bool, size: int = 18) -> str:
        if pysm_icons is None:
            return "OK" if passed else "ERROR"
        icon_name = "OK" if passed else "ERROR"
        try:
            return getattr(pysm_icons, icon_name)(size=size)
        except Exception:
            return "OK" if passed else "ERROR"

    def _show_photo_report(self, record: PhotoRecord) -> None:
        builder = DashboardBuilder(icon_size=18)
        builder.parts.append(self._base_report_html)
        builder.add_header_boxed("Сведения о выбранной фотографии")
        builder.parts.append(self._photo_details_html(record))
        self.report.setHtml(builder.get_html())

    def _show_assignment_student_report(
        self,
        student_id: str,
        *,
        location: str | None = None,
    ) -> None:
        result = self.state.build_result
        if not result:
            return
        student = self.roster.by_id.get(student_id)
        full_name = student.display_name if student else "Неизвестный ученик"
        builder = DashboardBuilder(icon_size=18)
        builder.parts.append(self._base_report_html)
        title = f"Фотографии ученика: {full_name}"
        if location:
            title += f" — {location}"
        builder.add_header_boxed(title)
        locations = [location] if location else self._known_locations(result)
        record_index = index_records_by_student_location(result)
        for location_name in locations:
            records = record_index.get((student_id, location_name), [])
            location_node = ResourceNode(
                location_name,
                Path(self.config.dest_dir) / location_name,
                "folder",
                is_critical=False,
            )
            badge_text = f"{len(records)} фото" if records else "Нет фотографий"
            badge_color = builder.theme.ok if records else "#a06000"
            badge = (
                f"<span style='margin-left:auto; padding:2px 7px; "
                f"color:{badge_color}; font-weight:bold;'>{badge_text}</span>"
            )
            builder.add_header_boxed(
                f"ЛОКАЦИЯ: {escape(location_name.upper())}",
                link_node=location_node,
                extra_html=badge,
                icon_size=24,
                vertical_padding=10,
            )
            if not records:
                builder.parts.append(
                    "<p style='color:#a06000; margin:4px 0'><b>Фотографии отсутствуют.</b></p>"
                )
                continue
            for record in records:
                builder.parts.append(
                    self._photo_details_html(
                        record,
                        heading=f"{record.number} — {record.analysis_filename}",
                        include_people=False,
                    )
                )
        self.report.setHtml(builder.get_html())

    def _photo_details_html(
        self,
        record: PhotoRecord,
        *,
        heading: str | None = None,
        include_people: bool = True,
    ) -> str:
        recognized = self._students_html(record.recognized_student_ids)
        assigned = self._students_html(record.assigned_student_ids)
        selected_by_clients = self._students_html(record.selected_student_ids)
        source_labels = []
        if record.selected_student_ids:
            source_labels.append("клиент")
        if record.photographer_selected:
            source_labels.append("фотограф")
        source = " + ".join(source_labels) or "не определён"
        files = []
        destination = Path(self.config.dest_dir).resolve()
        for relative, _origin, physical_paths in self._file_entries(record):
            for path in physical_paths:
                try:
                    path.resolve().relative_to(destination)
                    origin_icon = pysm_icons.FOLDER_VIRTUAL(size=16) if pysm_icons else "▤"
                except ValueError:
                    origin_icon = pysm_icons.FOLDER(size=16) if pysm_icons else "▣"
                reveal_file = self._report_action_link(path, "reveal-file", origin_icon)
                open_file = self._report_action_link(path, "open-file", escape(str(relative)))
                files.append(
                    "<li style='white-space:nowrap; margin:2px 0'>"
                    f"{reveal_file}&nbsp;{open_file}</li>"
                )
        files_html = f"<ul>{''.join(files)}</ul>" if files else "<p>Файлы не найдены.</p>"
        heading_html = f"<h4>{escape(heading)}</h4>" if heading else ""
        people_html = ""
        if include_people:
            people_html = (
                "<h4>Все распознанные люди</h4>"
                f"{recognized}"
                "<h4>Все назначения</h4>"
                f"{assigned}"
            )
        return (
            heading_html + "<table cellspacing='0' cellpadding='3'>"
            f"<tr><td><b>Номер:</b></td><td>{escape(record.number)}</td></tr>"
            f"<tr><td><b>Имя в анализе:</b></td><td>{escape(record.analysis_filename)}</td></tr>"
            f"<tr><td><b>Локация:</b></td><td>{escape(record.location)}</td></tr>"
            f"<tr><td><b>Источник выбора:</b></td><td>{escape(source)}</td></tr>"
            f"<tr><td><b>Распознано:</b></td><td>{len(record.recognized_student_ids)} чел.</td></tr>"
            f"<tr><td><b>Назначено:</b></td><td>{len(record.assigned_student_ids)} чел.</td></tr>"
            "</table>"
            f"{people_html}"
            "<h4>Персональный выбор клиентов</h4>"
            f"{selected_by_clients}"
            "<h4>Физические файлы</h4>"
            f"{files_html}"
        )

    def _students_html(self, student_ids: set[str]) -> str:
        if not student_ids:
            return "<p>Нет.</p>"
        names = sorted(
            (
                self.roster.by_id[student_id].display_name
                if student_id in self.roster.by_id else "Неизвестный ученик"
                for student_id in student_ids
            ),
            key=str.casefold,
        )
        return "<ul>" + "".join(f"<li>{escape(name)}</li>" for name in names) + "</ul>"
