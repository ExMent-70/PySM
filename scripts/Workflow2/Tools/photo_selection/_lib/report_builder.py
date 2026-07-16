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
try:
    from pysm_lib import pysm_context, theme_api
except ImportError:
    pysm_context = None
    theme_api = None

try:
    from pysm_lib.pysm_icons import icons as pysm_icons
except ImportError:
    pysm_icons = None

try:
    from pysm_lib.pysm_report_api import DashboardBuilder, ResourceNode
except ImportError:
    DashboardBuilder = None
    ResourceNode = None

IS_MANAGED_RUN = pysm_context is not None


class ReportMixin:
    def _show_student_report(self, row: int) -> None:
        """Render the complete *.list record and current photo selection."""
        student = self.roster.students[row]
        selection = self.document.students.get(student.student_id)
        builder = DashboardBuilder(icon_size=18)
        builder.add_header_boxed(
            f"УЧЕНИК: {escape(student.display_name.upper())}",
            extra_html=(
                f"<span style='margin-left:auto'>{escape(student.student_id)}</span>"
            ),
            icon_size=22,
            vertical_padding=9,
        )

        builder.parts.append(
            f"<table width='100%' cellspacing='0' cellpadding='4' "
            f"style='border-collapse:collapse; color:{builder.theme.text_main}'>"
        )
        labels = {
            "student_id": "Идентификатор",
            "surname": "Фамилия",
            "name": "Имя",
            "patronymic": "Отчество",
            "rank": "Роль",
            "shoot_order": "Порядок съёмки",
            "alpha_order": "Алфавитный порядок",
            "color1": "Основной цвет",
            "color1_fg": "Текст основного цвета",
            "color2": "Дополнительный цвет",
            "color2_fg": "Текст дополнительного цвета",
            "service_type": "Услуга",
            "service_cost": "Стоимость услуги",
            "extra_services": "Дополнительные услуги",
            "info": "Дополнительная информация",
        }
        for index, (key, value) in enumerate(student.raw_data.items()):
            background = builder.theme.bg_base if index % 2 == 0 else builder.theme.bg_alt
            builder.parts.append(
                f"<tr style='background-color:{background}'>"
                f"<td style='width:38%; padding:4px 7px'><b>"
                f"{escape(labels.get(key, key))}</b></td>"
                f"<td style='padding:4px 7px'>{self._student_value_html(key, value)}</td>"
                "</tr>"
            )
        builder.parts.append("</table>")

        numbers = selection.selected_numbers if selection else []
        badge_color = builder.theme.ok if numbers else builder.theme.text_sub
        builder.add_header_boxed(
            "ВЫБРАННЫЕ ФОТОГРАФИИ",
            extra_html=(
                f"<span style='margin-left:auto; color:{badge_color}; font-weight:bold'>"
                f"{len(numbers)} фото</span>"
            ),
        )
        if numbers:
            number_html = "&nbsp;&nbsp;".join(
                f"<code style='font-size:14px'><b>{escape(number)}</b></code>"
                for number in numbers
            )
            builder.parts.append(f"<p>{number_html}</p>")
        else:
            builder.parts.append(
                f"<p style='color:{builder.theme.text_sub}'>Номера не выбраны.</p>"
            )
        builder.parts.append(
            "<table cellspacing='0' cellpadding='3'>"
            f"<tr><td><b>Ответ получен:</b></td><td>"
            f"{'Да' if selection and selection.responded else 'Нет'}</td></tr>"
            f"<tr><td><b>Источник:</b></td><td>"
            f"{escape(selection.source if selection else '—')}</td></tr>"
            "</table>"
        )
        builder.add_list_zebra([
            ResourceNode("Файл списка учеников", self.roster.path, "file")
        ])
        self.import_result.setHtml(builder.get_html())

    def _student_value_html(self, key: str, value) -> str:
        """Format arbitrary list fields without losing nested information."""
        if value is None or value == "":
            return "—"
        if isinstance(value, bool):
            return "Да" if value else "Нет"
        if isinstance(value, dict):
            if not value:
                return "Нет"
            return "<ul>" + "".join(
                f"<li><b>{escape(str(child_key))}:</b> "
                f"{self._student_value_html(str(child_key), child_value)}</li>"
                for child_key, child_value in value.items()
            ) + "</ul>"
        if isinstance(value, list):
            if not value:
                return "Нет"
            return "<ul>" + "".join(
                f"<li>{self._student_value_html(key, item)}</li>" for item in value
            ) + "</ul>"
        text = escape(str(value))
        if key.startswith("color") and str(value).startswith("#"):
            return (
                f"<span style='background-color:{text}; border:1px solid #777; "
                f"padding:1px 12px'>&nbsp;</span>&nbsp; <code>{text}</code>"
            )
        return text

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

    def _emit_final_log_once(self) -> None:
        """Write the final readiness report to the PySM console once."""
        if self._final_log_emitted or not (IS_MANAGED_RUN and pysm_context):
            return
        try:
            pysm_context.log_html(self._completion_log_html())
            self._final_log_emitted = True
        except Exception:
            logger.warning("Не удалось вывести итоговый лог", exc_info=True)

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

    def _theme_text_color(self, style_name: str, fallback: str) -> str:
        if IS_MANAGED_RUN and theme_api:
            return theme_api.get_parsed_style(style_name).get("color", fallback)
        return fallback

    def _show_import_result(
        self,
        entries: list[ImportEntry],
        unresolved: list,
        *,
        source: str,
        status: str,
    ) -> None:
        success_color = self._theme_text_color("icon_success", "#218838")
        error_color = self._theme_text_color("status_error", "#c62828")
        warning_color = self._theme_text_color("icon_warning", "#b26a00")
        source_labels = {
            "csv": "CSV",
            "ai_json": "AI JSON",
        }
        if status.startswith("Ошибка"):
            status_color = error_color
        elif "отменён" in status.casefold():
            status_color = warning_color
        else:
            status_color = success_color
        parts = [
            "<h3>Последний импорт</h3>",
            f"<p><b>Источник:</b> {escape(source_labels.get(source, source))}<br>",
            f'<b>Статус:</b> <span style="color:{status_color}">{escape(status)}</span></p>',
            f'<h4 style="color:{success_color}">Корректные записи: {len(entries)}</h4>',
        ]
        if entries:
            parts.append("<ol>")
            for entry in entries:
                student = self.roster.by_id[entry.student_id]
                numbers = ", ".join(entry.selected_numbers) or "пустой ответ"
                parts.append(
                    "<li><b>"
                    + escape(student.display_name)
                    + "</b>"
                    + (
                        "<br>Из импорта: " + escape(entry.source_person)
                        if entry.source_person and entry.source_person != student.display_name
                        else ""
                    )
                    + "<br><code>"
                    + escape(numbers)
                    + "</code></li>"
                )
            parts.append("</ol>")
        else:
            parts.append("<p>Нет.</p>")

        parts.append(
            f'<h4 style="color:{error_color}">Требуют внимания: {len(unresolved)}</h4>'
        )
        if unresolved:
            parts.append("<ol>")
            for item in unresolved:
                if isinstance(item, dict):
                    person = str(item.get("source_person") or item.get("source_file") or "Без имени")
                    reason = str(item.get("reason") or "Не удалось сопоставить ученика")
                    numbers = item.get("selected_numbers") or []
                    number_text = ", ".join(str(number) for number in numbers) or "номера не найдены"
                    candidates = item.get("candidates") or []
                    candidate_text = ", ".join(str(value) for value in candidates)
                    parts.append(
                        "<li><b>"
                        + escape(person)
                        + "</b><br><code>"
                        + escape(number_text)
                        + "</code><br><span style=\"color:"
                        + warning_color
                        + "\">"
                        + escape(reason)
                        + "</span>"
                        + ("<br>Кандидаты: " + escape(candidate_text) if candidate_text else "")
                        + "</li>"
                    )
                else:
                    parts.append("<li>" + escape(str(item)) + "</li>")
            parts.append("</ol>")
        else:
            parts.append("<p>Нет.</p>")
        self.import_result.setHtml("".join(parts))
