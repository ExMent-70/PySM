"""Reports for the standalone selection importer."""
from __future__ import annotations
from html import escape
from .domain import ImportEntry
from pysm_lib import theme_api
from pysm_lib.pysm_report_api import DashboardBuilder, ResourceNode
IS_MANAGED_RUN = True

class ImportReportMixin:
    """Render roster and import diagnostics without scanning photographs."""

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
        self.import_result.setHtml(self._message_header_html() + builder.get_html())

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
        self.import_result.setHtml(self._message_header_html() + "".join(parts))
