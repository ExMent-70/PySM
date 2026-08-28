"""Theme-aware rendering of project metadata above resource sections."""

from __future__ import annotations

import html

from pysm_lib.pysm_report_api import ReportTheme

from .models import ProjectSummaryField


def render_project_summary_html(
    fields: tuple[ProjectSummaryField, ...],
) -> str:
    """Render optional project metadata in the shared report style."""

    if not fields:
        return ""

    theme = ReportTheme()
    header_style = (
        "font-family: sans-serif; font-size: 16px; font-weight: bold; "
        f"color: {theme.text_main}; margin: 0 0 5px 0; padding-bottom: 5px; "
        f"border-bottom: 1px solid {theme.accent};"
    )
    table_style = (
        "width: 100%; border-spacing: 0; font-family: sans-serif; "
        f"font-size: 13px; color: {theme.text_main}; margin-bottom: 14px;"
    )
    rows: list[str] = []
    for index, field in enumerate(fields):
        background = theme.bg_base if index % 2 == 0 else theme.bg_alt
        caption = html.escape(field.caption)
        value = html.escape(field.value or "—").replace("\n", "<br>")
        rows.append(
            f'<tr style="background-color: {background};">'
            '<td style="padding: 6px 8px; width: 32%; vertical-align: top; '
            f'font-weight: bold;">{caption}</td>'
            f'<td style="padding: 6px 8px; vertical-align: top;">{value}</td>'
            "</tr>"
        )

    return (
        f'<div style="{header_style}">Данные проекта</div>'
        f'<table style="{table_style}">{"".join(rows)}</table>'
    )
