"""Stage-oriented rendering of a Workflow2 project resource snapshot."""

from __future__ import annotations

import html
from pathlib import Path

from pysm_lib.pysm_icons import icons
from pysm_lib.pysm_report_api import ReportTheme

from .models import (
    ProcessingSessionSnapshot,
    ProjectReportOptions,
    ProjectResourceSnapshot,
)
from .summary_renderer import render_project_summary_html


def _icon(name: str, size: int, *, color: str | None = None) -> str:
    method = getattr(icons, name, icons.FILE)
    return method(size=size, color=color) if color else method(size=size)


class _WorkflowBuilder:
    """Small renderer for completion rows and resource tables."""

    def __init__(self, icon_size: int) -> None:
        self.icon_size = icon_size
        self.theme = ReportTheme()
        self.parts: list[str] = []
        self._row = 0

    def header(self, text: str) -> None:
        style = (
            "font-family:sans-serif;font-size:16px;font-weight:bold;"
            f"color:{self.theme.text_main};margin-top:18px;margin-bottom:6px;"
            f"padding-bottom:5px;border-bottom:2px solid {self.theme.accent};"
        )
        self.parts.append(f'<div style="{style}">{html.escape(text)}</div>')

    def subheader(self, text: str, path: Path | None = None, exists: bool = False) -> None:
        caption = html.escape(text)
        if path is not None:
            caption = self.link(path, caption, "FOLDER", exists)
        style = (
            f"font-family:sans-serif;font-size:14px;font-weight:bold;color:{self.theme.text_main};"
            f"background:{self.theme.header_bg};border:1px solid {self.theme.border};"
            "padding:7px 9px;margin-top:12px;"
        )
        self.parts.append(f'<div style="{style}">{caption}</div>')

    def link(self, path: Path, caption: str, icon_name: str, exists: bool) -> str:
        resource_icon = _icon(
            icon_name,
            self.icon_size,
            color=None if exists else self.theme.text_sub,
        )
        label = f'{resource_icon}&nbsp;{caption}'
        if not exists:
            return f'<span style="color:{self.theme.text_sub};">{label}</span>'
        return (
            f'<a href="{path.resolve().as_uri()}" style="text-decoration:none;'
            f'color:{self.theme.text_main};">{label}</a>'
        )

    def resource_icon_link(
        self,
        path: Path,
        caption: str,
        icon_name: str,
        available: bool,
        color: str | None = None,
    ) -> str:
        """Render a compact resource icon whose path is available as a hint."""

        resource_icon = _icon(icon_name, self.icon_size, color=color)
        tooltip = html.escape(f"{caption}: {path}", quote=True)
        if not available:
            return f'<span title="{tooltip}">{resource_icon}</span>'
        return (
            f'<a href="{path.resolve().as_uri()}" title="{tooltip}" '
            f'style="text-decoration:none;">{resource_icon}</a>'
        )

    def stage(
        self,
        text: str,
        completed: bool,
        *,
        path: Path | None = None,
        path_exists: bool | None = None,
        resource_caption: str = "Открыть ресурс",
        resource_icon: str = "FILE",
        warning: bool = False,
    ) -> None:
        background = self.theme.bg_base if self._row % 2 == 0 else self.theme.bg_alt
        self._row += 1
        if completed and not warning:
            status_icon = _icon("OK", self.icon_size)
        else:
            status_icon = _icon("WARNING", self.icon_size)
        resource = ""
        if path is not None:
            available = completed if path_exists is None else path_exists
            resource_color = None
            if not completed or warning:
                resource_color = (
                    self.theme.error if warning else self.theme.text_sub
                )
            resource = self.resource_icon_link(
                path,
                resource_caption,
                resource_icon,
                available,
                resource_color,
            )
        self.parts.append(
            f'<table style="width:100%;border-spacing:0;background:{background};'
            'font-family:sans-serif;font-size:13px;">'
            '<tr>'
            f'<td style="width:{self.icon_size + 12}px;padding:6px;text-align:center;">'
            f'{status_icon}</td>'
            f'<td style="width:{self.icon_size + 12}px;padding:6px;text-align:center;">'
            f'{resource}</td>'
            f'<td style="padding:6px;color:{self.theme.text_main};">'
            f'{html.escape(text)}</td></tr></table>'
        )

    def album_table(self, session: ProcessingSessionSnapshot) -> None:
        has_album_media = any(
            folder.psd_count or folder.jpg_count
            for folder in session.album_photo_folders
        )
        if not has_album_media:
            self.stage(
                "Фотографии для альбома не скопированы в папку для вёрстки.",
                False,
                path=session.album_session_path,
                path_exists=session.album_session_exists,
                resource_caption="Папка фотографий для альбома",
                resource_icon="FOLDER",
            )
            return
        self.stage(
            "Фотографии для альбома скопированы в папку для вёрстки.",
            True,
            path=session.album_session_path,
            path_exists=session.album_session_exists,
            resource_caption="Папка фотографий для альбома",
            resource_icon="FOLDER",
        )
        rows: list[str] = []
        for index, folder in enumerate(session.album_photo_folders):
            media_count = folder.psd_count + folder.jpg_count
            if media_count == 0:
                state = "Нет PSD/JPG"
                state_color = self.theme.text_sub
                state_icon = _icon("WARNING", self.icon_size)
            elif folder.xmp_count == 0:
                state = "Внимание: XMP не найдены"
                state_color = self.theme.error
                state_icon = _icon("WARNING", self.icon_size)
            else:
                state = "Готово"
                state_color = self.theme.ok
                state_icon = _icon("OK", self.icon_size)
            background = self.theme.bg_base if index % 2 == 0 else self.theme.bg_alt
            rows.append(
                f'<tr style="background:{background};">'
                '<td style="padding:6px;">'
                f'{self.link(folder.path, html.escape(folder.name), "FOLDER", True)}'
                '</td>'
                f'<td style="padding:6px;text-align:right;">{folder.psd_count}</td>'
                f'<td style="padding:6px;text-align:right;">{folder.jpg_count}</td>'
                f'<td style="padding:6px;text-align:right;">{folder.xmp_count}</td>'
                f'<td style="padding:6px;color:{state_color};">{state_icon}&nbsp;{state}</td>'
                '</tr>'
            )
        self.parts.append(
            '<table style="width:100%;border-spacing:0;font-family:sans-serif;font-size:13px;">'
            f'<tr style="background:{self.theme.header_bg};color:{self.theme.text_main};">'
            '<th style="padding:6px;text-align:left;">Папка альбома</th>'
            '<th style="padding:6px;text-align:right;">PSD</th>'
            '<th style="padding:6px;text-align:right;">JPG</th>'
            '<th style="padding:6px;text-align:right;">XMP</th>'
            '<th style="padding:6px;text-align:left;">Состояние</th></tr>'
            f'{"".join(rows)}</table>'
        )

    def file_table(
        self,
        title: str,
        files: tuple[Path, ...],
        *,
        parent_path: Path | None = None,
        parent_exists: bool = False,
    ) -> None:
        self.subheader(title, parent_path, parent_exists)
        if not files:
            self.stage("Файлы InDesign не найдены.", False)
            return
        rows = []
        for index, path in enumerate(files):
            background = self.theme.bg_base if index % 2 == 0 else self.theme.bg_alt
            rows.append(
                f'<tr style="background:{background};"><td style="padding:6px;">'
                f'{self.link(path, html.escape(path.name), "FILE_INDD", True)}'
                '</td></tr>'
            )
        self.parts.append(
            '<table style="width:100%;border-spacing:0;font-family:sans-serif;font-size:13px;">'
            f'{"".join(rows)}</table>'
        )


def _render_general(builder: _WorkflowBuilder, snapshot: ProjectResourceSnapshot) -> None:
    context = snapshot.context
    builder.header("1. Подготовка заказа")
    if context.project_path is not None:
        builder.stage(
            "Рабочая папка для файлов PSD/INDD создана."
            if snapshot.project_exists
            else "Рабочая папка для файлов PSD/INDD не создана.",
            snapshot.project_exists,
            path=context.project_path,
            path_exists=snapshot.project_exists,
            resource_caption="Папка заказа",
            resource_icon="FOLDER",
        )
    if context.capture_one_path is not None:
        builder.stage(
            "Рабочая папка для исходных файлов создана."
            if snapshot.capture_one_exists
            else "Рабочая папка для исходных файлов не создана.",
            snapshot.capture_one_exists,
            path=context.capture_one_path,
            path_exists=snapshot.capture_one_exists,
            resource_caption="Папка Capture One",
            resource_icon="FOLDER",
        )
    if snapshot.cosessiondb_path is not None:
        builder.stage(
            "Сессия Capture One создана."
            if snapshot.cosessiondb_exists
            else "Сессия Capture One не создана.",
            snapshot.cosessiondb_exists,
            path=snapshot.cosessiondb_path,
            path_exists=snapshot.cosessiondb_exists,
            resource_caption=snapshot.cosessiondb_path.name,
            resource_icon="FILE_C1",
        )
    if snapshot.list_path is not None:
        builder.stage(
            "Список класса или группы создан."
            if snapshot.list_exists
            else "Список класса или группы не создан.",
            snapshot.list_exists,
            path=snapshot.list_path,
            path_exists=snapshot.list_exists,
            resource_caption=snapshot.list_path.name,
            resource_icon="FILE_CODE",
        )
    if snapshot.contract_path is not None:
        builder.stage(
            "Договор на оказание фотоуслуг подготовлен."
            if snapshot.contract_exists
            else "Договор на оказание фотоуслуг не подготовлен.",
            snapshot.contract_exists,
            path=snapshot.contract_path,
            path_exists=snapshot.contract_exists,
            resource_caption=snapshot.contract_path.name,
            resource_icon="FILE_HTML",
        )
    if context.portrait_session.strip() and snapshot.reference_session_path is not None:
        builder.stage(
            f"Эталонная фотосессия задана: {context.portrait_session}.",
            snapshot.reference_session_exists,
            path=snapshot.reference_session_path,
            path_exists=snapshot.reference_session_exists,
            resource_caption="Результаты эталонной фотосессии",
            resource_icon="PHOTO_PORTRAIT",
            warning=not snapshot.reference_session_exists,
        )
    else:
        builder.stage(
            "Эталонная фотосессия для идентификации групповых фотографий не задана.",
            False,
            warning=True,
        )


def _render_photo_session(
    builder: _WorkflowBuilder,
    session: ProcessingSessionSnapshot,
) -> None:
    builder.subheader(
        f"Фотосессия {session.name}",
        session.capture_path,
        session.capture_exists,
    )
    builder.stage(
        (
            f"RAW-файлы скопированы: {session.raw_count} шт."
            if session.raw_count
            else "RAW-файлы не скопированы."
        ),
        session.raw_count > 0,
        path=session.capture_path,
        path_exists=session.raw_count > 0,
        resource_caption="Исходные RAW-файлы",
        resource_icon="FILE_RAW",
    )
    builder.stage(
        (
            f"RAW-файлы сконвертированы в JPG: {session.jpg_count} шт."
            if session.jpg_count
            else "RAW-файлы не сконвертированы в JPG."
        ),
        session.jpg_count > 0,
        path=session.jpg_path,
        path_exists=session.jpg_count > 0,
        resource_caption="Файлы JPG",
        resource_icon="FILE_IMAGE",
    )
    builder.stage(
        (
            f"Маски объектов созданы: {session.masks_count} шт."
            if session.masks_count
            else "Маски объектов не созданы."
        ),
        session.masks_count > 0,
        path=session.masks_path,
        path_exists=session.masks_count > 0,
        resource_caption="Маски Cutout",
        resource_icon="FILE_MASK",
    )
    builder.stage(
        "Техническая кластеризация выполнена."
        if session.cluster_cleaning
        else "Техническая кластеризация не выполнена.",
        session.cluster_cleaning,
    )
    builder.stage(
        (
            "Кластеризация отмечена выполненной, но файл info_faces.json не найден."
            if session.cluster_faces and not session.info_faces_exists
            else "Кластеризация портретных фотографий выполнена."
            if session.cluster_faces
            else "Кластеризация портретных фотографий не выполнена."
        ),
        session.cluster_faces,
        path=session.info_faces_path,
        path_exists=session.info_faces_exists,
        resource_caption="Результаты распознавания лиц",
        resource_icon="FILE_CODE",
        warning=session.cluster_faces and not session.info_faces_exists,
    )
    builder.stage(
        "Кластеризация фотографий по локациям и сюжетам выполнена."
        if session.cluster_locations
        else "Кластеризация фотографий по локациям и сюжетам не выполнена.",
        session.cluster_locations,
    )
    builder.stage(
        (
            "Идентификация отмечена выполненной, но файл совпадений не найден."
            if session.cluster_matches and not session.matches_exists
            else "Идентификация лиц на групповых фотографиях выполнена."
            if session.cluster_matches
            else "Идентификация лиц на групповых фотографиях не выполнена."
        ),
        session.cluster_matches,
        path=session.matches_path,
        path_exists=session.matches_exists,
        resource_caption="Совпадения портретов и групп",
        resource_icon="FILE_CODE",
        warning=session.cluster_matches and not session.matches_exists,
    )
    if session.errors_exists:
        builder.stage(
            "Файл контроля ошибок идентификации подготовлен.",
            True,
            path=session.errors_path,
            path_exists=True,
            resource_caption="Ошибки идентификации",
            resource_icon="FILE_CODE",
        )
    builder.stage(
        (
            "XMP-файлы подготовлены для синхронизации с RAW-конвертером: "
            f"{session.xmp_count} шт."
            if session.xmp_count
            else "XMP-файлы для синхронизации с RAW-конвертером не подготовлены."
        ),
        session.xmp_count > 0,
        path=session.capture_path,
        path_exists=session.xmp_count > 0,
        resource_caption="Папка с XMP",
        resource_icon="FILE_XMP",
    )
    builder.stage(
        "HTML-отчёт о кластеризации подготовлен."
        if session.html_report_exists
        else "HTML-отчёт о кластеризации не подготовлен.",
        session.html_report_exists,
        path=session.html_report_path,
        path_exists=session.html_report_exists,
        resource_caption="Отчёт о кластеризации",
        resource_icon="FILE_HTML",
    )
    builder.stage(
        "Фотографии для альбома отобраны пользователем или фотографом."
        if session.photo_selection_exists
        else "Фотографии для альбома не отобраны.",
        session.photo_selection_exists,
        path=session.photo_selection_path,
        path_exists=session.photo_selection_exists,
        resource_caption="Список выбранных фотографий",
        resource_icon="FILE_CODE",
    )
    builder.stage(
        "Список фотографий для автоматической вёрстки альбомов сформирован."
        if session.photo_assignments_exists
        else "Список фотографий для автоматической вёрстки не сформирован.",
        session.photo_assignments_exists,
        path=session.photo_assignments_path,
        path_exists=session.photo_assignments_exists,
        resource_caption="Фотографии для вёрстки",
        resource_icon="FILE_CODE",
    )
    builder.album_table(session)


def _render_order_results(
    builder: _WorkflowBuilder,
    snapshot: ProjectResourceSnapshot,
) -> None:
    builder.header("3. Верстка и печать фотографий/альбомов")
    builder.file_table(
        "Шаблоны и документы InDesign",
        snapshot.project_template_files,
        parent_path=snapshot.project_templates_path,
        parent_exists=snapshot.project_templates_exists,
    )
    if snapshot.ready_pages_path is not None:
        builder.stage(
            (
                "Электронные макеты альбомов готовы к печати: "
                f"{snapshot.ready_pages_jpg_count} шт."
                if snapshot.ready_pages_jpg_count
                else "Электронные макеты альбомов не готовы к печати."
            ),
            snapshot.ready_pages_jpg_count > 0,
            path=snapshot.ready_pages_path,
            path_exists=snapshot.ready_pages_jpg_count > 0,
            resource_caption="Готовые страницы",
            resource_icon="FILE_IMAGE",
        )
    if snapshot.graduates_path is not None:
        builder.stage(
            (
                "Электронные фотографии готовы для передачи заказчику: "
                f"{snapshot.graduates_jpg_count} шт."
                if snapshot.graduates_jpg_count
                else "Электронные фотографии не готовы для передачи заказчику."
            ),
            snapshot.graduates_jpg_count > 0,
            path=snapshot.graduates_path,
            path_exists=snapshot.graduates_jpg_count > 0,
            resource_caption="Папка «Выпускникам»",
            resource_icon="FILE_IMAGE",
        )


def render_workflow_html(
    snapshot: ProjectResourceSnapshot,
    options: ProjectReportOptions,
) -> str:
    """Render processing progress without reading the filesystem again."""

    builder = _WorkflowBuilder(options.icon_size_tree)
    builder.parts.append(render_project_summary_html(snapshot.context.summary_fields))
    _render_general(builder, snapshot)
    builder.header("2. AI-анализ, отбор и обработка фотографий")
    if snapshot.processing_sessions:
        for session in snapshot.processing_sessions:
            _render_photo_session(builder, session)
    else:
        builder.stage(
            "Связанные ресурсы фотосессий в Capture, Output/Analysis_* и "
            "папках фотографий альбома не найдены.",
            False,
        )
    _render_order_results(builder, snapshot)
    return "".join(builder.parts)
