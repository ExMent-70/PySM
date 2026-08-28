"""Dashboard rendering of a shared project resource snapshot."""

from __future__ import annotations

from pysm_lib.pysm_report_api import DashboardBuilder, ResourceNode

from .models import ProjectReportOptions, ProjectResourceSnapshot
from .summary_renderer import render_project_summary_html


def render_dashboard_html(
    snapshot: ProjectResourceSnapshot,
    options: ProjectReportOptions,
) -> str:
    """Render the dashboard without reading the filesystem again."""

    context = snapshot.context
    builder = DashboardBuilder(icon_size=options.icon_size_dashboard)
    builder.parts.append(render_project_summary_html(context.summary_fields))
    builder.add_header_simple("Блок 1. Общие ресурсы")
    builder.add_table_simple(
        [
            ResourceNode("RAW Base", context.session_root, "folder")
            if context.session_root is not None
            else None,
            ResourceNode("PSD Base", context.psd_root, "folder")
            if context.psd_root is not None
            else None,
            ResourceNode(
                "Каталог<br>шаблонов", context.idsgn_catalog.parent, "folder"
            )
            if context.idsgn_catalog is not None
            else None,
        ]
    )

    if context.capture_one_path is not None:
        builder.add_header_simple(
            "Блок 2. Исходные RAW-файлы. AI-анализ фотографий "
            f"({context.project_name})",
            ResourceNode(context.project_name, context.capture_one_path, "folder"),
        )
        builder.add_table_simple(
            [
                ResourceNode(
                    "Capture", context.capture_one_path / "Capture", "folder"
                ),
                ResourceNode("Output", context.capture_one_path / "Output", "folder"),
                ResourceNode(
                    "Selects", context.capture_one_path / "Selects", "folder"
                ),
                ResourceNode(
                    "Сессия C1",
                    context.capture_one_path / f"{context.project_name}.cosessiondb",
                    "c1",
                ),
                ResourceNode(
                    "Эталонная<br>фотосессия",
                    context.capture_one_path
                    / "Output"
                    / f"Analysis_{context.portrait_session}",
                    "folder",
                )
                if context.portrait_session
                else None,
            ]
        )

        for analysis in snapshot.analyses:
            xmp_html = (
                '<span style="color: #27AE60; font-weight: bold; '
                'margin-left: 10px;">+ XMP</span>'
                if analysis.has_xmp
                else ""
            )
            session_node = None
            if analysis.capture_path is not None and analysis.capture_exists:
                session_node = ResourceNode(
                    f"Фотосессия {analysis.suffix}: файлы RAW",
                    analysis.capture_path,
                    "folder",
                )
            builder.add_header_boxed(
                f"Фотосессия {analysis.suffix}: файлы RAW",
                session_node,
                extra_html=xmp_html,
            )
            builder.add_table_matrix(
                [
                    [
                        ResourceNode(
                            "AI-анализ", analysis.path, "folder", is_critical=False
                        ),
                        ResourceNode(
                            "JPG", analysis.jpg_path, "folder", is_critical=False
                        ),
                        ResourceNode(
                            "Masks", analysis.masks_path, "folder", is_critical=False
                        ),
                        ResourceNode(
                            "HTML<br>отчет",
                            analysis.html_report_path,
                            "html",
                            is_critical=False,
                        ),
                    ],
                    [
                        ResourceNode(
                            "Информация<br>о лицах(JSON)",
                            analysis.info_faces_path,
                            "code",
                            is_critical=False,
                        ),
                        ResourceNode(
                            "Портрет-Группа<br>(JSON)",
                            analysis.matches_path,
                            "code",
                            is_critical=False,
                        ),
                        ResourceNode(
                            "Ошибки<br>идентификации<br>(JSON)",
                            analysis.errors_path,
                            "code",
                            is_critical=False,
                        ),
                    ],
                    [
                        ResourceNode(
                            "Список выбранных<br>фотографий (JSON)",
                            analysis.photo_selection_path,
                            "code",
                            is_critical=False,
                        ),
                        ResourceNode(
                            "Фотографии<br>для вёрстки (JSON)",
                            analysis.photo_assignments_path,
                            "code",
                            is_critical=False,
                        ),
                    ],
                ]
            )

    if context.project_path is not None:
        work_path = context.project_path
        builder.add_header_simple(
            "Блок 3: Работа с альбомами (JPG/PSD/INDD)",
            ResourceNode("Work", work_path, "folder"),
        )
        builder.add_table_simple(
            [
                ResourceNode(
                    "Список",
                    work_path / f"{context.project_name}.list",
                    "code",
                ),
                ResourceNode(
                    "Договор",
                    work_path / f"{context.project_name}.html",
                    "html",
                ),
                ResourceNode("Выпускникам", work_path / "Выпускникам", "folder"),
                ResourceNode(
                    "В печать",
                    work_path / "Альбом" / "Готовые страницы",
                    "folder",
                ),
            ]
        )
        psd_path = work_path / "Альбом" / "Фото"
        builder.add_header_boxed(
            "Файлы PSD (сгруппированы по фотосессиям и сюжетам)",
            ResourceNode("PSD", psd_path, "folder"),
        )
        builder.add_grid(
            [ResourceNode(path.name, path, "folder") for path in snapshot.photo_subfolders],
            columns=5,
        )
        templates_path = work_path / "Альбом" / "_ШАБЛОНЫ_"
        builder.add_header_boxed(
            "Развороты альбомов (файлы InDesign)",
            ResourceNode("TPL", templates_path, "folder"),
        )
        builder.add_list_zebra(
            [
                ResourceNode(path.name, path, "indd")
                for path in snapshot.project_template_files
            ]
        )

    return builder.get_html()
