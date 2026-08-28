"""Standard tree rendering of a shared project resource snapshot."""

from __future__ import annotations

from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder

from .models import ProjectReportOptions, ProjectResourceSnapshot
from .summary_renderer import render_project_summary_html


def _analysis_node(analysis) -> ResourceNode:
    node = ResourceNode(analysis.name, analysis.path, "folder", is_critical=False)
    node.children.extend(
        (
            ResourceNode("JPG", analysis.jpg_path, "folder", is_critical=False),
            ResourceNode("Masks", analysis.masks_path, "folder", is_critical=False),
            ResourceNode(
                "Список выбранных фотографий",
                analysis.photo_selection_path,
                "code",
                is_critical=False,
            ),
            ResourceNode(
                "Фотографии для вёрстки",
                analysis.photo_assignments_path,
                "code",
                is_critical=False,
            ),
            ResourceNode(
                "info_faces.json", analysis.info_faces_path, "code", is_critical=False
            ),
            ResourceNode(
                "matches_portrait_to_group.json",
                analysis.matches_path,
                "code",
                is_critical=False,
            ),
            ResourceNode(
                "error_matches.json", analysis.errors_path, "code", is_critical=False
            ),
            ResourceNode(
                "face_clustering_report.html",
                analysis.html_report_path,
                "html",
                is_critical=False,
            ),
        )
    )
    return node


def render_standard_html(
    snapshot: ProjectResourceSnapshot,
    options: ProjectReportOptions,
) -> str:
    """Render the standard report without reading the filesystem again."""

    context = snapshot.context
    builder = StandardTreeBuilder(icon_size=options.icon_size_tree)
    builder.parts.append(render_project_summary_html(context.summary_fields))

    global_nodes: list[ResourceNode] = []
    if context.session_root is not None:
        global_nodes.append(
            ResourceNode(
                "RAW Base", context.session_root, "folder", "Корневая папка RAW"
            )
        )
    if context.psd_root is not None:
        global_nodes.append(
            ResourceNode(
                "Albums Base", context.psd_root, "folder", "Корневая папка Альбомов"
            )
        )
    if context.idsgn_catalog is not None:
        catalog_node = ResourceNode(
            "Каталог шаблонов",
            context.idsgn_catalog.parent,
            "folder",
            f"Файл: {context.idsgn_catalog.name}",
        )
        catalog_node.children = [
            ResourceNode(path.name, path, "indd") for path in snapshot.catalog_files
        ]
        global_nodes.append(catalog_node)
    builder.add_section("Блок 1: Глобальные ресурсы", global_nodes)

    if context.capture_one_path is not None:
        c1_nodes = [
            ResourceNode(
                context.project_name,
                context.capture_one_path,
                "folder",
                "Папка сессии",
            ),
            ResourceNode(
                f"{context.project_name}.cosessiondb",
                context.capture_one_path / f"{context.project_name}.cosessiondb",
                "c1",
                "Файл сессии",
            ),
        ]
        if context.portrait_session:
            c1_nodes.append(
                ResourceNode(
                    "Эталонная фотосессия",
                    context.capture_one_path
                    / "Output"
                    / f"Analysis_{context.portrait_session}",
                    "folder",
                    f"ID: {context.portrait_session}",
                )
            )

        capture_node = ResourceNode(
            "Capture", context.capture_one_path / "Capture", "folder"
        )
        capture_node.children = [
            ResourceNode(path.name, path, "folder")
            for path in snapshot.capture_subfolders
        ]
        c1_nodes.append(capture_node)

        output_node = ResourceNode(
            "Output", context.capture_one_path / "Output", "folder"
        )
        output_node.children = [_analysis_node(item) for item in snapshot.analyses]
        c1_nodes.append(output_node)

        selects_node = ResourceNode(
            "Selects", context.capture_one_path / "Selects", "folder"
        )
        selects_node.children = [
            ResourceNode(path.name, path, "folder")
            for path in snapshot.select_subfolders
        ]
        c1_nodes.append(selects_node)
        builder.add_section("Блок 2: Исходные RAW-файлы. AI-анализ", c1_nodes)

    if context.project_path is not None:
        work_path = context.project_path
        psd_nodes = [
            ResourceNode(context.project_name, work_path, "folder", "Рабочая папка"),
            ResourceNode(
                f"{context.project_name}.list",
                work_path / f"{context.project_name}.list",
                "code",
                "Файл списка",
            ),
            ResourceNode(
                "Приложение (HTML)",
                work_path / f"{context.project_name}.html",
                "html",
                is_critical=False,
            ),
            ResourceNode(
                "Выпускникам",
                work_path / "Выпускникам",
                "folder",
                is_critical=False,
            ),
        ]
        photos_node = ResourceNode(
            "Фото (PSD)", work_path / "Альбом" / "Фото", "folder"
        )
        photos_node.children = [
            ResourceNode(path.name, path, "folder")
            for path in snapshot.photo_subfolders
        ]
        psd_nodes.append(photos_node)
        psd_nodes.append(
            ResourceNode(
                "Готовые страницы",
                work_path / "Альбом" / "Готовые страницы",
                "folder",
                is_critical=False,
            )
        )
        templates_node = ResourceNode(
            "_ШАБЛОНЫ_",
            work_path / "Альбом" / "_ШАБЛОНЫ_",
            "folder",
            "Папка шаблонов",
        )
        templates_node.children = [
            ResourceNode(path.name, path, "indd")
            for path in snapshot.project_template_files
        ]
        psd_nodes.append(templates_node)
        builder.add_section("Блок 3: Работа с альбомами", psd_nodes)

    return builder.get_html()
