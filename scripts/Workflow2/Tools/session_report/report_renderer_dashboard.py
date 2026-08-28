"""Compatibility adapter for the shared dashboard project report renderer."""

from __future__ import annotations

from pathlib import Path

from scripts.Workflow2.Tools._common.project_resources import (
    ProjectReportContext,
    ProjectReportOptions,
    build_project_report,
)


def generate_dashboard_html(
    config,
    path_session_base,
    path_psd_base,
    path_c1_session,
    session_name,
    photo_session,
    wf_portrait_session,
    wf_idsgn_catalog_str,
) -> str:
    """Preserve the former renderer entry point without duplicating logic."""

    return build_project_report(
        ProjectReportContext(
            project_name=session_name,
            session_root=path_session_base,
            psd_root=path_psd_base,
            capture_one_path=path_c1_session,
            project_path=(path_psd_base / session_name) if path_psd_base else None,
            photo_session=photo_session or "",
            portrait_session=wf_portrait_session or "",
            idsgn_catalog=Path(wf_idsgn_catalog_str) if wf_idsgn_catalog_str else None,
        ),
        ProjectReportOptions(
            template="dashboard",
            scope=config.report_scope,
            icon_size_tree=getattr(config, "icon_size_tree", 24),
            icon_size_dashboard=config.icon_size_dashboard,
        ),
    ).html
