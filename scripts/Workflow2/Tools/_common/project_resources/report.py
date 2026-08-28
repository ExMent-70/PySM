"""Composition entry points for shared project resource reports."""

from __future__ import annotations

from .dashboard_renderer import render_dashboard_html
from .models import (
    ProjectReportContext,
    ProjectReportOptions,
    ProjectReportResult,
    ProjectResourceSnapshot,
)
from .scanner import collect_project_resources
from .standard_renderer import render_standard_html
from .workflow_renderer import render_workflow_html


def render_project_resources(
    snapshot: ProjectResourceSnapshot,
    options: ProjectReportOptions,
) -> str:
    """Render a previously collected snapshot in the selected presentation."""

    if options.template == "dashboard":
        return render_dashboard_html(snapshot, options)
    if options.template == "workflow":
        return render_workflow_html(snapshot, options)
    return render_standard_html(snapshot, options)


def build_project_report(
    context: ProjectReportContext,
    options: ProjectReportOptions,
) -> ProjectReportResult:
    """Collect and render one project without coupling to a presentation host."""

    snapshot = collect_project_resources(context, options)
    return ProjectReportResult(
        html=render_project_resources(snapshot, options),
        warnings=snapshot.warnings,
    )
