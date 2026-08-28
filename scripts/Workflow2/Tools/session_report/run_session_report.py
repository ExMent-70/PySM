"""Generate an HTML resource report for the current Workflow2 project."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


try:
    current_script_path = Path(__file__).resolve()
    script_dir = current_script_path.parent
    project_root = current_script_path.parents[4]
    tools_dir = script_dir.parent
    for import_path in (project_root, tools_dir):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))

    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from scripts.Workflow2.Tools._common.project_resources import (
        ProjectReportContext,
        ProjectReportOptions,
        ProjectResourceError,
        build_project_report,
    )

    IS_MANAGED_RUN = True
except ImportError as exc:
    print(f"Критическая ошибка импорта: {exc}", file=sys.stderr)
    traceback.print_exc()
    IS_MANAGED_RUN = False
    pysm_context = None


def get_config() -> argparse.Namespace:
    """Resolve command-line options through the standard PySM contract."""

    parser = argparse.ArgumentParser(
        description="Формирует HTML-отчёт о ресурсах текущего проекта."
    )
    parser.add_argument(
        "--template",
        dest="report_template",
        choices=("standard", "dashboard", "workflow"),
        default="standard",
    )
    parser.add_argument(
        "--scope",
        dest="report_scope",
        choices=("current", "full"),
        default="current",
    )
    parser.add_argument("--icon_size_tree", type=int, default=24)
    parser.add_argument("--icon_size_dashboard", type=int, default=48)
    return ConfigResolver(parser).resolve_all()


def _optional_path(value: object) -> Path | None:
    text = str(value or "").strip()
    return Path(text) if text else None


def main() -> int:
    """Build the current project report and publish it to the PySM log."""

    if not IS_MANAGED_RUN or not pysm_context:
        print("Ошибка: Скрипт запущен вне окружения PySM.", file=sys.stderr)
        return 1

    config = get_config()
    session_root = _optional_path(pysm_context.get("wf_session_path"))
    psd_root = _optional_path(pysm_context.get("wf_psd_path"))
    project_name = str(pysm_context.get("wf_session_name") or "").strip()
    try:
        context = ProjectReportContext(
            project_name=project_name,
            session_root=session_root,
            psd_root=psd_root,
            capture_one_path=(session_root / project_name) if session_root else None,
            project_path=(psd_root / project_name) if psd_root else None,
            photo_session=str(pysm_context.get("wf_photo_session") or ""),
            portrait_session=str(pysm_context.get("wf_portrait_session") or ""),
            idsgn_catalog=_optional_path(pysm_context.get("wf_idsgn_catalog")),
            cluster_run=pysm_context.get("var_claster_run"),
        )
        options = ProjectReportOptions(
            template=config.report_template,
            scope=config.report_scope or "current",
            icon_size_tree=config.icon_size_tree,
            icon_size_dashboard=config.icon_size_dashboard,
        )
        result = build_project_report(context, options)
    except ProjectResourceError as exc:
        print(f"Не удалось сформировать отчёт: {exc}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"Предупреждение: {warning}", file=sys.stderr)
    pysm_context.log_html(result.html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
