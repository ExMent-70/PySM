"""Typed contracts for the shared project resource report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProjectResourceError(ValueError):
    """The requested project report cannot be built safely."""


@dataclass(frozen=True)
class ProjectSummaryField:
    """One read-only project value rendered above the resource sections."""

    caption: str
    value: str


@dataclass(frozen=True)
class ProjectReportContext:
    """All project-specific values required by the resource scanner."""

    project_name: str
    psd_root: Path | None = None
    session_root: Path | None = None
    project_path: Path | None = None
    capture_one_path: Path | None = None
    context_path: Path | None = None
    photo_session: str = ""
    portrait_session: str = ""
    idsgn_catalog: Path | None = None
    cluster_run: Any = None
    summary_fields: tuple[ProjectSummaryField, ...] = ()

    def __post_init__(self) -> None:
        if not self.project_name.strip():
            raise ProjectResourceError("Не задано имя проекта для отчёта.")


@dataclass(frozen=True)
class ProjectReportOptions:
    """Presentation and filtering options shared by both report consumers."""

    template: str = "standard"
    scope: str = "current"
    icon_size_tree: int = 24
    icon_size_dashboard: int = 48

    def __post_init__(self) -> None:
        if self.template not in {"standard", "dashboard", "workflow"}:
            raise ProjectResourceError(
                f"Неизвестный шаблон отчёта: {self.template}."
            )
        if self.scope not in {"current", "full"}:
            raise ProjectResourceError(
                f"Неизвестная область отчёта: {self.scope}."
            )
        if self.icon_size_tree <= 0 or self.icon_size_dashboard <= 0:
            raise ProjectResourceError("Размеры иконок должны быть положительными.")


@dataclass(frozen=True)
class AnalysisResourceSnapshot:
    """One ``Analysis_*`` directory and its expected resources."""

    name: str
    suffix: str
    path: Path
    jpg_path: Path
    masks_path: Path
    photo_selection_path: Path
    photo_assignments_path: Path
    info_faces_path: Path
    matches_path: Path
    errors_path: Path
    html_report_path: Path
    capture_path: Path | None
    capture_exists: bool
    has_xmp: bool


@dataclass(frozen=True)
class AlbumPhotoFolderSnapshot:
    """One album-photo folder and the files required by the workflow."""

    name: str
    path: Path
    psd_count: int
    jpg_count: int
    xmp_count: int


@dataclass(frozen=True)
class ProcessingSessionSnapshot:
    """Read-only completion state for one project photo session."""

    name: str
    capture_path: Path
    capture_exists: bool
    analysis_path: Path
    raw_count: int
    xmp_count: int
    jpg_path: Path
    jpg_count: int
    masks_path: Path
    masks_count: int
    cluster_cleaning: bool
    cluster_faces: bool
    cluster_locations: bool
    cluster_matches: bool
    info_faces_path: Path
    info_faces_exists: bool
    matches_path: Path
    matches_exists: bool
    errors_path: Path
    errors_exists: bool
    html_report_path: Path
    html_report_exists: bool
    photo_selection_path: Path
    photo_selection_exists: bool
    photo_assignments_path: Path
    photo_assignments_exists: bool
    album_session_path: Path | None
    album_session_exists: bool
    album_photo_folders: tuple[AlbumPhotoFolderSnapshot, ...] = ()


@dataclass(frozen=True)
class ProjectResourceSnapshot:
    """Immutable result of one filesystem scan of a selected project."""

    context: ProjectReportContext
    capture_subfolders: tuple[Path, ...] = ()
    select_subfolders: tuple[Path, ...] = ()
    analyses: tuple[AnalysisResourceSnapshot, ...] = ()
    photo_subfolders: tuple[Path, ...] = ()
    project_templates_path: Path | None = None
    project_templates_exists: bool = False
    project_template_files: tuple[Path, ...] = ()
    catalog_files: tuple[Path, ...] = ()
    project_exists: bool = False
    capture_one_exists: bool = False
    cosessiondb_path: Path | None = None
    cosessiondb_exists: bool = False
    list_path: Path | None = None
    list_exists: bool = False
    contract_path: Path | None = None
    contract_exists: bool = False
    reference_session_path: Path | None = None
    reference_session_exists: bool = False
    processing_sessions: tuple[ProcessingSessionSnapshot, ...] = ()
    ready_pages_path: Path | None = None
    ready_pages_jpg_count: int = 0
    graduates_path: Path | None = None
    graduates_jpg_count: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectReportResult:
    """HTML and diagnostics produced from one resource snapshot."""

    html: str
    warnings: tuple[str, ...] = ()
