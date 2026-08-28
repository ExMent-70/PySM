"""Read-only filesystem scanner for a Workflow2 project resource report."""

from __future__ import annotations

from pathlib import Path

from .models import (
    AlbumPhotoFolderSnapshot,
    AnalysisResourceSnapshot,
    ProcessingSessionSnapshot,
    ProjectReportContext,
    ProjectReportOptions,
    ProjectResourceError,
    ProjectResourceSnapshot,
)


RAW_SUFFIXES = {".arw", ".cr2", ".cr3", ".nef", ".dng"}
JPG_SUFFIXES = {".jpg", ".jpeg"}
MASK_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _safe_children(folder: Path, warnings: list[str]) -> tuple[Path, ...]:
    try:
        if not folder.is_dir():
            return ()
        return tuple(folder.iterdir())
    except OSError as exc:
        warnings.append(f"Не удалось прочитать папку «{folder}»: {exc}")
        return ()


def _safe_is_dir(item: Path, warnings: list[str]) -> bool:
    try:
        return item.is_dir()
    except OSError as exc:
        warnings.append(f"Не удалось проверить папку «{item}»: {exc}")
        return False


def _safe_is_file(item: Path, warnings: list[str]) -> bool:
    try:
        return item.is_file()
    except OSError as exc:
        warnings.append(f"Не удалось проверить файл «{item}»: {exc}")
        return False


def _subfolders(folder: Path, warnings: list[str]) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                item
                for item in _safe_children(folder, warnings)
                if _safe_is_dir(item, warnings)
            ),
            key=lambda item: (item.name.casefold(), item.name),
        )
    )


def _files_with_suffixes(
    folder: Path,
    suffixes: set[str],
    warnings: list[str],
) -> tuple[Path, ...]:
    normalized = {suffix.casefold() for suffix in suffixes}
    return tuple(
        sorted(
            (
                item
                for item in _safe_children(folder, warnings)
                if _safe_is_file(item, warnings)
                and item.suffix.casefold() in normalized
            ),
            key=lambda item: (item.name.casefold(), item.name),
        )
    )


def _has_xmp(folder: Path | None, warnings: list[str]) -> bool:
    if folder is None:
        return False
    return any(
        _safe_is_file(item, warnings) and item.suffix.casefold() == ".xmp"
        for item in _safe_children(folder, warnings)
    )


def _count_files(
    folder: Path,
    suffixes: set[str],
    warnings: list[str],
) -> int:
    """Count matching direct children without raising on disappearing paths."""

    return len(_files_with_suffixes(folder, suffixes, warnings))


def _count_album_assets_recursive(
    folder: Path,
    warnings: list[str],
) -> tuple[int, int, int]:
    """Count PSD, JPG and XMP in one recursive pass without following links."""

    psd_count = 0
    jpg_count = 0
    xmp_count = 0
    pending = [folder]
    while pending:
        current = pending.pop()
        for item in _safe_children(current, warnings):
            try:
                if item.is_symlink():
                    if not item.is_file():
                        continue
                elif item.is_dir():
                    pending.append(item)
                    continue
                elif not item.is_file():
                    continue
                suffix = item.suffix.casefold()
                if suffix == ".psd":
                    psd_count += 1
                elif suffix in JPG_SUFFIXES:
                    jpg_count += 1
                elif suffix == ".xmp":
                    xmp_count += 1
            except OSError as exc:
                warnings.append(f"Не удалось проверить ресурс «{item}»: {exc}")
    return psd_count, jpg_count, xmp_count


def _count_capture_assets(folder: Path, warnings: list[str]) -> tuple[int, int]:
    """Count direct RAW and XMP files in one directory pass."""

    raw_count = 0
    xmp_count = 0
    for item in _safe_children(folder, warnings):
        if not _safe_is_file(item, warnings):
            continue
        suffix = item.suffix.casefold()
        if suffix in RAW_SUFFIXES:
            raw_count += 1
        elif suffix == ".xmp":
            xmp_count += 1
    return raw_count, xmp_count


def _cluster_flags(value: object, session_name: str) -> dict[str, bool]:
    """Normalize the documented ``var_claster_run.<session>`` structure."""

    if not isinstance(value, dict):
        return {}
    session_value = value.get(session_name)
    if not isinstance(session_value, dict):
        return {}
    return {
        str(key): str(item).strip().casefold() == "yes"
        for key, item in session_value.items()
    }


def _album_photo_folders(
    session_path: Path,
    warnings: list[str],
) -> tuple[AlbumPhotoFolderSnapshot, ...]:
    """Collect layout folders and recursively verify their XMP sidecars."""

    folders = _subfolders(session_path, warnings)
    if not folders and _safe_is_dir(session_path, warnings):
        folders = (session_path,)
    snapshots: list[AlbumPhotoFolderSnapshot] = []
    for folder in folders:
        psd_count, jpg_count, xmp_count = _count_album_assets_recursive(
            folder, warnings
        )
        snapshots.append(
            AlbumPhotoFolderSnapshot(
                name=folder.name,
                path=folder,
                psd_count=psd_count,
                jpg_count=jpg_count,
                xmp_count=xmp_count,
            )
        )
    return tuple(snapshots)


def _processing_snapshots(
    context: ProjectReportContext,
    options: ProjectReportOptions,
    capture_subfolders: tuple[Path, ...],
    analyses: tuple[AnalysisResourceSnapshot, ...],
    photo_subfolders: tuple[Path, ...],
    warnings: list[str],
) -> tuple[ProcessingSessionSnapshot, ...]:
    """Build a snapshot for every session evidenced by any workflow resource.

    A processed project may legitimately no longer contain its RAW directory.
    Discovering sessions only from ``Capture`` would then hide JPG, masks,
    clustering results and album assets that still exist in the other workflow
    branches.
    """

    if context.capture_one_path is None:
        return ()
    if options.scope == "current":
        session_names = (context.photo_session.strip(),)
    else:
        capture_names = {path.name for path in capture_subfolders}
        analysis_names = {analysis.suffix for analysis in analyses}
        album_names = {path.name for path in photo_subfolders}
        cluster_names: set[str] = set()
        if isinstance(context.cluster_run, dict):
            cluster_names.update(str(name).strip() for name in context.cluster_run)
        names = capture_names | analysis_names | album_names | cluster_names
        reference_name = context.portrait_session.strip()
        if (
            reference_name
            and reference_name in analysis_names
            and reference_name not in capture_names | album_names | cluster_names
        ):
            # A standalone Analysis_<wf_portrait_session> is the reference used
            # for matching group photos, not evidence of another order stage.
            names.discard(reference_name)
        names.discard("")
        session_names = tuple(sorted(names, key=lambda name: (name.casefold(), name)))

    snapshots: list[ProcessingSessionSnapshot] = []
    for name in session_names:
        capture_path = context.capture_one_path / "Capture" / name
        analysis_path = context.capture_one_path / "Output" / f"Analysis_{name}"
        jpg_path = analysis_path / "JPG"
        masks_path = analysis_path / "Masks" / "Cutout"
        info_faces_path = analysis_path / "info_faces.json"
        matches_path = analysis_path / "matches_portrait_to_group.json"
        errors_path = analysis_path / "error_matches.json"
        html_report_path = analysis_path / "face_clustering_report.html"
        photo_selection_path = analysis_path / "photo_selection.json"
        photo_assignments_path = analysis_path / "photo_assignments.json"
        flags = _cluster_flags(context.cluster_run, name)
        raw_count, xmp_count = _count_capture_assets(capture_path, warnings)
        album_session_path = (
            context.project_path / "Альбом" / "Фото" / name
            if context.project_path is not None
            else None
        )
        snapshots.append(
            ProcessingSessionSnapshot(
                name=name,
                capture_path=capture_path,
                capture_exists=_safe_is_dir(capture_path, warnings),
                analysis_path=analysis_path,
                raw_count=raw_count,
                xmp_count=xmp_count,
                jpg_path=jpg_path,
                jpg_count=_count_files(jpg_path, JPG_SUFFIXES, warnings),
                masks_path=masks_path,
                masks_count=_count_files(masks_path, MASK_SUFFIXES, warnings),
                cluster_cleaning=flags.get("cleaning", False),
                cluster_faces=flags.get("face", False),
                cluster_locations=flags.get("location", False),
                cluster_matches=flags.get("matches", False),
                info_faces_path=info_faces_path,
                info_faces_exists=_safe_is_file(info_faces_path, warnings),
                matches_path=matches_path,
                matches_exists=_safe_is_file(matches_path, warnings),
                errors_path=errors_path,
                errors_exists=_safe_is_file(errors_path, warnings),
                html_report_path=html_report_path,
                html_report_exists=_safe_is_file(html_report_path, warnings),
                photo_selection_path=photo_selection_path,
                photo_selection_exists=_safe_is_file(photo_selection_path, warnings),
                photo_assignments_path=photo_assignments_path,
                photo_assignments_exists=_safe_is_file(
                    photo_assignments_path, warnings
                ),
                album_session_path=album_session_path,
                album_session_exists=(
                    _safe_is_dir(album_session_path, warnings)
                    if album_session_path is not None
                    else False
                ),
                album_photo_folders=(
                    _album_photo_folders(album_session_path, warnings)
                    if album_session_path is not None
                    else ()
                ),
            )
        )
    return tuple(snapshots)


def _analysis_snapshots(
    context: ProjectReportContext,
    options: ProjectReportOptions,
    warnings: list[str],
) -> tuple[AnalysisResourceSnapshot, ...]:
    if options.scope == "current" and not context.photo_session.strip():
        raise ProjectResourceError(
            "Для области «Текущая фотосессия» не задана переменная "
            "wf_photo_session."
        )
    if context.capture_one_path is None:
        return ()

    output_path = context.capture_one_path / "Output"
    expected_name = (
        f"Analysis_{context.photo_session.strip()}"
        if options.scope == "current"
        else ""
    )
    analyses: list[AnalysisResourceSnapshot] = []
    for item in _subfolders(output_path, warnings):
        if not item.name.startswith("Analysis_"):
            continue
        if expected_name and item.name != expected_name:
            continue
        suffix = item.name.removeprefix("Analysis_")
        capture_path = context.capture_one_path / "Capture" / suffix
        analyses.append(
            AnalysisResourceSnapshot(
                name=item.name,
                suffix=suffix,
                path=item,
                jpg_path=item / "JPG",
                masks_path=item / "Masks" / "Cutout",
                photo_selection_path=item / "photo_selection.json",
                photo_assignments_path=item / "photo_assignments.json",
                info_faces_path=item / "info_faces.json",
                matches_path=item / "matches_portrait_to_group.json",
                errors_path=item / "error_matches.json",
                html_report_path=item / "face_clustering_report.html",
                capture_path=capture_path,
                capture_exists=_safe_is_dir(capture_path, warnings),
                has_xmp=_has_xmp(capture_path, warnings),
            )
        )
    return tuple(analyses)


def collect_project_resources(
    context: ProjectReportContext,
    options: ProjectReportOptions,
) -> ProjectResourceSnapshot:
    """Scan only the selected project and return an immutable resource snapshot."""

    warnings: list[str] = []
    capture_subfolders: tuple[Path, ...] = ()
    select_subfolders: tuple[Path, ...] = ()
    if context.capture_one_path is not None:
        capture_subfolders = _subfolders(
            context.capture_one_path / "Capture", warnings
        )
        select_subfolders = _subfolders(
            context.capture_one_path / "Selects", warnings
        )

    photo_subfolders: tuple[Path, ...] = ()
    project_template_files: tuple[Path, ...] = ()
    project_templates_path: Path | None = None
    project_templates_exists = False
    if context.project_path is not None:
        photo_subfolders = _subfolders(
            context.project_path / "Альбом" / "Фото", warnings
        )
        project_templates_path = context.project_path / "Альбом" / "_ШАБЛОНЫ_"
        project_templates_exists = _safe_is_dir(project_templates_path, warnings)
        project_template_files = _files_with_suffixes(
            project_templates_path,
            {".indd", ".idml"},
            warnings,
        )

    catalog_files: tuple[Path, ...] = ()
    if context.idsgn_catalog is not None:
        catalog_files = _files_with_suffixes(
            context.idsgn_catalog.parent,
            {".indd", ".idml"},
            warnings,
        )

    analyses = _analysis_snapshots(context, options, warnings)
    project_exists = (
        _safe_is_dir(context.project_path, warnings)
        if context.project_path is not None
        else False
    )
    capture_one_exists = (
        _safe_is_dir(context.capture_one_path, warnings)
        if context.capture_one_path is not None
        else False
    )
    cosessiondb_path = (
        context.capture_one_path / f"{context.project_name}.cosessiondb"
        if context.capture_one_path is not None
        else None
    )
    list_path = (
        context.project_path / f"{context.project_name}.list"
        if context.project_path is not None
        else None
    )
    contract_path = (
        context.project_path / f"{context.project_name}.html"
        if context.project_path is not None
        else None
    )
    reference_session_path = (
        context.capture_one_path
        / "Output"
        / f"Analysis_{context.portrait_session.strip()}"
        if context.capture_one_path is not None and context.portrait_session.strip()
        else None
    )
    ready_pages_path = (
        context.project_path / "Альбом" / "Готовые страницы"
        if context.project_path is not None
        else None
    )
    graduates_path = (
        context.project_path / "Выпускникам"
        if context.project_path is not None
        else None
    )

    return ProjectResourceSnapshot(
        context=context,
        capture_subfolders=capture_subfolders,
        select_subfolders=select_subfolders,
        analyses=analyses,
        photo_subfolders=photo_subfolders,
        project_templates_path=project_templates_path,
        project_templates_exists=project_templates_exists,
        project_template_files=project_template_files,
        catalog_files=catalog_files,
        project_exists=project_exists,
        capture_one_exists=capture_one_exists,
        cosessiondb_path=cosessiondb_path,
        cosessiondb_exists=(
            _safe_is_file(cosessiondb_path, warnings)
            if cosessiondb_path is not None
            else False
        ),
        list_path=list_path,
        list_exists=(
            _safe_is_file(list_path, warnings) if list_path is not None else False
        ),
        contract_path=contract_path,
        contract_exists=(
            _safe_is_file(contract_path, warnings)
            if contract_path is not None
            else False
        ),
        reference_session_path=reference_session_path,
        reference_session_exists=(
            _safe_is_dir(reference_session_path, warnings)
            if reference_session_path is not None
            else False
        ),
        processing_sessions=_processing_snapshots(
            context,
            options,
            capture_subfolders,
            analyses,
            photo_subfolders,
            warnings,
        ),
        ready_pages_path=ready_pages_path,
        ready_pages_jpg_count=(
            _count_files(ready_pages_path, JPG_SUFFIXES, warnings)
            if ready_pages_path is not None
            else 0
        ),
        graduates_path=graduates_path,
        graduates_jpg_count=(
            _count_files(graduates_path, JPG_SUFFIXES, warnings)
            if graduates_path is not None
            else 0
        ),
        warnings=tuple(dict.fromkeys(warnings)),
    )
