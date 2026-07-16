"""Compatibility imports for the canonical photo-selection assignment core."""

from scripts.Workflow2.FaceAnalysis._common.photo_selection_core import (
    BuildResult,
    Issue,
    KNOWN_FILE_ICON_SUFFIXES,
    LAYOUT_READY_SUFFIXES,
    PHOTOGRAPHER_PREFIX,
    PhotoRecord,
    build_assignments,
    extract_one_photo_number,
    extract_photo_numbers,
    has_layout_ready_destination_file,
    index_records_by_student_location,
    is_excluded_relative_path,
    load_roster,
    normalize_exclude_dirs,
    publishability_issues,
    save_assignments,
)

__all__ = [name for name in globals() if not name.startswith("_")]
