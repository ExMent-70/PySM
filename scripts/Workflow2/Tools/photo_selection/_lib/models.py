"""Shared GUI state for the combined photo-selection workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .assignment_core import BuildResult
from .copy_service import CopySummary


@dataclass
class PhotoSelectionSessionState:
    """Mutable state shared by import and assignment tabs."""

    selection_dirty: bool = False
    assignments_dirty: bool = False
    last_selection_saved_at: datetime | None = None
    last_assignments_saved_at: datetime | None = None
    build_result: BuildResult | None = None
    copy_summary: CopySummary | None = None
