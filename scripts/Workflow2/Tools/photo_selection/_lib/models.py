"""Shared GUI state for the combined photo-selection workflow."""

from __future__ import annotations

from dataclasses import dataclass

from .assignment_core import BuildResult
from .copy_service import CopySummary


@dataclass
class PhotoSelectionSessionState:
    """Mutable state shared by import and assignment tabs."""

    assignments_dirty: bool = False
    build_result: BuildResult | None = None
    copy_summary: CopySummary | None = None
