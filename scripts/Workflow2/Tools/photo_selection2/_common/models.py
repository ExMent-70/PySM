"""State of one read-only photo-selection stage."""

from __future__ import annotations

from dataclasses import dataclass

from .assignment_core import BuildResult
from .copy_service import CopySummary


@dataclass
class PhotoSelectionSessionState:
    """Results and publication status of the current stage."""

    assignments_dirty: bool = False
    build_result: BuildResult | None = None
    copy_summary: CopySummary | None = None
