"""Background workers for scanning, copying and publishing assignments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QThread, Signal

from .copy_service import CopySummary, copy_selected_files
from .assignment_core import (
    BuildResult,
    build_assignments,
    publishability_issues,
    save_assignments,
)

try:
    from pysm_lib.pysm_progress_reporter import IS_RUNNING_UNDER_PYSM, tqdm
except ImportError:
    IS_RUNNING_UNDER_PYSM = False
    tqdm = None


Operation = Literal["refresh", "copy", "build", "copy_and_build"]


@dataclass(frozen=True)
class BuildRequest:
    """Immutable paths used by a background operation."""

    student_list_file: Path
    analysis_dir: Path
    source_dir: Path
    dest_dir: Path
    exclude_dirs: tuple[str, ...]
    assignment_path: Path


@dataclass(frozen=True)
class OperationOutcome:
    """Result transferred from a worker to the GUI thread."""

    operation: Operation
    result: BuildResult
    copy_summary: CopySummary | None = None
    assignment_saved: bool = False


class PhotoSelectionOperationWorker(QThread):
    """Execute all filesystem-heavy stages outside the GUI thread."""

    stageChanged = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, request: BuildRequest, operation: Operation, parent=None):
        super().__init__(parent)
        self.request = request
        self.operation = operation

    def _build(self) -> BuildResult:
        request = self.request
        return build_assignments(
            student_list_file=request.student_list_file,
            analysis_dir=request.analysis_dir,
            source_dir=request.source_dir,
            dest_dir=request.dest_dir,
            exclude_dirs=request.exclude_dirs,
        )

    def run(self) -> None:
        try:
            self.stageChanged.emit("Сканирование исходной и целевой папок…")
            result = self._build()
            if result.has_errors:
                self.completed.emit(OperationOutcome(self.operation, result))
                return

            summary: CopySummary | None = None
            if self.operation in {"copy", "copy_and_build"}:
                self.stageChanged.emit("Копирование выбранных файлов…")
                summary = copy_selected_files(
                    result,
                    self.request.source_dir,
                    self.request.dest_dir,
                    progress_factory=tqdm if IS_RUNNING_UNDER_PYSM else None,
                )
                result.issues.extend(summary.issues)
                if any(issue.severity == "error" for issue in summary.issues):
                    self.completed.emit(
                        OperationOutcome(self.operation, result, summary)
                    )
                    return
                self.stageChanged.emit("Повторное сканирование целевой папки…")
                result = self._build()

            saved = False
            if self.operation in {"build", "copy_and_build"} and not result.has_errors:
                result.issues.extend(publishability_issues(result))
                if result.has_errors:
                    self.completed.emit(
                        OperationOutcome(self.operation, result, summary)
                    )
                    return
                self.stageChanged.emit("Создание photo_assignments.json…")
                save_assignments(self.request.assignment_path, result)
                saved = True

            self.completed.emit(
                OperationOutcome(self.operation, result, summary, saved)
            )
        except Exception as exc:
            self.failed.emit(str(exc))
