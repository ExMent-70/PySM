"""Immediate copying or moving with PySM progress; no window or Qt event loop."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

from ..._common.assignment_core import (
    BuildResult,
    normalize_exclude_dirs,
)
from ..._common.copy_core import build_copy_selection
from ..._common.copy_service import copy_selected_files
from ..._common.signatures import file_signature

try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder
except ImportError:
    pysm_context = ResourceNode = StandardTreeBuilder = None

try:
    from pysm_lib.pysm_progress_reporter import tqdm
except ImportError:
    from tqdm import tqdm


def _report_folders(config: Namespace) -> None:
    """Keep resource links optional; reporting cannot undo a completed transfer."""
    if pysm_context is None:
        print(f"Источник: {config.source_dir}\nНазначение: {config.dest_dir}", flush=True)
        return
    try:
        builder = StandardTreeBuilder(icon_size=28)
        builder.add_section("", [
            ResourceNode("Исходная папка", Path(config.source_dir), "folder"),
            ResourceNode("Целевая папка", Path(config.dest_dir), "folder"),
        ])
        pysm_context.log_html(builder.get_html())
    except Exception as exc:
        print(f"Предупреждение: не удалось вывести ссылки на папки: {exc}", file=sys.stderr, flush=True)


def _calculate(config: Namespace) -> BuildResult:
    """Discover selected files and locations without calculating assignments."""
    return build_copy_selection(
        analysis_dir=Path(config.analysis_dir),
        source_dir=Path(config.source_dir),
        dest_dir=Path(config.dest_dir),
        exclude_dirs=normalize_exclude_dirs(config.exclude_dirs),
    )


def _print_issues(result: BuildResult) -> None:
    """Keep blocking errors and non-blocking warnings visible in the PySM log."""
    for issue in result.issues:
        label = "Ошибка" if issue.severity == "error" else "Предупреждение"
        print(f"{label}: {issue.message}", flush=True)


def run_copy(config: Namespace) -> int:
    """Transfer immediately, report progress and return a workflow exit status.

    A fresh scan follows successful copying, as in the original button action.
    Name collisions follow on_conflict. Neither workflow JSON is written.
    """
    moving = config.mode == "move"
    operation = "Перемещение" if moving else "Копирование"
    print(f"<b>{operation.upper()} ВЫБРАННЫХ ФОТОГРАФИЙ</b><br>", flush=True)
    print(f"При совпадении имён: {config.on_conflict}", flush=True)
    selection_path = Path(config.analysis_dir) / "photo_selection.json"
    try:
        selection_signature = file_signature(selection_path)
        with tqdm(total=0, desc="Проверка выбранных фотографий", unit="file"):
            result = _calculate(config)
        if file_signature(selection_path) != selection_signature:
            raise ValueError("Выбор изменён во время проверки. Запустите копирование повторно.")
        if result.has_errors:
            _print_issues(result)
            return 1

        summary = copy_selected_files(
            result, Path(config.source_dir), Path(config.dest_dir),
            mode=config.mode,
            on_conflict=config.on_conflict,
            progress_factory=tqdm,
        )
        if moving:
            print(f"Перемещено: {summary.moved}. Пропущено: {summary.skipped}.", flush=True)
            if summary.copied > summary.moved:
                print(f"Скопировано без удаления оригинала: {summary.copied - summary.moved}.", flush=True)
        else:
            print(f"Скопировано: {summary.copied}. Пропущено: {summary.skipped}.", flush=True)
        result.issues.extend(summary.issues)
        if result.has_errors:
            _print_issues(result)
            return 1
        if file_signature(selection_path) != selection_signature:
            raise ValueError("Выбор изменён во время операции. Запустите скрипт повторно.")

        with tqdm(total=0, desc="Проверка результата операции", unit="file"):
            result = _calculate(config)
        if file_signature(selection_path) != selection_signature:
            raise ValueError("Выбор изменён во время проверки результата. Запустите копирование повторно.")
        _print_issues(result)
        if result.has_errors:
            return 1
        if summary.copied == 0 and summary.skipped == 0:
            print(f"Нет исходных файлов для {'перемещения' if moving else 'копирования'}.", flush=True)
        print(f"{operation} завершено.", flush=True)
        _report_folders(config)
        return 0
    except Exception as exc:
        print(f"Ошибка {'перемещения' if moving else 'копирования'}: {exc}", file=sys.stderr, flush=True)
        return 1
