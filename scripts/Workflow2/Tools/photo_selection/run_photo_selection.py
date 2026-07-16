"""PySM entry point for the per-student photo-selection editor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _lib.app import run_application

try:
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    ConfigResolver = None
    IS_MANAGED_RUN = False


def get_config() -> argparse.Namespace:
    """Resolve command-line arguments through the PySM context when available."""
    parser = argparse.ArgumentParser(
        description="Сбор персонального выбора номеров фотографий."
    )
    parser.add_argument("--student_list_file", required=True)
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--dest_dir", required=True)
    parser.add_argument("--session_name", required=True)
    parser.add_argument("--photo_session", required=True)
    parser.add_argument("--exclude_dirs", default="Masks")
    parser.add_argument(
        "--title",
        default="Выбор фотографий",
        help="Заголовок главного окна.",
    )
    parser.add_argument(
        "--message",
        default="",
        help="HTML-сообщение в начале правой информационной панели.",
    )
    resolver = ConfigResolver(parser) if IS_MANAGED_RUN else None
    return resolver.resolve_all() if resolver else parser.parse_args()


def main() -> int:
    return run_application(get_config())


if __name__ == "__main__":
    raise SystemExit(main())
