"""PySM entry point: Копирование выбранных фотографий."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

current_script_path = Path(__file__).resolve()
suite_dir = current_script_path.parent.parent
if not __package__:
    sys.path.insert(0, str(suite_dir.parent))
    __package__ = f"{suite_dir.name}.{current_script_path.parent.name}"
# A checkout can supply PySM locally; installed workflows get it via PYTHONPATH.
for candidate in current_script_path.parents:
    if (candidate / "pysm_lib" / "__init__.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from .._common.config import resolve_config


def get_config() -> argparse.Namespace:
    """Resolve the parameters of this stage through ConfigResolver."""
    return resolve_config("selection_copy")


def main() -> int:
    config = get_config()
    from ._lib.runner import run_copy
    return run_copy(config)


if __name__ == "__main__":
    raise SystemExit(main())
