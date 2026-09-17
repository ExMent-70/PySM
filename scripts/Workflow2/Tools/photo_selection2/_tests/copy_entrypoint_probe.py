"""Run the production copy entry point while forbidding GUI/event-loop startup."""

from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread


entry = Path(sys.argv[1])
sys.argv = [str(entry), *sys.argv[2:]]
assert QApplication.instance() is None
with patch.object(QApplication, "__init__", side_effect=AssertionError("Copy created an application")), \
     patch.object(QThread, "start", side_effect=AssertionError("Copy started a Qt thread")):
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
    else:
        code = 0
assert QApplication.instance() is None
assert not any(name.endswith("._common.assignment_window") for name in sys.modules)
assert not any(name.endswith("._common.image_pipeline") for name in sys.modules)
print("COPY_PROBE_NO_GUI", flush=True)
raise SystemExit(code)
