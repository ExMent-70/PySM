"""Exercise a real entry point and its Qt event loop on caller-owned fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[2]))
from photo_selection2._common import window_base


def main():
    """Close a successfully initialized real window without running its action."""
    entry = Path(sys.argv[1])
    sys.argv = [str(entry), *sys.argv[2:]]
    window_base.IS_MANAGED_RUN = False
    window_base.pysm_context = None
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    started = time.monotonic()
    outcome = {"opened": False}
    errors = []
    sys.excepthook = lambda *args: errors.append(str(args[1]))
    timer = QTimer()

    def check():
        if time.monotonic() - started > 10:
            outcome["error"] = "Initialization timed out"
            app.exit(2)
            return
        dialogs = [w for w in app.topLevelWidgets() if isinstance(w, QMessageBox) and w.isVisible()]
        if dialogs:
            outcome["error"] = dialogs[0].text()
            dialogs[0].accept()
            app.exit(2)
            return
        windows = [w for w in app.topLevelWidgets() if isinstance(w, QMainWindow) and w.isVisible()]
        if not windows:
            return
        window = windows[0]
        if getattr(window, "_worker", None) is not None:
            return
        tabs = getattr(window, "view_tabs", None)
        outcome.update(opened=True, tabs=tabs.count() if tabs is not None else 0, window=type(window).__name__,
                       module_file=sys.modules[type(window).__module__].__file__)
        window.close()
        if not hasattr(window, "_image_pipeline") or window._image_pipeline.is_closed:
            timer.stop()
            app.quit()

    timer.timeout.connect(check)
    timer.start(100)
    # Closing an image-backed window can be deferred to its shutdown callback.
    app.lastWindowClosed.connect(app.quit)
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as exc:
        outcome["exit"] = exc.code
    outcome["callback_errors"] = errors
    print(json.dumps(outcome, ensure_ascii=True))
    return 0 if outcome["opened"] and not errors and not outcome.get("error") and not outcome.get("exit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
