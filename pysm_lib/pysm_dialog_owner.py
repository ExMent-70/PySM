"""Connect a script's top-level windows to the PySM window on Windows."""

import ctypes
import logging
import os
import sys

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)
OWNER_ENV = "PYSM_PARENT_WINDOW_HWND"


def attach_to_pysm(window: QWidget) -> bool:
    """Set the native owner without making the window globally topmost."""
    if sys.platform != "win32":
        return False
    raw_owner = os.environ.get(OWNER_ENV)
    if not raw_owner:
        return False
    try:
        owner = int(raw_owner)
        if owner <= 0:
            return False
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.IsWindow.argtypes = [wintypes.HWND]
        user32.IsWindow.restype = wintypes.BOOL
        if not user32.IsWindow(owner):
            return False
        set_owner = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
        set_owner.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        set_owner.restype = ctypes.c_ssize_t
        ctypes.set_last_error(0)
        result = set_owner(int(window.winId()), -8, owner)  # GWLP_HWNDPARENT
        if not result and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        return True
    except (ValueError, OSError) as exc:
        logger.warning("Не удалось связать окно с PySM: %s", exc)
        return False


class _OwnerFilter(QObject):
    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Type.Show and isinstance(watched, QWidget)
                and watched.isWindow() and watched.parentWidget() is None):
            # Qt completes native window setup after sending Show.
            QTimer.singleShot(0, watched, lambda: attach_to_pysm(watched))
        return False


def install_dialog_owner(app) -> None:
    """Attach parentless windows, including later error dialogs, on first show."""
    if not hasattr(app, "_pysm_owner_filter"):
        app._pysm_owner_filter = _OwnerFilter(app)
        app.installEventFilter(app._pysm_owner_filter)
