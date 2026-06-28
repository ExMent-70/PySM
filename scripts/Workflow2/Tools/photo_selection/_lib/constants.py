"""Shared constants for the photo selection GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt

PHOTO_NUMBER_DIGITS = 6
WINDOW_STATE_VAR = "win_state.photo_selection"
ITEM_NUMBER_ROLE = int(Qt.ItemDataRole.UserRole)
ITEM_PATHS_ROLE = ITEM_NUMBER_ROLE + 1
ITEM_STUDENT_ROLE = ITEM_PATHS_ROLE + 1
ITEM_LOCATION_ROLE = ITEM_STUDENT_ROLE + 1
