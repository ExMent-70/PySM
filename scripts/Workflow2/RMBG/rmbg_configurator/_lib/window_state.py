"""Shared persistence of all RMBG Configurator window geometries."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from PySide6.QtWidgets import QSplitter, QWidget

from pysm_lib.window_state_manager import WindowStateManager


LOGGER = logging.getLogger(__name__)

WINDOW_STATE_VAR = "win_state.rmbg_configurator"
WINDOW_STATE_SCHEMA_VERSION = 1


class RmbgWindowStateStore:
    """Merge physical state from every Configurator dialog into one variable."""

    def __init__(self, context_api: Any) -> None:
        self._context_api = context_api
        self._payload = self._load_payload()

    def _load_payload(self) -> dict[str, Any]:
        try:
            saved = self._context_api.get_structured(WINDOW_STATE_VAR, {})
        except Exception:
            LOGGER.warning("Не удалось прочитать состояние окон RMBG", exc_info=True)
            saved = {}

        windows = saved.get("windows", {}) if isinstance(saved, Mapping) else {}
        if not isinstance(windows, Mapping):
            windows = {}
        return {
            "schema_version": WINDOW_STATE_SCHEMA_VERSION,
            "windows": deepcopy(dict(windows)),
        }

    def restore(
        self,
        window_id: str,
        window: QWidget,
        *,
        splitters: Mapping[str, QSplitter] | None = None,
    ) -> None:
        """Restore one dialog without affecting states of sibling dialogs."""

        state = self._payload["windows"].get(window_id)
        if not isinstance(state, Mapping):
            return
        WindowStateManager.restore_state(
            window=window,
            state_data=state,
            splitters=dict(splitters) if splitters else None,
        )

    def save(
        self,
        window_id: str,
        window: QWidget,
        *,
        splitters: Mapping[str, QSplitter] | None = None,
        commit: bool = False,
    ) -> None:
        """Save one dialog and publish the merged state to the PySM context."""

        self._payload["windows"][window_id] = WindowStateManager.save_state(
            window=window,
            splitters=dict(splitters) if splitters else None,
        )
        try:
            self._context_api.set_structured(
                WINDOW_STATE_VAR,
                deepcopy(self._payload),
                commit=commit,
            )
        except Exception:
            LOGGER.warning("Не удалось сохранить состояние окон RMBG", exc_info=True)
