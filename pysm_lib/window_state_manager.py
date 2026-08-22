# file: window_state_manager.py

"""
Модуль для сохранения и восстановления физического состояния GUI-окон.
Является универсальным компонентом и может переиспользоваться в любых
скриптах на базе PySide6/PyQt6.
"""

import logging
from collections.abc import Mapping
from typing import Any, Dict, Optional

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtWidgets import QSplitter, QWidget

logger = logging.getLogger(__name__)


class WindowStateManager:
    """
    Сервисный класс для сериализации и десериализации геометрии окна
    и пропорций сплиттеров в JSON-совместимый формат (Base64).
    """

    @staticmethod
    def save_state(
        window: QWidget,
        splitters: Optional[Dict[str, QSplitter]] = None,
    ) -> Dict[str, Any]:
        """
        Собирает геометрию окна и сплиттеров.

        :param window: Окно приложения, включая QMainWindow и QDialog.
        :param splitters: Словарь вида {'имя_сплиттера': QSplitter}.
        :return: Словарь с Base64-строками состояния.
        """
        state_data: Dict[str, Any] = {}

        try:
            state_data['geometry'] = (
                window.saveGeometry().toBase64().data().decode('utf-8')
            )
        except Exception:
            logger.warning("Не удалось сохранить геометрию окна", exc_info=True)

        save_main_window_state = getattr(window, "saveState", None)
        if callable(save_main_window_state):
            try:
                state_data['window_state'] = (
                    save_main_window_state().toBase64().data().decode('utf-8')
                )
            except Exception:
                logger.warning("Не удалось сохранить состояние QMainWindow", exc_info=True)

        if window.isFullScreen():
            state_data['window_mode'] = 'fullscreen'
        elif window.isMaximized():
            state_data['window_mode'] = 'maximized'
        else:
            # Свёрнутое окно при следующем запуске намеренно открывается обычным.
            state_data['window_mode'] = 'normal'

        if splitters:
            saved_splitters: Dict[str, str] = {}
            for name, splitter in splitters.items():
                try:
                    saved_splitters[name] = (
                        splitter.saveState().toBase64().data().decode('utf-8')
                    )
                except Exception:
                    logger.warning(
                        "Не удалось сохранить разделитель %s", name, exc_info=True
                    )
            if saved_splitters:
                state_data['splitters'] = saved_splitters

        logger.debug("Состояние окна успешно сохранено.")

        return state_data

    @staticmethod
    def restore_state(
        window: QWidget,
        state_data: Mapping[str, Any],
        splitters: Optional[Dict[str, QSplitter]] = None,
    ) -> None:
        """
        Восстанавливает геометрию окна и сплиттеров из словаря.
        """
        if not isinstance(state_data, Mapping) or not state_data:
            return

        if state_data.get('geometry'):
            try:
                window.restoreGeometry(
                    QByteArray.fromBase64(state_data['geometry'].encode('utf-8'))
                )
            except Exception:
                logger.warning("Не удалось восстановить геометрию окна", exc_info=True)

        restore_main_window_state = getattr(window, "restoreState", None)
        if callable(restore_main_window_state) and state_data.get('window_state'):
            try:
                restore_main_window_state(
                    QByteArray.fromBase64(state_data['window_state'].encode('utf-8'))
                )
            except Exception:
                logger.warning(
                    "Не удалось восстановить состояние QMainWindow", exc_info=True
                )

        mode = state_data.get('window_mode')
        if mode == 'fullscreen':
            window.setWindowState(Qt.WindowState.WindowFullScreen)
        elif mode == 'maximized':
            window.setWindowState(Qt.WindowState.WindowMaximized)
        elif mode == 'normal':
            window.setWindowState(Qt.WindowState.WindowNoState)

        saved_splitters = state_data.get('splitters')
        if splitters and isinstance(saved_splitters, Mapping):
            for name, splitter in splitters.items():
                encoded_state = saved_splitters.get(name)
                if encoded_state:
                    try:
                        splitter.restoreState(
                            QByteArray.fromBase64(encoded_state.encode('utf-8'))
                        )
                    except Exception:
                        logger.warning(
                            "Не удалось восстановить разделитель %s",
                            name,
                            exc_info=True,
                        )

        logger.debug("Состояние окна успешно восстановлено.")
