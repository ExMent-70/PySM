"""Shared window geometry, report links and application bootstrap."""
from __future__ import annotations
from html import escape
import logging
from pathlib import Path
import sys
from PySide6.QtCore import QProcess, QUrl, QUrlQuery
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.window_state_manager import WindowStateManager
except ImportError:
    pysm_context = theme_api = WindowStateManager = None
IS_MANAGED_RUN = pysm_context is not None
logger = logging.getLogger(__name__)

class WindowBase(QMainWindow):
    """Common UI services; no selection editing or file operations."""

    def _message_header_html(self):
        return self.user_message_html

    def _emit_final_log_once(self):
        if self._final_log_emitted or pysm_context is None:
            return
        try:
            pysm_context.log_html(self._completion_log_html())
            self._final_log_emitted = True
        except Exception:
            logger.warning("Не удалось вывести итоговый лог", exc_info=True)

    def _restore_window_state(self) -> None:
        if not (IS_MANAGED_RUN and pysm_context and WindowStateManager):
            return
        try:
            saved_state = pysm_context.get_structured(self.WINDOW_STATE_VAR, {})
            if isinstance(saved_state, dict) and saved_state:
                WindowStateManager.restore_state(
                    window=self,
                    state_data=saved_state,
                    splitters=self._splitters(),
                )
                active_tab = saved_state.get("active_tab")
                tabs = getattr(self, "view_tabs", None)
                if tabs is not None and isinstance(active_tab, int) and 0 <= active_tab < tabs.count():
                    tabs.setCurrentIndex(active_tab)
        except Exception:
            logger.warning("Не удалось восстановить состояние окна", exc_info=True)

    def _save_window_state(self) -> None:
        if not (IS_MANAGED_RUN and pysm_context and WindowStateManager):
            return
        try:
            state = WindowStateManager.save_state(
                window=self,
                splitters=self._splitters(),
            )
            tabs = getattr(self, "view_tabs", None)
            if tabs is not None:
                state["active_tab"] = tabs.currentIndex()
            pysm_context.set_structured(self.WINDOW_STATE_VAR, state)
        except Exception:
            logger.warning("Не удалось сохранить состояние окна", exc_info=True)

    def _open_report_link(self, url: QUrl) -> None:
        """Open local report links externally instead of navigating in QTextBrowser."""
        action = ""
        if url.scheme() == "pysm":
            action = url.host()
            path = Path(QUrlQuery(url).queryItemValue("path"))
        else:
            path = Path(url.toLocalFile())
        if not path.exists():
            QMessageBox.information(
                self,
                "Файл ещё не создан",
                f"Объект пока не существует:\n{path}",
            )
            return
        if action == "reveal-file":
            self._reveal_file(path)
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _reveal_file(path: Path) -> None:
        """Open the containing folder and select the file in Windows Explorer."""
        if sys.platform == "win32":
            QProcess.startDetached("explorer.exe", ["/select,", str(path.resolve())])
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    @staticmethod
    def _report_action_url(path: Path, action: str) -> QUrl:
        url = QUrl()
        url.setScheme("pysm")
        url.setHost(action)
        query = QUrlQuery()
        query.addQueryItem("path", str(path.resolve()))
        url.setQuery(query)
        return url

    @classmethod
    def _report_action_link(
        cls,
        path: Path,
        action: str,
        content_html: str,
    ) -> str:
        url = cls._report_action_url(path, action).toString()
        return (
            f'<a href="{escape(url, quote=True)}" '
            f'style="text-decoration:none">{content_html}</a>'
        )


def run_window(window_type, config):
    """Run one stage after its CLI has been resolved, including startup errors."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        if theme_api:
            theme_api.apply_theme_to_app(app)
        window = window_type(config)
        window.show()
        return app.exec()
    except Exception as exc:
        logger.exception("Не удалось запустить скрипт: %s", exc)
        QMessageBox.critical(None, "Ошибка запуска", str(exc))
        return 1
