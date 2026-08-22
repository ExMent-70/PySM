from __future__ import annotations

import os
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
)

from pysm_lib.window_state_manager import WindowStateManager


class WindowStateManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_qdialog_geometry_and_splitter_round_trip(self) -> None:
        source = QDialog()
        source.resize(760, 480)
        source_splitter = QSplitter(source)
        source_splitter.addWidget(QWidget())
        source_splitter.addWidget(QWidget())
        QHBoxLayout(source).addWidget(source_splitter)
        source.show()
        self.app.processEvents()
        source_splitter.setSizes([220, 520])

        state = WindowStateManager.save_state(
            source,
            splitters={"main": source_splitter},
        )

        self.assertIn("geometry", state)
        self.assertEqual(state["window_mode"], "normal")
        self.assertIn("main", state["splitters"])
        self.assertNotIn("window_state", state)

        restored = QDialog()
        restored_splitter = QSplitter(restored)
        restored_splitter.addWidget(QWidget())
        restored_splitter.addWidget(QWidget())
        QHBoxLayout(restored).addWidget(restored_splitter)
        WindowStateManager.restore_state(
            restored,
            state,
            splitters={"main": restored_splitter},
        )

        self.assertEqual(restored.size(), source.size())
        self.assertEqual(restored_splitter.saveState(), source_splitter.saveState())
        source.close()
        restored.close()

    def test_qmainwindow_native_state_remains_supported(self) -> None:
        window = QMainWindow()
        state = WindowStateManager.save_state(window)
        self.assertIn("window_state", state)

    def test_broken_geometry_does_not_block_valid_splitter_restore(self) -> None:
        class BrokenBase64:
            @staticmethod
            def encode(_encoding: str) -> bytes:
                raise ValueError("broken geometry")

        source = QSplitter()
        source.addWidget(QWidget())
        source.addWidget(QWidget())
        source.resize(600, 300)
        source.setSizes([180, 420])
        target = QSplitter()
        target.addWidget(QWidget())
        target.addWidget(QWidget())
        target.resize(600, 300)
        state = {
            "geometry": BrokenBase64(),
            "splitters": {
                "main": source.saveState().toBase64().data().decode("utf-8"),
            },
        }

        with self.assertLogs("pysm_lib.window_state_manager", level="WARNING"):
            WindowStateManager.restore_state(
                QDialog(),
                state,
                splitters={"main": target},
            )

        self.assertEqual(target.saveState(), source.saveState())


if __name__ == "__main__":
    unittest.main()
