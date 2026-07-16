"""Regression tests for hover tracking in the right-side face panel."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QImage, QPalette, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListWidgetItem

from scripts.Workflow2.FaceAnalysis.cluster_editor._lib.editor_delegates import (
    FACE_PIXMAP_ROLE,
    FACE_STATUS_COLOR_ROLE,
)
from scripts.Workflow2.FaceAnalysis.cluster_editor._lib.editor_widgets import (
    FaceDetailsWidget,
)


class FacePanelHoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_pointer_moves_are_reported_for_each_face_without_mouse_button(self) -> None:
        widget = FaceDetailsWidget(mode="location")
        widget.resize(360, 260)
        widget.addItems(["Лицо #1", "Лицо #2"])
        widget.show()
        widget.doItemsLayout()
        self.app.processEvents()

        self.assertTrue(widget.hasMouseTracking())
        self.assertTrue(widget.viewport().hasMouseTracking())
        self.assertTrue(widget.testAttribute(Qt.WidgetAttribute.WA_Hover))
        self.assertTrue(widget.viewport().testAttribute(Qt.WidgetAttribute.WA_Hover))

        entered_rows: list[int] = []
        widget.entered.connect(lambda index: entered_rows.append(index.row()))
        first = widget.visualItemRect(widget.item(0)).center()
        second = widget.visualItemRect(widget.item(1)).center()

        QTest.mouseMove(widget.viewport(), first)
        self.app.processEvents()
        QTest.mouseMove(widget.viewport(), second, delay=10)
        deadline = time.monotonic() + 1.0
        while 1 not in entered_rows and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)

        self.assertIn(0, entered_rows)
        self.assertIn(1, entered_rows)
        widget.close()

    def test_async_pixmaps_never_change_face_cell_geometry(self) -> None:
        for icon_size in (100, 130, 240, 400):
            for face_count in (1, 2, 3):
                with self.subTest(icon_size=icon_size, face_count=face_count):
                    widget = FaceDetailsWidget(mode="location")
                    widget.resize(920, 560)
                    widget.setIconSize(QSize(icon_size, icon_size))
                    widget.setGridSize(QSize(icon_size + 20, icon_size + 60))
                    for index in range(face_count):
                        item = QListWidgetItem(f"Лицо #{index + 1}\nУченик")
                        item.setData(FACE_STATUS_COLOR_ROLE, "#00aa00")
                        widget.addItem(item)
                    widget.setCurrentRow(0)
                    widget.show()
                    self.app.processEvents()

                    before = [
                        widget.visualItemRect(widget.item(index))
                        for index in range(face_count)
                    ]
                    portrait = QPixmap(max(1, icon_size * 3 // 4), icon_size)
                    portrait.fill(QColor("steelblue"))
                    for index in range(face_count):
                        widget.item(index).setData(FACE_PIXMAP_ROLE, portrait)
                    self.app.processEvents()
                    after = [
                        widget.visualItemRect(widget.item(index))
                        for index in range(face_count)
                    ]

                    self.assertEqual(after, before)
                    for index, rect in enumerate(after):
                        self.assertIs(widget.itemAt(rect.center()), widget.item(index))
                    widget.close()

    def test_delegate_draws_status_border_from_item_data(self) -> None:
        widget = FaceDetailsWidget(mode="location")
        widget.resize(220, 240)
        item = QListWidgetItem("Лицо #1\nУченик")
        item.setData(FACE_STATUS_COLOR_ROLE, "#00aa00")
        pixmap = QPixmap(75, 100)
        pixmap.fill(QColor("white"))
        item.setData(FACE_PIXMAP_ROLE, pixmap)
        widget.addItem(item)
        widget.show()
        self.app.processEvents()

        rendered = QImage(widget.viewport().size(), QImage.Format.Format_RGB32)
        rendered.fill(QColor("black"))
        widget.viewport().render(rendered)
        status_color = QColor("#00aa00")
        status_pixels = sum(
            rendered.pixelColor(x, y) == status_color
            for y in range(rendered.height())
            for x in range(rendered.width())
        )

        self.assertGreater(status_pixels, 0)
        widget.close()

    def test_delegate_draws_selected_and_hover_states_once(self) -> None:
        widget = FaceDetailsWidget(mode="location")
        widget.resize(360, 260)
        pixmap = QPixmap(100, 120)
        pixmap.fill(QColor("white"))
        for index in range(2):
            item = QListWidgetItem(f"Лицо #{index + 1}")
            item.setData(FACE_PIXMAP_ROLE, pixmap)
            widget.addItem(item)
        widget.setCurrentRow(0)
        widget.show()
        self.app.processEvents()

        second_rect = widget.visualItemRect(widget.item(1))
        QTest.mouseMove(widget.viewport(), second_rect.center())
        self.app.processEvents()
        rendered = QImage(widget.viewport().size(), QImage.Format.Format_RGB32)
        rendered.fill(QColor("black"))
        widget.viewport().render(rendered)

        hover_color = widget.itemDelegate().hover_color
        selected_color = widget.palette().color(QPalette.ColorRole.Highlight)
        colors = {
            rendered.pixelColor(x, y).name()
            for y in range(rendered.height())
            for x in range(rendered.width())
        }
        self.assertIn(hover_color.name(), colors)
        self.assertIn(selected_color.name(), colors)
        widget.close()

if __name__ == "__main__":
    unittest.main()
