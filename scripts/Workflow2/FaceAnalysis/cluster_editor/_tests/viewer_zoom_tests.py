"""Wheel zoom must not also scroll the image in the graphics view."""

import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem

from run_cluster_editor import ImageViewer
from _lib.data_models import Face, ImageRecord
from _lib.editor_viewer import _label_style
from pysm_lib.pysm_image_cache import AsyncImageLoader, QtImageCache


class ViewerZoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_face_labels_fit_and_align_with_visible_frame_edge(self):
        for width in (2, 15, 40, 200):
            with self.subTest(width=width):
                scene = QGraphicsScene()
                ImageViewer._draw_bounding_box(
                    SimpleNamespace(scene=scene), [100, 100, 100 + width, 150], True, "123"
                )
                rectangles = [item for item in scene.items() if isinstance(item, QGraphicsRectItem)]
                frame = next(item for item in rectangles if item.pen().style() != Qt.PenStyle.NoPen)
                background = next(item for item in rectangles if item.pen().style() == Qt.PenStyle.NoPen)
                label = next(item for item in scene.items() if isinstance(item, QGraphicsTextItem))
                self.assertAlmostEqual(background.sceneBoundingRect().left(), frame.sceneBoundingRect().left())
                self.assertAlmostEqual(label.sceneBoundingRect().center().x(), background.sceneBoundingRect().center().x())
                self.assertLessEqual(background.sceneBoundingRect().width(), width + 0.001)
                self.assertLessEqual(label.sceneBoundingRect().width(), width + 0.001)
                if width == 200:
                    self.assertEqual(label.font().pointSize(), 28)
                if width == 15:
                    self.assertLess(label.font().pointSize(), 28)

    def test_labels_share_width_and_center_within_narrowest_frame(self):
        for entries in ([('0', 40), ('23', 90)], [('1', 3), ('123', 50)]):
            style = _label_style(entries)
            widths = []
            for text, width in entries:
                scene = QGraphicsScene()
                ImageViewer._draw_bounding_box(
                    SimpleNamespace(scene=scene), [100, 100, 100 + width, 150], True,
                    text, label_style=style,
                )
                background = next(item for item in scene.items() if isinstance(item, QGraphicsRectItem) and item.pen().style() == Qt.PenStyle.NoPen)
                label = next(item for item in scene.items() if isinstance(item, QGraphicsTextItem))
                widths.append(background.sceneBoundingRect().width())
                self.assertAlmostEqual(label.sceneBoundingRect().center().x(), background.sceneBoundingRect().center().x())
                self.assertLess(label.sceneBoundingRect().width(), widths[-1])
            self.assertAlmostEqual(widths[0], widths[1])
            self.assertLessEqual(widths[0], min(width for _, width in entries) + 0.001)

    def test_wheel_scales_once_and_centers_without_viewport_scroll(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "JPG").mkdir()
            image = QImage(2400, 1600, QImage.Format.Format_RGB32)
            image.fill(0)
            self.assertTrue(image.save(str(root / "JPG" / "photo.jpg")))
            faces = [
                Face([100, 100, 180, 200], cluster_label=0),
                Face([300, 100, 420, 200], cluster_label=23),
            ]
            record = ImageRecord("photo.jpg", faces, (1600, 2400), face_count=2)
            manager = SimpleNamespace(
                records={"photo.jpg": record}, working_dir=root,
                student_label=lambda student_id: "",
            )
            cache = QtImageCache(root / "cache", memory_limit_bytes=32_000_000)
            loader = AsyncImageLoader(cache, max_threads=1)
            viewer = ImageViewer(manager, "photo.jpg", image_cache=cache, image_loader=loader)
            viewer.show()
            deadline = time.monotonic() + 5
            while viewer.pixmap_item.pixmap().isNull() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.app.processEvents()
            self.assertFalse(viewer.pixmap_item.pixmap().isNull())
            self.assertEqual(viewer.view.dragMode(), QGraphicsView.DragMode.ScrollHandDrag)
            labels = [item for item in viewer.scene.items() if isinstance(item, QGraphicsTextItem)]
            backgrounds = [
                item for item in viewer.scene.items()
                if isinstance(item, QGraphicsRectItem) and item.pen().style() == Qt.PenStyle.NoPen
            ]
            self.assertEqual({label.toPlainText() for label in labels}, {"0", "23"})
            self.assertEqual(len(backgrounds), 2)
            self.assertAlmostEqual(backgrounds[0].rect().width(), backgrounds[1].rect().width())

            for delta in [120] * 12 + [-120] * 4:
                viewport = viewer.view.viewport()
                before = viewer.view.transform().m11()
                event = QWheelEvent(
                    QPointF(20, 20), QPointF(20, 20), QPoint(), QPoint(0, delta),
                    Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.NoScrollPhase, False,
                )
                self.app.sendEvent(viewport, event)
                self.app.processEvents()
                after = viewer.view.transform().m11()
                self.assertAlmostEqual(after / before, 1.15 if delta > 0 else 1 / 1.15)
                actual = viewer.view.mapToScene(viewport.rect().center())
                center = viewer.pixmap_item.sceneBoundingRect().center()
                self.assertLess(abs(actual.x() - center.x()) * after, 3)
                self.assertLess(abs(actual.y() - center.y()) * after, 3)
                self.assertTrue(event.isAccepted())
            viewport = viewer.view.viewport()
            start = viewport.rect().center()
            before_drag = viewer.view.mapToScene(start)
            QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(viewport, start + QPoint(70, 50), delay=10)
            QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=start + QPoint(70, 50))
            self.app.processEvents()
            after_drag = viewer.view.mapToScene(start)
            self.assertGreater(abs(after_drag.x() - before_drag.x()), 1)
            self.assertGreater(abs(after_drag.y() - before_drag.y()), 1)
            viewer.reject()
            loader.cancel_all()
            loader.wait_for_done(5000)
