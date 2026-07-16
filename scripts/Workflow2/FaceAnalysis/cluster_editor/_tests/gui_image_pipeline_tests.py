"""Offscreen smoke tests for the shared GUI image pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cluster_editor import MainWindow
from _lib.data_models import Face, ImageRecord
from _lib.editor_delegates import FACE_PIXMAP_ROLE
from _lib.editor_dialogs import EnhanceSettingsDialog, FaceSelectorDialog
from _lib.editor_delegates import FACE_PIXMAP_ROLE
from _lib.editor_viewer import ImageViewer
from pysm_lib.pysm_image_cache import AsyncImageLoader, QtImageCache


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        app.processEvents()
        time.sleep(0.005)
    return bool(predicate())


class GuiImagePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_startup_and_face_panel_use_async_shared_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            working = root / "Session" / "Output" / "Analysis_Test"
            images = working / "JPG"
            images.mkdir(parents=True)
            source = images / "photo_000001.jpg"
            image = QImage(100, 100, QImage.Format.Format_RGB32)
            image.fill(QColor("steelblue"))
            self.assertTrue(image.save(str(source)))

            roster = root / "class.list"
            roster.write_text(json.dumps({
                "list_id": "A7K3",
                "students": [{
                    "student_id": "A7K3-S001",
                    "surname": "Иванов",
                    "name": "Иван",
                }],
            }, ensure_ascii=False), encoding="utf-8")
            (working / "info_faces.json").write_text(json.dumps({
                source.name: {
                    "face_count": 2,
                    "original_shape": [100, 100],
                    "location_cluster": 1,
                    "location_name": "Класс",
                    "faces": [
                        {
                            "bbox": [5, 20, 45, 80],
                            "face_index": 0,
                            "cluster_label": None,
                            "student_id": None,
                        },
                        {
                            "bbox": [55, 20, 95, 80],
                            "face_index": 1,
                            "cluster_label": None,
                            "student_id": None,
                        },
                    ],
                },
            }), encoding="utf-8")

            window = MainWindow(
                working,
                None,
                "location",
                2,
                "",
                "",
                roster,
            )
            window.show()
            self.assertTrue(_wait_until(lambda: window.data_load_thread is None))
            self.assertTrue(_wait_until(lambda: window.image_list_widget.count() == 1))
            self.assertTrue(_wait_until(lambda: not window._gallery_thumbnail_channels))
            self.assertTrue(_wait_until(lambda: not window._face_panel_channels))

            gallery_item = window.image_list_widget.item(0)
            gallery_pixmap = gallery_item.data(Qt.ItemDataRole.DecorationRole)
            self.assertFalse(gallery_pixmap.isNull())
            self.assertEqual(window.face_details_widget.count(), 2)
            self.assertIsInstance(
                window.face_details_widget.item(0).data(FACE_PIXMAP_ROLE),
                QPixmap,
            )
            self.assertIsInstance(
                window.face_details_widget.item(1).data(FACE_PIXMAP_ROLE),
                QPixmap,
            )
            self.assertGreaterEqual(window.image_cache.stats.source_decodes, 1)

            # Rebuild the panel while thumbnail delivery is paused. Both rows
            # must reserve their final icon geometry before async results.
            with mock.patch.object(window.image_loader, "request", return_value=1):
                window._update_face_panel(gallery_item)
            self.app.processEvents()

            first_item = window.face_details_widget.item(0)
            second_item = window.face_details_widget.item(1)
            first_rect = window.face_details_widget.visualItemRect(first_item)
            second_rect = window.face_details_widget.visualItemRect(second_item)
            icon_height = window.face_details_widget.iconSize().height()
            self.assertGreaterEqual(first_rect.height(), icon_height)
            self.assertGreaterEqual(second_rect.height(), icon_height)
            self.assertLess(first_rect.right(), second_rect.left())
            self.assertIs(
                window.face_details_widget.itemAt(second_rect.center()),
                second_item,
            )

            second_working = root / "Session" / "Output" / "Analysis_Second"
            second_images = second_working / "JPG"
            second_images.mkdir(parents=True)
            second_source = second_images / "other_000002.jpg"
            self.assertTrue(image.save(str(second_source)))
            second_json = second_working / "info_faces.json"
            second_json.write_text(json.dumps({
                second_source.name: {
                    "face_count": 1,
                    "original_shape": [100, 100],
                    "location_cluster": 1,
                    "location_name": "Класс",
                    "faces": [{
                        "bbox": [20, 20, 80, 80],
                        "face_index": 0,
                        "cluster_label": 0,
                        "student_id": "A7K3-S001",
                    }],
                },
            }), encoding="utf-8")
            success, message = window.begin_working_session_switch(second_json)
            self.assertTrue(success, message)
            self.assertTrue(_wait_until(lambda: window.data_load_thread is None))
            self.assertTrue(_wait_until(lambda: window.image_list_widget.count() == 1))
            self.assertEqual(
                window.image_list_widget.item(0).data(Qt.ItemDataRole.UserRole)["filename"],
                second_source.name,
            )
            self.assertEqual(window.working_dir, second_working)
            self.assertTrue(
                _wait_until(lambda: not window.image_pipeline._retired),
                "old image pipeline was not released after the session switch",
            )

            window.close()
            self.assertTrue(window.image_loader.wait_for_done(2000))
            self.app.processEvents()

    def test_dialogs_and_viewer_use_shared_async_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "JPG"
            images.mkdir()
            source = images / "photo.jpg"
            image = QImage(120, 80, QImage.Format.Format_RGB32)
            image.fill(QColor("darkcyan"))
            self.assertTrue(image.save(str(source)))

            cache = QtImageCache(root / "cache", memory_limit_bytes=16_000_000)
            loader = AsyncImageLoader(cache, max_threads=2)
            face = Face([20, 10, 80, 70], face_index=0)

            selector = FaceSelectorDialog(
                source,
                [face],
                image_cache=cache,
                image_loader=loader,
            )
            selector.show()
            self.assertTrue(_wait_until(lambda: not selector._image_channels))
            pixmap = selector.list_widget.item(0).data(FACE_PIXMAP_ROLE)
            self.assertIsInstance(pixmap, QPixmap)
            self.assertFalse(pixmap.isNull())
            selector.reject()

            record = ImageRecord(
                source.name,
                [face],
                (80, 120),
                face_count=1,
            )
            manager = SimpleNamespace(
                records={source.name: record},
                working_dir=root,
                student_label=lambda _student_id: "",
            )
            viewer = ImageViewer(
                manager,
                source.name,
                image_cache=cache,
                image_loader=loader,
            )
            viewer.show()
            self.assertTrue(
                _wait_until(lambda: not viewer.pixmap_item.pixmap().isNull())
            )
            viewer.reject()

            enhance = EnhanceSettingsDialog(
                source,
                [face.bbox],
                image_cache=cache,
                image_loader=loader,
            )
            enhance.show()
            self.assertTrue(
                _wait_until(lambda: not enhance.pixmap_item.pixmap().isNull())
            )
            enhance.reject()
            self.assertTrue(loader.wait_for_done(2000))


if __name__ == "__main__":
    unittest.main()
