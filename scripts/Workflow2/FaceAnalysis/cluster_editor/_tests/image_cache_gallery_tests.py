"""Tests for cluster editor gallery requests built for the shared image cache."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SCRIPT_DIR.parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pysm_lib.pysm_image_cache import QtImageCache
import run_cluster_editor as cluster_editor_module
from run_cluster_editor import MainWindow
from _lib.image_requests import normalized_face_crop


class GalleryImageCacheTests(unittest.TestCase):
    def test_large_gallery_build_yields_after_one_batch(self):
        window = MainWindow.__new__(MainWindow)
        window._gallery_build_generation = 1
        window._gallery_build_batch_size = 80
        window._gallery_build_state = {
            "generation": 1,
            "filenames": [f"photo_{index:06d}.jpg" for index in range(500)],
            "index": 0,
        }
        window._add_gallery_file_items = lambda _state, filename: [filename]
        window._update_gallery_progress = mock.Mock()
        window._queue_gallery_tasks = mock.Mock()
        window._finish_gallery_build = mock.Mock()

        with mock.patch.object(cluster_editor_module.QTimer, "singleShot") as timer:
            MainWindow._process_gallery_build_batch(window)

        self.assertEqual(window._gallery_build_state["index"], 80)
        self.assertEqual(
            len(window._queue_gallery_tasks.call_args.args[0]),
            80,
        )
        timer.assert_called_once()
        window._finish_gallery_build.assert_not_called()

    def test_cleaning_task_builds_normalized_crop_and_face_rect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "photo.jpg"
            Image.new("RGB", (200, 100), "white").save(image_path)

            window = MainWindow.__new__(MainWindow)
            window.image_cache = QtImageCache(root / "cache", memory_limit_bytes=16 * 1024 * 1024)

            request, rect_norm = MainWindow._gallery_request_for_task(
                window,
                {
                    "full_path": image_path,
                    "bbox": [50, 20, 90, 60],
                    "draw_face_rect": True,
                    "target_size": (128, 128),
                    "variant": "cluster_editor.test",
                },
            )

            self.assertEqual(request.source, image_path)
            self.assertEqual(request.target_size, (128, 128))
            self.assertEqual(request.mode, "fit")
            self.assertTrue(request.allow_upscale)
            self.assertEqual(request.variant, "cluster_editor.test")
            self.assertIsNotNone(request.crop)
            self.assertIsNotNone(rect_norm)
            assert request.crop is not None
            assert rect_norm is not None

            crop_x, crop_y, crop_width, crop_height = request.crop
            self.assertGreaterEqual(crop_x, 0)
            self.assertGreaterEqual(crop_y, 0)
            self.assertGreater(crop_width, 0)
            self.assertGreater(crop_height, 0)
            self.assertLessEqual(crop_x + crop_width, 1)
            self.assertLessEqual(crop_y + crop_height, 1)
            self.assertAlmostEqual(crop_width * 200, crop_height * 100)

            rect_x, rect_y, rect_width, rect_height = rect_norm
            self.assertGreaterEqual(rect_x, 0)
            self.assertGreaterEqual(rect_y, 0)
            self.assertGreater(rect_width, 0)
            self.assertGreater(rect_height, 0)
            self.assertLessEqual(rect_x + rect_width, 1)
            self.assertLessEqual(rect_y + rect_height, 1)

    def test_square_face_crop_keeps_scale_for_noise_and_edge_faces(self):
        source_size = (1000, 800)
        bboxes = (
            (480, 300, 520, 380),  # portrait bbox
            (460, 320, 540, 360),  # landscape bbox
            (0, 300, 40, 380),     # left image edge
            (960, 300, 1000, 380), # right image edge
        )

        for bbox in bboxes:
            with self.subTest(bbox=bbox):
                crop = normalized_face_crop(source_size, bbox, padding=0.5)
                self.assertIsNotNone(crop)
                assert crop is not None
                crop_width = crop[2] * source_size[0]
                crop_height = crop[3] * source_size[1]
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]

                self.assertAlmostEqual(crop_width, crop_height)
                self.assertAlmostEqual(
                    max(face_width, face_height) / crop_width,
                    0.5,
                )
                self.assertGreaterEqual(crop[0], 0)
                self.assertGreaterEqual(crop[1], 0)
                self.assertLessEqual(crop[0] + crop[2], 1)
                self.assertLessEqual(crop[1] + crop[3], 1)

    def test_missing_gallery_source_returns_no_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window = MainWindow.__new__(MainWindow)
            window.image_cache = QtImageCache(root / "cache", memory_limit_bytes=16 * 1024 * 1024)

            self.assertIsNone(
                MainWindow._gallery_request_for_task(
                    window,
                    {"full_path": root / "missing.jpg"},
                )
            )


if __name__ == "__main__":
    unittest.main()
