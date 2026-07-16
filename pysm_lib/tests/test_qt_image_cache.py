"""Qt-specific contracts for the shared image cache."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

try:
    from PIL import Image
except ImportError:
    Image = None

from pysm_lib.pysm_image_cache import (
    AsyncImageLoader,
    ImageRequest,
    QtImageCache,
)


def _create_image(path: Path, size: tuple[int, int] = (400, 200)) -> None:
    image = QImage(*size, QImage.Format.Format_ARGB32)
    image.fill(QColor(20, 40, 80, 180))
    painter = QPainter(image)
    painter.fillRect(10, 10, size[0] // 2, size[1] // 2, QColor(240, 80, 20, 220))
    painter.end()
    if not image.save(str(path)):
        raise RuntimeError(f"failed to create test image: {path}")


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        if app is not None:
            app.processEvents()
        time.sleep(0.005)
    return bool(predicate())


class QtImageCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_fit_decode_uses_memory_cache_and_preserves_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            _create_image(source)
            cache = QtImageCache(root / "cache", memory_limit_bytes=2_000_000)
            request = ImageRequest(source, (100, 100), mode="fit")

            first = cache.load(request)
            second = cache.load(request)

            self.assertEqual(first.size().toTuple(), (100, 50))
            self.assertTrue(first.hasAlphaChannel())
            self.assertEqual(second.size(), first.size())
            self.assertEqual(cache.stats.source_decodes, 1)
            self.assertEqual(cache.stats.memory_hits, 1)

    def test_persistent_derivative_is_reused_by_new_cache_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            _create_image(source)
            request = ImageRequest(source, (120, 80), mode="fit", variant="disk")

            first_cache = QtImageCache(root / "cache", memory_limit_bytes=2_000_000)
            first = first_cache.load(request, persist=True)
            second_cache = QtImageCache(root / "cache", memory_limit_bytes=2_000_000)
            second = second_cache.load(request, persist=True)

            self.assertFalse(first.isNull())
            self.assertEqual(second, first)
            self.assertEqual(first_cache.stats.source_decodes, 1)
            self.assertEqual(first_cache.stats.disk_writes, 1)
            self.assertEqual(second_cache.stats.disk_hits, 1)
            self.assertEqual(second_cache.stats.source_decodes, 0)

    def test_allow_upscale_is_part_of_derivative_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "small.png"
            _create_image(source, (40, 20))
            cache = QtImageCache(root / "cache", memory_limit_bytes=1_000_000)

            natural = cache.load(ImageRequest(source, (100, 100), mode="fit"))
            enlarged = cache.load(
                ImageRequest(source, (100, 100), mode="fit", allow_upscale=True)
            )

            self.assertEqual(natural.size().toTuple(), (40, 20))
            self.assertEqual(enlarged.size().toTuple(), (100, 50))
            self.assertEqual(cache.stats.source_decodes, 2)

    def test_source_size_does_not_decode_or_populate_memory_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            _create_image(source, (321, 123))
            cache = QtImageCache(root / "cache", memory_limit_bytes=1_000_000)

            size = cache.source_size(source)

            self.assertEqual(size, (321, 123))
            self.assertEqual(cache.stats.source_decodes, 0)
            self.assertEqual(cache.stats.memory_items, 0)

    def test_normalized_crop_is_applied_before_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            _create_image(source, (400, 200))
            cache = QtImageCache(root / "cache", memory_limit_bytes=1_000_000)

            cropped = cache.load(
                ImageRequest(
                    source,
                    (100, 100),
                    mode="stretch",
                    crop=(0.0, 0.0, 0.5, 1.0),
                )
            )

            self.assertEqual(cropped.size().toTuple(), (100, 100))

    def test_multiple_crops_reuse_one_decoded_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            _create_image(source, (400, 200))
            cache = QtImageCache(root / "cache", memory_limit_bytes=2_000_000)

            first = cache.load(ImageRequest(
                source,
                (100, 100),
                mode="stretch",
                crop=(0.0, 0.0, 0.5, 1.0),
                variant="first-face",
            ))
            second = cache.load(ImageRequest(
                source,
                (100, 100),
                mode="stretch",
                crop=(0.5, 0.0, 0.5, 1.0),
                variant="second-face",
            ))

            self.assertFalse(first.isNull())
            self.assertFalse(second.isNull())
            self.assertEqual(cache.stats.source_decodes, 1)

    def test_corrupt_persistent_derivative_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            _create_image(source)
            cache = QtImageCache(root / "cache", memory_limit_bytes=2_000_000)
            request = ImageRequest(source, (120, 80), variant="repair")
            key = cache.key_for(request)
            corrupt_path = cache._cache.disk.path_for(key, ".png")
            corrupt_path.parent.mkdir(parents=True)
            corrupt_path.write_bytes(b"not-an-image")

            image = cache.load(request, persist=True)

            self.assertFalse(image.isNull())
            self.assertNotEqual(corrupt_path.read_bytes(), b"not-an-image")
            self.assertEqual(cache.stats.source_decodes, 1)

    @unittest.skipUnless(Image is not None, "Pillow is required for EXIF fixture")
    def test_source_size_reports_auto_transformed_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "oriented.jpg"
            image = Image.new("RGB", (80, 40), "red")
            exif = Image.Exif()
            exif[274] = 6
            image.save(source, "JPEG", exif=exif)

            self.assertEqual(
                QtImageCache.source_size(source, auto_transform=False),
                (80, 40),
            )
            self.assertEqual(
                QtImageCache.source_size(source, auto_transform=True),
                (40, 80),
            )


class AsyncImageLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_identical_inflight_requests_are_coalesced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            _create_image(source, (1200, 800))
            cache = QtImageCache(root / "cache", memory_limit_bytes=8_000_000)
            loader = AsyncImageLoader(cache, max_threads=2)
            request = ImageRequest(source, (300, 200), variant="coalesced")
            original_decode = cache._decode_source
            release = threading.Event()

            def delayed_decode(image_request: ImageRequest) -> QImage:
                release.wait(2.0)
                return original_decode(image_request)

            cache._decode_source = delayed_decode
            results = []
            loader.imageReady.connect(results.append)

            loader.request(request, channel="first")
            loader.request(request, channel="second")
            release.set()

            self.assertTrue(_wait_until(lambda: len(results) == 2))
            self.assertEqual(loader.stats.coalesced, 1)
            self.assertEqual(cache.stats.source_decodes, 1)
            self.assertTrue(all(not result.image.isNull() for result in results))

    def test_newer_channel_request_suppresses_stale_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            _create_image(source, (1000, 600))
            cache = QtImageCache(root / "cache", memory_limit_bytes=8_000_000)
            loader = AsyncImageLoader(cache, max_threads=2)
            original_decode = cache._decode_source
            slow_started = threading.Event()
            release_slow = threading.Event()

            def controlled_decode(image_request: ImageRequest) -> QImage:
                if image_request.variant == "slow":
                    slow_started.set()
                    release_slow.wait(2.0)
                return original_decode(image_request)

            cache._decode_source = controlled_decode
            results = []
            loader.imageReady.connect(results.append)
            slow = ImageRequest(source, (320, 200), variant="slow")
            current = ImageRequest(source, (160, 100), variant="current")

            loader.request(slow, channel="preview")
            self.assertTrue(slow_started.wait(1.0))
            current_id = loader.request(current, channel="preview")
            release_slow.set()

            self.assertTrue(_wait_until(lambda: loader.stats.completed == 2))
            self.assertTrue(_wait_until(lambda: len(results) == 1))
            self.assertEqual(results[0].request_id, current_id)
            self.assertEqual(results[0].image.size().toTuple(), (160, 96))
            self.assertEqual(loader.stats.stale_dropped, 1)

    def test_cancel_removes_queued_request_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_source = root / "first.png"
            cancelled_source = root / "cancelled.png"
            _create_image(first_source, (1200, 800))
            _create_image(cancelled_source, (1200, 800))
            cache = QtImageCache(root / "cache", memory_limit_bytes=8_000_000)
            loader = AsyncImageLoader(cache, max_threads=1)
            original_decode = cache._decode_source
            first_started = threading.Event()
            release_first = threading.Event()
            decoded_variants: list[str] = []

            def controlled_decode(image_request: ImageRequest) -> QImage:
                decoded_variants.append(image_request.variant)
                if image_request.variant == "first":
                    first_started.set()
                    release_first.wait(2.0)
                return original_decode(image_request)

            cache._decode_source = controlled_decode
            results = []
            loader.imageReady.connect(results.append)

            loader.request(
                ImageRequest(first_source, (300, 200), variant="first"),
                channel="first",
            )
            self.assertTrue(first_started.wait(1.0))
            loader.request(
                ImageRequest(cancelled_source, (300, 200), variant="cancelled"),
                channel="cancelled",
            )
            loader.cancel("cancelled")
            release_first.set()

            self.assertTrue(_wait_until(lambda: len(results) == 1))
            self.assertTrue(loader.wait_for_done(2000))
            self.assertEqual(decoded_variants, ["first"])
            self.assertEqual(results[0].channel, "first")

    def test_cancelled_running_task_does_not_capture_identical_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            _create_image(source, (1200, 800))
            cache = QtImageCache(root / "cache", memory_limit_bytes=8_000_000)
            loader = AsyncImageLoader(cache, max_threads=2)
            request = ImageRequest(source, (300, 200), variant="replacement")
            original_decode = cache._decode_source
            first_started = threading.Event()
            release_first = threading.Event()
            decode_calls = 0

            def controlled_decode(image_request: ImageRequest) -> QImage:
                nonlocal decode_calls
                decode_calls += 1
                if decode_calls == 1:
                    first_started.set()
                    release_first.wait(2.0)
                return original_decode(image_request)

            cache._decode_source = controlled_decode
            results = []
            loader.imageReady.connect(results.append)

            loader.request(request, channel="preview")
            self.assertTrue(first_started.wait(1.0))
            loader.cancel("preview")
            replacement_id = loader.request(request, channel="preview")
            release_first.set()

            self.assertTrue(_wait_until(lambda: len(results) == 1))
            self.assertTrue(loader.wait_for_done(2000))
            self.assertEqual(results[0].request_id, replacement_id)
            self.assertFalse(results[0].image.isNull())
            self.assertEqual(cache.stats.source_decodes, 1)
            self.assertTrue(_wait_until(lambda: not loader._detached))


if __name__ == "__main__":
    unittest.main()
