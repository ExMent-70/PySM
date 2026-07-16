"""Contract tests for the shared image-cache foundation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from pysm_lib.pysm_image_cache import (
    ByteLRUCache,
    DiskImageCache,
    ImageCacheKey,
    ImageRequest,
)


def _process_cache_write(
    cache_root: str,
    source: str,
    payload: bytes,
) -> str:
    request = ImageRequest(Path(source), (200, 150), variant="process-race")
    key = ImageCacheKey.from_request(request)
    return str(DiskImageCache(Path(cache_root)).write_bytes(key, payload))


class ImageCacheInvalidationTests(unittest.TestCase):
    def test_source_change_creates_a_new_key_and_misses_old_derivative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "photo.jpg"
            source.write_bytes(b"first")
            request = ImageRequest(source, (320, 240), mode="fill")
            old_key = ImageCacheKey.from_request(request)
            disk_cache = DiskImageCache(root / "cache")
            disk_cache.write_bytes(old_key, b"cached-preview")

            source.write_bytes(b"later")
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            new_key = ImageCacheKey.from_request(request)

            self.assertNotEqual(old_key.digest, new_key.digest)
            self.assertEqual(disk_cache.read_bytes(old_key), b"cached-preview")
            self.assertIsNone(disk_cache.read_bytes(new_key))

    def test_processing_variant_and_version_are_part_of_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "photo.jpg"
            source.write_bytes(b"image")
            base = ImageCacheKey.from_request(
                ImageRequest(source, (100, 100), variant="catalog")
            )
            changed_variant = ImageCacheKey.from_request(
                ImageRequest(source, (100, 100), variant="cluster-editor")
            )
            changed_algorithm = ImageCacheKey.from_request(
                ImageRequest(
                    source,
                    (100, 100),
                    variant="catalog",
                    algorithm_version="2",
                )
            )

            self.assertNotEqual(base.digest, changed_variant.digest)
            self.assertNotEqual(base.digest, changed_algorithm.digest)


class ByteLRUCacheTests(unittest.TestCase):
    def test_access_promotes_item_before_byte_budget_eviction(self) -> None:
        cache: ByteLRUCache[str, str] = ByteLRUCache(max_bytes=10)
        cache.put("a", "A", 4)
        cache.put("b", "B", 4)

        self.assertEqual(cache.get("a"), "A")
        cache.put("c", "C", 4)

        self.assertNotIn("b", cache)
        self.assertIn("a", cache)
        self.assertIn("c", cache)
        self.assertEqual(cache.stats.current_bytes, 8)
        self.assertEqual(cache.stats.evictions, 1)

    def test_oversized_item_is_not_cached_or_allowed_over_budget(self) -> None:
        cache: ByteLRUCache[str, bytes] = ByteLRUCache(max_bytes=5)
        cache.put("old", b"old", 3)

        inserted = cache.put("large", b"too large", 9)

        self.assertFalse(inserted)
        self.assertNotIn("large", cache)
        self.assertLessEqual(cache.stats.current_bytes, cache.stats.max_bytes)
        self.assertIn("old", cache)


class DiskImageCacheConcurrencyTests(unittest.TestCase):
    def _key(self, root: Path) -> ImageCacheKey:
        source = root / "source.jpg"
        source.write_bytes(b"source")
        return ImageCacheKey.from_request(ImageRequest(source, (200, 150)))

    def test_concurrent_writers_publish_one_complete_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            key = self._key(root)
            cache_root = root / "cache"
            payloads = [bytes([index]) * 256_000 for index in range(1, 17)]

            def write(payload: bytes) -> Path:
                # Separate facade instances verify that coordination is shared.
                return DiskImageCache(cache_root).write_bytes(key, payload)

            with ThreadPoolExecutor(max_workers=16) as executor:
                paths = list(executor.map(write, payloads))

            self.assertEqual(len(set(paths)), 1)
            final_payload = paths[0].read_bytes()
            self.assertIn(final_payload, payloads)
            self.assertFalse(list(paths[0].parent.glob("*.tmp")))

    def test_concurrent_get_or_create_runs_producer_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            key = self._key(root)
            cache_root = root / "cache"
            producer_calls = 0
            counter_lock = threading.Lock()

            def producer() -> bytes:
                nonlocal producer_calls
                with counter_lock:
                    producer_calls += 1
                return b"generated-preview"

            def load() -> tuple[bytes, bool]:
                return DiskImageCache(cache_root).get_or_create_bytes(key, producer)

            with ThreadPoolExecutor(max_workers=12) as executor:
                results = list(executor.map(lambda _: load(), range(24)))

            self.assertEqual(producer_calls, 1)
            self.assertEqual(sum(created for _, created in results), 1)
            self.assertTrue(all(data == b"generated-preview" for data, _ in results))

    def test_independent_processes_publish_only_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            cache_root = root / "cache"
            payloads = [bytes([index]) * 128_000 for index in range(1, 5)]

            with ProcessPoolExecutor(max_workers=4) as executor:
                paths = list(executor.map(
                    _process_cache_write,
                    [str(cache_root)] * len(payloads),
                    [str(source)] * len(payloads),
                    payloads,
                ))

            self.assertEqual(len(set(paths)), 1)
            self.assertIn(Path(paths[0]).read_bytes(), payloads)


class DiskImageCacheBudgetTests(unittest.TestCase):
    def test_pruning_keeps_recent_derivatives_within_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            cache = DiskImageCache(root / "cache", max_bytes=20)
            keys = [
                ImageCacheKey.from_request(
                    ImageRequest(source, (10, 10), variant=f"item-{index}")
                )
                for index in range(3)
            ]

            cache.write_bytes(keys[0], b"a" * 10)
            time.sleep(0.01)
            cache.write_bytes(keys[1], b"b" * 10)
            self.assertEqual(cache.read_bytes(keys[0]), b"a" * 10)
            time.sleep(0.01)
            cache.write_bytes(keys[2], b"c" * 10)

            self.assertIsNotNone(cache.read_bytes(keys[0]))
            self.assertIsNone(cache.read_bytes(keys[1]))
            self.assertIsNotNone(cache.read_bytes(keys[2]))
            total = sum(
                path.stat().st_size
                for path in (root / "cache").rglob("*")
                if path.is_file()
            )
            self.assertLessEqual(total, 20)


if __name__ == "__main__":
    unittest.main()
