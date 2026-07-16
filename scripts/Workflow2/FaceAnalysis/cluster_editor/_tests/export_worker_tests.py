"""Regression tests for export lifecycle and output safety."""

from __future__ import annotations

from pathlib import Path
import concurrent.futures
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
FACE_ANALYSIS_ROOT = SCRIPT_DIR.parent
if str(FACE_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(FACE_ANALYSIS_ROOT))

from _lib import editor_workers
from _lib.editor_workers import ExportWorker, run_export_task
from _lib.export_controller import ExportController
from run_cluster_editor import _export_folder_name, _safe_export_path


class ExportWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _worker(self) -> ExportWorker:
        return ExportWorker(
            tasks=[],
            num_threads=1,
            enhancement_factors={},
            target_size=(100, 100),
            target_dpi=(300, 300),
            quality=90,
            apply_watermarks=False,
        )

    def test_pool_creation_failure_still_emits_finished_once(self) -> None:
        worker = self._worker()
        messages: list[str] = []
        worker.finished.connect(messages.append)

        with mock.patch.object(
            editor_workers.concurrent.futures,
            "ProcessPoolExecutor",
            side_effect=RuntimeError("pool failed"),
        ):
            worker.tasks = [{"source_path": "a.jpg", "output_path": "b.jpg"}]
            worker.run()

        self.assertEqual(len(messages), 1)
        self.assertIn("Ошибок: 1", messages[0])

    def test_pre_cancelled_worker_emits_finished_once(self) -> None:
        worker = self._worker()
        messages: list[str] = []
        worker.finished.connect(messages.append)
        worker.request_interruption()

        worker.run()

        self.assertEqual(len(messages), 1)
        self.assertIn("Экспорт прерван", messages[0])

    def test_running_export_can_be_interrupted(self) -> None:
        worker = self._worker()
        worker.tasks = [{"source_path": "a.jpg", "output_path": "b.jpg"}]
        messages: list[str] = []
        worker.finished.connect(messages.append)
        started = threading.Event()
        release = threading.Event()

        def slow_task(_task):
            started.set()
            release.wait(2.0)
            return "OK"

        with (
            mock.patch.object(
                editor_workers.concurrent.futures,
                "ProcessPoolExecutor",
                concurrent.futures.ThreadPoolExecutor,
            ),
            mock.patch.object(editor_workers, "run_export_task", slow_task),
        ):
            thread = threading.Thread(target=worker.run)
            thread.start()
            self.assertTrue(started.wait(1.0))
            worker.request_interruption()
            release.set()
            thread.join(2.0)

        deadline = time.monotonic() + 1.0
        while not messages and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(messages), 1)
        self.assertIn("Экспорт прерван", messages[0])


class ExportControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_controller_releases_thread_after_worker_finished(self) -> None:
        controller = ExportController()
        stopped: list[bool] = []
        controller.stopped.connect(lambda: stopped.append(True))

        def finish_immediately(worker):
            worker.finished.emit("done")

        with mock.patch.object(ExportWorker, "run", finish_immediately):
            controller.start([], 1, {}, (100, 100), (300, 300), 90, False)
            deadline = time.monotonic() + 2.0
            while not stopped and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)

        self.assertTrue(stopped)
        self.assertFalse(controller.is_running)

class ExportOutputTests(unittest.TestCase):
    def test_atomic_export_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jpg"
            output = root / "out" / "result.jpg"
            Image.new("RGB", (40, 30), "red").save(source, "JPEG")

            result = run_export_task({
                "source_path": source,
                "output_path": output,
                "student_name": "Иванов Иван",
                "target_size": (40, 30),
                "target_dpi": (300, 300),
                "quality": 90,
                "apply_watermarks": False,
                "factors": {},
            })

            self.assertEqual(result, "OK")
            self.assertTrue(output.is_file())
            self.assertFalse(list(output.parent.glob(".*.tmp")))

    def test_same_name_with_different_ids_produces_distinct_folders(self) -> None:
        first = _export_folder_name("Иванов Иван", "A7K3-S001", "cluster_1")
        second = _export_folder_name("Иванов Иван", "A7K3-S002", "cluster_2")
        self.assertNotEqual(first, second)

    def test_export_path_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "пределы каталога"):
                _safe_export_path(root, "student", "../../outside.jpg")


if __name__ == "__main__":
    unittest.main()
