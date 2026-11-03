"""Synthetic search and offscreen confirmation regressions; no real photo data."""

import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Event
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QMainWindow, QMessageBox, QProgressBar

from run_cluster_editor import MainWindow
from _lib.data_manager import ClusterDataManager
from _lib.data_models import Face, ImageRecord
from _lib.editor_dialogs import FaceSelectorDialog
from _lib.similar_faces import FaceAddress, FaceEmbeddingIndex, SearchCancelled, SimilarFace, face_address
from _lib.similar_faces_controller import SimilarFacesController
from pysm_lib.pysm_image_cache import AsyncImageLoader, QtImageCache


def write_embeddings(root, vectors, index):
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "faces_embeddings.npy", np.asarray(vectors, dtype=float))
    (root / "faces_index.json").write_text(json.dumps(index), encoding="utf-8")


def record(name, faces, portrait=False):
    return ImageRecord(name, faces, (100, 100), face_count=len(faces), image_type="portrait" if portrait else "group")


class SearchTests(unittest.TestCase):
    def test_fixed_samples_do_not_expand_transitively(self):
        import numpy as np

        # A->B and B->C pass; A->C does not. C must not join through B.
        vectors = [[np.cos(x), np.sin(x)] for x in (0.0, 0.6, 1.2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_embeddings(root, vectors, {"seed.jpg": [0], "b.jpg": [1], "c.jpg": [2]})
            index = FaceEmbeddingIndex()
            index.load(root, Event())
            hits, skipped = index.search(
                (FaceAddress("seed.jpg", 0, 0),),
                (FaceAddress("b.jpg", 0, 0), FaceAddress("c.jpg", 0, 0)), 0.2, Event(),
            )
            self.assertEqual([hit.address.filename for hit in hits], ["b.jpg"])
            self.assertEqual(skipped, 0)

    def test_multiple_seeds_deduplicate_and_sort_and_keep_original_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_embeddings(root, [[1, 0], [0, 1], [1, 1]], {"a": [0], "b": [1], "photo": [2, 0]})
            index = FaceEmbeddingIndex()
            index.load(root, Event())
            exact = FaceAddress("photo", 0, 1)
            near = FaceAddress("photo", 1, 0)
            hits, _ = index.search((FaceAddress("a", 0, 0), FaceAddress("b", 0, 0)), (near, exact, near), 0.3, Event())
            self.assertEqual([hit.address for hit in hits], [exact, near])
            self.assertEqual(hits[0].distance, 0.0)

    def test_invalid_vectors_indices_threshold_and_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_embeddings(root, [[1, 0], [0, 0], [float("nan"), 1]], {"a": [0], "bad": [1, 2, -1, True, 99]})
            index = FaceEmbeddingIndex()
            index.load(root, Event())
            hits, skipped = index.search((FaceAddress("a", 0, 0),), tuple(FaceAddress("bad", i, i) for i in range(6)), 1, Event())
            self.assertEqual((hits, skipped), ([], 6))
            with self.assertRaises(ValueError):
                index.search((FaceAddress("bad", 0, 0),), (), 0.3, Event())
            with self.assertRaises(ValueError):
                index.search((), (), float("nan"), Event())
            cancel = Event()
            cancel.set()
            with self.assertRaises(SearchCancelled):
                index.search((FaceAddress("a", 0, 0),), (), 0.3, cancel)

    def test_threshold_boundary_and_cache_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_embeddings(root, [[1, 0], [0, 1]], {"a": [0], "b": [1]})
            index = FaceEmbeddingIndex()
            index.load(root, Event())
            args = ((FaceAddress("a", 0, 0),), (FaceAddress("b", 0, 0),))
            self.assertEqual(len(index.search(*args, 1.0, Event())[0]), 1)
            self.assertEqual(len(index.search(*args, 0.99, Event())[0]), 0)
            write_embeddings(root, [[1, 0], [1, 0], [1, 0]], {"a": [0], "b": [1]})
            index.load(root, Event())
            self.assertEqual(len(index.search(*args, 0.0, Event())[0]), 1)

    def test_address_uses_persisted_index_including_zero(self):
        face = Face([0, 0, 1, 1], face_index=0)
        self.assertEqual(face_address("a", 5, face).slot, 0)
        face.face_index = None
        face.embedding_key = "a::3"
        self.assertEqual(face_address("a", 1, face).slot, 3)


class WindowHarness(QMainWindow):
    """Minimal real Qt parent using the production data manager and strategy."""

    def __init__(self, root):
        super().__init__()
        roster = root / "test.list"
        roster.write_text(json.dumps({"schema": 2, "list_id": "A7K3", "students": [{"student_id": "A7K3-S001", "surname": "Иванов", "name": "Иван"}]}), encoding="utf-8")
        self.data_manager = ClusterDataManager(root, mode="matches", student_list_file=roster)
        self.target = {"name": "Иван Иванов", "student_id": "A7K3-S001"}
        self.spin_similarity_distance = mock.Mock(value=lambda: 0.3)
        self.status_bar = QProgressBar(self)
        self.active_cluster_id = "error_matches"
        self._refresh_left_panel = mock.Mock()
        self._render_gallery = mock.Mock()
        self.image_cache = self.image_loader = None
        self.root = root

    def _get_cluster_item_data_by_id(self, cluster_id):
        return self.target

    def _get_image_path(self, filename):
        return self.root / filename


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_matches_window_has_search_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = WindowHarness(root)
            roster_path = harness.data_manager.student_roster.path
            harness.close()
            portrait = Face([0, 0, 5, 5], cluster_label=0, student_id="A7K3-S001")
            data = {"ref.jpg": record("ref.jpg", [portrait], portrait=True).to_dict()}
            (root / "info_faces.json").write_text(json.dumps(data), encoding="utf-8")
            window = MainWindow(root, None, "matches", 1, "", "", student_list_file=roster_path)
            window.resize(1920, 1000)
            window.show()
            deadline = time.monotonic() + 5
            while not window.data_manager.records and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            while window.data_load_thread is not None and window.data_load_thread.isRunning() and time.monotonic() < deadline:
                self.app.processEvents()
            self.assertTrue(window.data_manager.records)
            self.assertEqual(window.spin_similarity_distance.value(), 0.30)
            self.assertTrue(window.spin_similarity_distance.isVisible())
            window.spin_similarity_distance.setValue(0.15)
            self.assertEqual(window.spin_similarity_distance.value(), 0.15)
            if os.environ.get("CE_TEST_SCREENSHOT"):
                self.app.processEvents()
                window.grab().save(os.environ["CE_TEST_SCREENSHOT"])
            window.close()
            self.app.processEvents()

    def test_review_requires_confirmation_even_for_one_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = QtImageCache(Path(tmp) / "cache", memory_limit_bytes=1_000_000)
            loader = AsyncImageLoader(cache, max_threads=1)
            source = Path(tmp) / "test.jpg"
            image = QImage(10, 10, QImage.Format.Format_RGB32)
            image.fill(0)
            image.save(str(source))
            dialog = FaceSelectorDialog(source, [Face([0, 0, 5, 5])], image_cache=cache, image_loader=loader, review_distances=[0.1])
            self.assertEqual(dialog.get_selected_index(), -1)
            dialog.list_widget.clearSelection()
            dialog._on_accept()
            self.assertEqual(dialog.get_selected_index(), -1)
            self.assertFalse(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).isEnabled())
            dialog.done(FaceSelectorDialog.FINISH_REVIEW)
            self.assertEqual(dialog.result(), FaceSelectorDialog.FINISH_REVIEW)
            loader.cancel_all()
            loader.wait_for_done(5000)

    def test_background_search_filtering_confirmation_and_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = WindowHarness(root)
            manager = w.data_manager
            seed = Face([0, 0, 1, 1], student_id="A7K3-S001", extra_data={"matched_portrait_cluster_label": 0})
            unknown = Face([0, 0, 1, 1])
            known = Face([0, 0, 1, 1], student_id="A7K3-S001", extra_data={"matched_portrait_cluster_label": 0})
            trash = Face([0, 0, 1, 1], quality_status="trash")
            portrait = Face([0, 0, 1, 1], cluster_label=0, student_id="A7K3-S001")
            manager.records = {
                "seed.jpg": record("seed.jpg", [seed]),
                "photo.jpg": record("photo.jpg", [unknown, known, trash]),
                "ref.jpg": record("ref.jpg", [portrait], portrait=True),
            }
            (root / "photo.jpg").touch()
            write_embeddings(manager.embeddings_dir, [[1, 0]] * 5, {"seed.jpg": [0], "photo.jpg": [1, 2, 3], "ref.jpg": [4]})
            controller = SimilarFacesController(w)
            dialog = mock.Mock()
            dialog.exec.return_value = QDialog.DialogCode.Accepted
            dialog.get_selected_index.return_value = 0
            with mock.patch("_lib.similar_faces_controller.FaceSelectorDialog", return_value=dialog) as selector:
                with mock.patch("_lib.similar_faces_controller.russian_question", return_value=QMessageBox.StandardButton.Yes), mock.patch.object(QMessageBox, "information"):
                    controller.start("0", {"seed.jpg": 0})
                    deadline = time.monotonic() + 5
                    while controller.busy and time.monotonic() < deadline:
                        self.app.processEvents()
                        time.sleep(0.005)
                    self.assertFalse(controller.busy)
                self.assertEqual(selector.call_count, 1)
                self.assertEqual(selector.call_args.args[1], [unknown])
            self.assertEqual(unknown.student_id, "A7K3-S001")
            self.assertEqual(unknown.extra_data["matched_portrait_cluster_label"], 0)
            self.assertIsNone(trash.student_id)
            self.assertTrue(manager.save_data(), manager.last_error)
            saved = json.loads(manager.info_json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["photo.jpg"]["faces"][0]["student_id"], "A7K3-S001")
            self.assertTrue((root / "matches_portrait_to_group.json").is_file())
            self.assertTrue((root / "error_matches.json").is_file())
            w.close()

    def test_skip_finish_and_stale_candidates_do_not_assign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = WindowHarness(root)
            face = Face([0, 0, 1, 1])
            w.data_manager.records = {"photo.jpg": record("photo.jpg", [face])}
            (root / "photo.jpg").touch()
            controller = SimilarFacesController(w)
            controller.manager = w.data_manager
            controller.identities = {("photo.jpg", 0): face}
            controller.target_id = "0"
            controller.student_id = "A7K3-S001"
            controller.target_name = "Иван"
            address = FaceAddress("photo.jpg", 0, 0)
            hits = [SimilarFace(address, 0.0)]
            for outcome in (QDialog.DialogCode.Rejected, FaceSelectorDialog.FINISH_REVIEW):
                with mock.patch("_lib.similar_faces_controller.russian_question", return_value=QMessageBox.StandardButton.Yes), mock.patch.object(QMessageBox, "information"):
                    with mock.patch("_lib.similar_faces_controller.FaceSelectorDialog") as factory:
                        factory.FINISH_REVIEW = FaceSelectorDialog.FINISH_REVIEW
                        factory.return_value.exec.return_value = outcome
                        controller._review(hits, 0)
                self.assertIsNone(face.student_id)
            w.data_manager.records["photo.jpg"].faces[0] = Face([0, 0, 1, 1])
            self.assertIsNone(controller._current_face(address))
            controller.manager = mock.Mock()
            self.assertIsNone(controller._current_face(address))
            w.close()

    def test_manual_batch_uses_only_selected_faces_as_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = WindowHarness(Path(tmp))
            w.mode = "matches"
            w.similar_faces_controller = mock.Mock(busy=False)
            known = Face([0, 0, 1, 1], extra_data={"matched_portrait_cluster_label": 3})
            first, selected, automatic, skipped = [Face([0, 0, 1, 1]) for _ in range(4)]
            w.data_manager.records = {
                "a.jpg": record("a.jpg", [known, first, selected]),
                "b.jpg": record("b.jpg", [automatic]),
                "c.jpg": record("c.jpg", [skipped, Face([0, 0, 1, 1])]),
            }
            accepted = mock.Mock()
            accepted.exec.return_value = QDialog.DialogCode.Accepted
            accepted.get_selected_index.return_value = 1
            rejected = mock.Mock()
            rejected.exec.return_value = QDialog.DialogCode.Rejected
            with mock.patch("run_cluster_editor.FaceSelectorDialog", side_effect=[accepted, rejected]):
                MainWindow._process_drop_logic(w, "error_matches", "0", ["a.jpg", "b.jpg", "c.jpg"])
            w.similar_faces_controller.start.assert_called_once_with("0", {"a.jpg": 2, "b.jpg": 0})
            self.assertEqual(selected.student_id, "A7K3-S001")
            self.assertEqual(automatic.student_id, "A7K3-S001")
            self.assertIsNone(first.student_id)
            self.assertIsNone(skipped.student_id)
            self.assertEqual(known.extra_data["matched_portrait_cluster_label"], 3)
            w.close()

    def test_cancel_or_missing_embeddings_preserves_original_assignment(self):
        for cancel_search in (True, False):
            with self.subTest(cancel=cancel_search), tempfile.TemporaryDirectory() as tmp:
                w = WindowHarness(Path(tmp))
                source = Face([0, 0, 1, 1], student_id="A7K3-S001", extra_data={"matched_portrait_cluster_label": 0})
                unknown = Face([0, 0, 1, 1])
                w.data_manager.records = {"seed.jpg": record("seed.jpg", [source]), "other.jpg": record("other.jpg", [unknown])}
                controller = SimilarFacesController(w)
                with mock.patch.object(QMessageBox, "warning") as warning, mock.patch("_lib.similar_faces_controller.russian_question") as question:
                    controller.start("0", {"seed.jpg": 0})
                    if cancel_search:
                        controller.cancel_search()
                    deadline = time.monotonic() + 5
                    while controller.busy and time.monotonic() < deadline:
                        self.app.processEvents()
                        time.sleep(0.005)
                    self.assertFalse(controller.busy)
                    question.assert_not_called()
                    self.assertEqual(warning.call_count, 0 if cancel_search else 1)
                self.assertEqual(source.student_id, "A7K3-S001")
                self.assertIsNone(unknown.student_id)
                self.assertTrue(w.data_manager.has_changes())
                self.assertFalse(w.data_manager.info_json_path.exists())
                w.close()


if __name__ == "__main__":
    unittest.main()
