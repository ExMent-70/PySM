"""Failure-path tests for destructive and multi-file saves."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _lib import data_manager as data_manager_module
from _lib import json_io
from _lib.data_manager import ClusterDataManager
from _lib.data_models import Face, ImageRecord


def _write_roster(root: Path) -> Path:
    path = root / "class.list"
    path.write_text(
        json.dumps({
            "list_id": "A7K3",
            "students": [{
                "student_id": "A7K3-S001",
                "surname": "Иванов",
                "name": "Иван",
            }],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


class _EmbeddingLoaderStub:
    vectors = np.empty((0, 2), dtype=np.float32)
    index = {}

    def __init__(self, _root: Path) -> None:
        pass

    def load(self, _data_type: str):
        return self.vectors, self.index


class AtomicBundleTests(unittest.TestCase):
    def test_replacement_failure_restores_every_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text("old-first", encoding="utf-8")
            second.write_text("old-second", encoding="utf-8")
            original_replace = json_io.os.replace
            replacement_calls = 0

            def flaky_replace(source, target):
                nonlocal replacement_calls
                if str(source).endswith(".tmp"):
                    replacement_calls += 1
                    if replacement_calls == 2:
                        raise OSError("injected replacement failure")
                return original_replace(source, target)

            with mock.patch.object(json_io.os, "replace", side_effect=flaky_replace):
                with self.assertRaisesRegex(OSError, "injected"):
                    json_io.atomic_write_bundle({
                        first: lambda path: path.write_text("new-first", encoding="utf-8"),
                        second: lambda path: path.write_text("new-second", encoding="utf-8"),
                    })

            self.assertEqual(first.read_text(encoding="utf-8"), "old-first")
            self.assertEqual(second.read_text(encoding="utf-8"), "old-second")


class DataLoadTransactionTests(unittest.TestCase):
    def test_invalid_json_does_not_replace_current_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ClusterDataManager(
                root,
                mode="cleaning",
                student_list_file=_write_roster(root),
            )
            existing = ImageRecord(
                "existing.jpg",
                [Face([0, 0, 10, 10])],
                (100, 100),
                face_count=1,
            )
            manager.records = {"existing.jpg": existing}
            manager.info_json_path.write_text(
                json.dumps({
                    "bad.jpg": {
                        "face_count": 1,
                        "original_shape": [100, 100],
                        "faces": [{"bbox": [1, 2, 3]}],
                    }
                }),
                encoding="utf-8",
            )

            success, _ = manager.load_data()

            self.assertFalse(success)
            self.assertEqual(manager.records, {"existing.jpg": existing})


class CleaningSaveTests(unittest.TestCase):
    def _manager(self, root: Path) -> ClusterDataManager:
        manager = ClusterDataManager(
            root,
            mode="cleaning",
            student_list_file=_write_roster(root),
        )
        manager.info_json_path.write_text("{}", encoding="utf-8")
        return manager

    def test_preserves_removed_face_vectors_in_rebuilt_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            active = Face([0, 0, 10, 10], face_index=0)
            removed = Face([10, 10, 20, 20], face_index=1)
            active.embedding_key = "photo.jpg::0"
            removed.embedding_key = "photo.jpg::1"
            manager.records = {
                "photo.jpg": ImageRecord(
                    "photo.jpg",
                    [active],
                    (100, 100),
                    face_count=1,
                    removed_faces=[removed],
                )
            }
            _EmbeddingLoaderStub.vectors = np.asarray(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            )
            _EmbeddingLoaderStub.index = {"photo.jpg": [0, 1]}

            with mock.patch.object(
                data_manager_module,
                "EmbeddingLoader",
                _EmbeddingLoaderStub,
            ):
                self.assertTrue(manager.save_data(), manager.last_error)

            saved_vectors = np.load(root / "_Embeddings" / "faces_embeddings.npy")
            saved_index = json.loads(
                (root / "_Embeddings" / "faces_index.json").read_text(encoding="utf-8")
            )
            saved_json = json.loads(manager.info_json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_vectors.shape, (2, 2))
            self.assertEqual(saved_index, {"photo.jpg": [0, 1]})
            self.assertEqual(len(saved_json["photo.jpg"]["removed_faces"]), 1)

    def test_all_removed_faces_publish_empty_embedding_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            trashed = Face(
                [0, 0, 10, 10],
                quality_status="trash",
                face_index=0,
            )
            trashed.embedding_key = "photo.jpg"
            manager.records = {
                "photo.jpg": ImageRecord(
                    "photo.jpg",
                    [trashed],
                    (100, 100),
                    face_count=1,
                )
            }
            _EmbeddingLoaderStub.vectors = np.asarray([[1.0, 2.0]], dtype=np.float32)
            _EmbeddingLoaderStub.index = {"photo.jpg": [0]}

            with mock.patch.object(
                data_manager_module,
                "EmbeddingLoader",
                _EmbeddingLoaderStub,
            ):
                self.assertTrue(manager.save_data(), manager.last_error)

            saved_vectors = np.load(root / "_Embeddings" / "faces_embeddings.npy")
            saved_index = json.loads(
                (root / "_Embeddings" / "faces_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_vectors.shape, (0, 2))
            self.assertEqual(saved_index, {})
            self.assertEqual(json.loads(manager.info_json_path.read_text(encoding="utf-8")), {})

    def test_missing_vector_aborts_without_mutating_model_or_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            original_json = manager.info_json_path.read_bytes()
            face = Face([0, 0, 10, 10], face_index=3)
            face.embedding_key = "photo.jpg::3"
            record = ImageRecord(
                "photo.jpg",
                [face],
                (100, 100),
                face_count=1,
            )
            manager.records = {"photo.jpg": record}
            _EmbeddingLoaderStub.vectors = np.asarray([[1.0, 2.0]], dtype=np.float32)
            _EmbeddingLoaderStub.index = {"other.jpg": [0]}

            with mock.patch.object(
                data_manager_module,
                "EmbeddingLoader",
                _EmbeddingLoaderStub,
            ):
                self.assertFalse(manager.save_data())

            self.assertIs(manager.records["photo.jpg"], record)
            self.assertEqual(face.face_index, 3)
            self.assertEqual(manager.info_json_path.read_bytes(), original_json)


class MatchesBundleTests(unittest.TestCase):
    def test_main_and_derived_json_are_committed_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ClusterDataManager(
                root,
                mode="matches",
                student_list_file=_write_roster(root),
            )
            manager.info_json_path.write_text("{}", encoding="utf-8")
            reference_face = Face(
                [0, 0, 10, 10],
                cluster_label=0,
                student_id="A7K3-S001",
            )
            reference_face.commit_changes()
            matched_face = Face(
                [10, 10, 20, 20],
                student_id="A7K3-S001",
                extra_data={
                    "matched_portrait_cluster_label": 0,
                    "match_distance": 0.123456,
                },
            )
            matched_face.commit_changes()
            manager.records = {
                "portrait.jpg": ImageRecord(
                    "portrait.jpg",
                    [reference_face],
                    (100, 100),
                    face_count=1,
                    image_type="portrait",
                ),
                "group.jpg": ImageRecord(
                    "group.jpg",
                    [matched_face],
                    (100, 100),
                    face_count=1,
                    image_type="group",
                    original_image_type="group",
                ),
            }

            self.assertTrue(manager.save_data(), manager.last_error)

            main_data = json.loads(manager.info_json_path.read_text(encoding="utf-8"))
            matches = json.loads(
                (root / "matches_portrait_to_group.json").read_text(encoding="utf-8")
            )
            errors = json.loads(
                (root / "error_matches.json").read_text(encoding="utf-8")
            )
            self.assertIn("group.jpg", main_data)
            self.assertEqual(matches["0"]["group_photos"][0]["filename"], "group.jpg")
            self.assertEqual(errors["total"], 0)


if __name__ == "__main__":
    unittest.main()
