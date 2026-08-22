"""Синтетические проверки student_id-контракта cluster_face2."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
FACE_ANALYSIS_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
for import_path in (SCRIPT_DIR, FACE_ANALYSIS_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from _lib.analysis_manager import write_json_atomic  # noqa: E402
from _lib.strategies_analysis.matching import (  # noqa: E402
    ClusterProfile,
    MatchingStrategy,
)
from _lib.strategies_analysis.portraits import PortraitsStrategy  # noqa: E402
from _lib.student_ids import (  # noqa: E402
    StudentIdList,
    build_cluster_student_map,
    build_student_ids_order_context_key,
    load_student_ids_order,
    remove_legacy_name_fields,
)


class StudentIdContextTests(unittest.TestCase):
    def test_legacy_name_fields_are_removed_without_touching_other_identity(self) -> None:
        face = {
            "child_name": None,
            "matched_child_name": "Иванов Иван",
            "student_id": "A7K3-S001",
            "temp_child_name": "Temp_Cluster_0",
        }

        remove_legacy_name_fields(face)

        self.assertNotIn("child_name", face)
        self.assertNotIn("matched_child_name", face)
        self.assertEqual("A7K3-S001", face["student_id"])
        self.assertEqual("Temp_Cluster_0", face["temp_child_name"])

    def test_context_key_uses_photo_session(self) -> None:
        self.assertEqual(
            "wf_student_ids_order.Main_ids_order",
            build_student_ids_order_context_key("Main"),
        )

    def test_valid_json_array_preserves_order(self) -> None:
        result = load_student_ids_order(
            ["A7K3-S002", "", "A7K3-S001"],
            "wf_student_ids_order.Main_ids_order",
        )

        self.assertEqual("A7K3", result.list_id)
        self.assertEqual(("A7K3-S002", "A7K3-S001"), result.student_ids)

    def test_invalid_value_reports_array_position(self) -> None:
        with self.assertRaisesRegex(ValueError, "Элемент 2"):
            load_student_ids_order(
                ["A7K3-S001", "Иванов Иван"],
                "wf_student_ids_order.Main_ids_order",
            )

    def test_duplicate_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "повторяется"):
            load_student_ids_order(
                ["A7K3-S001", "A7K3-S001"],
                "wf_student_ids_order.Main_ids_order",
            )

    def test_mixed_list_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ожидался A7K3"):
            load_student_ids_order(
                ["A7K3-S001", "B8M4-S002"],
                "wf_student_ids_order.Main_ids_order",
            )

    def test_empty_array_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "не содержит student_id"):
            load_student_ids_order(
                [],
                "wf_student_ids_order.Main_ids_order",
            )

    def test_non_array_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON-массив"):
            load_student_ids_order(
                {"student_id": "A7K3-S001"},
                "wf_student_ids_order.Main_ids_order",
            )

    def test_portraits_strategy_reads_automatic_context_key(self) -> None:
        class FakeContext:
            @staticmethod
            def get(key, default=None):
                return "Main" if key == "wf_photo_session" else default

            @staticmethod
            def get_structured(key, default=None):
                if key == "wf_student_ids_order.Main_ids_order":
                    return ["A7K3-S001"]
                return default

        with patch(
            "_lib.strategies_analysis.portraits.pysm_context", FakeContext()
        ):
            result = PortraitsStrategy()._load_student_ids_order()

        self.assertEqual("A7K3", result.list_id)
        self.assertEqual(("A7K3-S001",), result.student_ids)


class ClusterMappingTests(unittest.TestCase):
    def test_clusters_are_mapped_by_first_photo_number(self) -> None:
        labels = np.array([8, 3, 8, -1, 3])
        filenames = ["IMG_020.jpg", "IMG_010.jpg", "IMG_021.jpg", "IMG_001.jpg", "IMG_011.jpg"]

        result = build_cluster_student_map(
            labels, filenames, ["A7K3-S001", "A7K3-S002"]
        )

        self.assertEqual({3: "A7K3-S001", 8: "A7K3-S002"}, result)

    def test_cluster_count_mismatch_disables_automatic_assignment(self) -> None:
        cases = (
            (np.array([0, 1]), ["A7K3-S001"]),
            (np.array([0]), ["A7K3-S001", "A7K3-S002"]),
        )
        for labels, student_ids in cases:
            with self.subTest(labels=labels.tolist(), student_ids=student_ids):
                filenames = [f"IMG_{index:03d}.jpg" for index in range(len(labels))]
                result = build_cluster_student_map(labels, filenames, student_ids)
                self.assertEqual({}, result)

    def test_portraits_mode_saves_unassigned_clusters_on_mismatch(self) -> None:
        class FakeBuilder:
            def __init__(self, *args, **kwargs):
                pass

            def add_section(self, *args, **kwargs):
                pass

            def get_html(self):
                return ""

        class FakeContext:
            @staticmethod
            def log_html(_html):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = SimpleNamespace(
                data_dir=root,
                json_data={
                    "IMG_001.jpg": {
                        "face_count": 1,
                        "faces": [{"child_name": "Старое имя"}],
                    },
                    "IMG_002.jpg": {
                        "face_count": 1,
                        "faces": [{"child_name": "Другое имя"}],
                    },
                },
                get_subset_embeddings=lambda _filter: (
                    ["IMG_001.jpg", "IMG_002.jpg"],
                    [0, 1],
                    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                ),
                save_json=lambda: None,
            )
            strategy = PortraitsStrategy()
            strategy._load_student_ids_order = lambda: StudentIdList(
                context_key="wf_student_ids_order.Main_ids_order",
                list_id="A7K3",
                student_ids=("A7K3-S001",),
            )
            config = Namespace(
                a_algorithm="dbscan",
                a_metric="cosine",
                a_sim_threshold=0.1,
                a_clear_min_claster_size=1,
                a_target_dir=str(root),
            )

            with patch(
                "_lib.strategies_analysis.portraits.StandardTreeBuilder", FakeBuilder
            ), patch(
                "_lib.strategies_analysis.portraits.pysm_context", FakeContext()
            ):
                strategy.run(config, manager)

            faces = [item["faces"][0] for item in manager.json_data.values()]
            self.assertEqual({0, 1}, {face["cluster_label"] for face in faces})
            self.assertTrue(all(face["student_id"] is None for face in faces))
            self.assertTrue(all("child_name" not in face for face in faces))


class MatchingContractTests(unittest.TestCase):
    def test_centroids_keep_student_id(self) -> None:
        manager = SimpleNamespace(
            json_data={
                "IMG_001.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 0, "student_id": "A7K3-S001"}],
                },
                "IMG_002.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 0, "student_id": "A7K3-S001"}],
                },
            },
            index_map={"IMG_001.jpg": [0], "IMG_002.jpg": [1]},
            embeddings=np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        )

        profiles = MatchingStrategy()._calculate_centroids(manager)

        self.assertEqual("A7K3-S001", profiles[0].student_id)
        self.assertAlmostEqual(1.0, float(np.linalg.norm(profiles[0].vector)), places=6)

    def test_mixed_ids_inside_cluster_are_rejected(self) -> None:
        manager = SimpleNamespace(
            json_data={
                "IMG_001.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 0, "student_id": "A7K3-S001"}],
                },
                "IMG_002.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 0, "student_id": "A7K3-S002"}],
                },
            },
            index_map={"IMG_001.jpg": [0], "IMG_002.jpg": [1]},
            embeddings=np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "разные student_id"):
            MatchingStrategy()._calculate_centroids(manager)

    def test_same_id_in_two_clusters_is_rejected(self) -> None:
        manager = SimpleNamespace(
            json_data={
                "IMG_001.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 0, "student_id": "A7K3-S001"}],
                },
                "IMG_002.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 1, "student_id": "A7K3-S001"}],
                },
            },
            index_map={"IMG_001.jpg": [0], "IMG_002.jpg": [1]},
            embeddings=np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "назначен портретным кластерам"):
            MatchingStrategy()._calculate_centroids(manager)

    def test_report_keeps_cluster_key_and_rounds_distance(self) -> None:
        manager = SimpleNamespace(
            json_data={
                "GROUP_001.jpg": {
                    "face_count": 2,
                    "faces": [
                        {
                            "matched_portrait_cluster_label": 0,
                            "student_id": "A7K3-S001",
                            "match_distance": 0.15727099315993642,
                        },
                        {
                            "matched_portrait_cluster_label": None,
                            "student_id": None,
                            "match_distance": 0.987654,
                        },
                    ],
                }
            }
        )
        profiles = {
            0: ClusterProfile(0, np.array([1.0, 0.0]), "A7K3-S001")
        }

        matches, errors = MatchingStrategy()._build_reports(manager, profiles)

        self.assertEqual("A7K3-S001", matches["0"]["student_id"])
        self.assertEqual(0.1573, matches["0"]["group_photos"][0]["min_distance"])
        self.assertNotIn("child_name", json.dumps(matches))
        self.assertEqual(
            0.9877,
            errors["unmatched_files"][0]["faces"][0]["nearest_match_distance"],
        )

    def test_report_rejects_inconsistent_student_id(self) -> None:
        manager = SimpleNamespace(
            json_data={
                "GROUP_001.jpg": {
                    "face_count": 2,
                    "faces": [
                        {
                            "matched_portrait_cluster_label": 0,
                            "student_id": "A7K3-S002",
                            "match_distance": 0.2,
                        }
                    ],
                }
            }
        )
        profiles = {
            0: ClusterProfile(0, np.array([1.0, 0.0]), "A7K3-S001")
        }

        with self.assertRaisesRegex(ValueError, "Несогласованные данные"):
            MatchingStrategy()._build_reports(manager, profiles)


class AtomicJsonTests(unittest.TestCase):
    def test_atomic_writer_creates_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.json"
            write_json_atomic(path, {"student_id": "A7K3-S001"})
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual({"student_id": "A7K3-S001"}, loaded)


if __name__ == "__main__":
    unittest.main()
