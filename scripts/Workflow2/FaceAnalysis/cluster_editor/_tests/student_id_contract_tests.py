"""Изолированные проверки student_id без запуска GUI и моделей."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _lib.data_manager import ClusterDataManager
from _lib.data_models import Face, ImageRecord
from _lib.strategies.matches import MatchesModeStrategy
from _lib.student_roster import load_student_roster


def write_roster(directory: Path, students: list[dict] | None = None) -> Path:
    path = directory / "class.list"
    data = {
        "list_id": "A7K3",
        "students": students or [
            {"student_id": "A7K3-S001", "surname": "Иванов", "name": "Иван"},
            {"student_id": "A7K3-S002", "surname": "Петрова", "name": "Анна"},
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def record(filename: str, face: Face, *, group: bool = False) -> ImageRecord:
    return ImageRecord(
        filename=filename,
        faces=[face],
        original_shape=(100, 100),
        image_type="group" if group else "portrait",
        original_image_type="group" if group else "portrait",
        face_count=2 if group else 1,
    )


class StudentRosterTests(unittest.TestCase):
    def test_loads_names_by_student_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            roster = load_student_roster(write_roster(Path(tmp)))
            self.assertEqual(roster.list_id, "A7K3")
            self.assertEqual(roster.name_for("A7K3-S001"), "Иванов Иван")
            self.assertEqual(
                roster.label_for("A7K3-S002"), "Петрова Анна [A7K3-S002]"
            )

    def test_rejects_duplicate_student_id(self):
        students = [
            {"student_id": "A7K3-S001", "surname": "Иванов", "name": "Иван"},
            {"student_id": "A7K3-S001", "surname": "Петров", "name": "Пётр"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "повторяется"):
                load_student_roster(write_roster(Path(tmp), students))

    def test_rejects_mixed_list_id(self):
        students = [
            {"student_id": "B7K3-S001", "surname": "Иванов", "name": "Иван"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "ожидался A7K3"):
                load_student_roster(write_roster(Path(tmp), students))


class FaceModelTests(unittest.TestCase):
    def test_serialization_drops_legacy_names_and_keeps_temp_name(self):
        face = Face.from_dict({
            "bbox": [1, 2, 3, 4],
            "student_id": "A7K3-S001",
            "child_name": "Старое имя",
            "matched_child_name": "Старый match",
            "temp_child_name": "Temp_Cluster_0",
        })
        data = face.to_dict()
        self.assertEqual(data["student_id"], "A7K3-S001")
        self.assertEqual(data["temp_child_name"], "Temp_Cluster_0")
        self.assertNotIn("child_name", data)
        self.assertNotIn("matched_child_name", data)

    def test_rejects_invalid_bbox(self):
        with self.assertRaisesRegex(ValueError, "bbox"):
            Face.from_dict({"bbox": [1, 2, 3]})

    def test_rejects_non_finite_bbox_and_untyped_student_id(self):
        with self.assertRaisesRegex(ValueError, "bbox"):
            Face.from_dict({"bbox": [1, 2, float("nan"), 4]})
        with self.assertRaisesRegex(ValueError, "student_id"):
            Face.from_dict({"bbox": [1, 2, 3, 4], "student_id": []})

    def test_rejects_fractional_face_index(self):
        with self.assertRaisesRegex(ValueError, "face_index"):
            Face.from_dict({"bbox": [1, 2, 3, 4], "face_index": 1.5})

    def test_rejects_filename_path_escape(self):
        payload = {
            "face_count": 1,
            "original_shape": [100, 100],
            "faces": [{"bbox": [1, 2, 3, 4]}],
        }
        for filename in ("../escape.jpg", "folder/escape.jpg", "C:\\escape.jpg"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "имя файла"):
                    ImageRecord.from_dict(filename, payload)

    def test_rejects_mismatched_face_count(self):
        with self.assertRaisesRegex(ValueError, "face_count"):
            ImageRecord.from_dict(
                "bad.jpg",
                {
                    "face_count": 2,
                    "original_shape": [100, 100],
                    "faces": [{"bbox": [1, 2, 3, 4]}],
                },
            )

    def test_rejects_fractional_original_shape(self):
        with self.assertRaisesRegex(ValueError, "original_shape"):
            ImageRecord.from_dict(
                "bad.jpg",
                {
                    "face_count": 1,
                    "original_shape": [100.5, 100],
                    "faces": [{"bbox": [1, 2, 3, 4]}],
                },
            )


class DataManagerTests(unittest.TestCase):
    def test_optional_roster_resolves_student_name_in_location_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ClusterDataManager(
                root,
                mode="location",
                student_list_file=write_roster(root),
            )

            self.assertEqual(manager.student_name("A7K3-S001"), "Иванов Иван")

    def test_location_mode_requires_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaisesRegex(ValueError, "Для всех режимов"):
                ClusterDataManager(root, mode="location")

    def test_location_mode_rejects_missing_explicit_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(FileNotFoundError):
                ClusterDataManager(
                    root,
                    mode="location",
                    student_list_file=root / "missing.list",
                )

    def test_unassigned_portrait_cluster_blocks_final_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ClusterDataManager(root, mode="face", student_list_file=write_roster(root))
            manager.records = {
                "a.jpg": record("a.jpg", Face([0, 0, 1, 1], 0, None))
            }

            self.assertFalse(manager.save_data())
            self.assertIn("не имеет student_id", manager.last_error)
            self.assertFalse((root / "info_faces.json").exists())

    def test_duplicate_student_between_portrait_clusters_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ClusterDataManager(root, mode="face", student_list_file=write_roster(root))
            manager.records = {
                "a.jpg": record("a.jpg", Face([0, 0, 1, 1], 0, "A7K3-S001")),
                "b.jpg": record("b.jpg", Face([0, 0, 1, 1], 1, "A7K3-S001")),
            }
            with self.assertRaisesRegex(ValueError, "назначен кластерам"):
                manager.validate_student_ids()

    def test_available_students_excludes_assigned_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ClusterDataManager(root, mode="face", student_list_file=write_roster(root))
            manager.records = {
                "a.jpg": record("a.jpg", Face([0, 0, 1, 1], 0, "A7K3-S001"))
            }
            self.assertEqual(
                [student.student_id for student in manager.available_students()],
                ["A7K3-S002"],
            )

    def test_standard_save_writes_only_student_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ClusterDataManager(root, mode="face", student_list_file=write_roster(root))
            face = Face.from_dict({
                "bbox": [0, 0, 1, 1],
                "cluster_label": 0,
                "student_id": "A7K3-S001",
                "child_name": "Устаревшее имя",
            })
            manager.records = {"a.jpg": record("a.jpg", face)}
            self.assertTrue(manager.save_data())
            saved = json.loads((root / "info_faces.json").read_text(encoding="utf-8"))
            saved_face = saved["a.jpg"]["faces"][0]
            self.assertEqual(saved_face["student_id"], "A7K3-S001")
            self.assertNotIn("child_name", saved_face)
            self.assertFalse(list(root.glob(".info_faces.json.*.tmp")))


class MatchesTests(unittest.TestCase):
    def test_assignment_and_unassignment_copy_and_clear_student_id(self):
        strategy = MatchesModeStrategy()
        group_face = Face([0, 0, 1, 1])
        records = {"group.jpg": record("group.jpg", group_face, group=True)}
        strategy.move_images(
            "error_matches", "0", ["group.jpg"], records,
            {"group.jpg": 0}, "A7K3-S001"
        )
        self.assertEqual(group_face.student_id, "A7K3-S001")
        self.assertEqual(group_face.extra_data["matched_portrait_cluster_label"], 0)
        strategy.move_images("0", "error_matches", ["group.jpg"], records)
        self.assertIsNone(group_face.student_id)
        self.assertIsNone(group_face.extra_data["matched_portrait_cluster_label"])

    def test_report_uses_student_id_and_rounds_distances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            strategy = MatchesModeStrategy()
            strategy.set_student_roster(roster)
            portrait = Face([0, 0, 1, 1], 0, "A7K3-S001")
            matched = Face([0, 0, 1, 1], student_id="A7K3-S001")
            matched.extra_data.update({
                "matched_portrait_cluster_label": 0,
                "match_distance": 0.15727099315993642,
            })
            records = {
                "portrait.jpg": record("portrait.jpg", portrait),
                "group.jpg": record("group.jpg", matched, group=True),
            }
            outputs = strategy.build_save_outputs(
                records,
                {"json_path": root / "info_faces.json"},
            )
            report = outputs[root / "matches_portrait_to_group.json"]
            self.assertEqual(report["0"]["student_id"], "A7K3-S001")
            self.assertNotIn("child_name", report["0"])
            self.assertEqual(report["0"]["group_photos"][0]["min_distance"], 0.1573)


if __name__ == "__main__":
    unittest.main()
