"""Проверки student_id-контракта select_photo без запуска GUI."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from matches_sync import atomic_write_json, sync_matches_data
from student_identity import (
    build_rename_stem,
    collect_photo_identity,
    format_students_for_table,
    load_student_roster,
    safe_windows_component,
    validate_identity_contract,
)


def write_roster(root: Path, students: list[dict] | None = None) -> Path:
    path = root / "class.list"
    path.write_text(
        json.dumps({
            "list_id": "A7K3",
            "students": students or [
                {"student_id": "A7K3-S001", "surname": "Иванов", "name": "Иван"},
                {"student_id": "A7K3-S002", "surname": "Петрова", "name": "Анна"},
                {"student_id": "A7K3-S003", "surname": "Сидоров", "name": "Пётр"},
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


class RosterAndIdentityTests(unittest.TestCase):
    def test_roster_resolves_name_and_rejects_bad_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            self.assertEqual(roster.name_for("A7K3-S001"), "Иванов Иван")

            bad_cases = (
                [
                    {"student_id": "A7K3-S001", "surname": "А", "name": "А"},
                    {"student_id": "A7K3-S001", "surname": "Б", "name": "Б"},
                ],
                [{"student_id": "B7K3-S001", "surname": "А", "name": "А"}],
                [{"student_id": "invalid", "surname": "А", "name": "А"}],
            )
            for students in bad_cases:
                with self.subTest(students=students), self.assertRaises(ValueError):
                    load_student_roster(write_roster(root, students))

    def test_group_collects_unique_students_in_face_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            roster = load_student_roster(write_roster(Path(tmp)))
            identity = collect_photo_identity([
                {"student_id": "A7K3-S002"},
                {"student_id": None},
                {"student_id": "A7K3-S001"},
                {"student_id": "A7K3-S002"},
            ], roster)
            self.assertEqual(identity.student_ids, ("A7K3-S002", "A7K3-S001"))
            self.assertEqual(identity.student_names, ("Петрова Анна", "Иванов Иван"))
            self.assertEqual(identity.display_students, "Петрова Анна, Иванов Иван")
            self.assertEqual(identity.rename_person_name, "")

    def test_rename_only_single_student_and_sanitizes_windows_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster_path = write_roster(root, [
                {"student_id": "A7K3-S001", "surname": "Ива/нов", "name": "И:ван"}
            ])
            roster = load_student_roster(roster_path)
            single = collect_photo_identity([{"student_id": "A7K3-S001"}], roster)
            multiple_roster = load_student_roster(write_roster(root))
            multiple = collect_photo_identity([
                {"student_id": "A7K3-S001"}, {"student_id": "A7K3-S002"}
            ], multiple_roster)
            self.assertEqual(build_rename_stem(single, "0123", "IMG_0123"), "Ива_нов И_ван-0123")
            self.assertEqual(build_rename_stem(multiple, "0123", "IMG_0123"), "")
            self.assertEqual(safe_windows_component(" name. "), "name")

    def test_students_table_text_is_compact_and_tooltip_is_multiline(self):
        single_text, single_hint = format_students_for_table(["Иванов Иван"])
        self.assertEqual(single_text, "Иванов Иван")
        self.assertIn("• Иванов Иван", single_hint)

        group_text, group_hint = format_students_for_table([
            "Иванов Иван",
            "Петрова Анна",
            "Сидоров Пётр",
        ])
        self.assertEqual(group_text, "Список учеников (3)")
        self.assertEqual(
            group_hint.splitlines(),
            [
                "Распознанные ученики:",
                "• Иванов Иван",
                "• Петрова Анна",
                "• Сидоров Пётр",
            ],
        )


class ValidationTests(unittest.TestCase):
    def test_valid_contract_allows_unused_roster_student(self):
        with tempfile.TemporaryDirectory() as tmp:
            roster = load_student_roster(write_roster(Path(tmp)))
            matches = {
                "0": {"student_id": "A7K3-S001", "group_photos": []},
                "1": {"student_id": "A7K3-S002", "group_photos": []},
            }
            metadata = {
                "portrait.jpg": {
                    "faces": [{"cluster_label": 0, "student_id": "A7K3-S001"}]
                },
                "group.jpg": {"faces": [
                    {
                        "matched_portrait_cluster_label": 1,
                        "student_id": "A7K3-S002",
                    },
                    {"student_id": None},
                ]},
            }
            validate_identity_contract(metadata, matches, roster)

    def test_unknown_missing_and_conflicting_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            roster = load_student_roster(write_roster(Path(tmp)))
            cases = (
                (
                    {"x.jpg": {"faces": [{"student_id": "A7K3-S999"}]}},
                    {},
                ),
                (
                    {"x.jpg": {"faces": [{"cluster_label": 0, "student_id": None}]}},
                    {},
                ),
                (
                    {"x.jpg": {"faces": [{
                        "matched_portrait_cluster_label": 0,
                        "student_id": "A7K3-S002",
                    }]}},
                    {"0": {"student_id": "A7K3-S001", "group_photos": []}},
                ),
                (
                    {},
                    {
                        "0": {"student_id": "A7K3-S001", "group_photos": []},
                        "1": {"student_id": "A7K3-S001", "group_photos": []},
                    },
                ),
                (
                    {
                        "a.jpg": {"faces": [{
                            "cluster_label": 0, "student_id": "A7K3-S001"
                        }]},
                        "b.jpg": {"faces": [{
                            "cluster_label": 1, "student_id": "A7K3-S001"
                        }]},
                    },
                    {},
                ),
            )
            for metadata, matches in cases:
                with self.subTest(metadata=metadata, matches=matches), self.assertRaises(ValueError):
                    validate_identity_contract(metadata, matches, roster)


class MatchesSyncTests(unittest.TestCase):
    def test_group_photo_is_added_to_every_student_present(self):
        matches = {
            "0": {"student_id": "A7K3-S001", "group_photos": []},
            "1": {"student_id": "A7K3-S002", "group_photos": []},
        }
        results = [{
            "number": "0123",
            "status": "Найден",
            "student_ids": ["A7K3-S001", "A7K3-S002"],
            "original_filename": "IMG_0123.jpg",
        }]
        updated = sync_matches_data(matches, results, {"0123"}, 4, 4)
        for record in updated.values():
            self.assertEqual(record["group_photos"][0]["filename"], "IMG_0123.jpg")
            self.assertIn("student_id", record)
            self.assertNotIn("child_name", record)
        self.assertEqual(set(updated), {"0", "1"})

    def test_sync_removes_stale_or_wrong_student_association(self):
        matches = {
            "0": {
                "student_id": "A7K3-S001",
                "group_photos": [{"filename": "IMG_0123.jpg", "min_distance": 0.1}],
            },
        }
        results = [{
            "number": "0123",
            "status": "Найден",
            "student_ids": ["A7K3-S002"],
            "original_filename": "IMG_0123.jpg",
        }]
        updated = sync_matches_data(matches, results, {"0123"}, 4, 4)
        self.assertEqual(updated["0"]["group_photos"], [])

    def test_atomic_writer_creates_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.json"
            atomic_write_json(path, {"0": {"student_id": "A7K3-S001"}})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["0"]["student_id"],
                "A7K3-S001",
            )
            self.assertFalse(list(path.parent.glob(".matches.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
