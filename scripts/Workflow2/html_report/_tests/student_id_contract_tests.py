"""Синтетические проверки student_id-контракта HTML-отчёта."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_data import (
    load_optional_json,
    load_required_json,
    prepare_matches,
    prepare_portrait_clusters,
)
from run_html_report import ReportGenerator
from student_roster import load_student_roster


def write_roster(root: Path, students: list[dict] | None = None) -> Path:
    path = root / "class.list"
    path.write_text(
        json.dumps(
            {
                "list_id": "A7K3",
                "students": students
                or [
                    {
                        "student_id": "A7K3-S001",
                        "surname": "Иванов",
                        "name": "Иван",
                    },
                    {
                        "student_id": "A7K3-S002",
                        "surname": "О'Коннор",
                        "name": 'Анна "Аня"',
                    },
                    {
                        "student_id": "A7K3-S003",
                        "surname": "Сидоров",
                        "name": "Пётр",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def face_info(filename: str, face: dict) -> dict:
    return {"filename": filename, "rel_path": f"JPG/{filename}"}


def portrait_data(*faces: tuple[str, int, str | None]) -> dict:
    return {
        filename: {
            "face_count": 1,
            "faces": [{"cluster_label": label, "student_id": student_id}],
        }
        for filename, label, student_id in faces
    }


class RosterAndJsonTests(unittest.TestCase):
    def test_roster_resolves_names_and_rejects_invalid_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            self.assertEqual(roster.name_for("A7K3-S001"), "Иванов Иван")

            invalid_cases = (
                [
                    {"student_id": "A7K3-S001", "surname": "А", "name": "А"},
                    {"student_id": "A7K3-S001", "surname": "Б", "name": "Б"},
                ],
                [{"student_id": "B7K3-S001", "surname": "А", "name": "А"}],
                [{"student_id": "invalid", "surname": "А", "name": "А"}],
                [{"student_id": "A7K3-S001", "surname": "", "name": "А"}],
            )
            for students in invalid_cases:
                with self.subTest(students=students), self.assertRaises(ValueError):
                    load_student_roster(write_roster(root, students))

    def test_required_and_optional_json_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"
            self.assertEqual(load_optional_json(missing), {})
            with self.assertRaises(FileNotFoundError):
                load_required_json(missing)
            broken = root / "broken.json"
            broken.write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_optional_json(broken)


class PortraitTests(unittest.TestCase):
    def test_cluster_gets_student_id_and_name_from_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            roster = load_student_roster(write_roster(Path(tmp)))
            data = portrait_data(
                ("b.jpg", 0, "A7K3-S001"),
                ("a.jpg", 0, "A7K3-S001"),
                ("noise.jpg", -1, None),
            )
            clusters, used = prepare_portrait_clusters(data, roster, face_info)
            self.assertEqual(clusters["0"]["display_name"], "Иванов Иван")
            self.assertEqual(clusters["0"]["student_id"], "A7K3-S001")
            self.assertEqual(
                [item["filename"] for item in clusters["0"]["files"]],
                ["a.jpg", "b.jpg"],
            )
            self.assertIsNone(clusters["-1"]["student_id"])
            self.assertEqual(used, {"A7K3-S001"})

    def test_missing_unknown_mixed_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            roster = load_student_roster(write_roster(Path(tmp)))
            cases = (
                portrait_data(("a.jpg", 0, None)),
                portrait_data(("a.jpg", 0, "A7K3-S999")),
                portrait_data(
                    ("a.jpg", 0, "A7K3-S001"),
                    ("b.jpg", 0, "A7K3-S002"),
                ),
                portrait_data(
                    ("a.jpg", 0, "A7K3-S001"),
                    ("b.jpg", 1, "A7K3-S001"),
                ),
            )
            for data in cases:
                with self.subTest(data=data), self.assertRaises(ValueError):
                    prepare_portrait_clusters(data, roster, face_info)


class MatchesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.roster = load_student_roster(write_roster(self.root))
        self.ref_data = portrait_data(
            ("portrait0.jpg", 0, "A7K3-S001"),
            ("portrait1.jpg", 1, "A7K3-S002"),
        )
        self.clusters, _ = prepare_portrait_clusters(
            self.ref_data, self.roster, face_info
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_matches_resolve_names_and_do_not_mutate_source(self):
        matches = {
            "0": {
                "student_id": "A7K3-S001",
                "group_photos": [
                    {"filename": "group.jpg", "min_distance": 0.123456, "num_faces": 2}
                ],
            }
        }
        target = {
            "group.jpg": {
                "face_count": 2,
                "faces": [
                    {
                        "matched_portrait_cluster_label": 0,
                        "student_id": "A7K3-S001",
                    }
                ],
            }
        }
        original = deepcopy(matches)
        prepared = prepare_matches(
            matches, target, self.clusters, self.roster, lambda name: f"JPG/{name}"
        )
        self.assertEqual(matches, original)
        self.assertEqual(prepared["0"]["display_name"], "Иванов Иван")
        self.assertEqual(prepared["0"]["student_id"], "A7K3-S001")
        self.assertEqual(prepared["0"]["group_photos"][0]["confidence"], 0.123456)

    def test_empty_photos_are_valid_but_not_rendered(self):
        prepared = prepare_matches(
            {"0": {"student_id": "A7K3-S001", "group_photos": []}},
            {},
            self.clusters,
            self.roster,
            str,
        )
        self.assertEqual(prepared, {})

    def test_invalid_matches_and_target_links_are_rejected(self):
        cases = (
            (
                {"0": {"student_id": "A7K3-S002", "group_photos": []}},
                {},
            ),
            (
                {"2": {"student_id": "A7K3-S001", "group_photos": []}},
                {},
            ),
            (
                {
                    "0": {"student_id": "A7K3-S001", "group_photos": []},
                    "1": {"student_id": "A7K3-S001", "group_photos": []},
                },
                {},
            ),
            (
                {"0": {"student_id": "A7K3-S001", "group_photos": [{}]}},
                {},
            ),
            (
                {
                    "0": {
                        "student_id": "A7K3-S001",
                        "group_photos": [{"filename": "missing.jpg"}],
                    }
                },
                {},
            ),
            (
                {"0": {"student_id": "A7K3-S001", "group_photos": []}},
                {
                    "group.jpg": {
                        "faces": [
                            {
                                "matched_portrait_cluster_label": 0,
                                "student_id": "A7K3-S002",
                            }
                        ]
                    }
                },
            ),
        )
        for matches, target in cases:
            with self.subTest(matches=matches, target=target), self.assertRaises(ValueError):
                prepare_matches(matches, target, self.clusters, self.roster, str)


class RenderingTests(unittest.TestCase):
    def test_report_contains_identity_fields_and_escaped_javascript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "Analysis_Test"
            (target / "JPG").mkdir(parents=True)
            roster = load_student_roster(write_roster(root))
            data = portrait_data(("portrait.jpg", 0, "A7K3-S002"))
            (target / "info_faces.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            matches = {
                "0": {
                    "student_id": "A7K3-S002",
                    "group_photos": [{"filename": "group.jpg", "min_distance": 0.1}],
                }
            }
            data["group.jpg"] = {
                "face_count": 2,
                "faces": [
                    {
                        "matched_portrait_cluster_label": 0,
                        "student_id": "A7K3-S002",
                    }
                ],
            }
            (target / "info_faces.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            (target / "matches_portrait_to_group.json").write_text(
                json.dumps(matches, ensure_ascii=False), encoding="utf-8"
            )

            report_path = ReportGenerator(target, target, roster).run()
            html = report_path.read_text(encoding="utf-8")
            self.assertIn("О&#39;Коннор Анна &#34;Аня&#34;", html)
            self.assertIn("Student ID: A7K3-S002", html)
            self.assertIn("Cluster ID: 0", html)
            self.assertIn("\\u0027", html)
            self.assertNotIn("child_name", html)
            self.assertFalse(list(target.glob(".face_clustering_report.html.*.tmp")))

    def test_cross_session_uses_reference_portraits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref"
            target = root / "target"
            ref.mkdir()
            target.mkdir()
            roster = load_student_roster(write_roster(root))
            (ref / "info_faces.json").write_text(
                json.dumps(portrait_data(("portrait.jpg", 0, "A7K3-S001"))),
                encoding="utf-8",
            )
            (target / "info_faces.json").write_text("{}", encoding="utf-8")
            context = ReportGenerator(target, ref, roster)._prepare_data()
            self.assertTrue(context["is_cross_session"])
            self.assertEqual(context["portrait_clusters"]["0"]["student_id"], "A7K3-S001")
            self.assertEqual(context["summary"]["unused_students"], 2)


if __name__ == "__main__":
    unittest.main()
