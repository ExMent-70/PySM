"""Синтетические проверки XMP-контракта student_id."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
FACE_ANALYSIS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for import_path in (SCRIPT_DIR, FACE_ANALYSIS_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from run_create_xmp import (  # noqa: E402
    MetadataProcessor,
    PhotoType,
    run_xmp_creation,
    validate_xmp_tasks,
)
from student_roster import load_student_roster  # noqa: E402


NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
}


def write_roster(root: Path, students: list[dict] | None = None) -> Path:
    path = root / "class.list"
    path.write_text(
        json.dumps(
            {
                "list_id": "A7K3",
                "students": students or [
                    {
                        "student_id": "A7K3-S001",
                        "surname": "Иванов",
                        "name": "Иван",
                    },
                    {
                        "student_id": "A7K3-S002",
                        "surname": "Петрова",
                        "name": "Анна",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def bag_values(root: ET.Element, field: str) -> list[str]:
    return [
        item.text or ""
        for item in root.findall(f".//{field}/rdf:Bag/rdf:li", NS)
    ]


class StudentRosterTests(unittest.TestCase):
    def test_resolves_real_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            roster = load_student_roster(write_roster(Path(tmp)))
            self.assertEqual(roster.name_for("A7K3-S001"), "Иванов Иван")

    def test_rejects_duplicate_and_mixed_ids(self):
        cases = (
            [
                {"student_id": "A7K3-S001", "surname": "А", "name": "А"},
                {"student_id": "A7K3-S001", "surname": "Б", "name": "Б"},
            ],
            [
                {"student_id": "B7K3-S001", "surname": "А", "name": "А"},
            ],
            [
                {"student_id": "not-an-id", "surname": "А", "name": "А"},
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, students in enumerate(cases):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_student_roster(write_roster(root, students))


class XmpIdentityTests(unittest.TestCase):
    def test_portrait_writes_name_and_student_id_subject_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            xmp_path = root / "portrait.xmp"
            processor = MetadataProcessor(None, False, roster)
            self.assertTrue(processor.process_file(
                xmp_path,
                "portrait.jpg",
                {
                    "face_count": 1,
                    "faces": [{
                        "cluster_label": 0,
                        "student_id": "A7K3-S001",
                        "emotion_faceonnx": "Happy",
                    }],
                },
                {},
                PhotoType.PORTRAIT,
                "Session",
                "SCHOOL",
            ))

            xml_root = ET.parse(xmp_path).getroot()
            self.assertIn(
                "PySM_PERSON_Иванов Иван", bag_values(xml_root, "dc:subject")
            )
            subject_codes = bag_values(xml_root, "Iptc4xmpCore:SubjectCode")
            self.assertIn("F0_person:Иванов Иван", subject_codes)
            self.assertIn("F0:A7K3-S001", subject_codes)
            self.assertFalse(any("student_id:" in item for item in subject_codes))
            transmission = xml_root.find(".//photoshop:TransmissionReference", NS)
            self.assertEqual(transmission.text, "Иванов Иван")
            headline = xml_root.find(".//photoshop:Headline", NS)
            self.assertEqual(headline.text, "Session")
            category = xml_root.find(".//photoshop:Category", NS)
            self.assertEqual(category.text, "SCHOOL")

    def test_group_writes_indexed_ids_and_real_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            xmp_path = root / "group.xmp"
            processor = MetadataProcessor(None, False, roster)
            self.assertTrue(processor.process_file(
                xmp_path,
                "group.jpg",
                {
                    "face_count": 2,
                    "faces": [
                        {"student_id": "A7K3-S001", "matched_portrait_cluster_label": 0},
                        {"student_id": "A7K3-S002", "matched_portrait_cluster_label": 1},
                    ],
                },
                {},
                PhotoType.GROUP,
                "Session",
            ))

            xml_root = ET.parse(xmp_path).getroot()
            subject_codes = bag_values(xml_root, "Iptc4xmpCore:SubjectCode")
            self.assertIn("F0:A7K3-S001", subject_codes)
            self.assertIn("F1:A7K3-S002", subject_codes)
            keywords = bag_values(xml_root, "dc:subject")
            self.assertIn("PySM_PERSON_Иванов Иван", keywords)
            self.assertIn("PySM_PERSON_Петрова Анна", keywords)

    def test_second_run_clears_stale_identity_and_keeps_unrelated_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            xmp_path = root / "stale.xmp"
            processor = MetadataProcessor(None, False, roster)
            identified = {
                "face_count": 1,
                "faces": [{"cluster_label": 0, "student_id": "A7K3-S001"}],
            }
            self.assertTrue(processor.process_file(
                xmp_path, "stale.jpg", identified, {}, PhotoType.PORTRAIT, "Session"
            ))

            from _common.xmp_editor import XmpEditor

            editor = XmpEditor(xmp_path)
            editor.set_simple_field("photoshop", "CaptionWriter", "Keep me")
            self.assertTrue(editor.save())

            unidentified_group = {"face_count": 2, "faces": [{"student_id": None}]}
            self.assertTrue(processor.process_file(
                xmp_path, "stale.jpg", unidentified_group, {}, PhotoType.GROUP, "Session"
            ))
            xml_root = ET.parse(xmp_path).getroot()
            self.assertFalse(any(
                item in {"F0:A7K3-S001", "F1:A7K3-S001"}
                for item in bag_values(xml_root, "Iptc4xmpCore:SubjectCode")
            ))
            self.assertFalse(any(
                item.startswith("PySM_PERSON_")
                for item in bag_values(xml_root, "dc:subject")
            ))
            transmission = xml_root.find(".//photoshop:TransmissionReference", NS)
            self.assertTrue(transmission is not None and not transmission.text)
            caption = xml_root.find(".//photoshop:CaptionWriter", NS)
            self.assertEqual(caption.text, "Keep me")


class PreflightTests(unittest.TestCase):
    def test_unknown_id_and_unassigned_identity_cluster_are_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            cases = (
                {"faces": [{"student_id": "A7K3-S999"}]},
                {"faces": [{"cluster_label": 0, "student_id": None}]},
                {"faces": [{"matched_portrait_cluster_label": 0, "student_id": None}]},
            )
            for index, file_data in enumerate(cases):
                tasks = [(root / f"{index}.xmp", f"{index}.jpg", file_data, f"{index}.jpg")]
                with self.subTest(index=index), self.assertRaises(ValueError):
                    validate_xmp_tasks(tasks, roster)

    def test_unmatched_group_face_and_noise_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            tasks = [(
                root / "group.xmp",
                "group.jpg",
                {"faces": [
                    {"student_id": None},
                    {"cluster_label": -1, "student_id": None},
                ]},
                "group.jpg",
            )]
            validate_xmp_tasks(tasks, roster)

    def test_preflight_failure_creates_no_partial_xmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            faces_data = {
                "valid.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 0, "student_id": "A7K3-S001"}],
                },
                "invalid.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 1, "student_id": None}],
                },
            }
            with patch("run_create_xmp.logger.info"), self.assertRaises(ValueError):
                run_xmp_creation(
                    faces_data, {}, root, "Session", 1, None,
                    False, False, False, roster,
                )
            self.assertFalse((root / "valid.xmp").exists())
            self.assertFalse((root / "invalid.xmp").exists())

    def test_scan_mode_validates_only_files_found_in_image_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            (root / "photo_001.jpg").write_bytes(b"")
            faces_data = {
                "IMG_001.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 0, "student_id": "A7K3-S001"}],
                },
                "IMG_999.jpg": {
                    "face_count": 1,
                    "faces": [{"cluster_label": 1, "student_id": None}],
                },
            }
            with patch("run_create_xmp.logger.info"):
                run_xmp_creation(
                    faces_data, {}, root, "Session", 1, None,
                    False, True, False, roster,
                )
            self.assertTrue((root / "photo_001.xmp").exists())
            self.assertFalse((root / "IMG_999.xmp").exists())


if __name__ == "__main__":
    unittest.main()
