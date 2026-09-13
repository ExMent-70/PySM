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
    extract_digits,
    load_optional_student_roster,
    run_xmp_creation,
    validate_xmp_tasks,
)
from _common.xmp_editor import XmpEditor  # noqa: E402
from _lib.student_roster import load_student_roster  # noqa: E402


NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/",
    "lightroom": "http://ns.adobe.com/lightroom/1.0/",
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
    "GettyImagesGIFT": "http://xmp.gettyimages.com/gift/1.0/",
}
LEGACY_GETTY_URI = "http://ns.gettyimages.com/gift/1.0/"


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

    def test_missing_optional_roster_logs_warning_and_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing.list"
            cases = (
                (None, "не указан"),
                (str(missing_path), "не найден"),
            )
            for raw_path, message_fragment in cases:
                with self.subTest(raw_path=raw_path), patch(
                    "run_create_xmp.logger.warning"
                ) as warning:
                    self.assertIsNone(load_optional_student_roster(raw_path))
                    self.assertIn(message_fragment, warning.call_args.args[0])


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
                        "pose": [1, 2, 3],
                        "temp_child_name": "Temp_Cluster_0",
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
            for field in ("dc:subject", "lightroom:hierarchicalSubject"):
                self.assertEqual(
                    bag_values(xml_root, field).count("PySM_Temp_Cluster_0"),
                    1,
                )
            subject_codes = bag_values(xml_root, "Iptc4xmpCore:SubjectCode")
            self.assertIn("F0_person:Иванов Иван", subject_codes)
            self.assertIn("F0:A7K3-S001", subject_codes)
            self.assertIn(
                "F0_temp_child_name:Temp_Cluster_0", subject_codes
            )
            self.assertFalse(any("student_id:" in item for item in subject_codes))
            transmission = xml_root.find(".//photoshop:TransmissionReference", NS)
            self.assertEqual(transmission.text, "Иванов Иван")
            headline = xml_root.find(".//photoshop:Headline", NS)
            self.assertEqual(headline.text, "Session")
            category = xml_root.find(".//photoshop:Category", NS)
            self.assertEqual(category.text, "SCHOOL")
            original_filename = xml_root.find(
                ".//GettyImagesGIFT:OriginalFilename", NS
            )
            self.assertEqual(original_filename.text, "portrait.jpg")
            usage_terms = xml_root.find(".//xmpRights:UsageTerms", NS)
            usage_value = usage_terms.find("./rdf:Alt/rdf:li", NS)
            self.assertEqual(usage_value.text, "1.000,2.000,3.000")
            self.assertEqual(
                usage_value.get("{http://www.w3.org/XML/1998/namespace}lang"),
                "x-default",
            )
            self.assertFalse((usage_terms.text or "").strip())
            self.assertNotIn(LEGACY_GETTY_URI, xmp_path.read_text(encoding="utf-8"))

    def test_without_roster_omits_names_but_keeps_id_and_temp_keyword(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xmp_path = root / "portrait.xmp"
            processor = MetadataProcessor(None, False, None)
            self.assertTrue(processor.process_file(
                xmp_path,
                "portrait.jpg",
                {
                    "face_count": 1,
                    "faces": [{
                        "cluster_label": 0,
                        "student_id": "A7K3-S001",
                        "temp_child_name": "Temp_Cluster_0",
                    }],
                },
                {},
                PhotoType.PORTRAIT,
                "Session",
            ))

            xml_root = ET.parse(xmp_path).getroot()
            for field in ("dc:subject", "lightroom:hierarchicalSubject"):
                keywords = bag_values(xml_root, field)
                self.assertIn("PySM_Temp_Cluster_0", keywords)
                self.assertFalse(
                    any(keyword.startswith("PySM_PERSON_") for keyword in keywords)
                )
            subject_codes = bag_values(xml_root, "Iptc4xmpCore:SubjectCode")
            self.assertIn("F0:A7K3-S001", subject_codes)
            self.assertFalse(any("_person:" in code for code in subject_codes))
            transmission = xml_root.find(".//photoshop:TransmissionReference", NS)
            self.assertTrue(transmission is not None and not transmission.text)

    def test_existing_legacy_getty_namespace_is_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            xmp_path = root / "legacy.xmp"
            xmp_path.write_text(
                f'''<x:xmpmeta xmlns:x="adobe:ns:meta/"
 xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:getty="{LEGACY_GETTY_URI}">
<rdf:RDF><rdf:Description rdf:about="">
<getty:OriginalFilename>old.jpg</getty:OriginalFilename>
</rdf:Description></rdf:RDF></x:xmpmeta>''',
                encoding="utf-8",
            )

            processor = MetadataProcessor(None, False, roster)
            self.assertTrue(processor.process_file(
                xmp_path,
                "new.jpg",
                {"face_count": 0, "faces": []},
                {},
                PhotoType.GROUP,
                "Session",
                "SCHOOL",
            ))

            xml_root = ET.parse(xmp_path).getroot()
            original_filename = xml_root.find(
                ".//GettyImagesGIFT:OriginalFilename", NS
            )
            self.assertEqual(original_filename.text, "new.jpg")
            self.assertIsNone(
                xml_root.find(f".//{{{LEGACY_GETTY_URI}}}OriginalFilename")
            )

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
    def test_extract_digits_uses_last_group_from_filename_stem(self):
        self.assertEqual(extract_digits("TEST3_270363.jpg"), "270363")
        self.assertEqual(extract_digits("1-TEST3_270363.xmp"), "270363")
        self.assertEqual(extract_digits("without_digits.jpg"), "")

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

    def test_without_roster_only_student_id_format_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_tasks = [(
                root / "valid.xmp",
                "valid.jpg",
                {"faces": [{"cluster_label": 0, "student_id": "B8M4-S123"}]},
                "valid.jpg",
            )]
            validate_xmp_tasks(valid_tasks, None)

            invalid_tasks = [(
                root / "invalid.xmp",
                "invalid.jpg",
                {"faces": [{"cluster_label": 0, "student_id": "bad-id"}]},
                "invalid.jpg",
            )]
            with self.assertRaisesRegex(ValueError, "должен иметь формат"):
                validate_xmp_tasks(invalid_tasks, None)

    def test_full_creation_without_roster_does_not_access_list_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            faces_data = {
                "portrait.jpg": {
                    "face_count": 1,
                    "faces": [{
                        "cluster_label": 0,
                        "student_id": "A7K3-S001",
                    }],
                }
            }

            with patch("run_create_xmp.logger.info") as info:
                run_xmp_creation(
                    faces_data, {}, root, "Session", 1, None,
                    False, False, False, None,
                )

            self.assertTrue((root / "portrait.xmp").exists())
            self.assertTrue(any(
                "список учеников не используется" in call.args[0]
                for call in info.call_args_list
            ))

    def test_duplicate_output_path_is_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            output = root / "same.xmp"
            tasks = [
                (output, "same.jpg", {"faces": []}, "same.jpg"),
                (output, "same.png", {"faces": []}, "same.png"),
            ]

            with self.assertRaisesRegex(ValueError, "назначен нескольким"):
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

    def test_scan_mode_matches_frame_number_after_numeric_session_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            for filename in ("TEST3_270363.jpg", "TEST3_270364.jpg"):
                (root / filename).write_bytes(b"")
            faces_data = {
                "IMG_270363.jpg": {
                    "face_count": 0,
                    "faces": [],
                    "location_name": "first_location",
                },
                "IMG_270364.jpg": {
                    "face_count": 0,
                    "faces": [],
                    "location_name": "second_location",
                },
            }

            with patch("run_create_xmp.logger.info"):
                run_xmp_creation(
                    faces_data, {}, root, "Session", 1, None,
                    False, True, False, roster,
                )

            first = ET.parse(root / "TEST3_270363.xmp").getroot()
            second = ET.parse(root / "TEST3_270364.xmp").getroot()
            self.assertEqual(
                first.find(".//Iptc4xmpCore:Location", NS).text,
                "first_location",
            )
            self.assertEqual(
                second.find(".//Iptc4xmpCore:Location", NS).text,
                "second_location",
            )

    def test_scan_mode_rejects_ambiguous_json_frame_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            image_path = root / "TEST3_270363.jpg"
            image_path.write_bytes(b"")
            faces_data = {
                "IMG_270363.jpg": {"face_count": 0, "faces": []},
                "COPY_270363.png": {"face_count": 0, "faces": []},
            }

            with patch("run_create_xmp.logger.info"), self.assertRaisesRegex(
                ValueError, "Неоднозначный номер кадра"
            ):
                run_xmp_creation(
                    faces_data, {}, root, "Session", 1, None,
                    False, True, False, roster,
                )
            self.assertFalse((root / "TEST3_270363.xmp").exists())

    def test_scan_mode_ignores_xmp_and_backup_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            backup_dir = root / "backup"
            xmp_dir = root / "XMP"
            backup_dir.mkdir()
            xmp_dir.mkdir()
            (root / "TEST3_270363.jpg").write_bytes(b"")
            (backup_dir / "TEST3_270363.jpg").write_bytes(b"")
            (xmp_dir / "TEST3_270364.jpg").write_bytes(b"")
            faces_data = {
                "IMG_270363.jpg": {"face_count": 0, "faces": []},
                "IMG_270364.jpg": {"face_count": 0, "faces": []},
            }

            with patch("run_create_xmp.logger.info"):
                run_xmp_creation(
                    faces_data, {}, root, "Session", 1, None,
                    False, True, True, roster,
                )

            self.assertTrue((xmp_dir / "TEST3_270363.xmp").exists())
            self.assertFalse((backup_dir / "XMP").exists())
            self.assertFalse((xmp_dir / "XMP").exists())

    def test_worker_failure_is_reported_as_a_failed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = load_student_roster(write_roster(root))
            faces_data = {"photo.jpg": {"face_count": 0, "faces": []}}

            with (
                patch("run_create_xmp.logger.info"),
                patch.object(MetadataProcessor, "process_file", return_value=False),
                self.assertRaisesRegex(RuntimeError, "Ошибок при сохранении"),
            ):
                run_xmp_creation(
                    faces_data, {}, root, "Session", 1, None,
                    False, False, False, roster,
                )


class XmpEditorSafetyTests(unittest.TestCase):
    def test_failed_replace_preserves_existing_xmp_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xmp_path = root / "existing.xmp"
            editor = XmpEditor(xmp_path)
            editor.set_simple_field("photoshop", "Headline", "before")
            self.assertTrue(editor.save())
            original = xmp_path.read_bytes()

            editor = XmpEditor(xmp_path)
            editor.set_simple_field("photoshop", "Headline", "after")
            with patch(
                "_common.xmp_editor.os.replace",
                side_effect=OSError("synthetic failure"),
            ):
                self.assertFalse(editor.save())

            self.assertEqual(xmp_path.read_bytes(), original)
            self.assertEqual(list(root.glob("*.xmp~")), [])


if __name__ == "__main__":
    unittest.main()
