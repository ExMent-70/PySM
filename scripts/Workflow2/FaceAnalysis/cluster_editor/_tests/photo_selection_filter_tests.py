"""Проверки общего фильтра выбранных фотографий."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _lib.photo_selection_filter import extract_photo_numbers, load_selected_photo_numbers


class PhotoSelectionFilterTests(unittest.TestCase):
    def test_extracts_only_standalone_six_digit_numbers(self):
        self.assertEqual(extract_photo_numbers("IMG_889714.jpg"), {"889714"})
        self.assertEqual(extract_photo_numbers("IMG_1256.jpg"), set())
        self.assertEqual(extract_photo_numbers("IMG_18897142.jpg"), set())

    def test_combines_unique_numbers_from_all_students(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photo_selection.json"
            path.write_text(json.dumps({
                "students": {
                    "A7K3-S001": {"selected_numbers": ["889714", "889715"]},
                    "A7K3-S002": {"selected_numbers": ["889714", "889716"]},
                }
            }), encoding="utf-8")

            self.assertEqual(
                load_selected_photo_numbers(path),
                {"889714", "889715", "889716"},
            )

    def test_missing_file_means_no_filter_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                load_selected_photo_numbers(Path(tmp) / "photo_selection.json")
            )

    def test_rejects_invalid_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photo_selection.json"
            path.write_text(json.dumps({
                "students": {
                    "A7K3-S001": {"selected_numbers": ["714"]},
                }
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "ровно шесть цифр"):
                load_selected_photo_numbers(path)


if __name__ == "__main__":
    unittest.main()
