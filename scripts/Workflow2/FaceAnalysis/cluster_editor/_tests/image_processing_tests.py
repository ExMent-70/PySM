"""Contracts shared by export and its GUI preview."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _lib.image_processing import create_watermark_layer


class WatermarkTests(unittest.TestCase):
    def test_same_settings_produce_identical_layer(self) -> None:
        settings = {
            "wm_stripe_alpha": 45,
            "wm_mask_fill": 20,
            "wm_pad_w": 0.1,
            "wm_pad_h": 0.2,
            "wm_text": "ПРОСМОТР",
            "wm_text_alpha": 150,
            "wm_seed": 42,
        }
        bboxes = [[40, 30, 90, 100]]

        first = create_watermark_layer((320, 240), bboxes, settings, "Ученик")
        second = create_watermark_layer((320, 240), bboxes, settings, "Ученик")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.tobytes(), second.tobytes())


if __name__ == "__main__":
    unittest.main()
