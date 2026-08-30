"""Регрессии контекста локаций и порядка лиц в панели matches."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_cluster_editor
from run_cluster_editor import MainWindow
from _lib.data_models import Face


def _face(matched_cluster: int | None) -> Face:
    face = Face([0, 0, 10, 10])
    if matched_cluster is not None:
        face.extra_data["matched_portrait_cluster_label"] = matched_cluster
    return face


class LocationContextContractTests(unittest.TestCase):
    def test_location_context_is_saved_with_structured_dot_key(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window.mode = "location"
        window.photo_session = "SCHOOL"
        window.data_manager = SimpleNamespace(
            last_error="",
            get_location_covers_dict=lambda: {"group_photo_01": "IMG_000001.jpg"},
        )
        context = SimpleNamespace(set_structured=mock.Mock(), set=mock.Mock())

        with mock.patch.object(run_cluster_editor, "IS_MANAGED_RUN", True):
            with mock.patch.object(run_cluster_editor, "pysm_context", context):
                self.assertTrue(MainWindow._update_pysm_context(window))

        context.set_structured.assert_called_once_with(
            "sys_location_name.SCHOOL",
            {
                "group_photo_01": "IMG_000001.jpg",
                "portrait_A6": "",
                "portrait_A5": "",
                "portrait_A4": "",
            },
        )
        context.set.assert_not_called()


class FacePanelOrderTests(unittest.TestCase):
    def test_error_matches_cluster_shows_unrecognized_faces_first(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window.mode = "matches"
        window.active_cluster_id = "error_matches"
        faces = [_face(4), _face(None), _face(2), _face(None)]

        ordered = MainWindow._ordered_face_panel_entries(window, faces)

        self.assertEqual([index for index, _face_obj in ordered], [1, 3, 0, 2])

    def test_portrait_cluster_shows_current_cluster_face_first(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window.mode = "matches"
        window.active_cluster_id = "4"
        faces = [_face(1), _face(None), _face(4), _face(4)]

        ordered = MainWindow._ordered_face_panel_entries(window, faces)

        self.assertEqual([index for index, _face_obj in ordered], [2, 3, 0, 1])

    def test_non_matches_mode_keeps_original_face_order(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window.mode = "location"
        window.active_cluster_id = "4"
        faces = [_face(4), _face(None), _face(1)]

        ordered = MainWindow._ordered_face_panel_entries(window, faces)

        self.assertEqual([index for index, _face_obj in ordered], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
