"""Location repair tests use synthetic files, never real photo sessions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from photo_selection2._common import location_service as locations


class LocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dest = self.root / "dest"
        self.dest.mkdir()
        self.info = self.root / "info_faces.json"
        self.metadata = {
            "IMG_270015.jpg": {"filename": "IMG_270015.jpg", "location_name": "new"},
            "IMG_270020.jpg": {"filename": "IMG_270020.jpg", "location_name": "other"},
        }
        self.write_info()

    def write_info(self):
        self.info.write_text(json.dumps(self.metadata), encoding="utf-8")

    def file(self, name, data=b"retouched layered PSD"):
        path = self.dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def plan(self):
        return locations.plan_locations(self.dest, self.info, ("Masks",))

    def test_move_preserves_nested_path_prefix_names_bytes_and_is_idempotent(self):
        names = ["old/nested/ph_IMG_270015База.psd", "IMG_270020Донор.jpg",
                 "old/CaptureOne/IMG_270015.cr3.[cc5f63dc-cd60-416e-9c7f-271955d3cadf].cot"]
        sources = [self.file(name) for name in names]
        before = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in sources}
        plan = self.plan()
        self.assertEqual(len(plan.executable), 3)
        summary = locations.repair_locations(plan)
        self.assertFalse(summary.issues)
        self.assertEqual(len(summary.moved), 3)
        self.assertFalse(any(p.exists() for p in sources))
        self.assertTrue((self.dest / "new/nested/ph_IMG_270015База.psd").is_file())
        self.assertTrue((self.dest / "other/IMG_270020Донор.jpg").is_file())
        self.assertTrue((self.dest / "old").is_dir())
        after = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in self.dest.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(self.plan().moves)
        self.assertFalse(locations.repair_locations(self.plan()).moved)

    def test_same_number_location_variants_merge_but_different_locations_block(self):
        self.metadata["variant"] = {"filename": "PH_IMG_270015База.jpg", "location_name": "NEW"}
        self.write_info()
        self.file("old/IMG_270015.psd")
        self.assertEqual(len(self.plan().executable), 1)
        self.metadata["variant"]["location_name"] = "another"
        self.write_info()
        self.assertFalse(self.plan().executable)
        self.assertTrue(any(i.code == "duplicate_number" for i in self.plan().issues))

    def test_all_duplicate_targets_are_conflicts_but_free_moves_continue(self):
        a = self.file("old1/IMG_270015.psd", b"first")
        b = self.file("old2/IMG_270015.psd", b"second")
        c = self.file("old2/IMG_270020.psd")
        plan = self.plan()
        self.assertEqual(len(plan.executable), 1)
        self.assertEqual(sum(bool(m.conflict) for m in plan.moves), 2)
        summary = locations.repair_locations(plan)
        self.assertEqual(len(summary.moved), 1)
        self.assertEqual(a.read_bytes(), b"first")
        self.assertEqual(b.read_bytes(), b"second")
        self.assertFalse(c.exists())

    def test_occupied_target_even_identical_content_is_never_replaced(self):
        source = self.file("old/IMG_270015.psd")
        target = self.file("new/IMG_270015.psd")
        self.assertFalse(self.plan().executable)
        locations.repair_locations(self.plan())
        self.assertEqual(source.read_bytes(), target.read_bytes())

    def test_merge_then_split_locations(self):
        self.file("first/IMG_270015.psd", b"first retouch")
        self.file("second/IMG_270020.psd", b"second retouch")
        self.metadata["IMG_270020.jpg"]["location_name"] = "new"
        self.write_info()
        self.assertEqual(len(locations.repair_locations(self.plan()).moved), 2)
        self.assertTrue((self.dest / "new/IMG_270020.psd").exists())
        self.metadata["IMG_270020.jpg"]["location_name"] = "split"
        self.write_info()
        self.assertEqual(len(locations.repair_locations(self.plan()).moved), 1)
        self.assertEqual((self.dest / "new/IMG_270015.psd").read_bytes(), b"first retouch")
        self.assertEqual((self.dest / "split/IMG_270020.psd").read_bytes(), b"second retouch")

    def test_target_created_between_check_and_rename_is_preserved(self):
        source = self.file("old/IMG_270015.psd")
        plan = self.plan()
        target = plan.moves[0].target
        rename = os.rename

        def race(src, dst):
            target.write_bytes(b"another editor")
            rename(src, dst)

        with patch.object(locations.os, "rename", side_effect=race):
            result = locations.repair_locations(plan)
        self.assertTrue(result.issues)
        self.assertFalse(result.moved)
        self.assertEqual(target.read_bytes(), b"another editor")
        self.assertTrue(source.is_file())

    def test_changed_file_metadata_or_new_file_invalidates_displayed_plan(self):
        source = self.file("old/IMG_270015.psd")
        plan = self.plan()
        source.write_bytes(b"new retouch")
        self.assertFalse(locations.repair_locations(plan).moved)
        plan = self.plan()
        self.metadata["IMG_270015.jpg"]["location_name"] = "newer"
        self.write_info()
        self.assertFalse(locations.repair_locations(plan).moved)
        plan = self.plan()
        self.file("other/IMG_270020.jpg")
        self.assertFalse(locations.repair_locations(plan).moved)
        self.assertTrue(source.is_file())

    def test_changed_analysis_mid_batch_stops_remaining_moves(self):
        self.file("old/IMG_270015.psd")
        self.file("old/IMG_270020.psd")

        def progress(done, total):
            self.metadata["IMG_270020.jpg"]["location_name"] = "changed"
            self.write_info()

        result = locations.repair_locations(self.plan(), progress=progress)
        self.assertEqual(len(result.moved), 1)
        self.assertTrue(result.issues)
        self.assertTrue((self.dest / "old/IMG_270020.psd").is_file())

    def test_locked_file_does_not_undo_successful_moves_and_can_be_retried(self):
        self.file("old/IMG_270015.psd")
        locked = self.file("old/IMG_270020.psd")
        rename = os.rename

        def partial(src, dst):
            if src == locked:
                raise PermissionError("locked by editor")
            rename(src, dst)

        with patch.object(locations.os, "rename", side_effect=partial):
            result = locations.repair_locations(self.plan())
        self.assertEqual(len(result.moved), 1)
        self.assertTrue(result.issues)
        self.assertEqual(len(locations.repair_locations(self.plan()).moved), 1)

    def test_open_windows_file_is_preserved_until_handle_is_closed(self):
        source = self.file("old/IMG_270015.psd")
        with source.open("rb") as handle:
            result = locations.repair_locations(self.plan())
            self.assertFalse(result.moved)
            self.assertTrue(result.issues)
            self.assertEqual(handle.read(), b"retouched layered PSD")
            self.assertTrue(source.exists())
        self.assertEqual(len(locations.repair_locations(self.plan()).moved), 1)

    def test_file_edited_between_moves_is_not_relocated(self):
        self.file("old/IMG_270015.psd")
        second = self.file("old/IMG_270020.psd")

        def edit_after_first_move(done, total):
            if done == 1:
                second.write_bytes(b"new retouch during repair")

        result = locations.repair_locations(self.plan(), progress=edit_after_first_move)
        self.assertEqual(len(result.moved), 1)
        self.assertTrue(result.issues)
        self.assertEqual(second.read_bytes(), b"new retouch during repair")
        self.assertFalse((self.dest / "other/IMG_270020.psd").exists())

    def test_unknown_numbers_and_excluded_folders_remain_unchanged(self):
        unknown = self.file("old/IMG_999999.psd")
        excluded = self.file("Masks/IMG_270015.psd")
        no_number = self.file("notes.txt")
        plan = self.plan()
        self.assertFalse(plan.moves)
        self.assertTrue(any(i.code == "location_unknown_number" for i in plan.issues))
        locations.repair_locations(plan)
        self.assertTrue(all(p.is_file() for p in (unknown, excluded, no_number)))

    def test_ambiguous_number_and_unsafe_location_block_repair(self):
        self.file("old/IMG_270015_270020.psd")
        self.file("old/IMG_270015.psd")
        self.assertFalse(self.plan().executable)
        self.metadata["IMG_270015.jpg"]["location_name"] = "../escape"
        self.write_info()
        self.assertFalse(self.plan().executable)

    def test_empty_location_uses_unknown_and_case_only_folder_is_valid(self):
        self.metadata["IMG_270020.jpg"]["location_name"] = ""
        self.write_info()
        self.file("NEW/IMG_270015.psd")
        self.file("old/IMG_270020.psd")
        plan = self.plan()
        self.assertEqual(len(plan.moves), 1)
        self.assertEqual(plan.moves[0].target.parent.name, "unknown")

    def test_missing_destination_is_reported_and_not_created(self):
        missing = self.root / "missing"
        plan = locations.plan_locations(missing, self.info)
        self.assertTrue(plan.issues)
        locations.repair_locations(plan)
        self.assertFalse(missing.exists())

    def test_reparse_folder_is_not_traversed(self):
        outside = self.root / "outside"
        outside.mkdir()
        photo = outside / "IMG_270015.psd"
        photo.write_bytes(b"outside destination")
        folder = self.dest / "linked"
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(folder), str(outside)],
                                capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.addCleanup(folder.rmdir)
        plan = self.plan()
        self.assertTrue(any(i.code == "location_scan_error" for i in plan.issues))
        self.assertFalse(plan.executable)
        self.assertFalse(plan.inventory)
        locations.repair_locations(plan)
        self.assertEqual(photo.read_bytes(), b"outside destination")

    def test_parent_file_is_a_conflict(self):
        self.file("old/IMG_270015.psd")
        self.file("new", b"not a directory")
        self.assertFalse(self.plan().executable)
        self.assertTrue(self.plan().moves[0].conflict)

    def test_occupied_nested_target_does_not_block_an_independent_move(self):
        self.file("old/nested/IMG_270015.psd")
        self.file("new/nested/IMG_270015.psd")
        self.file("old/IMG_270020.psd")
        plan = self.plan()
        self.assertEqual(sum(bool(m.conflict) for m in plan.moves), 1)
        self.assertEqual(len(plan.executable), 1)

    def test_client_change_stops_batch_and_keeps_remaining_source(self):
        self.file("old/IMG_270015.psd")
        second = self.file("old/IMG_270020.psd")
        current = [True]
        result = locations.repair_locations(
            self.plan(), inputs_current=lambda: current[0],
            progress=lambda done, total: current.__setitem__(0, False),
        )
        self.assertEqual(len(result.moved), 1)
        self.assertTrue(second.exists())
        self.assertTrue(result.issues)


if __name__ == "__main__":
    unittest.main()
