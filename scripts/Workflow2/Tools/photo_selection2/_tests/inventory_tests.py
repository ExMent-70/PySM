"""Read-only layout publication and placement policies on temporary files."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from photo_selection2._common import assignment_core as core
from photo_selection2._common import destination_inventory as inventory
from photo_selection2._common.copy_core import build_copy_selection
from photo_selection2._common.copy_service import copy_selected_files
from photo_selection2._common.config import make_parser


class FlatInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.dest = self.root / "dest"
        self.analysis = self.root / "analysis"
        for path in (self.source, self.dest, self.analysis):
            path.mkdir()
        self.roster = self.root / "class.list"
        self.write(self.roster, {"list_id": "94V3", "students": [
            {"student_id": "94V3-S001", "surname": "One", "name": "First"},
            {"student_id": "94V3-S002", "surname": "Two", "name": "Second"},
        ]})
        self.info = {"IMG_270001.jpg": {"filename": "IMG_270001.jpg", "location_name": "portrait", "faces": []},
                     "IMG_270002.jpg": {"filename": "IMG_270002.jpg", "location_name": "group_photo_all", "faces": [
                         {"student_id": "94V3-S001"}, {"student_id": "94V3-S002"}]}}
        self.info_path = self.analysis / "info_faces.json"
        self.write(self.info_path, self.info)
        self.write(self.analysis / "photo_selection.json", {"schema_version": 1, "list_id": "94V3",
                   "session_name": "test", "photo_session": "SCHOOL", "students": {
                       "94V3-S001": {"selected_numbers": ["270001"]}}})

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def build(self):
        return core.build_assignments(student_list_file=self.roster, analysis_dir=self.analysis,
                                      source_dir=self.source, dest_dir=self.dest)

    def test_flat_plan_assigns_ph_to_every_recognized_person(self):
        (self.dest / "IMG_270001.psd").write_bytes(b"retouched portrait")
        (self.dest / "PH_IMG_270002.jpg").write_bytes(b"group")
        result = self.build()
        self.assertFalse(result.has_errors)
        core.save_assignments(self.analysis / "photo_assignments.json", result)
        self.assertEqual(result.assignment_payload(), {"schema_version": 2, "list_id": "94V3", "assignments": {
            "94V3-S001": {"portrait": ["270001"], "group_photo_all": ["270002"]},
            "94V3-S002": {"group_photo_all": ["270002"]}}})

    def test_location_change_changes_json_without_moving_or_editing_image(self):
        photo = self.dest / "IMG_270001.psd"
        photo.write_bytes(b"retouched")
        before = photo.stat()
        first = self.build().assignment_payload()
        self.info["IMG_270001.jpg"]["location_name"] = "new"
        self.write(self.info_path, self.info)
        second = self.build().assignment_payload()
        self.assertNotEqual(first, second)
        self.assertEqual(second["assignments"]["94V3-S001"], {"new": ["270001"]})
        self.assertEqual(photo.read_bytes(), b"retouched")
        self.assertEqual(photo.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(list(self.dest.iterdir()), [photo])

    def test_changed_and_added_files_invalidate_snapshot(self):
        photo = self.dest / "IMG_270001.jpg"
        photo.write_bytes(b"original")
        for change in (lambda: photo.write_bytes(b"edited"),
                       lambda: (self.dest / "new.jpg").write_bytes(b"new")):
            snap = inventory.scan_destination(self.dest, self.info_path)
            change()
            with self.assertRaises(ValueError):
                inventory.assert_inventory_current(snap)

    def test_changed_analysis_rejects_publication_and_preserves_previous_json(self):
        (self.dest / "IMG_270001.jpg").write_bytes(b"image")
        result = self.build()
        target = self.analysis / "photo_assignments.json"
        target.write_bytes(b"previous")
        self.info["IMG_270001.jpg"]["location_name"] = "changed"
        self.write(self.info_path, self.info)
        with self.assertRaises(ValueError):
            core.save_assignments(target, result)
        self.assertEqual(target.read_bytes(), b"previous")

    def test_nested_image_does_not_satisfy_flat_readiness(self):
        nested = self.dest / "old"
        nested.mkdir()
        (nested / "IMG_270001.jpg").write_bytes(b"old image")
        self.assertTrue(core.publishability_issues(self.build()))
        self.assertTrue((nested / "IMG_270001.jpg").exists())

    def test_flat_default_and_by_location_have_independent_conflict_policies(self):
        for layout in ("flat", "by_location"):
            for mode in ("copy", "move"):
                for conflict in ("skip", "overwrite", "rename"):
                    with self.subTest(layout=layout, mode=mode, conflict=conflict):
                        source_root = self.source / f"{layout}-{mode}-{conflict}"
                        source_root.mkdir()
                        source = source_root / "nested" / "IMG_270001.jpg"
                        source.parent.mkdir()
                        source.write_bytes(b"new image")
                        target_root = self.dest / source_root.name
                        target_root.mkdir()
                        target = (target_root / "IMG_270001.jpg" if layout == "flat"
                                  else target_root / "portrait/nested/IMG_270001.jpg")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(b"existing")
                        selection = build_copy_selection(analysis_dir=self.analysis, source_dir=source_root, dest_dir=target_root)
                        summary = copy_selected_files(selection, source_root, target_root, layout=layout,
                                                      mode=mode, on_conflict=conflict)
                        self.assertFalse(summary.issues)
                        self.assertEqual(source.exists(), mode == "copy" or conflict == "skip")
                        self.assertEqual(target.read_bytes(), b"new image" if conflict == "overwrite" else b"existing")
                        if conflict == "rename":
                            self.assertEqual(target.with_stem("IMG_270001 (1)").read_bytes(), b"new image")

    def test_flat_preserves_capture_one_tree_beside_flat_raw_and_xmp(self):
        for mode in ("copy", "move"):
            for prefix, folder in (("", "CaptureOne"), ("shoot/portrait", "cApTuReOnE")):
                with self.subTest(mode=mode, prefix=prefix):
                    source_root = self.source / f"{mode}-{bool(prefix)}"
                    photo_root = source_root / prefix
                    target_root = self.dest / source_root.name
                    files = {
                        "IMG_270001.CR3": "IMG_270001.CR3",
                        "IMG_270001.xmp": "IMG_270001.xmp",
                        f"{folder}/Settings153/IMG_270001.CR3.cos": f"{folder}/Settings153/IMG_270001.CR3.cos",
                        f"{folder}/Settings160/IMG_270001.CR3.cos": f"{folder}/Settings160/IMG_270001.CR3.cos",
                        f"{folder}/Cache/Proxies/IMG_270001.CR3.cop": f"{folder}/Cache/Proxies/IMG_270001.CR3.cop",
                        f"{folder}/Cache/FocusMasks/IMG_270001.CR3.cof": f"{folder}/Cache/FocusMasks/IMG_270001.CR3.cof",
                        f"{folder}/Cache/Thumbnails/IMG_270001.CR3.[cc5f63dc-cd60-416e-9c7f-271955d3cadf].cot":
                            f"{folder}/Cache/Thumbnails/IMG_270001.CR3.[cc5f63dc-cd60-416e-9c7f-271955d3cadf].cot",
                    }
                    for relative in files:
                        path = photo_root / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(relative.encode())
                    unselected = photo_root / folder / "Settings153/IMG_270002.CR3.cos"
                    unselected.write_bytes(b"unselected")
                    target_root.mkdir()
                    selection = build_copy_selection(
                        analysis_dir=self.analysis, source_dir=source_root, dest_dir=target_root,
                    )
                    summary = copy_selected_files(selection, source_root, target_root, mode=mode)
                    self.assertFalse(summary.issues)
                    self.assertEqual(summary.copied, len(files))
                    self.assertEqual(summary.moved, len(files) if mode == "move" else 0)
                    for relative, target in files.items():
                        self.assertEqual((target_root / target).read_bytes(), relative.encode())
                        self.assertEqual((photo_root / relative).exists(), mode == "copy")
                    self.assertEqual(unselected.read_bytes(), b"unselected")
                    self.assertEqual(
                        {p.relative_to(target_root).as_posix() for p in target_root.rglob("*") if p.is_file()},
                        set(files.values()),
                    )

    def test_capture_one_conflicts_are_resolved_inside_service_folders(self):
        relative = Path("CaptureOne/Settings153/IMG_270001.CR3.cos")
        for mode in ("copy", "move"):
            for conflict in ("skip", "overwrite", "rename"):
                with self.subTest(mode=mode, conflict=conflict):
                    source_root = self.source / f"{mode}-{conflict}"
                    source = source_root / "portrait" / relative
                    source.parent.mkdir(parents=True)
                    source.write_bytes(b"new settings")
                    target_root = self.dest / source_root.name
                    target = target_root / relative
                    target.parent.mkdir(parents=True)
                    target.write_bytes(b"existing settings")
                    legacy = target_root / relative.name
                    legacy.write_bytes(b"previously flattened file")
                    selection = build_copy_selection(
                        analysis_dir=self.analysis, source_dir=source_root, dest_dir=target_root,
                    )
                    summary = copy_selected_files(
                        selection, source_root, target_root, mode=mode, on_conflict=conflict,
                    )
                    self.assertFalse(summary.issues)
                    self.assertEqual(source.exists(), mode == "copy" or conflict == "skip")
                    self.assertEqual(target.read_bytes(), b"new settings" if conflict == "overwrite" else b"existing settings")
                    self.assertEqual(legacy.read_bytes(), b"previously flattened file")
                    if conflict == "rename":
                        self.assertEqual(target.with_stem("IMG_270001.CR3 (1)").read_bytes(), b"new settings")

    def test_invalid_layout_does_not_transfer(self):
        result = build_copy_selection(analysis_dir=self.analysis, source_dir=self.source, dest_dir=self.dest)
        with self.assertRaises(ValueError):
            copy_selected_files(result, self.source, self.dest, layout="invalid")
        parser = make_parser("selection_copy")
        config = parser.parse_args(["--analysis_dir", str(self.analysis), "--source_dir", str(self.source), "--dest_dir", str(self.dest)])
        self.assertEqual(config.layout, "flat")

    def test_snapshot_is_read_only_and_checks_missing_destination(self):
        missing = self.root / "missing"
        snap = inventory.scan_destination(missing, self.info_path)
        self.assertTrue(snap.issues)
        self.assertFalse(missing.exists())

    @unittest.skipUnless(os.name == "nt", "Windows junction protection")
    def test_destination_junction_blocks_publication_without_editing_its_target(self):
        outside = self.root / "outside"
        outside.mkdir()
        photo = outside / "IMG_270001.jpg"
        photo.write_bytes(b"outside image")
        link = self.dest / "linked"
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True,
        )
        if created.returncode:
            self.skipTest("Cannot create a test junction on this filesystem")
        try:
            result = self.build()
            self.assertTrue(result.has_errors)
            with self.assertRaises(ValueError):
                core.save_assignments(self.analysis / "photo_assignments.json", result)
            excluded = inventory.scan_destination(self.dest, self.info_path, ("LINKED",))
            self.assertFalse(excluded.issues)
            self.assertEqual(excluded.inventory, ())
            self.assertEqual(photo.read_bytes(), b"outside image")
        finally:
            link.rmdir()
