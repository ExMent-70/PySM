"""Проверки самоссылок и переноса ссылок при импорте набора."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog

from pysm_lib.app_enums import SetRunMode
from pysm_lib.gui.widgets.instance_selection_dialog import InstanceSelectionDialog
from pysm_lib.models import (
    ScriptSetEntryModel,
    ScriptSetEntryValueEnabled,
    ScriptSetNodeModel,
    SetFolderNodeModel,
)
from pysm_lib.set_manager import SetManager
from pysm_lib.set_runner_orchestrator import SetRunnerOrchestrator


class InstanceRoutingSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_import_remaps_internal_ids_in_comma_separated_values(self):
        args = {
            "macro": ScriptSetEntryValueEnabled(value="a, b ,external,a", enabled=True),
            "single": ScriptSetEntryValueEnabled(value="a", enabled=True),
            "list": ScriptSetEntryValueEnabled(value=["a", "external", "b"], enabled=True),
            "text": ScriptSetEntryValueEnabled(value="unrelated, text", enabled=True),
        }
        source = ScriptSetNodeModel(name="Source", script_entries=[
            ScriptSetEntryModel(id="script_a", instance_id="a", command_line_args=args),
            ScriptSetEntryModel(id="script_b", instance_id="b"),
        ])
        with tempfile.TemporaryDirectory() as temporary:
            manager = SetManager(Path(temporary))
            parent = SetFolderNodeModel(name="Parent")
            manager.current_collection_model.root_nodes = [parent]
            manager._rebuild_nodes_cache()
            imported = manager.import_set_node(source.model_dump_json(), parent.id)

            self.assertIsNotNone(imported)
            entry_a, entry_b = imported.script_entries
            values = entry_a.command_line_args
            self.assertEqual(values["macro"].value, f"{entry_a.instance_id}, {entry_b.instance_id} ,external,{entry_a.instance_id}")
            self.assertEqual(values["single"].value, entry_a.instance_id)
            self.assertEqual(values["list"].value, [entry_a.instance_id, "external", entry_b.instance_id])
            self.assertEqual(values["text"].value, "unrelated, text")

    def test_current_instance_is_selectable_and_restored(self):
        entry = ScriptSetEntryModel(id="conditional", instance_id="self", name="Dialog")
        locale = SimpleNamespace(get=lambda key, **kw: f"{kw['name']} (current)" if 'name' in kw else key)
        with patch("pysm_lib.gui.widgets.instance_selection_dialog.icons.get_qicon", return_value=QIcon()), patch(
            "pysm_lib.gui.widgets.instance_selection_dialog.generate_instance_tooltip_html", return_value="tooltip"
        ):
            dialog = InstanceSelectionDialog(
                title="Select", script_entries=[("Set", entry)],
                get_script_name_func=lambda _: None, locale_manager=locale,
                theme_manager=None, current_value="self", current_instance_id="self",
            )
        item = dialog.tree_widget.topLevelItem(0).child(0)
        self.assertTrue(item.flags() & Qt.ItemFlag.ItemIsEnabled)
        self.assertTrue(item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
        self.assertEqual(item.checkState(0), Qt.CheckState.Checked)
        self.assertTrue(item.font(0).bold())
        self.assertIn("current", item.text(0))
        item.setCheckState(0, Qt.CheckState.Unchecked)
        item.setCheckState(0, Qt.CheckState.Checked)
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(dialog.get_selected_instance_id(), "self")
        dialog.close()

    def test_self_jump_repeats_current_script_in_both_run_modes(self):
        entry = ScriptSetEntryModel(id="conditional", instance_id="self", name="Dialog")
        root = ScriptSetNodeModel(name="Set", script_entries=[entry])
        for mode in (SetRunMode.CONDITIONAL_FULL, SetRunMode.SINGLE_FROM_SET):
            with self.subTest(mode=mode):
                orchestrator = SetRunnerOrchestrator(
                    set_node=root, run_mode=mode, continue_on_error=False,
                    get_script_info_func=lambda _: SimpleNamespace(name="Dialog", passport_valid=True),
                    config_manager=SimpleNamespace(), theme_manager=SimpleNamespace(),
                    locale_manager=SimpleNamespace(), context_file_path=Path("unused.context.json"),
                )
                frame = {"set_node": root, "queue": [entry], "idx": 0, "run_mode": mode}
                orchestrator.call_stack = [frame]
                orchestrator._handle_script_routing("self", "self")
                result = orchestrator._determine_next_script()
                self.assertIs(result, frame)
                self.assertEqual(result["queue"][result["idx"]].instance_id, "self")


if __name__ == "__main__":
    unittest.main()
