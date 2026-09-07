"""Regression tests for the gui_dialog_if_else branching behavior."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "run_gui_dialog_if_else.py"


def load_script_module():
    """Load the script entry point without requiring a package wrapper."""
    spec = importlib.util.spec_from_file_location(
        "run_gui_dialog_if_else_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_config(**overrides):
    """Build a complete dialog configuration for orchestration tests."""
    values = {
        "html_content": "<b>Выберите ветку</b>",
        "html_file": None,
        "html_output": "dialog",
        "html_align": "left",
        "html_margin": 0,
        "html_padding": 10,
        "html_style": "script_description",
        "dlg_msg_var": "workflow.choice",
        "dlg_msg_type": "yes_no_cancel",
        "dlg_msg_text_ok": "Да",
        "dlg_msg_text_no": "Нет",
        "dlg_msg_text_cancel": "Отмена",
        "dlg_msg_title": "Выбор",
        "dlg_msg_size_width": 700,
        "dlg_msg_size_height": 500,
        "instance_id_yes": "yes-id",
        "instance_id_no": "no-id",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class GuiDialogIfElseTests(unittest.TestCase):
    """Verify stable result persistence and routing semantics."""

    def test_main_saves_every_result_and_routes_only_yes_or_no(self):
        cases = [
            ("yes", 0, "yes-id"),
            ("no", 0, "no-id"),
            ("cancel", 1, None),
            ("unknown", 1, None),
        ]

        for result, expected_code, expected_target in cases:
            with self.subTest(result=result):
                module = load_script_module()
                config = make_config()
                saved_results = []
                routed_targets = []

                class FakeContext:
                    @staticmethod
                    def set_next_script(target_id):
                        routed_targets.append(target_id)

                module.get_config = lambda: config
                module.IS_MANAGED_RUN = True
                module.pysm_context = FakeContext()
                module.theme_api = SimpleNamespace(
                    get_parsed_style=lambda *args, **kwargs: {
                        "color": "#ffffff"
                    },
                )
                module.show_html_message_dialog = lambda **kwargs: result
                module.save_dialog_choice = (
                    lambda current_config, current_result: saved_results.append(
                        (current_config.dlg_msg_var, current_result)
                    )
                )

                self.assertEqual(module.main(), expected_code)
                self.assertEqual(
                    saved_results,
                    [("workflow.choice", result)],
                )
                self.assertEqual(
                    routed_targets,
                    [] if expected_target is None else [expected_target],
                )

    def test_validate_config_requires_both_branch_targets(self):
        module = load_script_module()

        with self.assertRaisesRegex(ValueError, "instance-id-no"):
            module.validate_config(make_config(instance_id_no="  "))

    def test_branch_target_maps_yes_and_no_only(self):
        module = load_script_module()

        self.assertEqual(
            module.branch_target("yes", "yes-id", "no-id"),
            ("YES", "yes-id"),
        )
        self.assertEqual(
            module.branch_target("no", "yes-id", "no-id"),
            ("NO", "no-id"),
        )
        self.assertIsNone(module.branch_target("cancel", "yes-id", "no-id"))
        self.assertIsNone(module.branch_target("unknown", "yes-id", "no-id"))


if __name__ == "__main__":
    unittest.main()
