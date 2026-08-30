"""Проверки разрешения контекста в заметках экземпляров скриптов."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from pysm_lib.models import ScriptSetEntryModel
from pysm_lib.set_runner_orchestrator import SetRunnerOrchestrator


class InstanceDescriptionContextTests(unittest.TestCase):
    @staticmethod
    def make_orchestrator():
        orchestrator = SetRunnerOrchestrator.__new__(SetRunnerOrchestrator)
        orchestrator._context_snapshot_cache = {
            "wf_source_file_path": {
                "type": "file_path",
                "value": r"D:\Photos\source.raw",
            },
            "wf_school_info": {
                "type": "json",
                "value": {"name": "School 42"},
            },
        }
        messages = []
        orchestrator.log_message = SimpleNamespace(
            emit=lambda *args: messages.append(args)
        )
        return orchestrator, messages

    def test_silent_description_resolves_context_without_changing_template(self) -> None:
        orchestrator, messages = self.make_orchestrator()
        entry = ScriptSetEntryModel(
            id="dir_operation",
            description=(
                "Copy {wf_source_file_path}\n"
                "Project: {wf_school_info.name}\n"
                "Unknown: {missing_var}"
            ),
            silent_mode=True,
        )

        orchestrator._log_script_start_info_silent(entry)

        self.assertIn(
            (
                "html_block",
                r"Copy D:\Photos\source.raw<br>Project: School 42<br>Unknown: {missing_var}",
            ),
            messages,
        )
        self.assertIn("{wf_source_file_path}", entry.description)

    def test_regular_description_resolves_context_before_console_output(self) -> None:
        orchestrator, messages = self.make_orchestrator()
        orchestrator.call_stack = [{}]
        orchestrator.locale_manager = SimpleNamespace(get=lambda key, **_kwargs: key)
        orchestrator.theme_manager = SimpleNamespace(
            get_active_theme_dynamic_styles=lambda: {}
        )
        entry = ScriptSetEntryModel(
            id="dir_operation",
            description="Copy {wf_source_file_path}",
        )
        script_info = SimpleNamespace(
            name="dir_operation",
            folder_abs_path=r"D:\PySM\scripts\dir_operation",
        )
        runner = SimpleNamespace(
            python_interpreter="python.exe",
            custom_command_args_dict={},
        )
        current_frame = {
            "set_node": SimpleNamespace(name="Workflow"),
            "idx": 0,
            "queue": [entry],
        }

        orchestrator._log_script_start_info(
            script_info,
            entry,
            runner,
            current_frame,
        )

        self.assertIn(
            ("html_block", r"  Copy D:\Photos\source.raw"),
            messages,
        )


if __name__ == "__main__":
    unittest.main()
