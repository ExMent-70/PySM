"""Тесты имён динамических кадров GOSUB/RETURN."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pysm_lib.app_enums import SetRunMode
from pysm_lib.models import ScriptSetEntryModel, ScriptSetNodeModel
from pysm_lib.set_runner_orchestrator import SetRunnerOrchestrator


class SetManagerStub:
    """Минимальный глобальный поиск экземпляров для тестов маршрутизации."""

    def __init__(self, entries):
        self.entries = {entry.instance_id: entry for entry in entries}

    def find_entry_and_parent_set(self, instance_id):
        entry = self.entries.get(instance_id)
        if not entry:
            return None
        return entry, SimpleNamespace(name="Внешний набор")


class DynamicRoutingNameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_context = tempfile.TemporaryDirectory()
        self.context_path = Path(self.temp_dir_context.name) / "context.json"

    def tearDown(self) -> None:
        self.temp_dir_context.cleanup()

    @staticmethod
    def make_entry(script_id: str, instance_id: str, name=None):
        return ScriptSetEntryModel(
            id=script_id,
            instance_id=instance_id,
            name=name,
        )

    def make_orchestrator(self, root_set, external_entries, script_names):
        info_by_id = {
            script_id: SimpleNamespace(name=name, passport_valid=True)
            for script_id, name in script_names.items()
        }
        set_manager = SetManagerStub(external_entries)
        orchestrator = SetRunnerOrchestrator(
            set_node=root_set,
            run_mode=SetRunMode.CONDITIONAL_FULL,
            continue_on_error=False,
            get_script_info_func=info_by_id.get,
            config_manager=SimpleNamespace(),
            theme_manager=SimpleNamespace(),
            locale_manager=SimpleNamespace(),
            context_file_path=self.context_path,
            get_set_manager_func=lambda: set_manager,
        )
        return orchestrator

    def test_nested_gosub_and_return_use_source_instance_names(self) -> None:
        source = self.make_entry(
            "script_source",
            "instance_source",
            "Запуск Capture One",
        )
        root_next = self.make_entry(
            "script_root_next",
            "instance_root_next",
            "Следующий этап",
        )
        nested_source = self.make_entry(
            "script_nested_source",
            "instance_nested_source",
            "Запуск Capture One разрешен? (да/нет)",
        )
        nested_target = self.make_entry(
            "script_nested_target",
            "instance_nested_target",
            "Завершить",
        )
        root_set = ScriptSetNodeModel(
            name="Этап R4",
            script_entries=[source, root_next],
        )
        orchestrator = self.make_orchestrator(
            root_set,
            [nested_source, nested_target],
            {
                "script_source": "py_goto",
                "script_root_next": "root_next",
                "script_nested_source": "py_if_else",
                "script_nested_target": "finish",
            },
        )
        orchestrator.call_stack = [
            {
                "set_node": root_set,
                "queue": [source, root_next],
                "idx": 0,
                "run_mode": SetRunMode.CONDITIONAL_FULL,
            }
        ]
        messages = []
        orchestrator.log_message.connect(
            lambda message_type, text: messages.append((message_type, text))
        )

        orchestrator._handle_script_routing(
            source.instance_id,
            nested_source.instance_id,
        )
        first_dynamic_frame = orchestrator._determine_next_script()

        self.assertEqual(first_dynamic_frame["set_node"].name, source.name)
        self.assertIn(f"↪ {source.name} (GOSUB).", messages[-1][1])

        orchestrator._handle_script_routing(
            nested_source.instance_id,
            nested_target.instance_id,
        )
        second_dynamic_frame = orchestrator._determine_next_script()

        self.assertEqual(second_dynamic_frame["set_node"].name, nested_source.name)
        self.assertIn(f"↪ {nested_source.name} (GOSUB).", messages[-1][1])

        returned_frame = orchestrator._determine_next_script()

        self.assertIs(returned_frame["set_node"], root_set)
        return_messages = [text for _kind, text in messages if "RETURN" in text]
        self.assertEqual(
            return_messages,
            [
                (
                    "↩ Возврат (RETURN) из "
                    f"'{nested_source.name}' в '{source.name}'"
                ),
                (
                    "↩ Возврат (RETURN) из "
                    f"'{source.name}' в '{root_set.name}'"
                ),
            ],
        )

    def test_base_script_name_is_used_when_instance_name_is_empty(self) -> None:
        source = self.make_entry("script_source", "instance_source", "   ")
        root_set = ScriptSetNodeModel(name="Основной набор", script_entries=[source])
        orchestrator = self.make_orchestrator(
            root_set,
            [],
            {"script_source": "Базовое имя скрипта"},
        )
        orchestrator.call_stack = [
            {
                "set_node": root_set,
                "queue": [source],
                "idx": 0,
                "run_mode": SetRunMode.CONDITIONAL_FULL,
            }
        ]

        self.assertEqual(
            orchestrator._next_dynamic_macro_name(source.instance_id),
            "Базовое имя скрипта",
        )

    def test_numbered_fallback_is_unique_per_orchestrator(self) -> None:
        root_set = ScriptSetNodeModel(name="Основной набор", script_entries=[])
        orchestrator = self.make_orchestrator(root_set, [], {})

        self.assertEqual(
            orchestrator._next_dynamic_macro_name("missing_1"),
            "Динамический макрос 1",
        )
        self.assertEqual(
            orchestrator._next_dynamic_macro_name("missing_2"),
            "Динамический макрос 2",
        )


if __name__ == "__main__":
    unittest.main()
