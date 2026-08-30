from types import SimpleNamespace
import unittest

from pysm_lib.gui.tooltip_generator import (
    TOOLTIP_ARGUMENT_VALUE_MAX_LENGTH,
    _truncate_tooltip_argument_value,
    generate_favorite_tooltip_html,
    generate_instance_tooltip_html,
    generate_script_tooltip_html,
)
from pysm_lib.locale_manager import LocaleManager
from pysm_lib.models import ScriptInfoModel, ScriptSetEntryModel


LONG_PARAMETER_VALUE = "первая строка\n" + "длинное значение " * 8
LONG_SINGLE_LINE_VALUE = "длинное однострочное значение " * 8


def _theme_manager_stub():
    return SimpleNamespace(get_active_theme_dynamic_styles=lambda: {})


def _script_info(description: str | None = None) -> ScriptInfoModel:
    return ScriptInfoModel(
        id="tooltip_test",
        name="tooltip_test",
        folder_abs_path=r"C:\scripts\tooltip_test",
        passport_valid=True,
        command_line_args_meta={
            "large_value": {
                "type": "string",
                "description": description or "Описание",
            }
        },
    )


def _instance_entry() -> ScriptSetEntryModel:
    return ScriptSetEntryModel(
        id="tooltip_test",
        instance_id="instance_tooltip_test",
        command_line_args={
            "large_value": {"value": LONG_PARAMETER_VALUE, "enabled": True}
        },
    )


def _routing_script_info() -> ScriptInfoModel:
    return ScriptInfoModel(
        id="routing_test",
        name="routing_test",
        folder_abs_path=r"C:\scripts\routing_test",
        passport_valid=True,
        command_line_args_meta={
            "goto_target": {
                "type": "instance",
                "description": "Целевые экземпляры",
            }
        },
    )


def _routing_entry() -> ScriptSetEntryModel:
    return ScriptSetEntryModel(
        id="routing_test",
        instance_id="instance_routing_test",
        command_line_args={
            "goto_target": {
                "value": "instance_target_one,instance_target_two",
                "enabled": True,
            }
        },
    )


class TooltipGeneratorTests(unittest.TestCase):
    def test_multiline_argument_value_is_single_line_and_truncated(self):
        result = _truncate_tooltip_argument_value(LONG_PARAMETER_VALUE)

        self.assertNotIn("\n", result)
        self.assertEqual(len(result), TOOLTIP_ARGUMENT_VALUE_MAX_LENGTH)
        self.assertTrue(result.endswith("..."))

    def test_long_single_line_argument_value_is_not_truncated(self):
        result = _truncate_tooltip_argument_value(LONG_SINGLE_LINE_VALUE)

        self.assertEqual(result, LONG_SINGLE_LINE_VALUE)

    def test_instance_and_favorite_tooltips_truncate_long_argument_values(self):
        for generator in (
            generate_instance_tooltip_html,
            generate_favorite_tooltip_html,
        ):
            with self.subTest(generator=generator.__name__):
                tooltip = generator(
                    _script_info(),
                    _instance_entry(),
                    LocaleManager("ru_RU"),
                    _theme_manager_stub(),
                )
                expected_value = _truncate_tooltip_argument_value(
                    LONG_PARAMETER_VALUE
                )

                self.assertIn(expected_value, tooltip)
                self.assertNotIn(LONG_PARAMETER_VALUE, tooltip)

    def test_script_tooltip_truncates_long_argument_description(self):
        tooltip = generate_script_tooltip_html(
            _script_info(LONG_PARAMETER_VALUE),
            LocaleManager("ru_RU"),
            _theme_manager_stub(),
        )
        expected_value = _truncate_tooltip_argument_value(LONG_PARAMETER_VALUE)

        self.assertIn(expected_value, tooltip)
        self.assertNotIn(LONG_PARAMETER_VALUE, tooltip)

    def test_routing_targets_are_rendered_on_separate_lines_with_names(self):
        resolved_names = {
            "instance_target_one": "[Этап 1] Импорт",
            "instance_target_two": "[Этап 2] Обработка",
        }

        for generator in (
            generate_instance_tooltip_html,
            generate_favorite_tooltip_html,
        ):
            with self.subTest(generator=generator.__name__):
                tooltip = generator(
                    _routing_script_info(),
                    _routing_entry(),
                    LocaleManager("ru_RU"),
                    _theme_manager_stub(),
                    instance_name_resolver=resolved_names.get,
                )

                first_target = "[Этап 1] Импорт (instance_target_one)"
                second_target = "[Этап 2] Обработка (instance_target_two)"
                self.assertIn("white-space: nowrap;", tooltip)
                self.assertIn("--goto_target: <br>", tooltip)
                self.assertIn(first_target, tooltip)
                self.assertIn(second_target, tooltip)
                self.assertLess(tooltip.index(first_target), tooltip.index(second_target))
                self.assertNotIn(
                    "instance_target_one,instance_target_two",
                    tooltip,
                )


if __name__ == "__main__":
    unittest.main()
