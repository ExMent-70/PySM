"""Contract tests for the public manual AI JSON workflow API."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pysm_lib.ai import (
    AiJsonDecodeError,
    AiJsonRequest,
    PromptTemplateError,
    extract_json_object,
    extract_json_value,
    get_json_array,
    render_prompt_template,
    require_json_object,
)
from pysm_lib.gui.ai import AiJsonDialogStatus, create_ai_json_dialog


class AiJsonCoreTests(unittest.TestCase):
    """Verify prompt rendering and provider-neutral JSON parsing."""

    def test_renderer_formats_json_values_and_preserves_source_text(self) -> None:
        prompt = render_prompt_template(
            "REFERENCE={{REFERENCE}}\nSOURCE={{RAW_TEXT}}",
            {
                "REFERENCE": [{"id": "S001", "name": "Иван"}],
                "RAW_TEXT": "Иван — 001234",
            },
        )

        self.assertIn('"id": "S001"', prompt)
        self.assertIn("SOURCE=Иван — 001234", prompt)

    def test_renderer_rejects_missing_placeholder_values(self) -> None:
        with self.assertRaisesRegex(PromptTemplateError, "MISSING"):
            render_prompt_template("{{MISSING}}", {})

    def test_extractor_ignores_prose_and_markdown_around_json(self) -> None:
        payload = extract_json_object(
            "Пояснение {это не JSON}.\n```json\n"
            '{"matched": [{"student_id": "A7K3-S001"}]}\n```'
        )

        self.assertEqual(payload["matched"][0]["student_id"], "A7K3-S001")

    def test_extractor_reports_an_unexpected_json_type(self) -> None:
        with self.assertRaisesRegex(AiJsonDecodeError, "JSON-массив"):
            extract_json_value("[1, 2, 3]", expected_type=dict)

    def test_request_validates_before_returning_a_domain_value(self) -> None:
        request = AiJsonRequest(
            prompt_template="SOURCE={{RAW_TEXT}}",
            response_validator=lambda payload: get_json_array(
                require_json_object(payload), "items", required=True
            ),
        )

        self.assertEqual(request.build_prompt("данные"), "SOURCE=данные")
        self.assertEqual(request.validate_response('{"items": [1, 2]}'), [1, 2])


class AiJsonDialogTests(unittest.TestCase):
    """Verify the reusable Qt dialog returns only an already validated value."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        for widget in QApplication.topLevelWidgets():
            widget.close()
        self.app.processEvents()

    def test_dialog_builds_prompt_and_returns_validated_result(self) -> None:
        request = AiJsonRequest(
            title="Test AI JSON",
            prompt_template="SOURCE={{RAW_TEXT}}",
            response_validator=lambda payload: require_json_object(payload)["value"],
        )
        dialog = create_ai_json_dialog(request)
        assert dialog.raw_text_edit is not None
        dialog.raw_text_edit.setPlainText("исходный текст")

        with patch("pysm_lib.gui.ai.manual_json_dialog.QMessageBox.information"):
            dialog._copy_prompt()
            dialog.response_text_edit.setPlainText('{"value": "готово"}')
            dialog._validate_response()

        self.assertEqual(dialog.prompt_preview.toPlainText(), "SOURCE=исходный текст")
        self.assertEqual(dialog.result.status, AiJsonDialogStatus.VALIDATED)
        self.assertTrue(dialog.result.accepted)
        self.assertEqual(dialog.result.value, "готово")


if __name__ == "__main__":
    unittest.main()
