"""Регрессионные проверки редактора переменных контекста."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pysm_lib.gui.dialogs.collection_passport_dialog import CollectionPassportDialog
from pysm_lib.gui.widgets.parameter_editor_widget import ParamTableColumn
from pysm_lib.models import ContextVariableModel, ScriptSetsCollectionModel


class _LocaleStub:
    def get(self, key: str, **_kwargs) -> str:
        return key


class CollectionPassportDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_description_edit_enables_ok_and_updates_dialog_data(self) -> None:
        dialog = CollectionPassportDialog(
            controller=SimpleNamespace(
                current_collection_file_path=None,
                get_known_args_with_details=lambda: {},
            ),
            collection_model=ScriptSetsCollectionModel(
                collection_name="Workflow",
                context_data={
                    "wf_photo_session": ContextVariableModel(
                        value="SCHOOL",
                        description="Old description",
                    )
                },
            ),
            locale_manager=_LocaleStub(),
            theme_manager=object(),
            script_entries=[],
            get_script_name_func=lambda _instance_id: None,
        )
        self.assertFalse(dialog.ok_button.isEnabled())

        description_item = dialog.context_editor.editor.table.item(
            0,
            ParamTableColumn.CONTEXT_DESCRIPTION,
        )
        description_item.setText("Updated description")

        self.assertTrue(dialog.ok_button.isEnabled())
        dialog.accept()
        self.assertEqual(
            dialog.get_data()["context_data"]["wf_photo_session"].description,
            "Updated description",
        )
        dialog.close()


if __name__ == "__main__":
    unittest.main()
