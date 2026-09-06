from __future__ import annotations

import os
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import QApplication, QTextEdit

from pysm_lib.app_enums import EditMode
from pysm_lib.gui.dialogs.script_properties_dialog import ScriptPropertiesDialog
from pysm_lib.gui.tooltip_generator import _generate_base_script_html
from pysm_lib.locale_manager import LocaleManager
from pysm_lib.models import ScriptInfoModel, ScriptSetEntryModel
from pysm_lib.theme_manager import ThemeManager


class ScriptPropertiesDialogLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_base_description_wraps_without_horizontal_scrollbar(self) -> None:
        locale_manager = LocaleManager("ru_RU")
        script_info = ScriptInfoModel(
            id="layout_test",
            name="layout_test",
            folder_abs_path=r"C:\scripts\layout_test",
            passport_valid=True,
            description="Описание тестового скрипта",
            command_line_args_meta={
                "long_argument": {
                    "type": "string",
                    "description": "Очень длинное описание параметра " * 20,
                }
            },
        )
        dialog = ScriptPropertiesDialog(
            EditMode.INSTANCE,
            script_info,
            locale_manager,
            ThemeManager(),
            instance_entry=ScriptSetEntryModel(id=script_info.id),
        )
        dialog.resize(800, 700)
        dialog.show()
        self.app.processEvents()

        description = dialog.details_toolbox.widget(0)
        self.assertIsInstance(description, QTextEdit)
        self.assertEqual(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            description.horizontalScrollBarPolicy(),
        )
        self.assertEqual(
            QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere,
            description.document().defaultTextOption().wrapMode(),
        )
        self.assertEqual(0, description.horizontalScrollBar().maximum())

        tooltip_html = _generate_base_script_html(script_info, locale_manager)
        dialog_html = _generate_base_script_html(
            script_info,
            locale_manager,
            wrap_argument_descriptions=True,
        )
        self.assertIn("white-space: nowrap", tooltip_html)
        self.assertIn("white-space: normal", dialog_html)
        dialog.close()


if __name__ == "__main__":
    unittest.main()
