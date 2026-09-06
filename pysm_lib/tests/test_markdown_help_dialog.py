from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import (
    QColor,
    QImage,
    QPalette,
    QTextCursor,
    QTextDocument,
    QTextFormat,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSlider

from pysm_lib.gui.dialogs.markdown_help_dialog import (
    MARKDOWN_FEATURES,
    MarkdownBrowser,
    MarkdownHelpDialog,
    MarkdownIntegrityError,
    normalize_markdown,
    resolve_link,
)


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]


class MarkdownHelpDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_github_markdown_syntax_matrix(self) -> None:
        source = """# Заголовок

**Жирный** *курсив* ~~зачёркнутый~~ `inline`

```python
print("code block")
```

> Цитата

- элемент
  - вложенный
1. первый
- [x] задача

| Колонка | Значение |
| --- | --- |
| строка | ячейка |

[ссылка](https://example.com) <https://example.org>

![картинка](missing.png)

---
"""
        with tempfile.TemporaryDirectory() as directory:
            source_path = pathlib.Path(directory) / "manual.md"
            browser = MarkdownBrowser(source_path.parent)
            browser.set_markdown_source(source, source_path)

        plain = browser.toPlainText()
        for marker in (
            "Заголовок",
            "Жирный",
            "курсив",
            "зачёркнутый",
            "inline",
            'print("code block")',
            "Цитата",
            "элемент",
            "вложенный",
            "первый",
            "задача",
            "Колонка",
            "строка",
            "ссылка",
            "https://example.org",
        ):
            self.assertIn(marker, plain)
        self.assertIn("<table", browser.document().toHtml())
        self.assertEqual(MARKDOWN_FEATURES, QTextDocument.MarkdownFeature.MarkdownDialectGitHub)

    def test_br_is_normalized_outside_code_and_document_tail_survives(self) -> None:
        for break_tag in ("<br>", "<br/>"):
            source = f"""| A | B |
| --- | --- |
| до {break_tag} после | ячейка |

## Раздел после таблицы

Контрольный конец документа.

```html
<br>
```

`<br>`
"""
            normalized = normalize_markdown(source)
            self.assertIn("до <br/> после", normalized)
            self.assertEqual(2, normalized.count("<br>"))
            with tempfile.TemporaryDirectory() as directory:
                source_path = pathlib.Path(directory) / "manual.md"
                browser = MarkdownBrowser(source_path.parent)
                browser.set_markdown_source(source, source_path)
                self.assertIn("Контрольный конец документа.", browser.toPlainText())

    def test_unsafe_html_is_displayed_literally_and_safe_attributes_are_removed(self) -> None:
        source = (
            '<script src="https://bad.invalid/x.js">x</script> '
            '<b onclick="x">ok</b>\n'
            '`unmatched <img src="https://bad.invalid/image.png">\n'
            '\\`escaped <img src="https://bad.invalid/second.png">\\`\n'
            '<img alt=`not-code` src="https://bad.invalid/third.png">'
        )
        normalized = normalize_markdown(source)
        self.assertIn("&lt;script", normalized)
        self.assertIn("&lt;/script&gt;", normalized)
        self.assertEqual(3, normalized.count("&lt;img"))
        self.assertNotIn("onclick", normalized)
        self.assertIn("<b>ok</b>", normalized)

    def test_unclosed_and_misnested_safe_html_is_balanced(self) -> None:
        normalized = normalize_markdown("<b>bold <i>italic</b> tail")
        self.assertEqual("<b>bold <i>italic</i></b> tail", normalized)

    def test_missing_integrity_marker_rejects_formatted_document(self) -> None:
        class BrokenProbe:
            def setBaseUrl(self, _url: QUrl) -> None:  # noqa: N802
                pass

            def setMarkdown(self, _source: str, _features: object) -> None:  # noqa: N802
                pass

            def toPlainText(self) -> str:  # noqa: N802
                return "Документ без маркера"

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            browser = MarkdownBrowser(root)
            with patch(
                "pysm_lib.gui.dialogs.markdown_help_dialog.QTextDocument",
                return_value=BrokenProbe(),
            ):
                with self.assertRaises(MarkdownIntegrityError) as raised:
                    browser.set_markdown_source("# Руководство", root / "manual.md")

        self.assertEqual("# Руководство", raised.exception.source_text)

    def test_relative_paths_and_navigation_update_the_base_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            help_dir = root / "help"
            image_dir = root / "images"
            help_dir.mkdir()
            image_dir.mkdir()
            manual = root / "manual.md"
            details = help_dir / "details.md"
            manual.write_text(
                "# Главная\n\n[Подробнее](help/details.md)\n\n![img](images/example.png)",
                encoding="utf-8",
            )
            details.write_text("# Подробности", encoding="utf-8")
            QImage(8, 8, QImage.Format.Format_ARGB32).save(str(image_dir / "example.png"))

            dialog = MarkdownHelpDialog(manual)
            resolution = resolve_link("help/details.md", manual, root)
            self.assertEqual("markdown", resolution.kind)
            self.assertEqual(details.resolve(), resolution.path)
            dialog._open_linked_markdown(str(details), "")

            self.assertEqual(details.resolve(), dialog.browser.current_file)
            self.assertEqual(
                QUrl.fromLocalFile(str(help_dir.resolve()) + os.sep),
                dialog.browser.document().baseUrl(),
            )
            self.assertTrue(dialog.back_button.isEnabled())
            dialog.go_back()
            self.assertEqual(manual.resolve(), dialog.browser.current_file)
            dialog.close()

    def test_local_image_is_loaded_scaled_and_remote_image_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            manual = root / "manual.md"
            manual.write_text("# Image", encoding="utf-8")
            source_image = QImage(1600, 20, QImage.Format.Format_ARGB32)
            self.assertTrue(source_image.save(str(image_dir / "wide.png")))
            browser = MarkdownBrowser(root)
            browser.resize(500, 300)
            browser.load_markdown_file(manual)

            loaded = browser.loadResource(
                QTextDocument.ResourceType.ImageResource,
                QUrl("images/wide.png"),
            )
            blocked = browser.loadResource(
                QTextDocument.ResourceType.ImageResource,
                QUrl("https://example.com/image.png"),
            )

            self.assertFalse(loaded.isNull())
            self.assertLessEqual(loaded.width(), browser.viewport().width() - 32)
            self.assertTrue(blocked.isNull())

    def test_link_security_and_no_automatic_opening(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manual = root / "manual.md"
            manual.write_text("# Manual", encoding="utf-8")
            inside = root / "inside.md"
            inside.write_text("# Inside", encoding="utf-8")
            outside = root.parent / "outside.md"

            self.assertEqual("external", resolve_link("https://example.com", manual, root).kind)
            self.assertEqual("anchor", resolve_link("#manual", manual, root).kind)
            self.assertEqual("markdown", resolve_link("inside.md", manual, root).kind)
            self.assertEqual("error", resolve_link("../outside.md", manual, root).kind)
            self.assertEqual("error", resolve_link("javascript:alert(1)", manual, root).kind)

            browser = MarkdownBrowser(root)
            browser.load_markdown_file(manual)
            self.assertFalse(browser.openLinks())
            self.assertFalse(browser.openExternalLinks())
            with patch(
                "pysm_lib.gui.dialogs.markdown_help_dialog.QDesktopServices"
            ) as desktop_services:
                desktop_services.openUrl.return_value = True
                browser._activate_link(QUrl("https://example.com"))
                desktop_services.openUrl.assert_called_once()
                browser._activate_link(QUrl("javascript:alert(1)"))
                desktop_services.openUrl.assert_called_once()

            # The file does not need to exist to validate absolute-path classification.
            absolute = resolve_link(str(outside.resolve()), manual, root)
            self.assertEqual("markdown", absolute.kind)

    def test_markdown_heading_anchor_is_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            browser = MarkdownBrowser(root)
            browser.set_markdown_source(
                "# Начало\n\n## Параметры запуска\n\nТекст",
                root / "manual.md",
            )
            self.assertTrue(browser.scroll_to_anchor("параметры-запуска"))

    def test_heading_spacing_is_applied_after_render_and_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            browser = MarkdownBrowser(root)
            browser.set_markdown_source(
                "# Первый уровень\n\nТекст\n\n## Второй уровень\n\n"
                "### Третий уровень",
                root / "manual.md",
            )

            def heading_margins() -> list[tuple[int, float, float]]:
                result = []
                block = browser.document().begin()
                while block.isValid():
                    block_format = block.blockFormat()
                    level = block_format.headingLevel()
                    if level:
                        result.append(
                            (level, block_format.topMargin(), block_format.bottomMargin())
                        )
                    block = block.next()
                return result

            expected = [(1, 18.0, 10.0), (2, 16.0, 8.0), (3, 12.0, 6.0)]
            self.assertEqual(expected, heading_margins())

            browser._rerender_current_document()
            self.assertEqual(expected, heading_margins())

    def test_blockquote_uses_theme_background_margins_and_painted_decoration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            browser = MarkdownBrowser(root)
            palette = QPalette(browser.palette())
            palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Link, QColor("#1683d8"))
            browser.setPalette(palette)
            browser.resize(640, 360)
            browser.set_markdown_source(
                "До цитаты\n\n> Первая строка\n>\n> Вторая строка\n\nПосле цитаты",
                root / "manual.md",
            )

            quote_blocks = []
            block = browser.document().begin()
            while block.isValid():
                block_format = block.blockFormat()
                quote_level = int(
                    block_format.property(QTextFormat.Property.BlockQuoteLevel) or 0
                )
                if quote_level > 0:
                    quote_blocks.append(block)
                block = block.next()

            self.assertEqual(2, len(quote_blocks))
            first_format = quote_blocks[0].blockFormat()
            last_format = quote_blocks[-1].blockFormat()
            self.assertEqual(28.0, first_format.rightMargin())
            self.assertEqual(7.0, first_format.topMargin())
            self.assertEqual(7.0, last_format.bottomMargin())
            self.assertEqual(QColor("#e8f3fb"), first_format.background().color())
            self.assertEqual(1, len(browser._quote_ranges))
            self.assertNotIn("”", browser.toPlainText())

            browser.show()
            self.app.processEvents()
            rendered = browser.grab()
            self.assertFalse(rendered.isNull())

            dark_palette = QPalette(browser.palette())
            dark_palette.setColor(QPalette.ColorRole.Base, QColor("#202124"))
            dark_palette.setColor(QPalette.ColorRole.Link, QColor("#8ab4f8"))
            browser.setPalette(dark_palette)
            self.app.processEvents()

            self.assertEqual(1, len(browser._quote_ranges))
            dark_quote = browser.document().findBlock(
                browser._quote_ranges[0][0]
            ).blockFormat()
            self.assertEqual(QColor("#333b4a"), dark_quote.background().color())
            browser.close()

    def test_same_document_anchor_uses_history_and_places_heading_at_top(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manual = root / "manual.md"
            manual.write_text(
                "# Начало\n\n[К разделу](#целевой-раздел)\n\n"
                + "\n\n".join(f"Текст до {index}" for index in range(60))
                + "\n\n## Целевой раздел\n\n"
                + "\n\n".join(f"Текст после {index}" for index in range(100)),
                encoding="utf-8",
            )
            dialog = MarkdownHelpDialog(manual)
            dialog.resize(800, 500)
            dialog.show()
            self.app.processEvents()
            scroll_bar = dialog.browser.verticalScrollBar()
            scroll_bar.setValue(120)

            self.assertFalse(dialog.back_button.isEnabled())
            self.assertFalse(dialog.forward_button.isEnabled())
            dialog.browser._activate_link(QUrl("#целевой-раздел"))
            self.app.processEvents()

            anchor_position = scroll_bar.value()
            target_block = dialog.browser._heading_blocks["целевой-раздел"]
            target_rect = dialog.browser.cursorRect(QTextCursor(target_block))
            self.assertLessEqual(abs(target_rect.top()), 2)
            self.assertTrue(dialog.back_button.isEnabled())
            self.assertFalse(dialog.forward_button.isEnabled())

            dialog.go_back()
            self.assertEqual(120, scroll_bar.value())
            self.assertFalse(dialog.back_button.isEnabled())
            self.assertTrue(dialog.forward_button.isEnabled())

            dialog.go_forward()
            self.assertEqual(anchor_position, scroll_bar.value())
            self.assertTrue(dialog.back_button.isEnabled())
            self.assertFalse(dialog.forward_button.isEnabled())
            dialog.close()

    def test_light_and_dark_palettes_produce_theme_derived_styles(self) -> None:
        browser = MarkdownBrowser(pathlib.Path.cwd())
        light = QPalette()
        light.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        light.setColor(QPalette.ColorRole.Text, QColor("#202020"))
        light.setColor(QPalette.ColorRole.AlternateBase, QColor("#eeeeee"))
        light.setColor(QPalette.ColorRole.Link, QColor("#0055aa"))
        browser.setPalette(light)
        browser._apply_document_style()
        light_style = browser.document().defaultStyleSheet()

        dark = QPalette(light)
        dark.setColor(QPalette.ColorRole.Base, QColor("#202124"))
        dark.setColor(QPalette.ColorRole.Text, QColor("#f1f3f4"))
        dark.setColor(QPalette.ColorRole.AlternateBase, QColor("#303134"))
        dark.setColor(QPalette.ColorRole.Link, QColor("#8ab4f8"))
        browser.setPalette(dark)
        browser._apply_document_style()
        dark_style = browser.document().defaultStyleSheet()

        self.assertIn("#ffffff", light_style)
        self.assertIn("#202020", light_style)
        self.assertIn("#202124", dark_style)
        self.assertIn("#f1f3f4", dark_style)
        self.assertIn("#8ab4f8", dark_style)
        self.assertNotEqual(light_style, dark_style)

    def test_vertical_scrollbar_drag_does_not_recreate_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            browser = MarkdownBrowser(root)
            browser.resize(500, 300)
            browser.set_markdown_source(
                "# Прокрутка\n\n" + "\n\n".join(f"Абзац {index}" for index in range(300)),
                root / "manual.md",
            )
            browser.show()
            self.app.processEvents()

            scroll_bar = browser.verticalScrollBar()
            self.assertGreater(scroll_bar.maximum(), 0)
            contents_changed = [0]
            browser.document().contentsChanged.connect(
                lambda: contents_changed.__setitem__(0, contents_changed[0] + 1)
            )

            option = QStyleOptionSlider()
            scroll_bar.initStyleOption(option)
            slider_rect = scroll_bar.style().subControlRect(
                QStyle.ComplexControl.CC_ScrollBar,
                option,
                QStyle.SubControl.SC_ScrollBarSlider,
                scroll_bar,
            )
            target = slider_rect.center()
            target.setY(scroll_bar.height() - 30)
            QTest.mousePress(
                scroll_bar,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                slider_rect.center(),
            )
            QTest.mouseMove(scroll_bar, target, 150)
            self.app.processEvents()
            dragged_value = scroll_bar.value()
            QTest.qWait(200)

            self.assertTrue(scroll_bar.isSliderDown())
            self.assertGreater(dragged_value, scroll_bar.maximum() // 2)
            self.assertEqual(dragged_value, scroll_bar.value())
            self.assertEqual(0, contents_changed[0])
            QTest.mouseRelease(
                scroll_bar,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                target,
            )
            browser.close()

    def test_project_crm_manual_keeps_the_troubleshooting_tail(self) -> None:
        manual = REPOSITORY_ROOT / "scripts/Workflow2/Tools/project_crm/manual.md"
        browser = MarkdownBrowser(manual.parent)
        browser.load_markdown_file(manual)
        plain = browser.toPlainText()
        for marker in (
            "Параметры запуска",
            "Понятие проекта",
            "Работа с мессенджером MAX",
            "Календарь съёмок",
            "Решение проблем",
            "Откройте Внешние календари…",
        ):
            self.assertIn(marker, plain)

    def test_every_manual_is_utf8_nonempty_and_keeps_its_end_marker(self) -> None:
        manuals = sorted(
            path
            for subtree in ("scripts", "scripts_utility", "script_api_demo")
            for path in (REPOSITORY_ROOT / subtree).rglob("manual.md")
        )
        self.assertTrue(manuals)
        for manual in manuals:
            with self.subTest(manual=manual.relative_to(REPOSITORY_ROOT)):
                source = manual.read_text(encoding="utf-8")
                self.assertTrue(source.strip())
                browser = MarkdownBrowser(manual.parent)
                browser.set_markdown_source(source, manual)
                self.assertTrue(browser.toPlainText().strip())


if __name__ == "__main__":
    unittest.main()
