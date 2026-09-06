"""Safe, reusable Markdown viewer for PySM manuals."""

from __future__ import annotations

import html
import os
import pathlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote

from PySide6.QtCore import QEvent, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFontDatabase,
    QImage,
    QImageReader,
    QPaintEvent,
    QPainter,
    QPalette,
    QPen,
    QTextCursor,
    QTextDocument,
    QTextFormat,
)
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


MARKDOWN_FEATURES = QTextDocument.MarkdownFeature.MarkdownDialectGitHub
ALLOWED_EXTERNAL_SCHEMES = frozenset({"http", "https", "mailto", "tg", "max"})
LOCAL_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
)
_VOID_HTML_TAGS = frozenset({"br", "hr"})
_SAFE_HTML_TAGS = frozenset(
    {"b", "strong", "i", "em", "u", "s", "del", "sub", "sup", "code"}
)
_HTML_TAG_RE = re.compile(
    r"<\s*(?P<closing>/?)\s*(?P<name>[A-Za-z][\w:-]*)\b[^<>]*?(?P<selfclose>/?)\s*>"
)
_FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_HEADING_BLOCK_MARGINS = {
    1: (18.0, 10.0),
    2: (16.0, 8.0),
    3: (12.0, 6.0),
    4: (12.0, 6.0),
    5: (10.0, 4.0),
    6: (10.0, 4.0),
}
_QUOTE_LEFT_MARGIN = 24.0
_QUOTE_RIGHT_MARGIN = 28.0
_QUOTE_OUTER_MARGIN = 7.0
_QUOTE_INNER_MARGIN = 0.0


def _blend_color(base: QColor, accent: QColor, factor: float) -> QColor:
    """Return a theme-aware color between the document base and accent."""

    bounded = max(0.0, min(float(factor), 1.0))
    return QColor(
        round(base.red() + (accent.red() - base.red()) * bounded),
        round(base.green() + (accent.green() - base.green()) * bounded),
        round(base.blue() + (accent.blue() - base.blue()) * bounded),
    )


class MarkdownFileError(Exception):
    """A manual cannot be read as a local UTF-8 Markdown file."""


class MarkdownIntegrityError(Exception):
    """Qt discarded the end marker while converting Markdown."""

    def __init__(self, message: str, source_text: str) -> None:
        super().__init__(message)
        self.source_text = source_text


@dataclass(frozen=True)
class LinkResolution:
    """Validated action produced from one clicked Markdown link."""

    kind: str
    path: Optional[pathlib.Path] = None
    url: Optional[QUrl] = None
    anchor: str = ""
    error: str = ""


@dataclass
class _HistoryEntry:
    path: pathlib.Path
    scroll_position: int = 0
    anchor: str = ""


def _normalize_html_fragment(fragment: str, open_tags: Optional[list[str]] = None) -> str:
    """Normalize the safe HTML subset in text outside Markdown code spans."""

    tag_stack = open_tags if open_tags is not None else []

    def replace_tag(match: re.Match[str]) -> str:
        tag_name = match.group("name").lower()
        is_closing = bool(match.group("closing"))
        if tag_name in _VOID_HTML_TAGS:
            return html.escape(match.group(0), quote=False) if is_closing else f"<{tag_name}/>"
        if tag_name in _SAFE_HTML_TAGS:
            if not is_closing:
                tag_stack.append(tag_name)
                return f"<{tag_name}>"
            if tag_name not in tag_stack:
                return html.escape(match.group(0), quote=False)
            closing_tags: list[str] = []
            while tag_stack:
                open_tag = tag_stack.pop()
                closing_tags.append(f"</{open_tag}>")
                if open_tag == tag_name:
                    break
            return "".join(closing_tags)
        # Display unsupported HTML literally. QTextDocument must never interpret it.
        return html.escape(match.group(0), quote=False)

    return _HTML_TAG_RE.sub(replace_tag, fragment)


def _normalize_non_block_code(text: str) -> str:
    """Normalize HTML while preserving only complete Markdown code spans."""

    output: list[str] = []
    open_tags: list[str] = []
    position = 0
    runs = [
        match
        for match in re.finditer(r"`+", text)
        if not _backtick_run_is_escaped(text, match.start())
        and text.rfind("<", 0, match.start()) <= text.rfind(">", 0, match.start())
    ]
    run_index = 0
    while run_index < len(runs):
        opening = runs[run_index]
        output.append(_normalize_html_fragment(text[position : opening.start()], open_tags))
        delimiter_length = len(opening.group(0))
        closing = None
        closing_index = run_index + 1
        while closing_index < len(runs):
            candidate = runs[closing_index]
            if len(candidate.group(0)) == delimiter_length:
                closing = candidate
                break
            closing_index += 1
        if closing is None:
            # An unmatched backtick is ordinary Markdown text, not a safe-code boundary.
            output.append(_normalize_html_fragment(text[opening.start() :], open_tags))
            output.extend(f"</{tag}>" for tag in reversed(open_tags))
            return "".join(output)
        output.append(text[opening.start() : closing.end()])
        position = closing.end()
        run_index = closing_index + 1
    output.append(_normalize_html_fragment(text[position:], open_tags))
    output.extend(f"</{tag}>" for tag in reversed(open_tags))
    return "".join(output)


def _backtick_run_is_escaped(text: str, position: int) -> bool:
    backslashes = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        backslashes += 1
        position -= 1
    return bool(backslashes % 2)


def normalize_markdown(markdown: str) -> str:
    """Normalize safe Qt HTML without changing fenced, indented, or inline code."""

    output: list[str] = []
    regular_text: list[str] = []
    fenced_character = ""
    fenced_length = 0

    def flush_regular_text() -> None:
        if regular_text:
            output.append(_normalize_non_block_code("".join(regular_text)))
            regular_text.clear()

    for original_line in markdown.splitlines(keepends=True):
        line = original_line.rstrip("\r\n")
        fence_match = _FENCE_RE.match(line)
        if fenced_character:
            output.append(original_line)
            if re.match(
                rf"^ {{0,3}}{re.escape(fenced_character)}{{{fenced_length},}}\s*$",
                line,
            ):
                fenced_character = ""
                fenced_length = 0
            continue
        if fence_match:
            flush_regular_text()
            fence = fence_match.group("fence")
            fenced_character = fence[0]
            fenced_length = len(fence)
            output.append(original_line)
            continue
        if line.startswith("    ") or line.startswith("\t"):
            flush_regular_text()
            output.append(original_line)
            continue

        regular_text.append(original_line)

    flush_regular_text()
    return "".join(output)


def _is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Return whether *path* is inside *root*, including across Windows drives."""

    try:
        common = os.path.commonpath((str(path), str(root)))
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(str(root))


def resolve_link(
    link: QUrl | str,
    current_file: pathlib.Path,
    allowed_root: pathlib.Path,
) -> LinkResolution:
    """Classify a clicked URL and enforce the local relative-path boundary."""

    url = QUrl(link) if isinstance(link, str) else QUrl(link)
    raw = unquote(url.toString())
    if raw.startswith("#"):
        return LinkResolution("anchor", anchor=unquote(raw[1:]))

    # QUrl interprets a Windows drive letter as a URL scheme.
    windows_absolute = bool(_WINDOWS_ABSOLUTE_RE.match(raw))
    scheme = url.scheme().lower()
    if scheme in ALLOWED_EXTERNAL_SCHEMES:
        return LinkResolution("external", url=url)
    if scheme and scheme != "file" and not windows_absolute:
        return LinkResolution(
            "error",
            error=f"Схема ссылки «{scheme}» запрещена.",
        )

    anchor = unquote(url.fragment())
    if scheme == "file":
        local_text = url.toLocalFile()
        is_absolute = True
    else:
        local_text = raw.split("#", 1)[0]
        is_absolute = windows_absolute or pathlib.Path(local_text).is_absolute()

    if not local_text:
        return LinkResolution("anchor", anchor=anchor)

    candidate = pathlib.Path(local_text)
    if not is_absolute:
        candidate = current_file.parent / candidate
    try:
        candidate = candidate.resolve(strict=False)
        root = allowed_root.resolve(strict=False)
    except OSError:
        return LinkResolution("error", error="Локальный путь ссылки нельзя разрешить.")

    if not is_absolute and not _is_within(candidate, root):
        return LinkResolution(
            "error",
            error="Относительная ссылка выходит за пределы папки скрипта.",
        )

    kind = "markdown" if candidate.suffix.lower() == ".md" else "local"
    return LinkResolution(kind, path=candidate, anchor=anchor)


def _heading_slug(text: str) -> str:
    """Build the useful subset of GitHub heading IDs, including Cyrillic."""

    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = re.sub(r"[^\w\- ]", "", normalized, flags=re.UNICODE)
    return re.sub(r"[ ]+", "-", normalized)


class MarkdownBrowser(QTextBrowser):
    """Theme-aware Markdown browser with controlled links and local resources."""

    markdownLinkActivated = Signal(str, str)
    anchorLinkActivated = Signal(str)
    warningRequested = Signal(str, str)
    resourceWarning = Signal(str)

    def __init__(
        self,
        allowed_root: pathlib.Path | str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.allowed_root = pathlib.Path(allowed_root).resolve(strict=False)
        self.current_file: Optional[pathlib.Path] = None
        self._current_markdown = ""
        self._heading_blocks: dict[str, object] = {}
        self._quote_ranges: list[tuple[int, int, int]] = []
        self._reported_resources: set[str] = set()

        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.anchorClicked.connect(self._activate_link)
        self._apply_document_style()

    def load_markdown_file(self, path: pathlib.Path | str) -> str:
        """Read and render one UTF-8 Markdown file, returning its source text."""

        source_path = pathlib.Path(path).resolve(strict=False)
        if not source_path.is_file():
            raise MarkdownFileError(f"Файл руководства не найден:\n{source_path}")
        try:
            source_text = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise MarkdownFileError(
                f"Руководство сохранено не в UTF-8:\n{source_path}"
            ) from error
        except OSError as error:
            raise MarkdownFileError(
                f"Не удалось прочитать руководство:\n{source_path}"
            ) from error
        self.set_markdown_source(source_text, source_path)
        return source_text

    def set_markdown_source(self, source_text: str, source_path: pathlib.Path | str) -> None:
        """Render already-read Markdown after checking Qt conversion integrity."""

        path = pathlib.Path(source_path).resolve(strict=False)
        normalized = normalize_markdown(source_text)
        marker = f"PYSM_MARKDOWN_END_{os.urandom(16).hex()}"
        probe = QTextDocument()
        probe.setBaseUrl(QUrl.fromLocalFile(str(path.parent) + os.sep))
        probe.setMarkdown(f"{normalized}\n\n{marker}\n", MARKDOWN_FEATURES)
        if marker not in probe.toPlainText():
            raise MarkdownIntegrityError(
                "Qt не смог полностью преобразовать Markdown. "
                "Показан исходный текст без форматирования.",
                source_text,
            )

        self.current_file = path
        self._current_markdown = normalized
        self._reported_resources.clear()
        self.document().setBaseUrl(QUrl.fromLocalFile(str(path.parent) + os.sep))
        self._apply_document_style()
        self.document().setMarkdown(normalized, MARKDOWN_FEATURES)
        self._apply_heading_block_margins()
        self._apply_quote_block_formatting()
        self._index_heading_blocks()

    def show_source_text(self, source_text: str, source_path: pathlib.Path | str) -> None:
        """Use a safe plain-text fallback after a failed integrity check."""

        path = pathlib.Path(source_path).resolve(strict=False)
        self.current_file = path
        self._current_markdown = ""
        self._quote_ranges.clear()
        self.document().setBaseUrl(QUrl.fromLocalFile(str(path.parent) + os.sep))
        self.setPlainText(source_text)

    def scroll_to_anchor(self, anchor: str) -> bool:
        """Scroll to a Markdown heading slug or an explicit QTextDocument anchor."""

        decoded = unquote(anchor).lstrip("#")
        block = self._heading_blocks.get(decoded.casefold())
        if block is not None and block.isValid():
            cursor = QTextCursor(block)
            scroll_bar = self.verticalScrollBar()
            target_position = scroll_bar.value() + self.cursorRect(cursor).top()
            self.setTextCursor(cursor)
            scroll_bar.setValue(target_position)
            return True
        self.scrollToAnchor(decoded)
        return False

    def loadResource(self, resource_type: int, name: QUrl) -> object:  # noqa: N802
        """Load only bounded local images; never fetch network resources."""

        if int(resource_type) != int(QTextDocument.ResourceType.ImageResource):
            return super().loadResource(resource_type, name)
        if self.current_file is None:
            return QImage()

        url = QUrl(name)
        scheme = url.scheme().lower()
        if scheme and scheme != "file":
            self._report_resource_warning(
                f"Удалённое изображение заблокировано: {url.toString()}"
            )
            return QImage()

        raw_path = url.toLocalFile() if scheme == "file" else unquote(url.path())
        image_path = pathlib.Path(raw_path)
        if not image_path.is_absolute():
            image_path = self.current_file.parent / image_path
        try:
            image_path = image_path.resolve(strict=False)
        except OSError:
            self._report_resource_warning("Путь изображения нельзя разрешить.")
            return QImage()

        if not _is_within(image_path, self.allowed_root):
            self._report_resource_warning(
                f"Изображение вне папки скрипта заблокировано: {image_path}"
            )
            return QImage()
        if image_path.suffix.lower() not in LOCAL_IMAGE_SUFFIXES or not image_path.is_file():
            self._report_resource_warning(f"Изображение не найдено: {image_path}")
            return QImage()

        reader = QImageReader(str(image_path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            self._report_resource_warning(f"Изображение нельзя прочитать: {image_path}")
            return QImage()

        maximum_width = max(64, self.viewport().width() - 32)
        if image.width() > maximum_width:
            image = image.scaledToWidth(
                maximum_width,
                Qt.TransformationMode.SmoothTransformation,
            )
        return image

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
        }:
            self._apply_document_style()
            self._rerender_current_document()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Paint quote accents without inserting decorative copied text."""

        super().paintEvent(event)
        if not self._quote_ranges:
            return

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        accent = self.palette().color(QPalette.ColorRole.Link)
        pen = QPen(accent)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        document_layout = self.document().documentLayout()
        vertical_offset = self.verticalScrollBar().value()
        horizontal_offset = self.horizontalScrollBar().value()
        document_margin = self.document().documentMargin()

        for start_position, end_position, _quote_level in self._quote_ranges:
            start_block = self.document().findBlock(start_position)
            end_block = self.document().findBlock(end_position)
            if not start_block.isValid() or not end_block.isValid():
                continue
            start_rect = document_layout.blockBoundingRect(start_block)
            end_rect = document_layout.blockBoundingRect(end_block)
            top = start_rect.top() - vertical_offset
            bottom = end_rect.bottom() - vertical_offset
            if bottom < 0 or top > self.viewport().height():
                continue

            block_format = start_block.blockFormat()
            line_x = (
                document_margin
                + block_format.leftMargin()
                - 8.0
                - horizontal_offset
            )
            painter.drawLine(
                round(line_x),
                round(max(0.0, top + 2.0)),
                round(line_x),
                round(min(float(self.viewport().height()), bottom - 2.0)),
            )

            if top >= -24.0:
                quote_font = painter.font()
                quote_font.setBold(True)
                quote_font.setPointSizeF(max(12.0, quote_font.pointSizeF() + 3.0))
                painter.setFont(quote_font)
                marker_x = (
                    self.viewport().width()
                    - document_margin
                    - block_format.rightMargin()
                    + 5.0
                )
                marker_y = top + painter.fontMetrics().ascent() + 2.0
                painter.drawText(round(marker_x), round(marker_y), "”")

        painter.end()

    def _activate_link(self, url: QUrl) -> None:
        if self.current_file is None:
            return
        resolution = resolve_link(url, self.current_file, self.allowed_root)
        if resolution.kind == "anchor":
            self.anchorLinkActivated.emit(resolution.anchor)
            return
        if resolution.kind == "markdown" and resolution.path is not None:
            if not resolution.path.is_file():
                self.warningRequested.emit(
                    "Ссылка не найдена",
                    f"Markdown-файл не найден:\n{resolution.path}",
                )
                return
            self.markdownLinkActivated.emit(str(resolution.path), resolution.anchor)
            return
        if resolution.kind == "external" and resolution.url is not None:
            if not QDesktopServices.openUrl(resolution.url):
                self.warningRequested.emit(
                    "Не удалось открыть ссылку",
                    resolution.url.toString(),
                )
            return
        if resolution.kind == "local" and resolution.path is not None:
            if not resolution.path.exists():
                self.warningRequested.emit(
                    "Ссылка не найдена",
                    f"Файл или папка не найдены:\n{resolution.path}",
                )
                return
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolution.path))):
                self.warningRequested.emit(
                    "Не удалось открыть ссылку",
                    str(resolution.path),
                )
            return
        self.warningRequested.emit("Ссылка заблокирована", resolution.error)

    def _index_heading_blocks(self) -> None:
        self._heading_blocks.clear()
        slug_counts: dict[str, int] = {}
        block = self.document().begin()
        while block.isValid():
            if block.blockFormat().headingLevel() > 0:
                base_slug = _heading_slug(block.text())
                occurrence = slug_counts.get(base_slug, 0)
                slug_counts[base_slug] = occurrence + 1
                slug = base_slug if occurrence == 0 else f"{base_slug}-{occurrence}"
                self._heading_blocks[slug.casefold()] = block
            block = block.next()

    def _report_resource_warning(self, message: str) -> None:
        if message not in self._reported_resources:
            self._reported_resources.add(message)
            self.resourceWarning.emit(message)

    def _rerender_current_document(self) -> None:
        if not self._current_markdown or self.current_file is None:
            return
        scroll_position = self.verticalScrollBar().value()
        self.document().setBaseUrl(
            QUrl.fromLocalFile(str(self.current_file.parent) + os.sep)
        )
        self.document().setMarkdown(self._current_markdown, MARKDOWN_FEATURES)
        self._apply_heading_block_margins()
        self._apply_quote_block_formatting()
        self._index_heading_blocks()
        QTimer.singleShot(
            0,
            lambda: self.verticalScrollBar().setValue(scroll_position),
        )

    def _apply_document_style(self) -> None:
        palette = self.palette()
        color = lambda role: palette.color(role).name()  # noqa: E731
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        self.document().setDefaultStyleSheet(
            f"""
            body {{ color: {color(QPalette.ColorRole.Text)};
                    background-color: {color(QPalette.ColorRole.Base)}; }}
            p, li {{ margin-top: 3px; margin-bottom: 3px; }}
            a {{ color: {color(QPalette.ColorRole.Link)}; text-decoration: underline; }}
            code {{ font-family: \"{fixed_font}\";
                    background-color: {color(QPalette.ColorRole.AlternateBase)}; }}
            pre {{ font-family: \"{fixed_font}\";
                   background-color: {color(QPalette.ColorRole.AlternateBase)};
                   border: 1px solid {color(QPalette.ColorRole.Mid)};
                   padding: 8px; white-space: pre-wrap; }}
            table {{ border-collapse: collapse; margin-top: 6px; margin-bottom: 8px; }}
            th {{ background-color: {color(QPalette.ColorRole.AlternateBase)};
                  border: 1px solid {color(QPalette.ColorRole.Mid)}; padding: 5px; }}
            td {{ border: 1px solid {color(QPalette.ColorRole.Mid)}; padding: 5px; }}
            hr {{ background-color: {color(QPalette.ColorRole.Mid)}; height: 1px; }}
            """
        )

    def _apply_heading_block_margins(self) -> None:
        """Apply spacing that Qt ignores in CSS after ``setMarkdown()``."""

        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        block = self.document().begin()
        while block.isValid():
            block_format = block.blockFormat()
            margins = _HEADING_BLOCK_MARGINS.get(block_format.headingLevel())
            if margins is not None:
                block_format.setTopMargin(margins[0])
                block_format.setBottomMargin(margins[1])
                cursor.setPosition(block.position())
                cursor.setBlockFormat(block_format)
            block = block.next()
        cursor.endEditBlock()

    def _apply_quote_block_formatting(self) -> None:
        """Style consecutive Markdown quote blocks using Qt block metadata."""

        quote_property = QTextFormat.Property.BlockQuoteLevel
        blocks: list[tuple[object, int]] = []
        block = self.document().begin()
        while block.isValid():
            level = int(block.blockFormat().property(quote_property) or 0)
            blocks.append((block, level))
            block = block.next()

        palette = self.palette()
        base = palette.color(QPalette.ColorRole.Base)
        accent = palette.color(QPalette.ColorRole.Link)
        blend_factor = 0.10 if base.lightnessF() >= 0.5 else 0.18
        background = _blend_color(base, accent, blend_factor)

        self._quote_ranges.clear()
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        group_start: int | None = None
        group_level = 0
        for index, (quote_block, level) in enumerate(blocks):
            if level <= 0:
                continue
            previous_level = blocks[index - 1][1] if index > 0 else 0
            next_level = blocks[index + 1][1] if index + 1 < len(blocks) else 0
            starts_group = previous_level != level
            ends_group = next_level != level

            block_format = quote_block.blockFormat()
            block_format.setLeftMargin(
                _QUOTE_LEFT_MARGIN + max(0, level - 1) * 14.0
            )
            block_format.setRightMargin(_QUOTE_RIGHT_MARGIN)
            block_format.setTopMargin(
                _QUOTE_OUTER_MARGIN if starts_group else _QUOTE_INNER_MARGIN
            )
            block_format.setBottomMargin(
                _QUOTE_OUTER_MARGIN if ends_group else _QUOTE_INNER_MARGIN
            )
            block_format.setBackground(background)
            cursor.setPosition(quote_block.position())
            cursor.setBlockFormat(block_format)

            if starts_group:
                group_start = quote_block.position()
                group_level = level
            if ends_group and group_start is not None:
                self._quote_ranges.append(
                    (group_start, quote_block.position(), group_level)
                )
                group_start = None
        cursor.endEditBlock()
        self.viewport().update()


class MarkdownHelpDialog(QDialog):
    """Dialog that owns Markdown page history and user-facing diagnostics."""

    def __init__(
        self,
        manual_path: pathlib.Path | str,
        title: str = "Руководство",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        initial_path = pathlib.Path(manual_path).resolve(strict=False)
        self._base_title = title
        self._history: list[_HistoryEntry] = []
        self._history_index = -1

        self.setMinimumSize(700, 500)
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.back_button = QPushButton("Назад")
        self.forward_button = QPushButton("Вперёд")
        self.open_source_button = QPushButton("Открыть исходный файл")
        self.path_label = QLabel()
        self.path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        toolbar.addWidget(self.back_button)
        toolbar.addWidget(self.forward_button)
        toolbar.addWidget(self.open_source_button)
        toolbar.addWidget(self.path_label, 1)
        layout.addLayout(toolbar)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        self.browser = MarkdownBrowser(initial_path.parent, self)
        layout.addWidget(self.browser, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.back_button.clicked.connect(self.go_back)
        self.forward_button.clicked.connect(self.go_forward)
        self.open_source_button.clicked.connect(self.open_source_file)
        self.browser.markdownLinkActivated.connect(self._open_linked_markdown)
        self.browser.anchorLinkActivated.connect(self._open_current_anchor)
        self.browser.warningRequested.connect(self._show_warning)
        self.browser.resourceWarning.connect(self._show_resource_warning)

        self._navigate(initial_path, push_history=True)

    def go_back(self) -> None:
        """Return to the previous Markdown page and restore its scroll position."""

        if self._history_index <= 0:
            return
        self._save_current_scroll_position()
        target_index = self._history_index - 1
        if self._load_history_entry(target_index):
            self._history_index = target_index
        self._update_navigation_buttons()

    def go_forward(self) -> None:
        """Move to the next Markdown page in history."""

        if self._history_index >= len(self._history) - 1:
            return
        self._save_current_scroll_position()
        target_index = self._history_index + 1
        if self._load_history_entry(target_index):
            self._history_index = target_index
        self._update_navigation_buttons()

    def open_source_file(self) -> None:
        """Open the current Markdown file after an explicit button click."""

        if self.browser.current_file is None:
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.browser.current_file))):
            self._show_warning("Не удалось открыть файл", str(self.browser.current_file))

    def _open_linked_markdown(self, path: str, anchor: str) -> None:
        self._navigate(pathlib.Path(path), anchor=anchor, push_history=True)

    def _open_current_anchor(self, anchor: str) -> None:
        if self.browser.current_file is None:
            return
        self._save_current_scroll_position()
        if not self.browser.scroll_to_anchor(anchor):
            self._show_warning("Раздел не найден", f"Якорь не найден: #{anchor}")
            return

        del self._history[self._history_index + 1 :]
        self._history.append(
            _HistoryEntry(
                self.browser.current_file,
                self.browser.verticalScrollBar().value(),
                anchor,
            )
        )
        self._history_index = len(self._history) - 1
        self._update_navigation_buttons()

    def _navigate(
        self,
        path: pathlib.Path,
        anchor: str = "",
        push_history: bool = False,
    ) -> bool:
        self._save_current_scroll_position()
        if not self._load_page(path, anchor=anchor):
            self._update_navigation_buttons()
            return False
        if push_history:
            del self._history[self._history_index + 1 :]
            self._history.append(_HistoryEntry(path.resolve(strict=False), 0, anchor))
            self._history_index = len(self._history) - 1
        self._update_navigation_buttons()
        return True

    def _load_history_entry(self, index: int) -> bool:
        entry = self._history[index]
        current_file = self.browser.current_file
        if current_file is None or current_file != entry.path:
            if not self._load_page(entry.path):
                return False
        if entry.anchor and not entry.scroll_position:
            QTimer.singleShot(
                0,
                lambda: self.browser.scroll_to_anchor(entry.anchor),
            )
        else:
            self.browser.verticalScrollBar().setValue(entry.scroll_position)
        return True

    def _load_page(self, path: pathlib.Path, anchor: str = "") -> bool:
        self.status_label.setVisible(False)
        try:
            source_text = self.browser.load_markdown_file(path)
        except MarkdownIntegrityError as error:
            self.browser.show_source_text(error.source_text, path)
            self._set_page_metadata(path, error.source_text)
            self._show_resource_warning(str(error))
            return True
        except MarkdownFileError as error:
            if self.browser.current_file is None:
                self.browser.setPlainText(str(error))
                self.path_label.setText(str(path))
                self.setWindowTitle(self._base_title)
                self._show_resource_warning(str(error))
            else:
                self._show_warning("Не удалось открыть руководство", str(error))
            return False

        self._set_page_metadata(path, source_text)
        if anchor:
            QTimer.singleShot(0, lambda: self.browser.scroll_to_anchor(anchor))
        return True

    def _set_page_metadata(self, path: pathlib.Path, source_text: str) -> None:
        self.path_label.setText(str(path))
        self.path_label.setToolTip(str(path))
        heading = _first_heading(source_text) or path.stem
        self.setWindowTitle(f"{self._base_title} — {heading}")

    def _save_current_scroll_position(self) -> None:
        if 0 <= self._history_index < len(self._history):
            self._history[self._history_index].scroll_position = (
                self.browser.verticalScrollBar().value()
            )

    def _update_navigation_buttons(self) -> None:
        self.back_button.setEnabled(self._history_index > 0)
        self.forward_button.setEnabled(self._history_index < len(self._history) - 1)
        self.open_source_button.setEnabled(self.browser.current_file is not None)

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _show_resource_warning(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.setVisible(True)


def _first_heading(markdown: str) -> str:
    """Return the first ATX heading for a compact page title."""

    fenced = False
    fence_character = ""
    fence_length = 0
    for line in markdown.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fenced:
            if re.match(
                rf"^ {{0,3}}{re.escape(fence_character)}{{{fence_length},}}\s*$",
                line,
            ):
                fenced = False
            continue
        if fence_match:
            fence = fence_match.group("fence")
            fenced = True
            fence_character = fence[0]
            fence_length = len(fence)
            continue
        heading_match = re.match(r"^ {0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
        if heading_match:
            return heading_match.group(1)
    return ""
