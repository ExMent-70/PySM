"""GUI for test-mask sessions, cached generation and visual comparison."""

from __future__ import annotations

import html
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL.ImageQt import ImageQt
from PySide6.QtCore import (
    QPointF,
    QRectF,
    QSize,
    QEvent,
    QObject,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QBrush,
    QCloseEvent,
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QListWidgetItem,
)

from _common.config_schema import RmbgSettings
from _common.image_io import discover_images

from .preview_engine import generate_mask_set
from .preview_store import PreviewStore, TestMaskSet, TestSession
from .template_dialog import SECTION_LABELS, SETTING_LABELS
from .template_store import TemplateStore
from .window_state import RmbgWindowStateStore


class CreateSessionDialog(QDialog):
    """Select a stable source collection copied into one comparison session."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        window_state_store: RmbgWindowStateStore | None = None,
    ) -> None:
        super().__init__(parent)
        self._window_state_store = window_state_store
        self.setWindowTitle("Новая тестовая сессия")
        self.resize(620, 420)
        self.source_paths: tuple[Path, ...] = ()

        root = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Например: Портреты и волосы")
        form.addRow("Название сессии:", self.name_edit)
        root.addLayout(form)
        root.addWidget(
            QLabel(
                "Выбранные изображения будут один раз скопированы в Sources. "
                "Все тестовые наборы этой сессии создаются на одинаковых файлах."
            )
        )
        self.source_list = QListWidget()
        self.source_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        root.addWidget(self.source_list, 1)

        actions = QHBoxLayout()
        add_files = QPushButton("Добавить файлы…")
        add_folder = QPushButton("Добавить папку…")
        remove = QPushButton("Убрать выбранные")
        add_files.clicked.connect(self._add_files)
        add_folder.clicked.connect(self._add_folder)
        remove.clicked.connect(self._remove_selected)
        actions.addWidget(add_files)
        actions.addWidget(add_folder)
        actions.addWidget(remove)
        actions.addStretch(1)
        root.addLayout(actions)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Создать")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        if self._window_state_store is not None:
            self._window_state_store.restore("create_test_session", self)

    def done(self, result: int) -> None:
        if self._window_state_store is not None:
            self._window_state_store.save("create_test_session", self)
        super().done(result)

    def _add_paths(self, paths: tuple[Path, ...]) -> None:
        existing = {
            str(self.source_list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.source_list.count())
        }
        for path in paths:
            resolved = str(path.resolve())
            if resolved in existing:
                continue
            item = QListWidgetItem(path.name)
            item.setToolTip(resolved)
            item.setData(Qt.ItemDataRole.UserRole, resolved)
            self.source_list.addItem(item)
            existing.add(resolved)

    def _add_files(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(
            self,
            "Выберите тестовые изображения",
            "",
            "Изображения (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)",
        )
        self._add_paths(tuple(Path(path) for path in paths))

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку изображений")
        if folder:
            self._add_paths(discover_images(Path(folder), recursive=True))

    def _remove_selected(self) -> None:
        for item in self.source_list.selectedItems():
            self.source_list.takeItem(self.source_list.row(item))

    def _accept_if_valid(self) -> None:
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Тестовая сессия", "Введите название сессии.")
            return
        if not self.source_list.count():
            QMessageBox.warning(
                self,
                "Тестовая сессия",
                "Добавьте хотя бы одно тестовое изображение.",
            )
            return
        self.source_paths = tuple(
            Path(str(self.source_list.item(index).data(Qt.ItemDataRole.UserRole)))
            for index in range(self.source_list.count())
        )
        self.accept()


class _PreviewWorker(QObject):
    progress = Signal(str, int, int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, job: dict, cancel_event: threading.Event) -> None:
        super().__init__()
        self.job = job
        self.cancel_event = cancel_event

    @Slot()
    def run(self) -> None:
        try:
            result = generate_mask_set(
                **self.job,
                progress=lambda message, value, total: self.progress.emit(
                    message,
                    value,
                    total,
                ),
                cancel_event=self.cancel_event,
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.completed.emit(result)


class ComparisonCanvas(QWidget):
    """Smooth cached A/B viewer with cursor-centered zoom, pan and wipe."""

    zoom_changed = Signal(int)
    view_changed = Signal(float, float, float)
    SPLIT_MODES = {"mask", "cutout"}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._image_a: QImage | None = None
        self._image_b: QImage | None = None
        self._mode = "mask"
        self._label_a = "A"
        self._label_b = "B"
        self._zoom_factor = 1.0
        self._pan = QPointF()
        self._split = 0.5
        self._blink_side: str | None = None
        self._drag_mode: str | None = None
        self._last_mouse = QPointF()
        self._empty_message = "Нет изображения"

    def sizeHint(self) -> QSize:
        return QSize(900, 620)

    @property
    def split(self) -> float:
        return self._split

    @property
    def zoom_percent(self) -> int:
        return round(self._scale() * 100.0)

    @property
    def pan_offset(self) -> QPointF:
        return QPointF(self._pan)

    @property
    def view_state(self) -> tuple[float, float, float]:
        if self._image_a is None:
            return 1.0, 0.0, 0.0
        scale = self._scale()
        target = self._target_rect(scale)
        return (
            scale,
            (self.width() / 2.0 - target.left()) / scale,
            (self.height() / 2.0 - target.top()) / scale,
        )

    def set_comparison(
        self,
        image_a: QImage,
        image_b: QImage | None,
        *,
        mode: str,
        label_a: str,
        label_b: str,
        reset_view: bool = False,
    ) -> None:
        self._image_a = image_a
        self._image_b = image_b
        self._mode = mode
        self._label_a = label_a
        self._label_b = label_b
        if reset_view:
            self.fit_to_view()
        else:
            self._clamp_pan()
            self._emit_zoom()
            self.update()

    def clear_image(self, message: str = "Нет изображения") -> None:
        """Clear stale content and show a short problem description."""

        self._image_a = None
        self._image_b = None
        self._pan = QPointF()
        self._empty_message = message
        self.update()

    def fit_to_view(self) -> None:
        self._zoom_factor = 1.0
        self._pan = QPointF()
        self._emit_zoom()
        self.update()
        self._emit_view()

    def show_actual_size(self) -> None:
        fit_scale = self._fit_scale()
        self._zoom_factor = 1.0 / fit_scale if fit_scale > 0 else 1.0
        self._pan = QPointF()
        self._clamp_pan()
        self._emit_zoom()
        self.update()
        self._emit_view()

    def set_view_state(self, scale: float, center_x: float, center_y: float) -> None:
        """Apply an absolute image scale and center without emitting a sync loop."""

        if self._image_a is None or scale <= 0:
            return
        self._zoom_factor = scale / max(self._fit_scale(), 1e-6)
        centered = self._centered_top_left(scale)
        desired = QPointF(
            self.width() / 2.0 - center_x * scale,
            self.height() / 2.0 - center_y * scale,
        )
        self._pan = desired - centered
        self._clamp_pan()
        self._emit_zoom()
        self.update()

    def set_blink_side(self, side: str | None) -> None:
        self._blink_side = side if side in {"a", "b"} else None
        self.update()

    def zoom_at(self, position: QPointF, factor: float) -> None:
        if self._image_a is None or factor <= 0:
            return
        old_scale = self._scale()
        old_target = self._target_rect(old_scale)
        image_point = QPointF(
            (position.x() - old_target.left()) / old_scale,
            (position.y() - old_target.top()) / old_scale,
        )
        min_factor = max(0.05 / max(self._fit_scale(), 1e-6), 0.1)
        max_factor = max(16.0 / max(self._fit_scale(), 1e-6), min_factor)
        self._zoom_factor = min(
            max(self._zoom_factor * factor, min_factor),
            max_factor,
        )
        new_scale = self._scale()
        centered = self._centered_top_left(new_scale)
        desired = QPointF(
            position.x() - image_point.x() * new_scale,
            position.y() - image_point.y() * new_scale,
        )
        self._pan = desired - centered
        self._clamp_pan()
        self._emit_zoom()
        self.update()
        self._emit_view()

    def pan_by(self, delta: QPointF) -> None:
        self._pan += delta
        self._clamp_pan()
        self.update()
        self._emit_view()

    def set_split_from_view_x(self, x: float) -> None:
        if self._image_a is None:
            return
        target = self._target_rect()
        if target.width() <= 0:
            return
        self._split = min(max((x - target.left()) / target.width(), 0.0), 1.0)
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.zoom_at(event.position(), 1.18 ** steps)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in {Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton}:
            self.setFocus()
            self._last_mouse = event.position()
            if (
                event.button() == Qt.MouseButton.LeftButton
                and self._is_over_divider(event.position())
            ):
                self._drag_mode = "split"
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self._drag_mode = "pan"
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode == "split":
            self.set_split_from_view_x(event.position().x())
            event.accept()
            return
        if self._drag_mode == "pan":
            delta = event.position() - self._last_mouse
            self._last_mouse = event.position()
            self.pan_by(delta)
            event.accept()
            return
        self.setCursor(
            Qt.CursorShape.SizeHorCursor
            if self._is_over_divider(event.position())
            else Qt.CursorShape.OpenHandCursor
        )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode is not None:
            self._drag_mode = None
            self.setCursor(
                Qt.CursorShape.SizeHorCursor
                if self._is_over_divider(event.position())
                else Qt.CursorShape.OpenHandCursor
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if abs(self._scale() - 1.0) <= 0.02:
            self.fit_to_view()
        else:
            self.show_actual_size()
        event.accept()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#202124"))
        if self._image_a is None:
            painter.setPen(QColor("#D0D0D0"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._empty_message,
            )
            return

        target = self._target_rect()
        if (
            self._mode in self.SPLIT_MODES
            and self._image_b is not None
            and self._blink_side is not None
        ):
            painter.drawImage(
                target,
                self._image_a if self._blink_side == "a" else self._image_b,
            )
        elif self._mode in self.SPLIT_MODES and self._image_b is not None:
            divider = target.left() + target.width() * self._split
            painter.save()
            painter.setClipRect(
                QRectF(target.left(), target.top(), divider - target.left(), target.height())
            )
            painter.drawImage(target, self._image_a)
            painter.restore()
            painter.save()
            painter.setClipRect(
                QRectF(divider, target.top(), target.right() - divider, target.height())
            )
            painter.drawImage(target, self._image_b)
            painter.restore()
            self._draw_divider(painter, target, divider)
        else:
            painter.drawImage(target, self._image_a)
        self._draw_legend(painter)

    def resizeEvent(self, event) -> None:
        self._clamp_pan()
        self._emit_zoom()
        super().resizeEvent(event)

    def _fit_scale(self) -> float:
        if self._image_a is None or self._image_a.isNull():
            return 1.0
        margin = 16.0
        return max(
            0.001,
            min(
                max(1.0, self.width() - margin * 2) / self._image_a.width(),
                max(1.0, self.height() - margin * 2) / self._image_a.height(),
                1.0,
            ),
        )

    def _scale(self) -> float:
        return self._fit_scale() * self._zoom_factor

    def _centered_top_left(self, scale: float) -> QPointF:
        assert self._image_a is not None
        return QPointF(
            (self.width() - self._image_a.width() * scale) / 2.0,
            (self.height() - self._image_a.height() * scale) / 2.0,
        )

    def _target_rect(self, scale: float | None = None) -> QRectF:
        assert self._image_a is not None
        actual_scale = self._scale() if scale is None else scale
        top_left = self._centered_top_left(actual_scale) + self._pan
        return QRectF(
            top_left,
            QSize(
                max(1, round(self._image_a.width() * actual_scale)),
                max(1, round(self._image_a.height() * actual_scale)),
            ),
        )

    def _clamp_pan(self) -> None:
        if self._image_a is None:
            self._pan = QPointF()
            return
        scale = self._scale()
        overflow_x = max(0.0, (self._image_a.width() * scale - self.width()) / 2.0)
        overflow_y = max(0.0, (self._image_a.height() * scale - self.height()) / 2.0)
        self._pan.setX(min(max(self._pan.x(), -overflow_x), overflow_x))
        self._pan.setY(min(max(self._pan.y(), -overflow_y), overflow_y))

    def _is_over_divider(self, position: QPointF) -> bool:
        if self._image_a is None or self._mode not in self.SPLIT_MODES:
            return False
        target = self._target_rect()
        divider = target.left() + target.width() * self._split
        return (
            target.top() <= position.y() <= target.bottom()
            and abs(position.x() - divider) <= 10.0
        )

    def _draw_divider(self, painter: QPainter, target: QRectF, x: float) -> None:
        visible_top = max(target.top(), 0.0)
        visible_bottom = min(target.bottom(), float(self.height()))
        painter.setPen(QPen(QColor("#00B7FF"), 2.0))
        painter.drawLine(QPointF(x, visible_top), QPointF(x, visible_bottom))

    def _draw_legend(self, painter: QPainter) -> None:
        margin = 12.0
        max_chip_width = max(120, int(self.width() * 0.44))
        if self._mode in self.SPLIT_MODES:
            text_a = self._elide_chip_text(
                painter,
                f"A · {self._label_a}",
                max_chip_width,
            )
            self._draw_chip(painter, margin, margin, text_a, QColor("#00D5FF"))
            text = self._elide_chip_text(
                painter,
                f"B · {self._label_b}",
                max_chip_width,
            )
            width = self._chip_width(painter, text)
            self._draw_chip(
                painter,
                max(margin, self.width() - width - margin),
                margin,
                text,
                QColor("#FF4FD8"),
            )
            if self._blink_side is not None:
                blink_text = f"BLINK · {self._blink_side.upper()}"
                blink_width = self._chip_width(painter, blink_text)
                blink_color = (
                    QColor("#00D5FF")
                    if self._blink_side == "a"
                    else QColor("#FF4FD8")
                )
                self._draw_chip(
                    painter,
                    max(margin, (self.width() - blink_width) / 2.0),
                    margin + 36,
                    blink_text,
                    blink_color,
                )
            return

        text_a = self._elide_chip_text(
            painter,
            f"A · {self._label_a}",
            max_chip_width,
        )
        text_b = self._elide_chip_text(
            painter,
            f"B · {self._label_b}",
            max_chip_width,
        )
        self._draw_chip(painter, margin, margin, text_a, QColor("#00D5FF"))
        width_b = self._chip_width(painter, text_b)
        self._draw_chip(
            painter,
            max(margin, self.width() - width_b - margin),
            margin,
            text_b,
            QColor("#FF4FD8"),
        )
        explanation = (
            "Жёлтый · совпадение контуров"
            if self._mode == "contours"
            else "Яркость цвета · величина различия alpha"
        )
        self._draw_chip(
            painter,
            margin,
            margin + 36,
            explanation,
            QColor("#FFE45C"),
        )

    @staticmethod
    def _chip_width(painter: QPainter, text: str) -> float:
        return float(painter.fontMetrics().horizontalAdvance(text) + 20)

    @staticmethod
    def _elide_chip_text(painter: QPainter, text: str, max_width: int) -> str:
        return painter.fontMetrics().elidedText(
            text,
            Qt.TextElideMode.ElideMiddle,
            max(20, max_width - 20),
        )

    def _draw_chip(
        self,
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        color: QColor,
    ) -> None:
        rect = QRectF(x, y, self._chip_width(painter, text), 28)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(QBrush(QColor(20, 20, 20, 220)))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(color)
        painter.drawText(rect.adjusted(10, 0, -10, 0), Qt.AlignmentFlag.AlignVCenter, text)

    def _emit_zoom(self) -> None:
        self.zoom_changed.emit(self.zoom_percent)

    def _emit_view(self) -> None:
        scale, center_x, center_y = self.view_state
        self.view_changed.emit(scale, center_x, center_y)


class DetailedComparisonDialog(QDialog):
    """Compare two arbitrary sets using a wipe, difference or contour view."""

    def __init__(
        self,
        session: TestSession,
        mask_sets: tuple[TestMaskSet, ...],
        *,
        window_state_store: RmbgWindowStateStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._window_state_store = window_state_store
        self.session = session
        self.mask_sets = mask_sets
        self._source_cache: dict[str, Image.Image] = {}
        self._mask_cache: dict[tuple[str, str], np.ndarray] = {}
        self._view_cache: dict[tuple[str, str, str, bool, str], QImage] = {}
        self._comparison_cache: dict[tuple[str, str, str, str], QImage] = {}
        self._active_source_id: str | None = None
        self._custom_background = QColor("#808080")
        self._blink_side = "a"
        self._space_blink = False
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(450)
        self.setWindowTitle("Детальное сравнение масок A/B")
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.resize(1220, 860)

        root = QVBoxLayout(self)
        self.source_combo = QComboBox()
        for source in session.sources:
            self.source_combo.addItem(source.filename, source.source_id)
        self.set_a = QComboBox()
        self.set_b = QComboBox()
        for item in mask_sets:
            label = f"{item.number:03d} — {item.name}"
            self.set_a.addItem(label, item.set_id)
            self.set_b.addItem(label, item.set_id)
        if self.set_b.count() > 1:
            self.set_b.setCurrentIndex(1)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Маска A/B", "mask")
        self.mode_combo.addItem("Cutout A/B", "cutout")
        self.mode_combo.addItem("Карта различий", "difference")
        self.mode_combo.addItem("Цветные контуры A/B", "contours")
        for combo in (self.source_combo, self.set_a, self.set_b, self.mode_combo):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(16)
        self.source_combo.setMinimumContentsLength(12)
        self.mode_combo.setMinimumContentsLength(26)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Изображение:"))
        source_row.addWidget(self.source_combo, 1)
        source_row.addWidget(QLabel("Вид:"))
        source_row.addWidget(self.mode_combo, 2)
        root.addLayout(source_row)

        sets_row = QHBoxLayout()
        sets_row.addWidget(QLabel("A:"))
        sets_row.addWidget(self.set_a, 1)
        self.swap_button = QPushButton("A ↔ B")
        sets_row.addWidget(self.swap_button)
        sets_row.addWidget(QLabel("B:"))
        sets_row.addWidget(self.set_b, 1)
        root.addLayout(sets_row)

        toolbar = QHBoxLayout()
        self.fit_button = QPushButton("Вписать")
        self.actual_size_button = QPushButton("100%")
        self.zoom_label = QLabel("Масштаб: —")
        self.source_under_mask = QCheckBox("Исходник под масками")
        self.blink_button = QPushButton("Blink A/B")
        self.blink_button.setCheckable(True)
        self.background_combo = QComboBox()
        for label, value in (
            ("Шахматный", "checker"),
            ("Белый", "white"),
            ("Чёрный", "black"),
            ("Серый", "gray"),
            ("Зелёный", "green"),
            ("Пурпурный", "magenta"),
            ("Пользовательский", "custom"),
        ):
            self.background_combo.addItem(label, value)
        self.background_color_button = QPushButton("Цвет…")
        self.swap_button.setToolTip("Поменять наборы A и B местами.")
        self.fit_button.setToolTip("Вписать изображение целиком в область просмотра.")
        self.actual_size_button.setToolTip("Показать один пиксель изображения одним пикселем экрана.")
        self.source_under_mask.setToolTip(
            "В режиме «Маска A/B» показывает исходное изображение под "
            "полупрозрачной красной маской.\n"
            "Отключите параметр, чтобы оценивать чистую чёрно-белую маску.\n"
            "В остальных режимах исходник уже входит в представление."
        )
        self.blink_button.setToolTip(
            "Попеременно показывает A и B на всём холсте.\n"
            "Можно также удерживать пробел; после отпускания разделитель вернётся."
        )
        self.background_combo.setToolTip(
            "Фон Cutout для поиска светлых, тёмных и цветных ореолов."
        )
        toolbar.addWidget(self.fit_button)
        toolbar.addWidget(self.actual_size_button)
        toolbar.addWidget(self.zoom_label)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        display_options = QHBoxLayout()
        display_options.addWidget(self.blink_button)
        display_options.addWidget(self.source_under_mask)
        display_options.addWidget(QLabel("Фон Cutout:"))
        display_options.addWidget(self.background_combo)
        display_options.addWidget(self.background_color_button)
        display_options.addStretch(1)
        root.addLayout(display_options)

        self.navigation_help = QLabel(
            "Колесо — масштаб относительно курсора\n"
            "Перетаскивание — перемещение; голубая линия — сравнение A/B\n"
            "Двойной щелчок — 100%/вписать"
        )
        self.navigation_help.setWordWrap(True)
        root.addWidget(self.navigation_help)

        self.mode_help = QLabel()
        self.mode_help.setWordWrap(True)
        root.addWidget(self.mode_help)
        self.canvas = ComparisonCanvas()
        self.canvas.setToolTip(
            "Колесо мыши изменяет масштаб относительно курсора.\n"
            "Перетаскивание изображения перемещает увеличенный фрагмент.\n"
            "Перетаскивание голубой линии сравнивает варианты A и B.\n"
            "Двойной щелчок переключает 100% и вписывание в окно."
        )
        root.addWidget(self.canvas, 1)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        self.metrics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.metrics)

        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        close.rejected.connect(self.reject)
        root.addWidget(close)
        for control in (self.source_combo, self.set_a, self.set_b, self.mode_combo):
            control.currentIndexChanged.connect(self._render)
        self.swap_button.clicked.connect(self._swap_sets)
        self.fit_button.clicked.connect(self.canvas.fit_to_view)
        self.actual_size_button.clicked.connect(self.canvas.show_actual_size)
        self.source_under_mask.toggled.connect(self._render)
        self.background_combo.currentIndexChanged.connect(self._background_changed)
        self.background_color_button.clicked.connect(self._choose_background_color)
        self.blink_button.toggled.connect(self._blink_toggled)
        self._blink_timer.timeout.connect(self._advance_blink)
        self.canvas.zoom_changed.connect(
            lambda value: self.zoom_label.setText(f"Масштаб: {value}%")
        )
        for widget in (self, *self.findChildren(QWidget)):
            widget.installEventFilter(self)
        self._update_background_button()
        self._render()
        if self._window_state_store is not None:
            self._window_state_store.restore("detailed_comparison", self)

    def _set_by_id(self, set_id: str) -> TestMaskSet:
        return next(item for item in self.mask_sets if item.set_id == set_id)

    def _source_by_id(self, source_id: str):
        return next(item for item in self.session.sources if item.source_id == source_id)

    def _mask_path(self, mask_set: TestMaskSet, source_id: str) -> Path:
        record = next(
            item for item in mask_set.source_masks if item["source_id"] == source_id
        )
        return mask_set.path / record["mask"]

    def _source_image(self, source_id: str) -> Image.Image:
        cached = self._source_cache.get(source_id)
        if cached is None:
            source = self._source_by_id(source_id)
            cached = _load_source_image(self.session.path / source.relative_path)
            self._source_cache[source_id] = cached
        return cached

    def _mask(self, mask_set: TestMaskSet, source_id: str) -> np.ndarray:
        key = (mask_set.set_id, source_id)
        cached = self._mask_cache.get(key)
        if cached is None:
            cached = _load_mask(
                self._mask_path(mask_set, source_id),
                self._source_image(source_id).size,
            )
            self._mask_cache[key] = cached
        return cached

    @staticmethod
    def _to_qimage(image: Image.Image) -> QImage:
        return QImage(ImageQt(image)).copy()

    def _background_key(self) -> str:
        mode = str(self.background_combo.currentData())
        return f"custom:{self._custom_background.name()}" if mode == "custom" else mode

    def _view_image(
        self,
        source_id: str,
        mask_set: TestMaskSet,
        mode: str,
    ) -> QImage:
        show_source = mode == "mask" and self.source_under_mask.isChecked()
        background_key = self._background_key() if mode == "cutout" else ""
        key = (source_id, mask_set.set_id, mode, show_source, background_key)
        cached = self._view_cache.get(key)
        if cached is None:
            mask = self._mask(mask_set, source_id)
            if show_source:
                rendered = _mask_overlay_image(self._source_image(source_id), mask)
            elif mode == "mask":
                rendered = _mask_image(mask)
            else:
                rendered = _cutout_image(
                    self._source_image(source_id),
                    mask,
                    background=str(self.background_combo.currentData()),
                    custom_color=self._custom_background.getRgb()[:3],
                )
            cached = self._to_qimage(rendered)
            self._view_cache[key] = cached
        return cached

    def _comparison_image(
        self,
        source_id: str,
        set_a: TestMaskSet,
        set_b: TestMaskSet,
        mode: str,
    ) -> QImage:
        key = (source_id, set_a.set_id, set_b.set_id, mode)
        cached = self._comparison_cache.get(key)
        if cached is None:
            rendered = _render_comparison(
                self._source_image(source_id),
                self._mask(set_a, source_id),
                self._mask(set_b, source_id),
                mode,
                0.5,
            )
            cached = self._to_qimage(rendered)
            self._comparison_cache[key] = cached
        return cached

    @Slot()
    def _swap_sets(self) -> None:
        index_a = self.set_a.currentIndex()
        index_b = self.set_b.currentIndex()
        self.set_a.blockSignals(True)
        self.set_b.blockSignals(True)
        self.set_a.setCurrentIndex(index_b)
        self.set_b.setCurrentIndex(index_a)
        self.set_a.blockSignals(False)
        self.set_b.blockSignals(False)
        self._render()

    @Slot()
    def _background_changed(self) -> None:
        self._update_background_button()
        self._render()

    @Slot()
    def _choose_background_color(self) -> None:
        selected = QColorDialog.getColor(
            self._custom_background,
            self,
            "Цвет фона Cutout",
        )
        if not selected.isValid():
            return
        self._custom_background = selected
        self._update_background_button()
        self._render()

    def _update_background_button(self) -> None:
        is_custom = self.background_combo.currentData() == "custom"
        self.background_color_button.setEnabled(
            is_custom and self.mode_combo.currentData() == "cutout"
        )
        self.background_color_button.setStyleSheet(
            f"background-color: {self._custom_background.name()};"
        )

    @Slot(bool)
    def _blink_toggled(self, enabled: bool) -> None:
        if enabled:
            if self.mode_combo.currentData() not in ComparisonCanvas.SPLIT_MODES:
                self.blink_button.setChecked(False)
                return
            self._start_blink()
        elif not self._space_blink:
            self._stop_blink()

    def _start_blink(self) -> None:
        if self.mode_combo.currentData() not in ComparisonCanvas.SPLIT_MODES:
            return
        self._blink_side = "a"
        self.canvas.set_blink_side(self._blink_side)
        self._blink_timer.start()

    @Slot()
    def _advance_blink(self) -> None:
        self._blink_side = "b" if self._blink_side == "a" else "a"
        self.canvas.set_blink_side(self._blink_side)

    def _stop_blink(self) -> None:
        self._blink_timer.stop()
        self.canvas.set_blink_side(None)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if isinstance(event, QKeyEvent) and event.key() == Qt.Key.Key_Space:
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                if self.mode_combo.currentData() in ComparisonCanvas.SPLIT_MODES:
                    self._space_blink = True
                    self._start_blink()
                    return True
            elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                self._space_blink = False
                if not self.blink_button.isChecked():
                    self._stop_blink()
                return True
        return super().eventFilter(watched, event)

    def done(self, result: int) -> None:
        self._space_blink = False
        self._stop_blink()
        if self._window_state_store is not None:
            self._window_state_store.save("detailed_comparison", self)
        super().done(result)

    @Slot()
    def _render(self) -> None:
        if not self.mask_sets or self.source_combo.currentData() is None:
            return
        source_id = str(self.source_combo.currentData())
        set_a = self._set_by_id(str(self.set_a.currentData()))
        set_b = self._set_by_id(str(self.set_b.currentData()))
        mask_a = self._mask(set_a, source_id)
        mask_b = self._mask(set_b, source_id)
        mode = str(self.mode_combo.currentData())
        self.source_under_mask.setEnabled(mode == "mask")
        self.background_combo.setEnabled(mode == "cutout")
        self._update_background_button()
        blink_available = mode in ComparisonCanvas.SPLIT_MODES
        self.blink_button.setEnabled(blink_available)
        if not blink_available:
            self._space_blink = False
            if self.blink_button.isChecked():
                self.blink_button.setChecked(False)
            else:
                self._stop_blink()
        label_a = f"{set_a.number:03d} — {set_a.name}"
        label_b = f"{set_b.number:03d} — {set_b.name}"
        split_mode = mode in ComparisonCanvas.SPLIT_MODES
        image_a = (
            self._view_image(source_id, set_a, mode)
            if split_mode
            else self._comparison_image(source_id, set_a, set_b, mode)
        )
        image_b = self._view_image(source_id, set_b, mode) if split_mode else None
        self.canvas.set_comparison(
            image_a,
            image_b,
            mode=mode,
            label_a=label_a,
            label_b=label_b,
            reset_view=source_id != self._active_source_id,
        )
        self._active_source_id = source_id
        if mode in ComparisonCanvas.SPLIT_MODES:
            source_note = ""
            if mode == "mask":
                source_note = (
                    " Исходник показан под полупрозрачной красной маской."
                    if self.source_under_mask.isChecked()
                    else " Показана чистая чёрно-белая маска."
                )
            self.mode_help.setText(
                f"<b>A — {label_a}</b> показан слева от линии; "
                f"<b>B — {label_b}</b> — справа. Перетаскивайте разделитель "
                f"непосредственно на изображении.{source_note}"
            )
        elif mode == "difference":
            self.mode_help.setText(
                f"Сравниваются alpha-маски <b>A — {label_a}</b> и "
                f"<b>B — {label_b}</b>. Циан означает alpha A &gt; B, "
                "пурпурный — alpha B &gt; A; яркость показывает величину различия."
            )
        else:
            self.mode_help.setText(
                f"Контур <b>A — {label_a}</b> показан цианом, контур "
                f"<b>B — {label_b}</b> — пурпурным, точное совпадение — жёлтым."
            )
        difference = np.abs(mask_a - mask_b)
        intersection = np.logical_and(mask_a >= 0.5, mask_b >= 0.5).sum()
        union = np.logical_or(mask_a >= 0.5, mask_b >= 0.5).sum()
        iou = float(intersection / union) if union else 1.0
        changed = float(np.mean(difference >= (1.0 / 255.0)) * 100.0)
        delta = mask_a - mask_b
        a_more = float(np.mean(delta >= (1.0 / 255.0)) * 100.0)
        b_more = float(np.mean(delta <= -(1.0 / 255.0)) * 100.0)
        self.metrics.setText(
            f"<b>A ↔ B:</b> {label_a} ↔ {label_b} &nbsp;|&nbsp; "
            f"<b>Средняя разница alpha:</b> {difference.mean():.4f} &nbsp;|&nbsp; "
            f"<b>Изменившиеся пиксели:</b> {changed:.2f}% &nbsp;|&nbsp; "
            f"<b>A больше:</b> {a_more:.2f}% &nbsp;|&nbsp; "
            f"<b>B больше:</b> {b_more:.2f}% &nbsp;|&nbsp; "
            f"<b>IoU при пороге 0,5:</b> {iou:.4f}"
        )


class OverviewCanvas(ComparisonCanvas):
    """Compact comparison canvas without per-image overlay legends."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 130)

    def sizeHint(self) -> QSize:
        return QSize(440, 330)

    def _draw_legend(self, _painter: QPainter) -> None:
        return


class SingleSetViewerDialog(QDialog):
    """Inspect every source, mask and cutout produced by one test set."""

    def __init__(
        self,
        session: TestSession,
        mask_set: TestMaskSet,
        *,
        window_state_store: RmbgWindowStateStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._window_state_store = window_state_store
        self.session = session
        self.mask_set = mask_set
        self._source_cache: dict[str, Image.Image] = {}
        self._mask_cache: dict[str, np.ndarray] = {}
        self._view_cache: dict[tuple[str, str, bool, str], QImage] = {}
        self._active_source_id: str | None = None
        self._last_load_error: tuple[str, str, str] | None = None
        self._custom_background = QColor("#808080")
        self.setWindowTitle(
            f"Просмотр тестового набора {mask_set.number:03d} — {mask_set.name}"
        )
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.resize(1100, 820)

        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.source_combo = QComboBox()
        for source in session.sources:
            self.source_combo.addItem(source.filename, source.source_id)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Маска", "mask")
        self.mode_combo.addItem("Cutout", "cutout")
        self.mode_combo.addItem("Исходное изображение", "source")
        for combo, length in ((self.source_combo, 12), (self.mode_combo, 24)):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(length)
        controls.addWidget(QLabel("Изображение:"))
        controls.addWidget(self.source_combo, 1)
        controls.addWidget(QLabel("Вид:"))
        controls.addWidget(self.mode_combo, 2)
        root.addLayout(controls)

        toolbar = QHBoxLayout()
        self.fit_button = QPushButton("Вписать")
        self.actual_size_button = QPushButton("100%")
        self.zoom_label = QLabel("Масштаб: —")
        toolbar.addWidget(self.fit_button)
        toolbar.addWidget(self.actual_size_button)
        toolbar.addWidget(self.zoom_label)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        display_options = QHBoxLayout()
        self.source_under_mask = QCheckBox("Исходник под маской")
        self.background_combo = QComboBox()
        for label, value in (
            ("Шахматный фон", "checker"),
            ("Белый", "white"),
            ("Чёрный", "black"),
            ("Серый", "gray"),
            ("Зелёный", "green"),
            ("Пурпурный", "magenta"),
            ("Пользовательский", "custom"),
        ):
            self.background_combo.addItem(label, value)
        self.background_color_button = QPushButton("Цвет…")
        display_options.addWidget(self.source_under_mask)
        display_options.addWidget(QLabel("Фон Cutout:"))
        display_options.addWidget(self.background_combo)
        display_options.addWidget(self.background_color_button)
        display_options.addStretch(1)
        root.addLayout(display_options)

        self.set_summary = QLabel(self._settings_summary())
        self.set_summary.setWordWrap(True)
        self.set_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.set_summary)

        self.navigation_help = QLabel(
            "Колесо — масштаб относительно курсора\n"
            "Перетаскивание — перемещение; двойной щелчок — 100%/вписать"
        )
        self.navigation_help.setWordWrap(True)
        root.addWidget(self.navigation_help)

        self.canvas = OverviewCanvas()
        self.canvas.setMinimumSize(480, 320)
        self.canvas.setToolTip(
            "Колесо мыши изменяет масштаб относительно курсора.\n"
            "Перетаскивание перемещает увеличенный фрагмент.\n"
            "Двойной щелчок переключает 100% и вписывание."
        )
        root.addWidget(self.canvas, 1)

        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        close.rejected.connect(self.reject)
        root.addWidget(close)

        self.fit_button.setToolTip("Вписать изображение целиком в область просмотра.")
        self.actual_size_button.setToolTip(
            "Показать один пиксель изображения одним пикселем экрана."
        )
        self.source_under_mask.setToolTip(
            "Показывает исходную фотографию под полупрозрачной красной маской.\n"
            "Отключите параметр для просмотра чистой чёрно-белой маски."
        )
        self.background_combo.setToolTip(
            "Подложка Cutout помогает обнаруживать светлые, тёмные и цветные ореолы."
        )
        self.source_combo.currentIndexChanged.connect(self._render)
        self.mode_combo.currentIndexChanged.connect(self._render)
        self.source_under_mask.toggled.connect(self._render)
        self.background_combo.currentIndexChanged.connect(self._background_changed)
        self.background_color_button.clicked.connect(self._choose_background_color)
        self.fit_button.clicked.connect(self.canvas.fit_to_view)
        self.actual_size_button.clicked.connect(self.canvas.show_actual_size)
        self.canvas.zoom_changed.connect(
            lambda value: self.zoom_label.setText(f"Масштаб: {value}%")
        )
        self._update_background_button()
        self._render()
        if self._window_state_store is not None:
            self._window_state_store.restore("single_set_viewer", self)

    def _settings_summary(self) -> str:
        settings = self.mask_set.settings
        model = html.escape(settings.resolved_model_name().value)
        refinement = html.escape(settings.resolved_refinement().value)
        name = html.escape(self.mask_set.name)
        resolution = settings.model.process_resolution or "auto"
        mask = settings.mask
        return (
            f"<b>Набор:</b> {self.mask_set.number:03d} — {name} &nbsp;|&nbsp; "
            f"<b>Модель:</b> {model} &nbsp;|&nbsp; "
            f"<b>Разрешение:</b> {resolution} &nbsp;|&nbsp; "
            f"<b>Refinement:</b> {refinement}<br>"
            f"<b>Маска:</b> чувствительность {mask.sensitivity:g}; "
            f"размытие {mask.blur}; смещение края {mask.offset}; "
            f"растушёвка {mask.feather}; "
            f"отверстия {'да' if mask.fill_holes else 'нет'}; "
            f"мелкие области {'да' if mask.remove_small_regions else 'нет'}; "
            f"инверсия {'да' if mask.invert else 'нет'}."
        )

    def _source_by_id(self, source_id: str):
        try:
            return next(
                item for item in self.session.sources if item.source_id == source_id
            )
        except StopIteration as exc:
            raise FileNotFoundError(
                "Выбранное изображение отсутствует в описании тестовой сессии."
            ) from exc

    def _mask_path(self, source_id: str) -> Path:
        try:
            record = next(
                item
                for item in self.mask_set.source_masks
                if item.get("source_id") == source_id
            )
        except StopIteration as exc:
            source = self._source_by_id(source_id)
            raise FileNotFoundError(
                f"В наборе нет маски для изображения «{source.filename}»."
            ) from exc
        relative_path = record.get("mask")
        if not isinstance(relative_path, str) or not relative_path:
            raise FileNotFoundError("В описании набора не указан файл маски.")
        return self.mask_set.path / relative_path

    def _source_image(self, source_id: str) -> Image.Image:
        cached = self._source_cache.get(source_id)
        if cached is not None:
            return cached
        source = self._source_by_id(source_id)
        path = self.session.path / source.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Исходное изображение не найдено: {path}")
        try:
            cached = _load_source_image(path)
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось прочитать исходное изображение «{source.filename}»: {exc}"
            ) from exc
        self._source_cache[source_id] = cached
        return cached

    def _mask(self, source_id: str) -> np.ndarray:
        cached = self._mask_cache.get(source_id)
        if cached is not None:
            return cached
        path = self._mask_path(source_id)
        if not path.is_file():
            raise FileNotFoundError(f"Файл маски не найден: {path}")
        try:
            cached = _load_mask(path, self._source_image(source_id).size)
        except Exception as exc:
            raise RuntimeError(f"Не удалось прочитать маску «{path.name}»: {exc}") from exc
        self._mask_cache[source_id] = cached
        return cached

    def _background_key(self) -> str:
        mode = str(self.background_combo.currentData())
        return f"custom:{self._custom_background.name()}" if mode == "custom" else mode

    def _view_image(self, source_id: str, mode: str) -> QImage:
        show_source = mode == "mask" and self.source_under_mask.isChecked()
        background_key = self._background_key() if mode == "cutout" else ""
        key = (source_id, mode, show_source, background_key)
        cached = self._view_cache.get(key)
        if cached is not None:
            return cached
        source = self._source_image(source_id)
        if mode == "source":
            rendered = source
        elif mode == "mask":
            mask = self._mask(source_id)
            rendered = _mask_overlay_image(source, mask) if show_source else _mask_image(mask)
        else:
            rendered = _cutout_image(
                source,
                self._mask(source_id),
                background=str(self.background_combo.currentData()),
                custom_color=self._custom_background.getRgb()[:3],
            )
        cached = QImage(ImageQt(rendered)).copy()
        self._view_cache[key] = cached
        return cached

    @Slot()
    def _render(self) -> None:
        source_data = self.source_combo.currentData()
        if source_data is None:
            self.canvas.clear_image("В тестовой сессии нет исходных изображений.")
            return
        source_id = str(source_data)
        mode = str(self.mode_combo.currentData())
        self.source_under_mask.setEnabled(mode == "mask")
        self.background_combo.setEnabled(mode == "cutout")
        self._update_background_button()
        try:
            image = self._view_image(source_id, mode)
        except Exception as exc:
            self.canvas.clear_image("Не удалось загрузить выбранный результат.")
            self.zoom_label.setText("Масштаб: —")
            error_key = (source_id, mode, str(exc))
            if error_key != self._last_load_error:
                self._last_load_error = error_key
                QMessageBox.warning(
                    self,
                    "Не удалось открыть результат набора",
                    f"Изображение: {self.source_combo.currentText()}\n"
                    f"Вид: {self.mode_combo.currentText()}\n\n{exc}",
                )
            return
        self._last_load_error = None
        self.canvas.set_comparison(
            image,
            None,
            mode="overview",
            label_a="",
            label_b="",
            reset_view=source_id != self._active_source_id,
        )
        self._active_source_id = source_id

    @Slot()
    def _background_changed(self) -> None:
        self._update_background_button()
        self._render()

    @Slot()
    def _choose_background_color(self) -> None:
        selected = QColorDialog.getColor(
            self._custom_background,
            self,
            "Цвет фона Cutout",
        )
        if not selected.isValid():
            return
        self._custom_background = selected
        self._update_background_button()
        self._render()

    def _update_background_button(self) -> None:
        is_custom = self.background_combo.currentData() == "custom"
        self.background_color_button.setEnabled(
            is_custom and self.mode_combo.currentData() == "cutout"
        )
        self.background_color_button.setStyleSheet(
            f"background-color: {self._custom_background.name()};"
        )

    def done(self, result: int) -> None:
        if self._window_state_store is not None:
            self._window_state_store.save("single_set_viewer", self)
        super().done(result)


class MultiSetOverviewDialog(QDialog):
    """Show two to four cached mask sets in a synchronized interactive grid."""

    def __init__(
        self,
        session: TestSession,
        mask_sets: tuple[TestMaskSet, ...],
        *,
        window_state_store: RmbgWindowStateStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._window_state_store = window_state_store
        self.session = session
        self.mask_sets = mask_sets
        self._source_cache: dict[str, Image.Image] = {}
        self._mask_cache: dict[tuple[str, str], np.ndarray] = {}
        self._view_cache: dict[tuple[str, str, str, bool, str], QImage] = {}
        self._active_source_id: str | None = None
        self._syncing_view = False
        self._custom_background = QColor("#808080")
        self.canvases: list[OverviewCanvas] = []
        self.panels: list[QWidget] = []
        self.setWindowTitle("Обзор нескольких наборов масок")
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.resize(1250, 850)
        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.source_combo = QComboBox()
        for source in session.sources:
            self.source_combo.addItem(source.filename, source.source_id)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Маска", "mask")
        self.mode_combo.addItem("Cutout", "cutout")
        for combo, length in ((self.source_combo, 12), (self.mode_combo, 24)):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(length)
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, len(mask_sets))
        self.columns_spin.setValue(min(2, len(mask_sets)))
        controls.addWidget(QLabel("Изображение:"))
        controls.addWidget(self.source_combo, 1)
        controls.addWidget(QLabel("Вид:"))
        controls.addWidget(self.mode_combo, 2)
        controls.addWidget(QLabel("Изображений в строке:"))
        controls.addWidget(self.columns_spin)
        root.addLayout(controls)

        toolbar = QHBoxLayout()
        self.fit_button = QPushButton("Вписать")
        self.actual_size_button = QPushButton("100%")
        self.zoom_label = QLabel("Масштаб: —")
        self.source_under_mask = QCheckBox("Исходник под масками")
        self.background_combo = QComboBox()
        for label, value in (
            ("Шахматный фон", "checker"),
            ("Белый", "white"),
            ("Чёрный", "black"),
            ("Серый", "gray"),
            ("Зелёный", "green"),
            ("Пурпурный", "magenta"),
            ("Пользовательский", "custom"),
        ):
            self.background_combo.addItem(label, value)
        self.background_color_button = QPushButton("Цвет…")
        toolbar.addWidget(self.fit_button)
        toolbar.addWidget(self.actual_size_button)
        toolbar.addWidget(self.zoom_label)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        display_options = QHBoxLayout()
        display_options.addWidget(self.source_under_mask)
        display_options.addWidget(QLabel("Фон Cutout:"))
        display_options.addWidget(self.background_combo)
        display_options.addWidget(self.background_color_button)
        display_options.addStretch(1)
        root.addLayout(display_options)

        self.navigation_help = QLabel(
            "Колесо над любой панелью — синхронный масштаб всех изображений\n"
            "Перетаскивание — синхронное перемещение; двойной щелчок — 100%/вписать"
        )
        self.navigation_help.setWordWrap(True)
        root.addWidget(self.navigation_help)

        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.grid_widget, 1)
        for mask_set in self.mask_sets:
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(3, 3, 3, 3)
            title = QLabel(f"{mask_set.number:03d} — {mask_set.name}")
            title.setStyleSheet("font-weight: bold;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            canvas = OverviewCanvas()
            canvas.setToolTip(
                "Колесо мыши синхронно масштабирует все варианты.\n"
                "Перетаскивание синхронно перемещает увеличенный фрагмент.\n"
                "Двойной щелчок переключает 100% и вписывание."
            )
            canvas.view_changed.connect(
                lambda scale, center_x, center_y, source=canvas: self._sync_view(
                    source,
                    scale,
                    center_x,
                    center_y,
                )
            )
            panel_layout.addWidget(title)
            panel_layout.addWidget(canvas, 1)
            self.panels.append(panel)
            self.canvases.append(canvas)

        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        close.rejected.connect(self.reject)
        root.addWidget(close)

        self.fit_button.setToolTip("Вписать изображения целиком во все панели.")
        self.actual_size_button.setToolTip(
            "Показать один пиксель изображения одним пикселем экрана во всех панелях."
        )
        self.source_under_mask.setToolTip(
            "Показывает исходную фотографию под полупрозрачной красной маской.\n"
            "Отключите параметр для просмотра чистых чёрно-белых масок."
        )
        self.background_combo.setToolTip(
            "Подложка Cutout помогает обнаруживать светлые, тёмные и цветные ореолы."
        )
        self.columns_spin.setToolTip(
            "Количество сравниваемых вариантов в одной строке. Содержимое кэша "
            "при перестроении не пересоздаётся."
        )
        self.source_combo.currentIndexChanged.connect(self._render)
        self.mode_combo.currentIndexChanged.connect(self._render)
        self.source_under_mask.toggled.connect(self._render)
        self.background_combo.currentIndexChanged.connect(self._background_changed)
        self.background_color_button.clicked.connect(self._choose_background_color)
        self.columns_spin.valueChanged.connect(self._rebuild_grid)
        self.fit_button.clicked.connect(self._fit_all)
        self.actual_size_button.clicked.connect(self._actual_size_all)
        self._update_background_button()
        self._rebuild_grid()
        self._render()
        if self._window_state_store is not None:
            self._window_state_store.restore("multi_set_overview", self)

    def _source_image(self, source_id: str) -> Image.Image:
        cached = self._source_cache.get(source_id)
        if cached is None:
            source = next(
                item for item in self.session.sources if item.source_id == source_id
            )
            cached = _load_source_image(self.session.path / source.relative_path)
            self._source_cache[source_id] = cached
        return cached

    def _mask(self, mask_set: TestMaskSet, source_id: str) -> np.ndarray:
        key = (mask_set.set_id, source_id)
        cached = self._mask_cache.get(key)
        if cached is None:
            record = next(
                item for item in mask_set.source_masks if item["source_id"] == source_id
            )
            cached = _load_mask(
                mask_set.path / record["mask"],
                self._source_image(source_id).size,
            )
            self._mask_cache[key] = cached
        return cached

    def _background_key(self) -> str:
        mode = str(self.background_combo.currentData())
        return f"custom:{self._custom_background.name()}" if mode == "custom" else mode

    def _view_image(self, mask_set: TestMaskSet, source_id: str) -> QImage:
        mode = str(self.mode_combo.currentData())
        show_source = mode == "mask" and self.source_under_mask.isChecked()
        background_key = self._background_key() if mode == "cutout" else ""
        key = (source_id, mask_set.set_id, mode, show_source, background_key)
        cached = self._view_cache.get(key)
        if cached is not None:
            return cached
        source = self._source_image(source_id)
        mask = self._mask(mask_set, source_id)
        if show_source:
            rendered = _mask_overlay_image(source, mask)
        elif mode == "mask":
            rendered = _mask_image(mask)
        else:
            rendered = _cutout_image(
                source,
                mask,
                background=str(self.background_combo.currentData()),
                custom_color=self._custom_background.getRgb()[:3],
            )
        cached = QImage(ImageQt(rendered)).copy()
        self._view_cache[key] = cached
        return cached

    @Slot()
    def _render(self) -> None:
        source_id = str(self.source_combo.currentData())
        mode = str(self.mode_combo.currentData())
        self.source_under_mask.setEnabled(mode == "mask")
        self.background_combo.setEnabled(mode == "cutout")
        self.background_color_button.setEnabled(
            mode == "cutout" and self.background_combo.currentData() == "custom"
        )
        reset_view = source_id != self._active_source_id
        for canvas, mask_set in zip(self.canvases, self.mask_sets, strict=True):
            canvas.set_comparison(
                self._view_image(mask_set, source_id),
                None,
                mode="overview",
                label_a="",
                label_b="",
                reset_view=False,
            )
        self._active_source_id = source_id
        if reset_view:
            QTimer.singleShot(0, self._fit_all)

    @Slot()
    def _rebuild_grid(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0)
        columns = self.columns_spin.value()
        for index, panel in enumerate(self.panels):
            self.grid.addWidget(panel, index // columns, index % columns)
        for column in range(len(self.panels)):
            self.grid.setColumnStretch(column, 0)
        for column in range(columns):
            self.grid.setColumnStretch(column, 1)
        QTimer.singleShot(0, self._fit_all)

    @Slot()
    def _fit_all(self) -> None:
        if not self.canvases:
            return
        self._syncing_view = True
        try:
            first = self.canvases[0]
            first.fit_to_view()
            state = first.view_state
            for canvas in self.canvases[1:]:
                canvas.set_view_state(*state)
        finally:
            self._syncing_view = False
        self.zoom_label.setText(f"Масштаб: {self.canvases[0].zoom_percent}%")

    @Slot()
    def _actual_size_all(self) -> None:
        if not self.canvases:
            return
        self.canvases[0].show_actual_size()

    @Slot()
    def _background_changed(self) -> None:
        self._update_background_button()
        self._render()

    @Slot()
    def _choose_background_color(self) -> None:
        selected = QColorDialog.getColor(
            self._custom_background,
            self,
            "Цвет фона Cutout",
        )
        if not selected.isValid():
            return
        self._custom_background = selected
        self._update_background_button()
        self._render()

    def _update_background_button(self) -> None:
        is_custom = self.background_combo.currentData() == "custom"
        self.background_color_button.setEnabled(
            is_custom and self.mode_combo.currentData() == "cutout"
        )
        self.background_color_button.setStyleSheet(
            f"background-color: {self._custom_background.name()};"
        )

    def _sync_view(
        self,
        source: OverviewCanvas,
        scale: float,
        center_x: float,
        center_y: float,
    ) -> None:
        if self._syncing_view:
            return
        self._syncing_view = True
        try:
            for canvas in self.canvases:
                if canvas is not source:
                    canvas.set_view_state(scale, center_x, center_y)
        finally:
            self._syncing_view = False
        self.zoom_label.setText(f"Масштаб: {round(scale * 100)}%")

    def done(self, result: int) -> None:
        if self._window_state_store is not None:
            self._window_state_store.save("multi_set_overview", self)
        super().done(result)


class TestMaskManagerDialog(QDialog):
    """Manage sessions and immutable test sets generated from current settings."""

    def __init__(
        self,
        store: PreviewStore,
        template_store: TemplateStore,
        settings_provider: Callable[[], RmbgSettings],
        settings_applier: Callable[[RmbgSettings], None],
        *,
        window_state_store: RmbgWindowStateStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.template_store = template_store
        self.settings_provider = settings_provider
        self.settings_applier = settings_applier
        self._window_state_store = window_state_store
        self._thread: QThread | None = None
        self._worker: _PreviewWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._progress_dialog: QProgressDialog | None = None
        self._generation_result: TestMaskSet | None = None
        self._generation_error: str | None = None
        self._generation_set_id: str | None = None
        self.setWindowTitle("Тестирование и сравнение масок RMBG")
        self.resize(1050, 680)

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.session_combo = QComboBox()
        self.new_session_button = QPushButton("Новая сессия…")
        self.delete_session_button = QPushButton("Удалить сессию")
        top.addWidget(QLabel("Тестовая сессия:"))
        top.addWidget(self.session_combo, 1)
        top.addWidget(self.new_session_button)
        top.addWidget(self.delete_session_button)
        root.addLayout(top)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.set_tree = QTreeWidget()
        self.set_tree.setHeaderLabels(
            ["№", "Название", "Модель", "Разрешение", "Refinement", "Статус"]
        )
        self.set_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.set_tree.setRootIsDecorated(False)
        self.main_splitter.addWidget(self.set_tree)

        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_caption = QLabel("Все настройки выбранного набора")
        self.settings_caption.setStyleSheet("font-weight: bold;")
        settings_layout.addWidget(self.settings_caption)
        self.settings_tree = QTreeWidget()
        self.settings_tree.setColumnCount(2)
        self.settings_tree.setHeaderLabels(["Параметр", "Значение"])
        self.settings_tree.setRootIsDecorated(True)
        self.settings_tree.setAlternatingRowColors(True)
        self.settings_tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.settings_tree.header().setStretchLastSection(True)
        self.settings_tree.header().resizeSection(0, 245)
        settings_layout.addWidget(self.settings_tree, 1)
        self.main_splitter.addWidget(settings_panel)
        self.main_splitter.setSizes([440, 610])
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 2)
        root.addWidget(self.main_splitter, 1)

        self.selection_summary = QLabel("Выберите тестовый набор.")
        root.addWidget(self.selection_summary)

        self.actions_layout = QHBoxLayout()
        self.set_actions_button = QPushButton("Набор")
        self.set_actions_menu = QMenu(self.set_actions_button)
        self.generate_action = self.set_actions_menu.addAction(
            "Создать из текущих настроек…"
        )
        self.set_actions_menu.addSeparator()
        self.view_set_action = self.set_actions_menu.addAction(
            "Просмотреть результаты выбранного набора"
        )
        self.apply_action = self.set_actions_menu.addAction(
            "Применить настройки выбранного набора"
        )
        self.template_action = self.set_actions_menu.addAction(
            "Сохранить выбранный набор как шаблон…"
        )
        self.set_actions_button.setMenu(self.set_actions_menu)
        self.set_actions_button.setToolTip(
            "Создание тестового набора и работа с его сохранёнными настройками."
        )

        self.comparison_actions_button = QPushButton("Сравнение")
        self.comparison_actions_menu = QMenu(self.comparison_actions_button)
        self.compare_action = self.comparison_actions_menu.addAction(
            "Детальное сравнение A/B"
        )
        self.overview_action = self.comparison_actions_menu.addAction(
            "Обзор 2–4 наборов"
        )
        self.comparison_actions_button.setMenu(self.comparison_actions_menu)
        self.comparison_actions_button.setToolTip(
            "Выберите способ сравнения отмеченных готовых наборов."
        )

        self.manage_actions_button = QPushButton("Удаление и кэш")
        self.manage_actions_menu = QMenu(self.manage_actions_button)
        self.delete_sets_action = self.manage_actions_menu.addAction(
            "Удалить выбранные наборы"
        )
        self.delete_all_action = self.manage_actions_menu.addAction(
            "Удалить все наборы"
        )
        self.manage_actions_menu.addSeparator()
        self.clear_cache_action = self.manage_actions_menu.addAction(
            "Очистить промежуточный кэш"
        )
        self.manage_actions_button.setMenu(self.manage_actions_menu)
        self.manage_actions_button.setToolTip(
            "Удаление результатов тестирования и управление промежуточным кэшем."
        )

        self.close_button = QPushButton("Закрыть")
        self.actions_layout.addWidget(self.set_actions_button)
        self.actions_layout.addWidget(self.comparison_actions_button)
        self.actions_layout.addWidget(self.manage_actions_button)
        self.actions_layout.addStretch(1)
        self.actions_layout.addWidget(self.close_button)
        root.addLayout(self.actions_layout)

        self.session_combo.currentIndexChanged.connect(self._reload_sets)
        self.new_session_button.clicked.connect(self._create_session)
        self.delete_session_button.clicked.connect(self._delete_session)
        self.set_tree.itemSelectionChanged.connect(self._update_selection)
        self.set_tree.itemDoubleClicked.connect(self._view_double_clicked_set)
        self.generate_action.triggered.connect(self._generate)
        self.view_set_action.triggered.connect(self._view_selected)
        self.apply_action.triggered.connect(self._apply_selected)
        self.template_action.triggered.connect(self._save_as_template)
        self.compare_action.triggered.connect(self._compare)
        self.overview_action.triggered.connect(self._overview)
        self.delete_sets_action.triggered.connect(self._delete_selected)
        self.delete_all_action.triggered.connect(self._delete_all)
        self.clear_cache_action.triggered.connect(self._clear_cache)
        self.close_button.clicked.connect(self.reject)
        self._reload_sessions()
        if self._window_state_store is not None:
            self._window_state_store.restore(
                "test_mask_manager",
                self,
                splitters={"main": self.main_splitter},
            )

    def _current_session(self) -> TestSession | None:
        session_id = self.session_combo.currentData()
        return self.store.get_session(str(session_id)) if session_id else None

    def _selected_sets(self, *, complete_only: bool = False) -> tuple[TestMaskSet, ...]:
        session = self._current_session()
        if session is None:
            return ()
        ids = {
            str(item.data(0, Qt.ItemDataRole.UserRole))
            for item in self.set_tree.selectedItems()
        }
        selected = tuple(
            item for item in self.store.list_sets(session.session_id)
            if item.set_id in ids
        )
        return tuple(
            item for item in selected if not complete_only or item.status == "complete"
        )

    def _reload_sessions(self, selected_id: str | None = None) -> None:
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for session in self.store.list_sessions():
            self.session_combo.addItem(session.name, session.session_id)
            if session.session_id == selected_id:
                self.session_combo.setCurrentIndex(self.session_combo.count() - 1)
        self.session_combo.blockSignals(False)
        self._reload_sets()

    @Slot()
    def _reload_sets(self, selected_id: str | None = None) -> None:
        previous_ids = {
            str(item.data(0, Qt.ItemDataRole.UserRole))
            for item in self.set_tree.selectedItems()
        }
        desired_ids = {selected_id} if selected_id is not None else previous_ids
        self.set_tree.blockSignals(True)
        self.set_tree.clear()
        selected_row: QTreeWidgetItem | None = None
        session = self._current_session()
        if session is not None:
            for item in self.store.list_sets(session.session_id):
                descriptor = item.settings.resolved_model_name().value
                resolution = item.settings.model.process_resolution or "auto"
                row = QTreeWidgetItem(
                    [
                        f"{item.number:03d}",
                        item.name,
                        descriptor,
                        str(resolution),
                        item.settings.resolved_refinement().value,
                        item.status,
                    ]
                )
                row.setData(0, Qt.ItemDataRole.UserRole, item.set_id)
                if item.error:
                    row.setToolTip(5, item.error)
                self.set_tree.addTopLevelItem(row)
                if item.set_id in desired_ids:
                    row.setSelected(True)
                    selected_row = row
        self.set_tree.blockSignals(False)
        self.set_tree.resizeColumnToContents(0)
        self.set_tree.resizeColumnToContents(1)
        if selected_row is not None:
            self.set_tree.setCurrentItem(selected_row)
            self.set_tree.scrollToItem(selected_row)
        self._update_selection()
        self._update_buttons()

    @Slot()
    def _update_selection(self) -> None:
        selected = self._selected_sets()
        if len(selected) == 1:
            item = selected[0]
            self.settings_caption.setText(
                f"Все настройки: {item.number:03d} — {item.name}"
            )
            self.selection_summary.setText(
                f"Выбран набор: {item.number:03d} — {item.name}"
            )
            self._show_settings(item.settings)
        elif selected:
            self.settings_caption.setText("Настройки набора")
            self.selection_summary.setText(f"Выбрано наборов: {len(selected)}")
            self._show_settings_placeholder(
                "Для просмотра всех настроек выберите один набор."
            )
        else:
            self.settings_caption.setText("Настройки набора")
            self.selection_summary.setText("Выберите тестовый набор.")
            self._show_settings_placeholder("Выберите один тестовый набор.")
        self._update_buttons()

    def _show_settings(self, settings: RmbgSettings) -> None:
        """Render the exact immutable settings snapshot stored with a test set."""

        payload = settings.to_context_value()
        grouped: tuple[tuple[str, dict[str, object]], ...] = (
            (
                "general",
                {
                    "schema_version": payload["schema_version"],
                    "profile_name": payload["profile_name"],
                },
            ),
            *(
                (section, payload[section])
                for section in (
                    "task",
                    "model",
                    "segmentation",
                    "mask",
                    "output",
                    "performance",
                )
            ),
        )
        self.settings_tree.clear()
        for section, values in grouped:
            group = QTreeWidgetItem([SECTION_LABELS[section], ""])
            group.setFirstColumnSpanned(True)
            self.settings_tree.addTopLevelItem(group)
            for key, value in values.items():
                row = QTreeWidgetItem(
                    [SETTING_LABELS.get(key, key), self._format_setting_value(value)]
                )
                row.setToolTip(0, f"{section}.{key}")
                row.setToolTip(1, str(value))
                group.addChild(row)
            group.setExpanded(True)

    def _show_settings_placeholder(self, message: str) -> None:
        self.settings_tree.clear()
        placeholder = QTreeWidgetItem([message, ""])
        placeholder.setFirstColumnSpanned(True)
        self.settings_tree.addTopLevelItem(placeholder)

    @staticmethod
    def _format_setting_value(value: object) -> str:
        if isinstance(value, bool):
            return "Да" if value else "Нет"
        if value is None or value == "":
            return "—"
        return str(value)

    def _update_buttons(self) -> None:
        session = self._current_session()
        selected = self._selected_sets()
        complete = tuple(item for item in selected if item.status == "complete")
        one_complete = len(selected) == 1 and len(complete) == 1
        busy = self._thread is not None and self._thread.isRunning()
        self.generate_action.setEnabled(session is not None and not busy)
        self.delete_session_button.setEnabled(session is not None and not busy)
        self.view_set_action.setEnabled(one_complete and not busy)
        self.apply_action.setEnabled(len(complete) == 1 and not busy)
        self.template_action.setEnabled(len(complete) == 1 and not busy)
        self.compare_action.setEnabled(len(complete) >= 2 and not busy)
        self.overview_action.setEnabled(2 <= len(complete) <= 4 and not busy)
        self.delete_sets_action.setEnabled(bool(selected) and not busy)
        has_sets = session is not None and bool(self.store.list_sets(session.session_id))
        self.delete_all_action.setEnabled(has_sets and not busy)
        self.clear_cache_action.setEnabled(session is not None and not busy)
        self.set_actions_button.setEnabled(
            any(
                action.isEnabled()
                for action in (
                    self.generate_action,
                    self.view_set_action,
                    self.apply_action,
                    self.template_action,
                )
            )
        )
        self.comparison_actions_button.setEnabled(
            self.compare_action.isEnabled() or self.overview_action.isEnabled()
        )
        self.manage_actions_button.setEnabled(
            any(
                action.isEnabled()
                for action in (
                    self.delete_sets_action,
                    self.delete_all_action,
                    self.clear_cache_action,
                )
            )
        )

    def _create_session(self) -> None:
        dialog = CreateSessionDialog(
            self,
            window_state_store=self._window_state_store,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            session = self.store.create_session(
                dialog.name_edit.text(),
                dialog.source_paths,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось создать сессию", str(exc))
            return
        self._reload_sessions(session.session_id)

    def _delete_session(self) -> None:
        session = self._current_session()
        if session is None:
            return
        answer = QMessageBox.warning(
            self,
            "Удаление тестовой сессии",
            f"Полностью удалить сессию «{session.name}», исходники, наборы и кэш?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete_session(session.session_id)
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось удалить сессию", str(exc))
            return
        self._reload_sessions()

    def _generate(self) -> None:
        session = self._current_session()
        if session is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Новый тестовый набор",
            "Короткое название набора:",
        )
        if not accepted:
            return
        try:
            settings = self.settings_provider()
            mask_set = self.store.create_set(
                session.session_id,
                name=name,
                settings=settings,
            )
            self._generation_set_id = mask_set.set_id
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось начать генерацию", str(exc))
            return

        self._reload_sets(selected_id=mask_set.set_id)

        self._cancel_event = threading.Event()
        self._thread = QThread(self)
        job = {
            "store": self.store,
            "session_id": session.session_id,
            "set_id": mask_set.set_id,
            "settings": settings,
        }
        self._worker = _PreviewWorker(job, self._cancel_event)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._generation_completed)
        self._worker.failed.connect(self._generation_failed)
        self._worker.completed.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._generation_finished)
        self._thread.finished.connect(self._thread.deleteLater)

        self._progress_dialog = QProgressDialog(
            "Подготовка тестовой генерации…",
            "Отмена",
            0,
            1,
            self,
        )
        self._progress_dialog.setWindowTitle("Генерация тестовых масок")
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.canceled.connect(self._cancel_generation)
        self._thread.start()
        self._update_buttons()

    @Slot(str, int, int)
    def _on_progress(self, message: str, value: int, total: int) -> None:
        if self._progress_dialog is None:
            return
        self._progress_dialog.setLabelText(message)
        self._progress_dialog.setRange(0, max(1, total))
        self._progress_dialog.setValue(value)

    @Slot()
    def _cancel_generation(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()

    @Slot(object)
    def _generation_completed(self, mask_set: TestMaskSet) -> None:
        self._generation_result = mask_set

    @Slot(str)
    def _generation_failed(self, message: str) -> None:
        self._generation_error = message

    @Slot()
    def _generation_finished(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
        result = self._generation_result
        error = self._generation_error
        generated_set_id = self._generation_set_id
        self._progress_dialog = None
        self._worker = None
        self._thread = None
        self._cancel_event = None
        self._generation_result = None
        self._generation_error = None
        self._generation_set_id = None
        self._reload_sets(
            selected_id=result.set_id if result is not None else generated_set_id
        )
        if result is not None:
            QMessageBox.information(
                self,
                "Тестовый набор готов",
                f"Создан набор {result.number:03d} — {result.name}.",
            )
        elif error is not None:
            QMessageBox.warning(self, "Генерация не завершена", error)

    def _apply_selected(self) -> None:
        selected = self._selected_sets(complete_only=True)
        if len(selected) != 1:
            return
        self.settings_applier(selected[0].settings)
        QMessageBox.information(
            self,
            "Настройки применены",
            "Параметры набора загружены в Configurator. Для записи в контекст "
            "нажмите «Сохранить» в основном окне.",
        )

    def _view_selected(self) -> None:
        session = self._current_session()
        selected = self._selected_sets()
        if (
            session is None
            or len(selected) != 1
            or selected[0].status != "complete"
        ):
            return
        self._open_set_viewer(session, selected[0])

    @Slot(QTreeWidgetItem, int)
    def _view_double_clicked_set(
        self,
        row: QTreeWidgetItem,
        _column: int,
    ) -> None:
        """Open the completed set represented by the double-clicked row."""

        session = self._current_session()
        busy = self._thread is not None and self._thread.isRunning()
        if session is None or busy:
            return
        set_id = str(row.data(0, Qt.ItemDataRole.UserRole) or "")
        mask_set = next(
            (
                item
                for item in self.store.list_sets(session.session_id)
                if item.set_id == set_id
            ),
            None,
        )
        if mask_set is None or mask_set.status != "complete":
            return
        self.set_tree.clearSelection()
        row.setSelected(True)
        self.set_tree.setCurrentItem(row)
        self._update_selection()
        self._open_set_viewer(session, mask_set)

    def _open_set_viewer(
        self,
        session: TestSession,
        mask_set: TestMaskSet,
    ) -> None:
        SingleSetViewerDialog(
            session,
            mask_set,
            window_state_store=self._window_state_store,
            parent=self,
        ).exec()

    def _save_as_template(self) -> None:
        selected = self._selected_sets(complete_only=True)
        if len(selected) != 1:
            return
        mask_set = selected[0]
        name, accepted = QInputDialog.getText(
            self,
            "Шаблон из тестового набора",
            "Название шаблона:",
            text=mask_set.name,
        )
        if not accepted:
            return
        description, accepted = QInputDialog.getMultiLineText(
            self,
            "Описание шаблона",
            "Описание:",
            text=(
                f"Создан из тестового набора {mask_set.number:03d} "
                f"в сессии «{self._current_session().name}»."
            ),
        )
        if not accepted:
            return
        try:
            self.template_store.create(
                name=name,
                description=description,
                settings=mask_set.settings,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось создать шаблон", str(exc))
            return
        QMessageBox.information(self, "Шаблон RMBG", "Шаблон сохранён.")

    def _compare(self) -> None:
        selected = self._selected_sets(complete_only=True)
        if len(selected) < 2:
            return
        DetailedComparisonDialog(
            self._current_session(),
            selected,
            window_state_store=self._window_state_store,
            parent=self,
        ).exec()

    def _overview(self) -> None:
        selected = self._selected_sets(complete_only=True)
        if not 2 <= len(selected) <= 4:
            QMessageBox.information(
                self,
                "Обзор наборов",
                "Для общего обзора выберите от двух до четырёх готовых наборов.",
            )
            return
        MultiSetOverviewDialog(
            self._current_session(),
            selected,
            window_state_store=self._window_state_store,
            parent=self,
        ).exec()

    def _delete_selected(self) -> None:
        session = self._current_session()
        selected = self._selected_sets()
        if session is None or not selected:
            return
        answer = QMessageBox.question(
            self,
            "Удаление тестовых наборов",
            f"Удалить выбранные наборы ({len(selected)}) и неиспользуемый кэш?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete_sets(
                session.session_id,
                tuple(item.set_id for item in selected),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось удалить наборы", str(exc))
        self._reload_sets()

    def _delete_all(self) -> None:
        session = self._current_session()
        if session is None:
            return
        answer = QMessageBox.warning(
            self,
            "Удаление всех наборов",
            "Удалить все тестовые наборы и промежуточный кэш этой сессии? "
            "Копии исходных изображений останутся.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete_all_sets(session.session_id)
            self.store.clear_cache(session.session_id)
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось удалить наборы", str(exc))
        self._reload_sets()

    def _clear_cache(self) -> None:
        session = self._current_session()
        if session is None:
            return
        answer = QMessageBox.question(
            self,
            "Очистка промежуточного кэша",
            "Удалить базовые и refined-маски? Готовые тестовые наборы останутся, "
            "но следующая генерация может снова запустить модель.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.clear_cache(session.session_id)
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось очистить кэш", str(exc))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(
                self,
                "Генерация выполняется",
                "Отмените генерацию и дождитесь остановки worker перед закрытием окна.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def done(self, result: int) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(
                self,
                "Генерация выполняется",
                "Отмените генерацию и дождитесь остановки worker перед закрытием окна.",
            )
            return
        if self._window_state_store is not None:
            self._window_state_store.save(
                "test_mask_manager",
                self,
                splitters={"main": self.main_splitter},
            )
        super().done(result)


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        mask = np.asarray(opened, dtype=np.float32) / 65535.0
    if mask.shape != (size[1], size[0]):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)


def _load_source_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(mask * 255.0).astype(np.uint8), mode="L").convert("RGB")


def _mask_overlay_image(source: Image.Image, mask: np.ndarray) -> Image.Image:
    """Show a soft mask over the source without hiding image details."""

    base = np.asarray(source, dtype=np.float32)
    overlay = np.empty_like(base)
    overlay[...] = (255.0, 52.0, 52.0)
    opacity = np.clip(mask, 0.0, 1.0)[..., None] * 0.48
    rendered = np.clip(base * (1.0 - opacity) + overlay * opacity, 0.0, 255.0)
    return Image.fromarray(np.rint(rendered).astype(np.uint8), mode="RGB")


def _checkerboard(size: tuple[int, int], tile: int = 24) -> Image.Image:
    width, height = size
    y, x = np.indices((height, width))
    pattern = ((x // tile + y // tile) % 2).astype(np.uint8)
    values = np.where(pattern[..., None] == 0, 210, 245).astype(np.uint8)
    rgb = np.repeat(values, 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _cutout_image(
    source: Image.Image,
    mask: np.ndarray,
    *,
    background: str = "checker",
    custom_color: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    foreground = source.convert("RGBA")
    foreground.putalpha(Image.fromarray(np.rint(mask * 255.0).astype(np.uint8)))
    colors = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "green": (0, 177, 64),
        "magenta": (220, 0, 170),
        "custom": custom_color,
    }
    canvas = (
        _checkerboard(source.size)
        if background == "checker"
        else Image.new("RGB", source.size, colors.get(background, custom_color))
    ).convert("RGBA")
    return Image.alpha_composite(canvas, foreground).convert("RGB")


def _render_comparison(
    source: Image.Image,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    mode: str,
    split: float,
) -> Image.Image:
    if mode == "difference":
        delta = mask_a - mask_b
        strength = np.sqrt(np.abs(delta))[..., None]
        base = np.asarray(source, dtype=np.float32) * 0.28
        color = np.zeros_like(base)
        color[delta >= 0.0] = (0.0, 213.0, 255.0)
        color[delta < 0.0] = (255.0, 79.0, 216.0)
        rendered = np.clip(
            base * (1.0 - strength) + color * strength,
            0.0,
            255.0,
        ).astype(np.uint8)
        return Image.fromarray(rendered, mode="RGB")
    if mode == "contours":
        rendered = np.asarray(source, dtype=np.uint8).copy()
        edge_a = cv2.Canny(np.rint(mask_a * 255.0).astype(np.uint8), 64, 160) > 0
        edge_b = cv2.Canny(np.rint(mask_b * 255.0).astype(np.uint8), 64, 160) > 0
        rendered[edge_a] = (0, 255, 255)
        rendered[edge_b] = (255, 0, 255)
        rendered[edge_a & edge_b] = (255, 255, 0)
        return Image.fromarray(rendered, mode="RGB")
    image_a = _mask_image(mask_a) if mode == "mask" else _cutout_image(source, mask_a)
    image_b = _mask_image(mask_b) if mode == "mask" else _cutout_image(source, mask_b)
    divider = int(source.width * split)
    rendered = image_b.copy()
    rendered.paste(image_a.crop((0, 0, divider, source.height)), (0, 0))
    pixels = np.asarray(rendered).copy()
    if 0 <= divider < source.width:
        pixels[:, max(0, divider - 1):min(source.width, divider + 2)] = (0, 180, 255)
    return Image.fromarray(pixels, mode="RGB")
