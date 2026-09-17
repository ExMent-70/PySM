"""Assignment table rendering helpers for the photo selection window."""

from __future__ import annotations

from pathlib import Path
import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QTreeWidgetItem

from .assignment_core import (
    BuildResult, KNOWN_FILE_ICON_SUFFIXES, PhotoRecord,
    index_records_by_student_location, is_excluded_relative_path,
)
from .constants import ITEM_NUMBER_ROLE, ITEM_PATHS_ROLE, ITEM_STUDENT_ROLE, ITEM_LOCATION_ROLE
try:
    from pysm_lib import theme_api
except ImportError:
    theme_api = None

try:
    from pysm_lib.pysm_icons import icons as pysm_icons
except ImportError:
    pysm_icons = None

logger = logging.getLogger(__name__)


class AssignmentViewsMixin:
    def _render_assignment_summary(self) -> None:
        result = self.state.build_result
        if result is None:
            self.report.setHtml(
                self._message_header_html()
                + "<p>Список назначений ещё не рассчитан.</p>"
            )
            return
        self._base_report_html = self._completion_report_html(result)
        self.report.setHtml(self._base_report_html)

    def _render_assignment_views(self) -> None:
        result = self.state.build_result
        if result is None:
            return
        file_entries = {
            record.number: self._file_entries(record)
            for record in result.records.values()
            if record.assigned_student_ids or record.photographer_selected
        }
        self._render_photo_table(result, file_entries)
        self._render_student_location_table(result, file_entries)
        self._render_assignment_summary()

    def _render_photo_table(
        self,
        result: BuildResult,
        file_entries: dict[str, list[tuple[Path, str, list[Path]]]],
    ) -> None:
        visible = [
            record for record in result.records.values()
            if record.selected_student_ids or record.photographer_selected
        ]
        self.photo_table.setUpdatesEnabled(False)
        self.photo_table.setSortingEnabled(False)
        try:
            self.photo_table.clear()
            for record in sorted(visible, key=lambda item: item.number):
                files = file_entries.get(record.number, [])
                compact_students = (
                    len(record.recognized_student_ids) > 1
                    or len(record.assigned_student_ids) > 1
                )
                recognized_text, recognized_hint = self._student_cell_content(
                    record.recognized_student_ids,
                    compact=compact_students,
                )
                assigned_text, assigned_hint = self._student_cell_content(
                    record.assigned_student_ids,
                    compact=compact_students,
                )
                values = (
                    record.number,
                    record.location,
                    "",
                    recognized_text,
                    assigned_text,
                    f"{len(files)} файл(ов)" if files else "—",
                )
                parent = QTreeWidgetItem(self.photo_table, values)
                parent.setData(0, ITEM_NUMBER_ROLE, record.number)
                for column, value in enumerate(values):
                    parent.setToolTip(column, value)
                parent.setToolTip(3, recognized_hint)
                parent.setToolTip(4, assigned_hint)
                self._set_source_indicator(parent, 2, record)
                if pysm_icons:
                    try:
                        parent.setIcon(5, pysm_icons.get_qicon("FOLDER", 20))
                    except Exception:
                        logger.debug("Не удалось создать иконку папки", exc_info=True)
                for relative, origin, physical_paths in files:
                    child = QTreeWidgetItem(parent, ("", "", origin, "", "", str(relative)))
                    child.setData(0, ITEM_PATHS_ROLE, [str(path) for path in physical_paths])
                    self._set_file_origin_indicator(child, 2, origin)
                    child.setToolTip(5, "\n".join(str(path) for path in physical_paths))
                    self._set_file_icon(child, relative)
        finally:
            self.photo_table.setSortingEnabled(True)
            self.photo_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
            self.photo_table.collapseAll()
            self.photo_table.setUpdatesEnabled(True)

    def _render_student_location_table(
        self,
        result: BuildResult,
        file_entries: dict[str, list[tuple[Path, str, list[Path]]]],
    ) -> None:
        locations = self._known_locations(result)
        record_index = index_records_by_student_location(result)
        table = self.student_location_table
        table.setUpdatesEnabled(False)
        table.setSortingEnabled(False)
        try:
            table.clear()
            warning_brush = self._warning_brush()
            for student in sorted(self.roster.students, key=lambda item: item.display_name.casefold()):
                records_by_location = {
                    location: record_index.get((student.student_id, location), [])
                    for location in locations
                }
                selected_location_count = sum(
                    bool(records) for records in records_by_location.values()
                )
                status = f"{selected_location_count}/{len(locations)}"
                student_item = QTreeWidgetItem(
                    table,
                    (student.display_name, status, "", "", ""),
                )
                student_item.setData(0, ITEM_STUDENT_ROLE, student.student_id)
                student_item.setToolTip(0, student.display_name)
                student_item.setToolTip(
                    1,
                    f"Выбрано локаций: {selected_location_count} из {len(locations)}",
                )
                student_item.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
                if pysm_icons:
                    try:
                        student_item.setIcon(
                            0,
                            pysm_icons.get_qicon(
                                "PHOTO_PORTRAIT_EDIT", 20, color="Green"
                            ),
                        )
                    except Exception:
                        logger.debug("Не удалось создать иконку ученика", exc_info=True)
                for location in locations:
                    records = records_by_location[location]
                    if not records:
                        location_item = QTreeWidgetItem(
                            student_item,
                            ("", "", location, "Выбор отсутствует", ""),
                        )
                        location_item.setData(
                            0, ITEM_STUDENT_ROLE, student.student_id
                        )
                        location_item.setData(0, ITEM_LOCATION_ROLE, location)
                        self._mark_missing_selection(location_item, warning_brush)
                        continue
                    location_item = QTreeWidgetItem(
                        student_item,
                        ("", "", location, f"Выбрано: {len(records)}", ""),
                    )
                    location_item.setData(0, ITEM_STUDENT_ROLE, student.student_id)
                    location_item.setData(0, ITEM_LOCATION_ROLE, location)
                    for record in records:
                        files = file_entries.get(record.number, [])
                        values = (
                            record.number,
                            "",
                            "",
                            "",
                            f"{len(files)} файл(ов)" if files else "—",
                        )
                        photo_item = QTreeWidgetItem(location_item, values)
                        photo_item.setData(0, ITEM_NUMBER_ROLE, record.number)
                        for column, value in enumerate(values):
                            photo_item.setToolTip(column, value)
                        self._set_source_indicator(photo_item, 3, record)
                        if pysm_icons:
                            try:
                                photo_item.setIcon(
                                    4, pysm_icons.get_qicon("FOLDER", 20)
                                )
                            except Exception:
                                logger.debug(
                                    "Не удалось создать иконку папки",
                                    exc_info=True,
                                )
                        for relative, origin, physical_paths in files:
                            file_item = QTreeWidgetItem(
                                photo_item,
                                ("", "", "", origin, str(relative)),
                            )
                            file_item.setData(
                                0,
                                ITEM_PATHS_ROLE,
                                [str(path) for path in physical_paths],
                            )
                            self._set_file_origin_indicator(file_item, 3, origin)
                            file_item.setToolTip(
                                4,
                                "\n".join(str(path) for path in physical_paths),
                            )
                            self._set_file_icon(file_item, relative, column=4)
        finally:
            table.setSortingEnabled(True)
            table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
            table.collapseAll()
            table.setUpdatesEnabled(True)

    def _known_locations(self, result: BuildResult) -> list[str]:
        return sorted(
            {
                record.location
                for record in result.records.values()
                if record.location and record.location.casefold() != "unknown"
            },
            key=str.casefold,
        )

    def _student_cell_content(self, student_ids: set[str], *, compact: bool) -> tuple[str, str]:
        full_list = ", ".join(
            self.roster.by_id.get(student_id).display_name
            if student_id in self.roster.by_id else "Неизвестный ученик"
            for student_id in sorted(student_ids)
        )
        text = f"{len(student_ids)} чел." if compact else full_list
        return text, full_list

    def _file_entries(self, record: PhotoRecord) -> list[tuple[Path, str, list[Path]]]:
        grouped: dict[str, dict] = {}
        source_dir = Path(self.config.source_dir)
        destination_dir = Path(self.config.dest_dir)
        exclude_dirs = tuple(self._exclude_dirs())

        def add(path: Path, relative: Path, origin: str) -> None:
            key = str(relative).casefold()
            entry = grouped.setdefault(key, {
                "relative": relative,
                "origins": set(),
                "paths": [],
            })
            entry["origins"].add(origin)
            entry["paths"].append(path)

        for path in record.source_files:
            try:
                relative = path.relative_to(source_dir)
            except ValueError:
                relative = Path(path.name)
            if is_excluded_relative_path(relative, exclude_dirs):
                continue
            add(path, relative, "исходник")

        for path in record.destination_files:
            try:
                relative = path.relative_to(destination_dir)
            except ValueError:
                relative = Path(path.name)
            if is_excluded_relative_path(relative, exclude_dirs):
                continue
            if (
                len(relative.parts) > 1
                and relative.parts[0].casefold() == record.location.casefold()
            ):
                relative = Path(*relative.parts[1:])
            add(path, relative, "папка локации")

        result = []
        for entry in grouped.values():
            origins = entry["origins"]
            origin = (
                "исходник + папка локации"
                if len(origins) > 1
                else next(iter(origins))
            )
            result.append((entry["relative"], origin, entry["paths"]))
        return sorted(
            result,
            key=lambda item: (item[0].suffix.casefold(), str(item[0]).casefold()),
        )

    @staticmethod
    def _warning_brush() -> QBrush:
        color = "#a06000"
        if theme_api:
            try:
                color = theme_api.get_parsed_style(
                    "icon_warning", default="color: #a06000;"
                ).get("color", color)
            except Exception:
                logger.debug("Не удалось получить цвет предупреждения темы", exc_info=True)
        return QBrush(QColor(color))

    @staticmethod
    def _mark_missing_selection(item: QTreeWidgetItem, brush: QBrush) -> None:
        item.setForeground(2, brush)
        item.setForeground(3, brush)
        font = item.font(3)
        font.setBold(True)
        item.setFont(3, font)
        item.setToolTip(3, "Для ученика не назначена фотография этой локации")
        if pysm_icons:
            try:
                item.setIcon(2, pysm_icons.get_qicon("WARNING", 20))
            except Exception:
                logger.debug("Не удалось создать иконку предупреждения", exc_info=True)

    @staticmethod
    def _set_source_indicator(item: QTreeWidgetItem, column: int, record: PhotoRecord) -> None:
        icon_names = []
        hints = []
        if record.selected_student_ids:
            icon_names.append("PHOTO_PORTRAIT")
            hints.append("Фотография выбрана клиентом")
        if record.photographer_selected:
            icon_names.append("CAMERA")
            hints.append("Фотография выбрана фотографом")
        item.setText(column, "" if icon_names else "—")
        item.setToolTip(column, "\n".join(hints))
        if not pysm_icons or not icon_names:
            return
        try:
            item.setIcon(column, AssignmentViewsMixin._combined_icon(icon_names))
        except Exception:
            logger.debug("Не удалось создать индикатор источника", exc_info=True)

    @staticmethod
    def _set_file_origin_indicator(item: QTreeWidgetItem, column: int, origin: str) -> None:
        icon_names = []
        hints = []
        if "исходник" in origin:
            icon_names.append("FOLDER")
            hints.append("Файл расположен в исходной папке")
        if "папка локации" in origin:
            icon_names.append("FOLDER_VIRTUAL")
            hints.append("Файл расположен в папке локации")
        item.setText(column, "" if icon_names else "—")
        item.setToolTip(column, "\n".join(hints))
        if not pysm_icons or not icon_names:
            return
        try:
            item.setIcon(column, AssignmentViewsMixin._combined_icon(icon_names))
        except Exception:
            logger.debug("Не удалось создать индикатор расположения файла", exc_info=True)

    @staticmethod
    def _combined_icon(icon_names: list[str]) -> QIcon:
        icons = [pysm_icons.get_qicon(name, 18) for name in icon_names]
        if len(icons) == 1:
            return icons[0]
        icon_size = 18
        gap = 2
        pixmap = QPixmap(len(icons) * icon_size + (len(icons) - 1) * gap, icon_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        for index, icon in enumerate(icons):
            painter.drawPixmap(
                index * (icon_size + gap),
                0,
                icon.pixmap(icon_size, icon_size),
            )
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _icon_name(path: Path) -> str:
        suffix = path.suffix.casefold()
        if suffix in {".jpg", ".jpeg"}:
            return "FILE_JPG"
        if suffix == ".psd":
            return "FILE_PSD"
        if suffix == ".xmp":
            return "FILE_XMP"
        if suffix in KNOWN_FILE_ICON_SUFFIXES:
            return "FILE_RAW"
        return "FILE"

    def _set_file_icon(self, item: QTreeWidgetItem, path: Path, column: int = 5) -> None:
        if not pysm_icons:
            return
        try:
            item.setIcon(column, pysm_icons.get_qicon(self._icon_name(path), 20))
        except Exception:
            logger.debug("Не удалось создать иконку для %s", path, exc_info=True)
