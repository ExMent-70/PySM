"""GUI management for named RMBG configuration templates."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from _common.config_schema import RmbgSettings
from pysm_lib.pysm_icons import icons

from .template_store import RmbgTemplate, TemplateStore
from .window_state import RmbgWindowStateStore


SECTION_LABELS = {
    "general": "Общие сведения",
    "task": "Задача",
    "model": "Модель и вычисления",
    "segmentation": "Сегментация",
    "mask": "Маска",
    "output": "Результат",
    "performance": "Производительность",
}

SETTING_LABELS = {
    "schema_version": "Версия схемы",
    "profile_name": "Название профиля",
    "type": "Тип задачи",
    "preset": "Тип изображений",
    "selection": "Выбор модели",
    "name": "Модель",
    "model_dir": "Папка моделей",
    "process_resolution": "Разрешение обработки",
    "device": "Устройство",
    "precision": "Точность",
    "unload_after_run": "Выгружать после запуска",
    "prompt": "Текстовый запрос",
    "threshold": "Порог сегментации",
    "merge_instances": "Объединять экземпляры",
    "max_segments": "Максимум сегментов",
    "sensitivity": "Чувствительность",
    "blur": "Размытие",
    "offset": "Смещение края",
    "feather": "Растушёвка",
    "fill_holes": "Заполнять отверстия",
    "max_hole_area": "Макс. площадь отверстия",
    "remove_small_regions": "Удалять мелкие области",
    "min_region_area": "Мин. площадь области",
    "invert": "Инвертировать маску",
    "refinement": "Уточнение края",
    "sdmatte_variant": "Вариант SDMatte",
    "sdmatte_resolution": "Разрешение SDMatte",
    "sdmatte_transparent_object": "Учитывать прозрачные объекты",
    "sdmatte_constraint": "Строгость исходной маски",
    "save_cutout": "Сохранять cutout",
    "save_mask": "Сохранять маску",
    "save_composite": "Сохранять composite",
    "background_mode": "Режим фона",
    "background_color": "Цвет фона",
    "background_image": "Фоновое изображение",
    "background_fit": "Размещение фона",
    "background_position": "Положение обрезки",
    "image_suffix": "Суффикс изображения",
    "mask_suffix": "Суффикс маски",
    "composite_suffix": "Суффикс composite",
    "image_format": "Формат изображений",
    "png_compress_level": "Сжатие PNG",
    "jpeg_quality": "Качество JPEG",
    "batch_size": "GPU batch",
    "io_workers": "Потоки чтения/записи",
    "max_loaded_models": "Моделей в памяти",
    "allow_cpu_fallback": "Переход на CPU при OOM",
}


class TemplateManagerDialog(QDialog):
    """Create, update, delete and select one persistent RMBG template."""

    def __init__(
        self,
        store: TemplateStore,
        current_settings: Callable[[], RmbgSettings],
        *,
        window_state_store: RmbgWindowStateStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._current_settings = current_settings
        self._window_state_store = window_state_store
        self.selected_settings: RmbgSettings | None = None
        self.setWindowTitle("Шаблоны настроек RMBG")
        self.setMinimumSize(1000, 640)
        self.resize(1180, 760)

        root = QVBoxLayout(self)
        explanation = QLabel(
            "Шаблон содержит все настройки RMBG, но не runtime-пути input_dir, "
            "output_dir и background_dir. Описание сохраняется вместе с шаблоном."
        )
        explanation.setWordWrap(True)
        root.addWidget(explanation)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.template_list = QListWidget()
        self.template_list.setMinimumWidth(340)
        self.template_list.setIconSize(QSize(24, 24))
        self.template_list.currentItemChanged.connect(self._show_selected)
        self.main_splitter.addWidget(self.template_list)

        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        editor_layout.addWidget(QLabel("Название шаблона:"))
        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(120)
        editor_layout.addWidget(self.name_edit)
        editor_layout.addWidget(QLabel("Описание шаблона:"))
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText(
            "Назначение шаблона, тип изображений, особенности параметров и "
            "условия, для которых он был подобран."
        )
        self.description_edit.setMaximumHeight(120)
        editor_layout.addWidget(self.description_edit)

        editor_layout.addWidget(QLabel("Все параметры выбранного шаблона:"))
        self.settings_tree = QTreeWidget()
        self.settings_tree.setColumnCount(2)
        self.settings_tree.setHeaderLabels(["Параметр", "Значение"])
        self.settings_tree.setRootIsDecorated(True)
        self.settings_tree.setAlternatingRowColors(True)
        self.settings_tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.settings_tree.header().setStretchLastSection(True)
        self.settings_tree.header().resizeSection(0, 260)
        editor_layout.addWidget(self.settings_tree, 1)

        actions = QHBoxLayout()
        self.create_button = QPushButton("Сохранить текущие как новый")
        self.update_button = QPushButton("Обновить текущими")
        self.metadata_button = QPushButton("Сохранить название и описание")
        self.delete_button = QPushButton("Удалить")
        actions.addWidget(self.create_button)
        actions.addWidget(self.update_button)
        actions.addWidget(self.metadata_button)
        actions.addWidget(self.delete_button)
        editor_layout.addLayout(actions)

        self.create_button.clicked.connect(self._create)
        self.update_button.clicked.connect(self._update_from_current)
        self.metadata_button.clicked.connect(self._update_metadata)
        self.delete_button.clicked.connect(self._delete)
        self.main_splitter.addWidget(editor)
        self.main_splitter.setSizes([380, 760])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        root.addWidget(self.main_splitter, 1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Close
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).setText(
            "Загрузить в Configurator"
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self._apply_selected
        )
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self._reload()
        if self._window_state_store is not None:
            self._window_state_store.restore(
                "template_manager",
                self,
                splitters={"main": self.main_splitter},
            )

    def done(self, result: int) -> None:
        if self._window_state_store is not None:
            self._window_state_store.save(
                "template_manager",
                self,
                splitters={"main": self.main_splitter},
            )
        super().done(result)

    def _reload(self, selected_id: str | None = None) -> None:
        try:
            templates = self._store.list_templates()
        except Exception as exc:
            QMessageBox.critical(self, "Файл шаблонов", str(exc))
            templates = ()
        self.template_list.clear()
        selected_item = None
        for template in templates:
            item = QListWidgetItem(
                icons.get_qicon("FILE_MASK", size=24),
                template.name,
            )
            item.setData(Qt.ItemDataRole.UserRole, template.template_id)
            item.setToolTip(template.description or "Описание не заполнено")
            self.template_list.addItem(item)
            if template.template_id == selected_id:
                selected_item = item
        if selected_item is not None:
            self.template_list.setCurrentItem(selected_item)
        elif self.template_list.count():
            self.template_list.setCurrentRow(0)
        else:
            self.name_edit.clear()
            self.description_edit.clear()
            self.settings_tree.clear()
        self._update_buttons()

    def _selected_template(self) -> RmbgTemplate | None:
        item = self.template_list.currentItem()
        if item is None:
            return None
        try:
            return self._store.get(str(item.data(Qt.ItemDataRole.UserRole)))
        except Exception as exc:
            QMessageBox.warning(self, "Шаблоны RMBG", str(exc))
            return None

    def _show_selected(self, current: QListWidgetItem | None, _previous) -> None:
        del _previous
        if current is None:
            self._update_buttons()
            return
        template = self._selected_template()
        if template is not None:
            self.name_edit.setText(template.name)
            self.description_edit.setPlainText(template.description)
            self._show_settings(template.settings)
        self._update_buttons()

    def _show_settings(self, settings: RmbgSettings) -> None:
        """Render every persisted setting as a readable grouped tree."""

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
                    [SETTING_LABELS.get(key, key), self._format_value(value)]
                )
                row.setToolTip(0, f"{section}.{key}")
                row.setToolTip(1, str(value))
                group.addChild(row)
            group.setExpanded(True)

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, bool):
            return "Да" if value else "Нет"
        if value is None or value == "":
            return "—"
        return str(value)

    def _update_buttons(self) -> None:
        has_selection = self.template_list.currentItem() is not None
        self.update_button.setEnabled(has_selection)
        self.metadata_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).setEnabled(
            has_selection
        )

    def _create(self) -> None:
        try:
            created = self._store.create(
                name=self.name_edit.text(),
                description=self.description_edit.toPlainText(),
                settings=self._current_settings(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось создать шаблон", str(exc))
            return
        self._reload(created.template_id)

    def _update_from_current(self) -> None:
        template = self._selected_template()
        if template is None:
            return
        try:
            updated = self._store.update(
                template.template_id,
                name=self.name_edit.text(),
                description=self.description_edit.toPlainText(),
                settings=self._current_settings(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось обновить шаблон", str(exc))
            return
        self._reload(updated.template_id)

    def _update_metadata(self) -> None:
        template = self._selected_template()
        if template is None:
            return
        try:
            updated = self._store.update(
                template.template_id,
                name=self.name_edit.text(),
                description=self.description_edit.toPlainText(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось обновить шаблон", str(exc))
            return
        self._reload(updated.template_id)

    def _delete(self) -> None:
        template = self._selected_template()
        if template is None:
            return
        answer = QMessageBox.question(
            self,
            "Удаление шаблона",
            f"Удалить шаблон «{template.name}»?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._store.delete(template.template_id)
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось удалить шаблон", str(exc))
            return
        self._reload()

    def _apply_selected(self) -> None:
        template = self._selected_template()
        if template is None:
            return
        self.selected_settings = template.settings
        self.accept()
