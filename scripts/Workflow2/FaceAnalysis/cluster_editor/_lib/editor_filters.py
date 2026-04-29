# analize/cluster_editor/_lib/editor_filters.py
# -*- coding: utf-8 -*-

"""
Модуль управления фильтрами галереи.
Обеспечивает логику взаимоисключающих кнопок и проверку элементов на соответствие фильтрам.
"""

from PySide6.QtCore import QObject, Slot

class GalleryFilterManager(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window

    def bind_ui(self):
        """Подключает сигналы от кнопок UI к слотам менеджера."""
        w = self.window
        w.btn_filter_male.toggled.connect(self.on_male_toggled)
        w.btn_filter_female.toggled.connect(self.on_female_toggled)
        w.btn_filter_eyes.toggled.connect(self.apply_filters)
        w.btn_filter_mouth.toggled.connect(self.apply_filters)
        w.btn_filter_portrait.toggled.connect(self.on_portrait_toggled)
        w.btn_filter_group.toggled.connect(self.on_group_toggled)
        w.spin_group_count.valueChanged.connect(self.apply_filters)
        w.btn_filter_beauty.toggled.connect(self.on_beauty_toggled)
        w.spin_beauty_score.valueChanged.connect(self.apply_filters)

    @Slot(bool)
    def on_male_toggled(self, checked: bool):
        if checked:
            self.window.btn_filter_female.blockSignals(True)
            self.window.btn_filter_female.setChecked(False)
            self.window.btn_filter_female.blockSignals(False)
        self.apply_filters()

    @Slot(bool)
    def on_female_toggled(self, checked: bool):
        if checked:
            self.window.btn_filter_male.blockSignals(True)
            self.window.btn_filter_male.setChecked(False)
            self.window.btn_filter_male.blockSignals(False)
        self.apply_filters()

    @Slot(bool)
    def on_portrait_toggled(self, checked: bool):
        if checked:
            self.window.btn_filter_group.blockSignals(True)
            self.window.btn_filter_group.setChecked(False)
            self.window.btn_filter_group.blockSignals(False)
            self.window.spin_group_count.setEnabled(False)
        self.apply_filters()

    @Slot(bool)
    def on_group_toggled(self, checked: bool):
        if checked:
            self.window.btn_filter_portrait.blockSignals(True)
            self.window.btn_filter_portrait.setChecked(False)
            self.window.btn_filter_portrait.blockSignals(False)
        self.window.spin_group_count.setEnabled(checked)
        self.apply_filters()

    @Slot(bool)
    def on_beauty_toggled(self, checked: bool):
        self.window.spin_beauty_score.setEnabled(checked)
        self.apply_filters()

    @Slot()
    def apply_filters(self, *args, **kwargs):
        """Дает команду главному окну перерисовать галерею."""
        if self.window.active_cluster_id:
            self.window._render_gallery(self.window.active_cluster_id)

    def has_active_filters(self) -> bool:
        """Проверяет, включен ли хотя бы один фильтр."""
        w = self.window
        return any([
            w.btn_filter_male.isChecked(),
            w.btn_filter_female.isChecked(),
            w.btn_filter_eyes.isChecked(),
            w.btn_filter_mouth.isChecked(),
            w.btn_filter_portrait.isChecked(),
            w.btn_filter_group.isChecked(),
            w.btn_filter_beauty.isChecked()
        ])

    def passes(self, user_data: dict) -> bool:
        """Определяет, проходит ли конкретная фотография через текущие фильтры."""
        w = self.window
        overlays = user_data.get("overlays", list())
        face_count = user_data.get("face_count", 1)
        beauty_score = user_data.get("beauty_score", -1)

        # Логика "И" (AND)
        if w.btn_filter_male.isChecked() and "GENDER_MALE" not in overlays: return False
        if w.btn_filter_female.isChecked() and "GENDER_FEMALE" not in overlays: return False
        if w.btn_filter_eyes.isChecked() and "EYE_CLOSED" not in overlays: return False
        if w.btn_filter_mouth.isChecked() and "MOUTH_OPEN" not in overlays: return False
        
        if w.btn_filter_portrait.isChecked() and face_count != 1: return False
        if w.btn_filter_group.isChecked() and not (1 < face_count < w.spin_group_count.value()): return False
        
        if w.btn_filter_beauty.isChecked() and beauty_score <= w.spin_beauty_score.value(): return False

        return True