# pysm_lib/gui/widgets/icon_selection_dialog.py

from typing import Optional, Tuple, List
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QToolButton, QVBoxLayout, QScrollArea, 
    QWidget, QHBoxLayout, QLabel, QPushButton, QColorDialog, QFrame
)
from PySide6.QtGui import QColor
from PySide6.QtCore import QSize, Qt
from ...pysm_icons import icons, ICON_CATEGORIES
from ...locale_manager import LocaleManager

class IconSelectionDialog(QDialog):
    def __init__(self, locale_manager: LocaleManager, current_color: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self.setWindowTitle(locale_manager.get("dialogs.icon_selection.title"))
        self.setMinimumWidth(450)
        self.setMinimumHeight(550)
        
        self.selected_icon: Optional[str] = None
        self.selected_color: Optional[str] = current_color
        self.icon_buttons: List[Tuple[QToolButton, str]] =[]
        
        main_layout = QVBoxLayout(self)
        
        # --- ПАНЕЛЬ ВЫБОРА ЦВЕТА ---
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel(locale_manager.get("dialogs.icon_selection.color_label")))
        
        # Кнопка "Сбросить цвет (По умолчанию)"
        self.btn_default_color = QPushButton("∅")
        self.btn_default_color.setToolTip(locale_manager.get("dialogs.icon_selection.default_color"))
        self.btn_default_color.setFixedSize(28, 28)
        self.btn_default_color.clicked.connect(lambda: self._apply_color(None))
        color_layout.addWidget(self.btn_default_color)
        
        # Предустановленные приятные цвета
        preset_colors =["#E74C3C", "#E67E22", "#F1C40F", "#27AE60", "#3498DB", "#9B59B6", "#7F8C8D"]
        for color_hex in preset_colors:
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #777; border-radius: 12px;")
            btn.clicked.connect(lambda checked=False, c=color_hex: self._apply_color(c))
            color_layout.addWidget(btn)
            
        # Кастомный цвет
        self.btn_custom_color = QPushButton(locale_manager.get("dialogs.icon_selection.custom_color"))
        self.btn_custom_color.clicked.connect(self._pick_custom_color)
        color_layout.addWidget(self.btn_custom_color)
        color_layout.addStretch()
        
        main_layout.addLayout(color_layout)
        
        # Линия-разделитель
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)
        
        # --- СЕТКА ИКОНОК ПО КАТЕГОРИЯМ ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        vbox = QVBoxLayout(container)
        
        for cat_key, icon_list in ICON_CATEGORIES.items():           
            translated = locale_manager.get(f"dialogs.icon_categories.{cat_key}")
            cat_name = cat_key.capitalize() if translated == f"dialogs.icon_categories.{cat_key}" else translated
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            
            lbl = QLabel(f"<b style='font-size: 14px;'>{cat_name}</b>")
            lbl.setContentsMargins(0, 10, 0, 5)
            vbox.addWidget(lbl)            
            
            grid_widget = QWidget()
            grid = QGridLayout(grid_widget)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(5)
            
            row, col = 0, 0
            max_cols = 8

            for icon_name in icon_list:
                btn = QToolButton()
                btn.setIcon(icons.get_qicon(icon_name, size=32, color=self.selected_color))
                btn.setIconSize(QSize(32, 32))
                btn.setToolTip(icon_name)
                btn.setAutoRaise(True)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setProperty("class", "favorite-btn")
                
                btn.clicked.connect(lambda checked=False, name=icon_name: self._on_icon_selected(name))
                
                self.icon_buttons.append((btn, icon_name))
                grid.addWidget(btn, row, col)

    
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
                    
            vbox.addWidget(grid_widget)
            
        vbox.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll, 1)

    def _apply_color(self, color_hex: Optional[str]):
        self.selected_color = color_hex
        # Мгновенно перерисовываем все иконки в диалоге для предпросмотра
        for btn, name in self.icon_buttons:
            btn.setIcon(icons.get_qicon(name, size=32, color=self.selected_color))

    def _pick_custom_color(self):
        initial = QColor(self.selected_color) if self.selected_color else QColor(Qt.GlobalColor.black)
        color = QColorDialog.getColor(initial, self)
        if color.isValid():
            self._apply_color(color.name())

    def _on_icon_selected(self, name: str):
        self.selected_icon = name
        self.accept()

    def get_selected(self) -> Tuple[Optional[str], Optional[str]]:
        if self.result() == QDialog.DialogCode.Accepted:
            return self.selected_icon, self.selected_color
        return None, None