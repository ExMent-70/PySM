# gui_lib/custom_widgets.py

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QWidget, QLineEdit, QToolButton, QHBoxLayout, QStyle, QFileDialog
)

class PathSelectorWidget(QWidget):
    """
    Кастомный виджет для выбора пути к папке или файлу.
    """
    def __init__(self, dialog_title: str, select_type: str = 'folder', file_filter: str = "*.*"):
        super().__init__()
        self.dialog_title = dialog_title
        # --- ИЗМЕНЕНИЕ: Добавляем тип и фильтр ---
        self.select_type = select_type  # 'folder' или 'file'
        self.file_filter = file_filter

        # --- Создание виджетов (без изменений) ---
        self.line_edit = QLineEdit()
        self.line_edit.setReadOnly(False)

        self.browse_button = QToolButton()
        icon_browse = self.style().standardIcon(
            QStyle.StandardPixmap.SP_DirOpenIcon if select_type == 'folder' else QStyle.StandardPixmap.SP_FileIcon
        )
        self.browse_button.setIcon(icon_browse)
        self.browse_button.setToolTip("Выбрать...")
        self.browse_button.setCursor(self.cursor())

        self.open_button = QToolButton()
        icon_open = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)
        self.open_button.setIcon(icon_open)
        self.open_button.setToolTip("Открыть расположение...")
        self.open_button.setCursor(self.cursor())
        
        # --- Макет (без изменений) ---
        layout = QHBoxLayout(self.line_edit)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(self.browse_button)
        layout.addWidget(self.open_button)
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.line_edit)

        # --- Соединение сигналов (без изменений) ---
        self.browse_button.clicked.connect(self._on_browse)
        self.open_button.clicked.connect(self._on_open)
    
    def text(self) -> str:
        return self.line_edit.text()

    def setText(self, text: str):
        self.line_edit.setText(text)

    @Slot()
    def _on_browse(self):
        """Слот для кнопки 'Выбрать' (теперь универсальный)."""
        start_path = self.text()
        start_dir = ""
        
        if start_path and Path(start_path).exists():
            start_dir = str(Path(start_path).parent)
        else:
            start_dir = str(Path.home())
        
        # --- ИЗМЕНЕНИЕ: Логика выбора файла или папки ---
        if self.select_type == 'folder':
            path = QFileDialog.getExistingDirectory(self, self.dialog_title, start_dir)
        else: # 'file'
            path, _ = QFileDialog.getOpenFileName(self, self.dialog_title, start_dir, self.file_filter)
        
        if path:
            self.line_edit.setText(str(Path(path)))

    @Slot()
    def _on_open(self):
        """Слот для кнопки 'Открыть' (теперь открывает расположение файла)."""
        path_str = self.text()
        if not path_str:
            return
            
        path = Path(path_str)
        # --- ИЗМЕНЕНИЕ: Открываем папку, содержащую файл/папку ---
        dir_to_open = path.parent if path.is_file() else path

        if dir_to_open.is_dir():
            try:
                if sys.platform == "win32":
                    os.startfile(dir_to_open)
                elif sys.platform == "darwin":
                    subprocess.run(["open", dir_to_open])
                else:
                    subprocess.run(["xdg-open", dir_to_open])
            except Exception as e:
                print(f"Не удалось открыть папку {dir_to_open}: {e}")