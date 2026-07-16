# -*- coding: utf-8 -*-

"""
Модуль для построения графического интерфейса (UI Builder) главного окна редактора.
Отделяет логику визуального представления от бизнес-логики.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSplitter,
    QLineEdit, QMenu, QListWidget, QSlider, QTextEdit, QProgressBar, QSpinBox,
)
from PySide6.QtCore import Qt

from .editor_widgets import ImageDragListWidget, ClusterDropListWidget, FaceDetailsWidget
from .editor_delegates import FACE_MIN, FACE_MAX, FACE_SIZE, FACE_SIZE_PORTRAIT

# --- Интеграция с контекстом системы ---
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_theme_api import set_widget_class
    from pysm_lib.pysm_icons import icons as pysm_icons    
    try:
        from pysm_lib.window_state_manager import WindowStateManager
    except ImportError:
        try:
            from pysm_lib.win_state_claster_editor import WindowStateManager
        except ImportError:
            WindowStateManager = None
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    pysm_icons = None
    WindowStateManager = None
    def set_widget_class(widget, css_class):
        pass


class EditorUIBuilder:
    """Класс-строитель, отвечающий за инициализацию всех виджетов главного окна."""

    @staticmethod
    def build_ui(window):
        """Создает и размещает все виджеты на переданном экземпляре QMainWindow."""
        window.setGeometry(0, 0, 1460, 900)
        
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        window.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(window.main_splitter, 1)        

        # =====================================================================
        # 1. LEFT PANEL (Список кластеров)
        # =====================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_label_text = "Список кластеров"
        if window.mode == 'matches': left_label_text = "Эталоны (Портреты)"
        elif window.mode == 'cleaning': left_label_text = "Технические группы"
        
        window.cluster_list_title = QLabel(
            f"{window.photo_session}: {left_label_text}"
        )
        left_layout.addWidget(window.cluster_list_title)
        
        window.search_bar = QLineEdit()
        window.search_bar.setPlaceholderText("Поиск...")
        window.search_bar.textChanged.connect(window._on_search_text_changed)
        left_layout.addWidget(window.search_bar)

        window.cluster_list_widget = ClusterDropListWidget(window)
        window.cluster_list_widget.setItemDelegate(window.cluster_delegate)
        window.cluster_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        window.cluster_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        window.cluster_list_widget.setSpacing(10)
        
        window.cluster_list_widget.itemDoubleClicked.connect(window.menu_manager.rename_cluster_action)
                
        window.cluster_list_widget.currentItemChanged.connect(window._on_cluster_selected)
        window.cluster_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        
        window.cluster_list_widget.customContextMenuRequested.connect(window.menu_manager.show_cluster_context_menu)
        
        window.cluster_list_widget.setAcceptDrops(True)
        window.cluster_list_widget.setDragDropMode(QListWidget.DragDropMode.DropOnly) 
        window.cluster_list_widget.itemsDropped.connect(window._handle_drop)

        left_layout.addWidget(window.cluster_list_widget, 1)

        btn_layout = QHBoxLayout()
        if window.mode != 'cleaning':
            window.export_button = QPushButton("Экспорт")
            export_menu = QMenu(window)
            export_menu.addAction("Все кластеры").triggered.connect(window._on_export_all_triggered)
            export_menu.addAction("Активный кластер").triggered.connect(window._on_export_active_triggered)
            window.export_button.setMenu(export_menu)
            btn_layout.addWidget(window.export_button)

        window.save_button = QPushButton("Сохранить")
        if window.mode == 'cleaning':
            window.save_button.setText("Удалить мусор и Сохранить")
            window.save_button.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        window.save_button.clicked.connect(lambda: window._save_changes(silent=False))
        btn_layout.addWidget(window.save_button)
        
        left_layout.addLayout(btn_layout)

        # =====================================================================
        # 2. CENTER PANEL (Галерея и Фильтры)
        # =====================================================================
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        window.gallery_label = QLabel("Галерея")
        center_layout.addWidget(window.gallery_label)
        
        window.image_list_widget = ImageDragListWidget(window)
        window.image_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        window.image_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        window.image_list_widget.setSpacing(10)
        window.image_list_widget.setItemDelegate(window.image_delegate)
        window.image_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        window.image_list_widget.setDragEnabled(True)
        window.image_list_widget.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        window.image_list_widget.itemDoubleClicked.connect(window._open_image_viewer)
        window.image_list_widget.currentItemChanged.connect(window._update_face_panel)
        
        window.image_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        
        window.image_list_widget.customContextMenuRequested.connect(window.menu_manager.show_gallery_context_menu)
        
        center_layout.addWidget(window.image_list_widget, 1)

        # --- Панель фильтров ---
        filter_widget = QWidget()
        filter_layout = QVBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 5)
        
        row1 = QHBoxLayout()
        row1.addStretch()
        row1.addSpacing(10)
        row1.addWidget(QLabel("Пол:"))
        
        window.btn_filter_male = QPushButton("")
        if pysm_icons: window.btn_filter_male.setIcon(pysm_icons.get_qicon("GENDER_MALE", 16))
        window.btn_filter_male.setCheckable(True)
        window.btn_filter_male.setToolTip("Только мужчины")
        
        window.btn_filter_female = QPushButton("")
        if pysm_icons: window.btn_filter_female.setIcon(pysm_icons.get_qicon("GENDER_FEMALE", 16))
        window.btn_filter_female.setCheckable(True)
        window.btn_filter_female.setToolTip("Только женщины")
        
        row1.addWidget(window.btn_filter_male)
        row1.addWidget(window.btn_filter_female)
        row1.addSpacing(10)
        
        row1.addWidget(QLabel("Лицо:"))
        window.btn_filter_eyes = QPushButton("")
        if pysm_icons: window.btn_filter_eyes.setIcon(pysm_icons.get_qicon("EYE_CLOSED", 16))
        window.btn_filter_eyes.setCheckable(True)
        window.btn_filter_eyes.setToolTip("Есть лицо с закрытыми глазами")
        
        window.btn_filter_mouth = QPushButton("")
        if pysm_icons: window.btn_filter_mouth.setIcon(pysm_icons.get_qicon("MOUTH_OPEN", 16))
        window.btn_filter_mouth.setCheckable(True)
        window.btn_filter_mouth.setToolTip("Есть лицо с открытым ртом")
        
        row1.addWidget(window.btn_filter_eyes)
        row1.addWidget(window.btn_filter_mouth)

        window.btn_filter_beauty = QPushButton("✨ AI Оценка")
        window.btn_filter_beauty.setCheckable(True)
        window.btn_filter_beauty.setToolTip("AI-оценка выше указанного значения")
        
        window.spin_beauty_score = QSpinBox()
        window.spin_beauty_score.setPrefix("> ")
        window.spin_beauty_score.setRange(0, 100)
        window.spin_beauty_score.setValue(15)
        window.spin_beauty_score.setEnabled(False)
        
        row1.addWidget(window.btn_filter_beauty)
        row1.addWidget(window.spin_beauty_score)
        row1.addSpacing(10)
        
        row1.addWidget(QLabel("Тип фото:"))       
        window.btn_filter_portrait = QPushButton("")
        if pysm_icons: window.btn_filter_portrait.setIcon(pysm_icons.get_qicon("PHOTO_PORTRAIT", 16))
        window.btn_filter_portrait.setCheckable(True)
        window.btn_filter_portrait.setToolTip("Только портретные фотографии")
        
        window.btn_filter_group = QPushButton("")
        if pysm_icons: window.btn_filter_group.setIcon(pysm_icons.get_qicon("PHOTO_GROUP", 16))
        window.btn_filter_group.setCheckable(True)
        window.btn_filter_group.setToolTip("Только групповые фотографии")
        
        window.spin_group_count = QSpinBox()
        window.spin_group_count.setPrefix("< ")
        window.spin_group_count.setSuffix(" чел.")
        window.spin_group_count.setRange(2, 100)
        window.spin_group_count.setValue(5)
        window.spin_group_count.setEnabled(False)
        
        row1.addWidget(window.btn_filter_portrait)
        row1.addWidget(window.btn_filter_group)
        row1.addWidget(window.spin_group_count)
        row1.addSpacing(10)

        window.btn_filter_selected_photos = QPushButton("")
        if pysm_icons:
            window.btn_filter_selected_photos.setIcon(
                pysm_icons.get_qicon("SELECT_FILES", 16)
            )
        window.btn_filter_selected_photos.setCheckable(True)
        window.btn_filter_selected_photos.setToolTip(
            "Только фотографии, выбранные пользователями"
        )
        window.btn_filter_selected_photos.toggled.connect(
            window._on_selected_photos_toggled
        )
        row1.addWidget(window.btn_filter_selected_photos)
        row1.addStretch()
        
        filter_layout.addLayout(row1)
        center_layout.addWidget(filter_widget)
        # ----------------------------------

        # =====================================================================
        # 3. RIGHT PANEL (Детали лица)
        # =====================================================================
        window.main_splitter.addWidget(left_widget)
        window.main_splitter.addWidget(center_widget)
        
        if window.data_manager.strategy.show_face_details_panel():
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            right_layout.setContentsMargins(0, 0, 0, 0)
            
            window.photo_info_label = QLabel("Информация о фото")
            right_layout.addWidget(window.photo_info_label)
            
            window.photo_info_viewer = QTextEdit()
            window.photo_info_viewer.setReadOnly(True)
            right_layout.addWidget(window.photo_info_viewer, 15) 
            
            right_layout.addWidget(QLabel("Лица на фото"))
            window.face_details_widget = FaceDetailsWidget(window, mode=window.mode)
            if IS_MANAGED_RUN: set_widget_class(window.face_details_widget, "face-panel")
            
            window.face_details_widget.itemClicked.connect(window._on_face_item_clicked)
            window.face_details_widget.itemDoubleClicked.connect(window._on_face_item_double_clicked)
            window.face_details_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            
            window.face_details_widget.customContextMenuRequested.connect(window.menu_manager.show_face_details_context_menu)            

            right_layout.addWidget(window.face_details_widget, 51)
            
            window.face_size_slider = QSlider(Qt.Orientation.Horizontal)
            window.face_size_slider.setRange(FACE_MIN, FACE_MAX)
            if window.mode == 'face':
                window.face_size_slider.setValue(FACE_SIZE_PORTRAIT)
            else:
                window.face_size_slider.setValue(FACE_SIZE)
                
            window.face_size_slider.valueChanged.connect(window._on_face_size_changed)
            right_layout.addWidget(window.face_size_slider)
            
            right_layout.addWidget(QLabel("Информация о выбранном лице"))
            window.face_info_viewer = QTextEdit()
            window.face_info_viewer.setReadOnly(True)
            right_layout.addWidget(window.face_info_viewer, 24)

            window.main_splitter.addWidget(right_widget)
            
            window.main_splitter.setStretchFactor(0, 30)
            window.main_splitter.setStretchFactor(1, 35)
            window.main_splitter.setStretchFactor(2, 35)
        else:
            window.main_splitter.setStretchFactor(0, 35)
            window.main_splitter.setStretchFactor(1, 65)

        window.status_bar = QProgressBar()
        window.status_bar.setTextVisible(True)
        main_layout.addWidget(window.status_bar)
        
        
        # --- Загрузка состояния окна и сплиттеров ---
        if IS_MANAGED_RUN and pysm_context and window.win_state_var_name and WindowStateManager:
            mode_var_name = f"{window.win_state_var_name}.{window.mode}"
            saved_state = pysm_context.get_structured(mode_var_name, dict())
            if saved_state:
                WindowStateManager.restore_state(
                    window=window,
                    state_data=saved_state,
                    splitters={'main': window.main_splitter}
                )
