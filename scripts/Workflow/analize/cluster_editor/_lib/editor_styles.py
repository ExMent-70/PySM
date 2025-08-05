# analize/cluster_editor/_lib/editor_styles.py
"""
Модуль для хранения всех констант QSS (стилей) и размеров
для редактора кластеров.
"""

# --- Константы размеров ---
THUMBNAIL_SIZE = 180
PREVIEW_SIZE = 180

# --- Основной стиль окна ---
MAIN_WINDOW_STYLE = "QWidget { background-color: #252525; color: #eee; }"

# --- Стиль для кнопок ---
BUTTON_STYLE = """
    QPushButton { 
        background-color: #007bff; 
        color: white; 
        font-weight: bold; 
        border: none; 
        padding: 10px; 
        border-radius: 5px;
    }
    QPushButton:hover { 
        background-color: #0056b3; 
    }
    QPushButton:disabled { 
        background-color: #555; 
        color: #999;
    }
"""

# --- Стиль для выпадающего меню ---
MENU_STYLE = """
    QMenu {
        background-color: #3a3a3a;
        color: #eee;
        border: 1px solid #555;
        padding: 5px;
    }
    QMenu::item {
        padding: 8px 25px 8px 20px;
        border: 1px solid transparent;
        border-radius: 4px;
    }
    QMenu::item:selected {
        background-color: #007bff;
        color: white;
    }
    QMenu::item:disabled {
        color: #777;
    }
    QMenu::separator {
        height: 1px;
        background: #555;
        margin-left: 10px;
        margin-right: 5px;
    }
"""

# --- Стиль для полос прокрутки ---
SCROLLBAR_STYLE = """
    QScrollBar:vertical { 
        border: none; background-color: #252525; 
        width: 12px; margin: 0px; 
    }
    QScrollBar:horizontal { 
        border: none; background-color: #252525; 
        height: 12px; margin: 0px; 
    }
    QScrollBar::handle { 
        background-color: #555; border-radius: 6px; 
    }
    QScrollBar::handle:hover { background-color: #777; }
    QScrollBar::handle:pressed { background-color: #999; }
    QScrollBar::handle:vertical { min-height: 25px; }
    QScrollBar::handle:horizontal { min-width: 25px; }
    QScrollBar::add-line, QScrollBar::sub-line { 
        height: 0px; width: 0px; background: none; border: none; 
    }
    QScrollBar::add-page, QScrollBar::sub-page { background: none; }
"""


LIST_WIDGET_STYLE = """
    QListWidget {
        background-color: #252525; /* Темный фон */
        border: 1px solid #444;    /* Рамка, как у ScrollArea */
        border-radius: 5px;
        padding: 5px;              /* Небольшой внутренний отступ */
    }
    QListWidget::item {
        /* Убираем стандартное выделение, так как наш делегат рисует свое */
        outline: 0;
        border: none;
        background-color: transparent;
        color: transparent; /* Делаем стандартный текст невидимым */
    }
    QListWidget::item:selected {
        background-color: transparent;
        color: transparent;
    }
    QListWidget::item:hover {
        background-color: transparent;
    }
"""


# --- Стили для виджетов-контейнеров и прочего ---
SCROLL_AREA_STYLE = "QScrollArea { border: 1px solid #444; border-radius: 5px; }"
TITLE_LABEL_STYLE = "font-size: 12pt; margin-left: 5px;"
SEARCH_BAR_STYLE = """
    QLineEdit {
        border: 1px solid #444;
        border-radius: 5px;
        padding: 5px;
        background-color: #3a3a3a;
        color: #eee;
    }
"""


# --- Стиль для QProgressBar ---
PROGRESS_BAR_STYLE_ACTIVE = """
    QProgressBar {
        border: 1px solid #444;
        border-radius: 5px;
        background-color: #3a3a3a;
        text-align: center;
        color: #eee;
        font-weight: bold;
    }
    QProgressBar::chunk {
        background-color: #007bff;
        border-radius: 5px;
    }
"""

# --- НОВЫЙ СТИЛЬ: для неактивного состояния ---
PROGRESS_BAR_STYLE_INACTIVE = """
    QProgressBar {
        border: 1px solid transparent; /* Прозрачная рамка */
        border-radius: 5px;
        background-color: transparent; /* Прозрачный фон */
        text-align: center;
        color: transparent; /* Прозрачный текст */
    }
    QProgressBar::chunk {
        background-color: transparent; /* Прозрачная заполненная часть */
    }
"""

# --- ИЗМЕНЯЕМЫЙ БЛОК: SLIDER_STYLE в editor_styles.py ---

# --- Стиль для QSlider ---
SLIDER_STYLE = """
    QSlider::groove:horizontal {
        border: 1px solid #444;
        height: 4px; /* << Уменьшили толщину "рельсы" */
        background: #3a3a3a;
        margin: 2px 0;
        border-radius: 2px; /* Уменьшили радиус скругления */
    }
    QSlider::handle:horizontal {
        background: #007bff;
        border: 1px solid #0056b3;
        width: 16px; /* Ширина ползунка */
        height: 16px; /* Высота ползунка (опционально, для круглого вида) */
        margin: -6px 0; /* << Скорректировали отступ, чтобы ползунок был по центру */
        border-radius: 8px; /* Делает ползунок круглым */
    }
    QSlider::handle:horizontal:hover {
        background: #0056b3;
    }
    QSlider::sub-page:horizontal {
        background: #007bff;
        border: 1px solid #444;
        height: 4px; /* << Должна совпадать с groove */
        border-radius: 2px;
    }
    QSlider::add-page:horizontal {
        background: #555;
        border: 1px solid #444;
        height: 4px; /* << Должна совпадать с groove */
        border-radius: 2px;
    }
"""

# --- Стиль для компактных оранжевых кнопок ---
BUTTON_STYLE_ORANGE_COMPACT = """
    QPushButton { 
        background-color: #fd7e14; /* Оранжевый цвет (Bootstrap Orange) */
        color: white; 
        font-weight: bold; 
        border: none; 
        padding: 5px 10px; /* Уменьшенные вертикальные поля */
        border-radius: 5px;
    }
    QPushButton:hover { 
        background-color: #e87311; /* Более темный оранжевый при наведении */
    }
    QPushButton:disabled { 
        background-color: #555; 
        color: #999;
    }
"""