# gui_lib/main_window.py

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QPlainTextEdit, QProgressBar, QComboBox, QFileDialog, QMessageBox, QToolBox, QSizePolicy
)

# Импортируем наши новые модули
from .ui_texts import UI_TEXTS
from .custom_widgets import PathSelectorWidget
from .ui_logic import GuiLogicHandler
from .ui_components import (
    PathsGroup, TasksGroup, ProcessingGroup, ReportGroup,
    ClusteringGroup, MovingGroup, DebugGroup
)
from .workers import QtLogHandler, ProcessingWorker

# Импорты из основного проекта
from main import setup_logging

logger = logging.getLogger("GUI")

# Переносим константу сюда, чтобы избежать циклического импорта
SCRIPT_DIR_INSIDE_LIB = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR_INSIDE_LIB / "face_config.toml"


class MainWindow(QWidget):
    """
    Главное окно приложения. Собирает UI из компонентов
    и управляет общей логикой приложения.
    """
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI_TEXTS["window_title"])

        # --- ИЗМЕНЕНИЕ: Устанавливаем минимальный размер и центрируем окно ---
        
        # 1. Задаем минимально допустимый размер окна
        self.setMinimumSize(850, 800) # Например, 900x700 пикселей

        # 2. Логика для центрирования окна на экране
        try:
            # Получаем геометрию основного экрана
            screen_geometry = QApplication.primaryScreen().geometry()
            # Получаем геометрию нашего окна
            window_geometry = self.frameGeometry()
            # Вычисляем центральную точку и перемещаем окно
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            self.move(window_geometry.topLeft())
        except Exception as e:
            # Если что-то пошло не так (например, в headless-системе),
            # просто используем старый метод
            logger.warning(f"Не удалось отцентрировать окно, используется геометрия по умолчанию: {e}")
            self.setGeometry(100, 100, 850, 800)

        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # --- Инициализация основных сущностей (без изменений) ---
        self.current_config_path = str(DEFAULT_CONFIG_PATH)
        self.loaded_config_data: Optional[Dict] = None
        self.logic_handler = GuiLogicHandler()
        
        self._create_ui_components()

        self.worker_thread: Optional[QThread] = None
        self.processing_worker: Optional[ProcessingWorker] = None
        
        self.init_ui()
        self._setup_logging()
        
        self.load_initial_config()
        self._connect_signals()

    def _create_ui_components(self):
        """Создает экземпляры всех групп виджетов."""
        self.paths_group = PathsGroup()
        self.tasks_group = TasksGroup()
        self.processing_group = ProcessingGroup()
        self.report_group = ReportGroup()
        self.clustering_group = ClusteringGroup()
        self.moving_group = MovingGroup()
        self.debug_group = DebugGroup()

    def init_ui(self):
        """Собирает главный UI из готовых компонентов с новым дизайном."""
        main_layout = QVBoxLayout(self)

        # 1. Верхняя часть: Файл конфигурации (без изменений)
        self._create_config_file_section(main_layout)

        # 2. Средняя часть: Основные настройки (в две колонки)
        settings_layout = QHBoxLayout()
        
        # --- ЛЕВАЯ КОЛОНКА ---
        left_column_layout = QVBoxLayout()
        left_column_layout.addWidget(self.paths_group)
        left_column_layout.addWidget(self.tasks_group)
        log_level_layout = QHBoxLayout()
        log_level_layout.addWidget(QLabel(UI_TEXTS["log_level_label"]))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.currentTextChanged.connect(self.change_log_level)
        log_level_layout.addWidget(self.log_level_combo)
        log_level_layout.addStretch(1)
        left_column_layout.addLayout(log_level_layout)
        left_column_layout.addStretch(1)

        # --- ПРАВАЯ КОЛОНКА (Аккордеон) ---
        self.tool_box = QToolBox()
        self.tool_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)        
        # --- ИЗМЕНЕНИЕ: Настраиваем виджеты перед добавлением ---
        
        # Вкладка 1: Параметры базового анализа
        self.processing_group.setTitle("")  # Убираем заголовок
        self.processing_group.setFlat(True) # Убираем рамку
        self.processing_group.setStyleSheet("QGroupBox { border: 0; margin-top: 0em; }")
        self.tool_box.addItem(self.processing_group, UI_TEXTS["processing_group_title"])
        
        # Вкладка 2: Параметры кластеризации
        self.clustering_group.setTitle("")  # Убираем заголовок
        self.clustering_group.setFlat(True) # Убираем рамку
        self.processing_group.setStyleSheet("QGroupBox { border: 0; margin-top: 1em; }")
        self.tool_box.addItem(self.clustering_group, UI_TEXTS["clustering_group_title"])
        
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        
        # Вкладка 3: Прочие параметры (здесь оставляем группы как есть)
        other_settings_widget = QWidget()
        other_settings_layout = QVBoxLayout(other_settings_widget)
        other_settings_layout.setContentsMargins(0, 0, 0, 0)
        other_settings_layout.addWidget(self.moving_group)
        other_settings_layout.addWidget(self.report_group)
        other_settings_layout.addWidget(self.debug_group)
        other_settings_layout.addStretch(1)
        self.tool_box.addItem(other_settings_widget, "Прочие параметры")
        
        # Добавляем колонки в основной макет настроек
        settings_layout.addLayout(left_column_layout, 1)
        settings_layout.addWidget(self.tool_box, 1)
        
        main_layout.addLayout(settings_layout)

        # 3. Нижняя часть: Кнопки, Лог, Прогресс
        self._create_bottom_section(main_layout)
        self.setLayout(main_layout)

    def _setup_logging(self):
        """Настраивает перехват логов для вывода в GUI."""
        self.qt_log_handler = QtLogHandler(self.log_signal)
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        gui_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        self.qt_log_handler.setFormatter(gui_formatter)
        logging.getLogger().addHandler(self.qt_log_handler)
        logger.info("QtLogHandler добавлен к корневому логгеру.")
    
    def _connect_signals(self):
        """Соединяет сигналы и слоты."""
        self.log_signal.connect(self._add_log_message)
        main_task_cb = self.tasks_group.get_main_task_checkbox()
        main_task_cb.stateChanged.connect(self._on_main_task_state_changed)
    
    # Блок 4: Изменение _create_config_file_section
    def _create_config_file_section(self, parent_layout: QVBoxLayout):
        """Создает секцию для выбора файла конфигурации."""
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel(UI_TEXTS["config_label"]))
        
        self.config_path_selector = PathSelectorWidget(
            dialog_title=UI_TEXTS["dialog_select_config_title"],
            select_type='file',
            file_filter=UI_TEXTS["dialog_select_config_filter"]
        )
        self.config_path_selector.setText(str(DEFAULT_CONFIG_PATH))
        
        # --- ИЗМЕНЕНИЕ: Устанавливаем поле только для чтения ---
        self.config_path_selector.line_edit.setReadOnly(True)
        
        self.config_path_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        config_layout.addWidget(self.config_path_selector)
        
        self.load_config_btn = QPushButton(UI_TEXTS["config_load_btn"])
        self.load_config_btn.clicked.connect(self.load_config_action)
        config_layout.addWidget(self.load_config_btn)
        
        parent_layout.addLayout(config_layout)



    def _create_bottom_section(self, parent_layout: QVBoxLayout):
        """Создает нижнюю часть UI (теперь без уровня логирования)."""
        # --- УДАЛЕНО: Уровень логирования переехал в init_ui ---

        # Кнопки управления
        button_layout = QHBoxLayout()
        self.save_config_btn = QPushButton(UI_TEXTS["save_config_btn"])
        self.save_config_btn.clicked.connect(self.save_config_action)
        button_layout.addWidget(self.save_config_btn)
        self.run_btn = QPushButton(UI_TEXTS["run_btn"])
        self.run_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_processing_action)
        button_layout.addWidget(self.run_btn)
        exit_btn = QPushButton(UI_TEXTS["exit_btn"])
        exit_btn.clicked.connect(self.close)
        button_layout.addWidget(exit_btn)
        parent_layout.addLayout(button_layout)

        # Область вывода лога
        parent_layout.addWidget(QLabel(UI_TEXTS["log_area_label"]))
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(2000)
        size_policy_log = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy_log.setVerticalStretch(1)
        self.log_output.setSizePolicy(size_policy_log)
        parent_layout.addWidget(self.log_output)
        
        # Прогресс бар
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel(UI_TEXTS["progress_label"]))
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        parent_layout.addLayout(progress_layout)



    @Slot(str)
    def _add_log_message(self, message: str):
        """Добавляет сообщение в виджет лога."""
        self.log_output.appendPlainText(message)

    @Slot(int)
    def _on_main_task_state_changed(self, state: int):
        """Обновляет состояние зависимых чекбоксов при изменении главного."""
        is_enabled = state == Qt.CheckState.Checked.value
        self.tasks_group.update_dependent_tasks_state(is_enabled)
    
    @Slot()
    def load_config_action(self):
        """Загружает конфигурацию из файла, указанного в поле ввода."""
        # Получаем путь из нашего нового виджета
        filepath = self.config_path_selector.text()
        if not (filepath and Path(filepath).is_file()):
            QMessageBox.warning(self, UI_TEXTS["msg_error"], UI_TEXTS["msg_file_not_found"].format(filepath))
            return
        
        loaded_data = self.logic_handler.load_config_data(filepath)
        if loaded_data:
            self.load_and_update_gui(filepath, loaded_data)

    @Slot()
    def save_config_action(self):
        """Сохраняет текущую конфигурацию GUI в файл."""
        filepath = self.config_path_selector.text()
        if not filepath:
            start_dir = str(Path(self.current_config_path).parent) if self.current_config_path else str(Path.cwd())
            default_name = Path(self.current_config_path).name if self.current_config_path else "face_config.toml"
            filepath, _ = QFileDialog.getSaveFileName(self, UI_TEXTS["dialog_save_config_title"], str(Path(start_dir) / default_name), UI_TEXTS["dialog_save_config_filter"])
            if not filepath: return
        
        config_to_save = self.get_config_from_gui()
        try:
            self.logic_handler.save_config_to_file(filepath, config_to_save)
            self.current_config_path = filepath
            self.config_path_selector.setText(filepath)
            QMessageBox.information(self, UI_TEXTS["msg_info"], UI_TEXTS["msg_config_saved"])
        except IOError as e:
            QMessageBox.critical(self, UI_TEXTS["msg_error"], str(e))
        except Exception as e:
            logger.error(f"Неожиданная ошибка при сохранении: {e}", exc_info=True)
            QMessageBox.critical(self, UI_TEXTS["msg_error"], f"Ошибка:\n{e}")
        self.config_path_selector.setText(self.current_config_path)

        
    @Slot()
    def run_processing_action(self):
        """Запускает процесс обработки в отдельном потоке после сохранения конфига."""
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, UI_TEXTS["msg_warning"], UI_TEXTS["msg_processing_running"])
            return

        config_path_to_use = self.config_path_selector.text() or self.current_config_path
        if not config_path_to_use:
            config_path_to_use, _ = QFileDialog.getSaveFileName(self, UI_TEXTS["dialog_save_config_title"], str(DEFAULT_CONFIG_PATH), UI_TEXTS["dialog_save_config_filter"])
            if not config_path_to_use:
                logger.warning(UI_TEXTS["msg_save_cancelled"])
                return
            if not config_path_to_use.lower().endswith(".toml"):
                config_path_to_use += ".toml"
            self.current_config_path = config_path_to_use
            self.config_path_selector.setText(config_path_to_use)

        confirmation_text = UI_TEXTS["msg_confirm_run_text"].format(config_path_to_use)
        reply = QMessageBox.question(self, UI_TEXTS["msg_confirm_run_title"], confirmation_text, buttons=QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, defaultButton=QMessageBox.StandardButton.Ok)

        if reply == QMessageBox.StandardButton.Cancel:
            logger.info("Запуск обработки отменен пользователем.")
            return

        logger.info(f"Сохранение настроек перед запуском в: {config_path_to_use}")
        config_saved = False
        try:
            config_to_save = self.get_config_from_gui()
            self.logic_handler.save_config_to_file(config_path_to_use, config_to_save)
            config_saved = True
        except IOError as io_err:
            logger.error(f"Ошибка сохранения файла конфигурации перед запуском: {io_err}")
            QMessageBox.critical(self, UI_TEXTS["msg_error"], str(io_err))
        except Exception as other_err:
            logger.error(f"Критическая ошибка при сохранении конфига перед запуском: {other_err}", exc_info=True)
            QMessageBox.critical(self, UI_TEXTS["msg_error"], f"Критическая ошибка при сохранении настроек:\n{other_err}")

        if not config_saved:
            return

        self.current_config_path = config_path_to_use
        
        logger.info(f"Запуск обработки с конфигурацией: {self.current_config_path}")
        self.log_output.clear()
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("background-color: orange; font-weight: bold;")
        
        self.worker_thread = QThread()
        self.processing_worker = ProcessingWorker(self.current_config_path)
        self.processing_worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.processing_worker.run)
        self.processing_worker.processing_finished.connect(self.on_processing_finished)
        self.processing_worker.log_message.connect(self._add_log_message)
        self.processing_worker.progress_updated.connect(self.update_progress_bar)
        self.processing_worker.error_occurred.connect(self.on_processing_error)
        
        self.processing_worker.processing_finished.connect(self.worker_thread.quit)
        
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.processing_worker.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_finished)
        
        self.worker_thread.start()

    @Slot(bool)
    def on_processing_finished(self, success: bool):
        """Слот, вызываемый при успешном или неуспешном завершении обработки."""
        self.run_btn.setEnabled(True)
        self.run_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        if success:
            self.progress_bar.setValue(100)
        
        QMessageBox.information(
            self,
            UI_TEXTS["msg_info"],
            UI_TEXTS["msg_processing_status"].format(
                UI_TEXTS["msg_status_success"] if success else UI_TEXTS["msg_status_error"]
            ),
        )
        
    @Slot(str)
    def on_processing_error(self, error_message: str):
        """Слот, вызываемый при критической ошибке в потоке."""
        QMessageBox.critical(self, UI_TEXTS["msg_error"], UI_TEXTS["msg_critical_error"].format(error_message))
        self.run_btn.setEnabled(True)
        self.run_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()

    @Slot(int, int)
    def update_progress_bar(self, current: int, total: int):
        """Обновляет прогресс бар."""
        self.progress_bar.setValue(int((current / total) * 100) if total > 0 else 0)

    @Slot(str)
    def change_log_level(self, level_str: str):
        """Изменяет уровень логирования."""
        level = getattr(logging, level_str.upper(), logging.INFO)
        logging.getLogger().setLevel(level)
        if self.qt_log_handler:
            self.qt_log_handler.setLevel(level)
        logger.info(f"Уровень логирования изменен на: {level_str}")
            
    def load_initial_config(self):
        """Загружает конфигурацию при запуске приложения."""
        if not Path(self.current_config_path).is_file():
            logger.warning(f"Файл конфигурации по умолчанию не найден: {self.current_config_path}")
            self.set_gui_from_config({})
            return
        loaded_data = self.logic_handler.load_config_data(self.current_config_path)
        self.load_and_update_gui(self.current_config_path, loaded_data)

    def load_and_update_gui(self, filepath: str, config_data: Optional[dict]):
        """Обновляет все компоненты GUI из словаря конфигурации."""
        if not config_data:
            logger.warning(f"Нет данных для обновления GUI из файла {filepath}")
            return
        self.current_config_path = filepath
        self.config_path_selector.setText(filepath)
        self.set_gui_from_config(config_data)
        logger.info(f"GUI обновлен из файла: {filepath}")

    def get_config_from_gui(self) -> Dict[str, Any]:
        """Собирает данные со всех компонентов UI в единый словарь."""
        config = {}
        config.update(self.paths_group.get_data())
        config.update(self.tasks_group.get_data())
        config.update(self.processing_group.get_data())
        config.update(self.report_group.get_data())
        config.update(self.clustering_group.get_data())
        config.update(self.moving_group.get_data())
        config.update(self.debug_group.get_data())
        config["logging_level"] = self.log_level_combo.currentText()
        return config

    def set_gui_from_config(self, config_data: Dict[str, Any]):
        """Устанавливает данные во все компоненты UI из словаря."""
        self.paths_group.set_data(config_data)
        self.tasks_group.set_data(config_data)
        self.processing_group.set_data(config_data)
        self.report_group.set_data(config_data)
        self.clustering_group.set_data(config_data)
        self.moving_group.set_data(config_data)
        self.debug_group.set_data(config_data)
        self.log_level_combo.setCurrentText(config_data.get("logging_level", "INFO"))

    def closeEvent(self, event):
        """Корректно обрабатывает закрытие окна, дожидаясь завершения потока."""
        logger.info("Окно GUI закрывается.")

        if self.worker_thread and self.worker_thread.isRunning():
            logger.warning("Обработка прерывается из-за закрытия окна. Ожидание завершения потока...")
            self.worker_thread.quit()
            if not self.worker_thread.wait(3000):
                logger.error("Рабочий поток не завершился вовремя.")
        
        if self.qt_log_handler:
            logging.getLogger().removeHandler(self.qt_log_handler)
            self.qt_log_handler.close()
            self.qt_log_handler = None
            
        logger.info("Закрытие окна принято.")
        event.accept()

    @Slot()
    def _on_thread_finished(self):
        """Слот, вызываемый по сигналу QThread.finished. Очищает ссылки."""
        logger.debug("Сигнал QThread.finished получен. Очистка ссылок на воркер и поток.")
        self.processing_worker = None
        self.worker_thread = None