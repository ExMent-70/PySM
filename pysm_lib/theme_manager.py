# pysm_lib/theme_manager.py
"""
Модуль, отвечающий за управление темами оформления приложения.

Этот модуль инкапсулирует всю логику, связанную со сканированием, загрузкой,
обработкой и предоставлением "тематических пакетов". Он также включает
простой препроцессор для QSS-файлов, позволяющий использовать переменные.
"""

import logging
import pathlib
import re
import shutil
from typing import Dict, List, Optional, Any

import toml
from pydantic import BaseModel, Field, RootModel, ValidationError

from .app_constants import APPLICATION_ROOT_DIR
from .locale_manager import LocaleManager

locale_manager = LocaleManager()
logger = logging.getLogger(f"PyScriptManager.{__name__}")


# 1. Блок: Pydantic-модели для структуры данных тем
# ==============================================================================

def get_default_dynamic_styles() -> Dict[str, str]:
    """
    Фабрика для создания словаря с динамическими стилями по умолчанию.
    Эти стили используются для генерации темы "default", если она отсутствует,
    а также как запасной вариант в случае ошибок.
    """
    return {
        "api_image_description": "color: #555555; font-size: 11pt; font-style: italic;",
        "api_link": "color: #005f73;",
        "set_header": "color: #FFFFFF; background-color: #000080; font-weight: bold;",
        "script_header_block": "color: #FFFFFF; background-color: #2E8B57; font-weight: bold;",
        "script_success_block": "color: #FFFFFF; background-color: #2E8B57; font-weight: bold;",
        "script_error_block": "color: #FFFFFF; background-color: #B22222; font-weight: bold;",
        "script_stdout": "color: #000000;",
        "script_stderr": "color: #FF0000;",
        "console_background": "background-color: #ffffff;",
        "status_running": "background-color: #daf7a6;",
        "status_error": "background-color: #ffc300;",
        "status_skipped": "background-color: #e0e0e0;",
        "tooltip_arg_value": "color: #0077b6;",
        "delegate_hover_border": "color: #0078d7;",
        "delegate_changed_indicator": "color: #f0ad4e;",
        "delegate_preview_background": "color: #e8e8e8;",
        "delegate_secondary_text": "color: #555555;",
        "markup_number": "background-color: #27ae60; border-color: #bbdefb; color: #ffffff;",
        "table_status_found": "color: #ffffff; background-color: #27ae60;",
        "table_status_not_found": "color: #ffffff; background-color: #c0392b;"        
    }


class DynamicStylesModel(RootModel[Dict[str, str]]):
    """
    Модель для хранения динамических стилей из файла theme.toml.
    Использует RootModel для гибкости, позволяя теме быть простым словарем.
    """
    root: Dict[str, str] = Field(default_factory=get_default_dynamic_styles)

    def get_styles_as_dict(self) -> Dict[str, str]:
        """Возвращает стили в виде словаря."""
        return self.root


class ThemeDataModel(BaseModel):
    """Модель для хранения всех данных загруженной темы в памяти."""
    name: str
    qss_content: str
    dynamic_styles: DynamicStylesModel


# 2. Блок: Основной класс ThemeManager
# ==============================================================================

class ThemeManager:
    """
    Управляет сканированием, загрузкой и предоставлением тем оформления.

    Работает с "тематическими пакетами" - папками внутри директории /themes.
    Включает в себя простой препроцессор для QSS-файлов для поддержки переменных.
    """

    def __init__(self):
        """Инициализирует менеджер тем."""
        self.themes_root_dir = APPLICATION_ROOT_DIR / "themes"
        self.themes_root_dir.mkdir(exist_ok=True)
        
        self.available_themes: Optional[List[str]] = None
        self.active_theme_data: Optional[ThemeDataModel] = None

    def _ensure_default_theme_exists(self):
        """Проверяет наличие 'default' темы и создает ее, если она отсутствует."""
        default_dir = self.themes_root_dir / "default"
        qss_file = default_dir / "style.qss"
        toml_file = default_dir / "theme.toml"

        try:
            default_dir.mkdir(exist_ok=True)
            if not qss_file.is_file():
                # Создаем пустой QSS-файл с комментарием
                qss_file.write_text("/* PyScriptManager Default Theme */\n", encoding="utf-8")
            
            if not toml_file.is_file():
                # Создаем TOML-файл со стилями по умолчанию
                default_styles = get_default_dynamic_styles()
                with open(toml_file, "w", encoding="utf-8") as f:
                    toml.dump(default_styles, f)
        except Exception as e:
            logger.critical(f"Не удалось создать файлы для базовой темы 'default': {e}", exc_info=True)

    def _process_qss_variables(self, qss_content: str, theme_name: str) -> str:
        """
        Находит определения переменных (@key = value), заменяет их в тексте
        и удаляет строки с определениями.
        """
        # Паттерн для поиска определений: @имя_переменной = значение_до_конца_строки
        variable_def_pattern = re.compile(r"^\s*@([a-zA-Z0-9_-]+)\s*=\s*(.+?);?\s*$", re.MULTILINE)
        
        variables = {}
        try:
            # 1. Находим все определения и складываем их в словарь
            for match in variable_def_pattern.finditer(qss_content):
                key = match.group(1).strip()
                value = match.group(2).strip()
                if value.endswith(';'):
                    value = value[:-1]
                variables[key] = value

        except Exception as e:
            logger.error(f"Ошибка парсинга переменных в теме '{theme_name}': {e}")
            return qss_content # В случае ошибки возвращаем исходный текст

        if not variables:
            return qss_content # Если переменных нет, ничего не делаем

        # 2. Удаляем строки с определениями из основного текста
        processed_content = variable_def_pattern.sub("", qss_content)
        
        # 3. Заменяем все вхождения переменных (@key) на их значения
        for name, value in variables.items():
            # Используем \b (граница слова), чтобы не заменять @bgColor внутри @bgColorBase
            processed_content = re.sub(rf"@{name}\b", value, processed_content)
            
        return processed_content

    def rescan_themes(self) -> List[str]:
        """(Пере)сканирует директорию themes и обновляет список доступных тем."""
        self._ensure_default_theme_exists()
        themes = []
        if not self.themes_root_dir.is_dir():
            self.available_themes = ["default"] 
            return self.available_themes

        for theme_dir in self.themes_root_dir.iterdir():
            if theme_dir.is_dir():
                qss_file = theme_dir / "style.qss"
                toml_file = theme_dir / "theme.toml"
                if qss_file.is_file() and toml_file.is_file():
                    themes.append(theme_dir.name)
        
        if "default" not in themes:
            logger.warning("Базовая тема 'default' не найдена. Функциональность может быть нарушена.")
            themes.append("default")
        
        self.available_themes = sorted(themes)
        return self.available_themes

    def get_available_themes(self) -> List[str]:
        """
        Возвращает список имен найденных тем.
        Выполняет сканирование только при первом вызове ("ленивая" инициализация).
        """
        if self.available_themes is None:
            self.rescan_themes()
        return self.available_themes if self.available_themes is not None else ["default"]

    def load_theme(self, theme_name: str) -> ThemeDataModel:
        """
        Загружает данные указанной темы. Гарантированно возвращает валидную модель.

        В случае ошибки загрузки запрашиваемой темы, пытается загрузить 'default'.
        В случае критической ошибки, возвращает "аварийную" пустую модель.
        """
        available = self.get_available_themes()
        
        target_theme_name = theme_name
        if target_theme_name not in available:
            logger.error(f"Тема '{target_theme_name}' не найдена. Откат к 'default'.")
            target_theme_name = "default"
        
        if "default" not in available and target_theme_name == "default":
            logger.critical("Критическая ошибка: тема 'default' отсутствует. Будет создана пустая тема.")
            return ThemeDataModel(name="<error>", qss_content="", dynamic_styles=DynamicStylesModel())

        theme_dir = self.themes_root_dir / target_theme_name
        qss_file = theme_dir / "style.qss"
        toml_file = theme_dir / "theme.toml"

        try:
            qss_raw_content = qss_file.read_text(encoding="utf-8")
            qss_content = self._process_qss_variables(qss_raw_content, target_theme_name)
            
            dynamic_styles_data = toml.load(toml_file)
            dynamic_styles_model = DynamicStylesModel.model_validate(dynamic_styles_data)
            
            logger.info(f"Тема '{target_theme_name}' успешно загружена и обработана.")
            return ThemeDataModel(
                name=target_theme_name,
                qss_content=qss_content,
                dynamic_styles=dynamic_styles_model
            )
        except Exception as e:
            logger.error(f"Не удалось загрузить тему '{target_theme_name}': {e}", exc_info=True)
            
            if target_theme_name != "default":
                logger.warning("Произошла ошибка. Выполняется откат к теме 'default'.")
                return self.load_theme("default")
            
            logger.critical("Критическая ошибка: не удалось загрузить даже базовую тему 'default'.")
            return ThemeDataModel(name="<error>", qss_content="", dynamic_styles=DynamicStylesModel())

    def set_active_theme(self, theme_name: str) -> bool:
        """
        Устанавливает тему как активную, загружая ее данные.
        Возвращает True, если запрошенная тема была загружена успешно (без отката).
        """
        loaded_theme = self.load_theme(theme_name)
        self.active_theme_data = loaded_theme
        logger.info(f"Активная тема установлена: '{self.active_theme_data.name}'")
        return self.active_theme_data.name == theme_name

    def get_active_theme_qss(self) -> str:
        """Возвращает обработанный QSS-контент активной темы."""
        if self.active_theme_data:
            return self.active_theme_data.qss_content
        return ""

    def get_active_theme_dynamic_styles(self) -> Dict[str, str]:
        """Возвращает словарь динамических стилей активной темы."""
        if self.active_theme_data:
            return self.active_theme_data.dynamic_styles.get_styles_as_dict()
        return get_default_dynamic_styles()

    def get_active_theme_name(self) -> str:
        """Возвращает имя активной темы."""
        return self.active_theme_data.name if self.active_theme_data else "default"

    def get_qss_file_path_for_theme(self, theme_name: str) -> Optional[pathlib.Path]:
        """Возвращает абсолютный путь к style.qss для указанной темы."""
        if theme_name in self.get_available_themes():
            return self.themes_root_dir / theme_name / "style.qss"
        return None

    def save_dynamic_styles(self, theme_name: str, styles: Dict[str, Any]) -> bool:
        """Сохраняет словарь динамических стилей в файл theme.toml."""
        if theme_name not in self.get_available_themes():
            return False
        
        toml_path = self.themes_root_dir / theme_name / "theme.toml"
        try:
            with open(toml_path, "w", encoding="utf-8") as f:
                toml.dump(styles, f)
            logger.info(f"Динамические стили для темы '{theme_name}' сохранены.")
            return True
        except Exception as e:
            logger.error(f"Не удалось сохранить theme.toml для темы '{theme_name}': {e}")
            return False

    def create_theme_from_existing(self, source_theme_name: str, new_theme_name: str) -> bool:
        """Создает новую тему путем копирования существующей."""
        source_dir = self.themes_root_dir / source_theme_name
        new_dir = self.themes_root_dir / new_theme_name

        if not source_dir.is_dir():
            logger.error(f"Исходная тема '{source_theme_name}' для копирования не найдена.")
            return False
        if new_dir.exists():
            logger.error(f"Тема с именем '{new_theme_name}' уже существует.")
            return False

        try:
            shutil.copytree(source_dir, new_dir)
            self.rescan_themes()
            logger.info(f"Тема '{new_theme_name}' успешно создана из '{source_theme_name}'.")
            return True
        except Exception as e:
            logger.error(f"Не удалось создать тему '{new_theme_name}': {e}", exc_info=True)
            return False

    def delete_theme(self, theme_name: str) -> bool:
        """Удаляет папку с темой."""
        if theme_name == "default":
            logger.warning("Попытка удаления базовой темы 'default'. Операция отменена.")
            return False
        
        theme_dir = self.themes_root_dir / theme_name
        if not theme_dir.is_dir():
            logger.warning(f"Тема '{theme_name}' для удаления не найдена.")
            return True 
        
        try:
            shutil.rmtree(theme_dir)
            self.rescan_themes()
            logger.info(f"Тема '{theme_name}' успешно удалена.")
            return True
        except Exception as e:
            logger.error(f"Не удалось удалить тему '{theme_name}': {e}", exc_info=True)
            return False