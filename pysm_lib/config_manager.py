# pysm_lib/config_manager.py

import toml
import pathlib
import sys
from typing import (
    List,
    Union,
    Literal,
    Dict,
    Any,
)
import logging

# --- 1. БЛОК: ИЗМЕНЕННЫЕ ИМПОРТЫ ---
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ConfigDict,
    field_validator,
    RootModel,  # <--- НОВЫЙ ИМПОРТ
)

from .app_constants import APPLICATION_ROOT_DIR
from .path_utils import to_relative_if_possible, resolve_path, find_best_relative_path
from .locale_manager import LocaleManager

locale_manager = LocaleManager()
logger = logging.getLogger(f"PyScriptManager.{__name__}")

CONFIG_FILE_NAME = "config.toml"
CONFIG_FILE_PATH = APPLICATION_ROOT_DIR / CONFIG_FILE_NAME


class PathsConfig(BaseModel):
    python_interpreter: str = Field(
        default_factory=lambda: str(pathlib.Path(sys.executable).resolve())
    )
    additional_env_paths: List[str] = Field(
        default_factory=list,
        description=locale_manager.get(
            "config_manager.model_descriptions.additional_env_paths"
        ),
    )
    python_paths: List[str] = Field(
        default_factory=list,
        description=locale_manager.get(
            "config_manager.model_descriptions.python_paths"
        ),
    )


class UIConfig(BaseModel):
    last_used_script_set: str = Field(default="")
    last_used_sets_collection_file: str = Field(default="")


ValidLogLevels = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingConfig(BaseModel):
    log_level: ValidLogLevels = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def check_log_level_case_insensitive(cls, value: str) -> str:
        return value.upper()


# --- 2. БЛОК: КЛАСС ConsoleStylesConfig (ПОЛНОСТЬЮ ПЕРЕРАБОТАН) ---
def get_default_styles() -> Dict[str, str]:
    """Фабрика для создания словаря со стилями по умолчанию."""
    return {
        "api_image_description": "color: #555555; font-size: 11pt; font-style: italic; background-color: transparent; padding: 2px;",
        "api_link": "color: #005f73; font-size: 12pt; text-decoration: none; font-weight: normal;",
        "set_header": "color: #FFFFFF; background-color: #000080; font-weight: bold;",
        "set_info": "color: #000080;",
        "script_header_block": "color: #FFFFFF; background-color: #2E8B57; font-weight: bold;",
        "script_success_block": "color: #FFFFFF; background-color: #2E8B57; font-weight: bold;",
        "script_error_block": "color: #FFFFFF; background-color: #B22222; font-weight: bold;",
        "script_info": "color: #555555; font-family: 'Courier New', monospace;",
        "script_stdout": "color: #000000;",
        "script_stderr": "color: #FF0000;",
        "runner_info": "color: #808080; font-style: italic;",
        "script_arg_value": "color: #000080;",
        "tooltip_script_args_block": "background-color: #e8f0fe; padding: 5px; border-radius: 3px;",
        "tooltip_instance_args_block": "background-color: #eaf5ea; padding: 5px; border-radius: 3px;",        
        "console_background": "background-color: #ffffff;",
        "collection_info": "background-color: #e8f0fe; padding: 5px; border-radius: 3px;",
        "status_running": "background-color: #daf7a6;",
        "status_success": "background-color: #ffffff;",
        "status_error": "background-color: #ffc300;",
        "status_pending": "background-color: #ffffff;",
        "status_skipped": "background-color: #e0e0e0;",
        "script_description": "color: #ffffff; background-color: #0077b6; font-size: 10pt;",
        "tooltip_arg_value": "color: #0077b6;"
    }

class ConsoleStylesConfig(RootModel[Dict[str, str]]):
    """
    Гибкая модель для хранения стилей консоли.
    Использует RootModel для представления себя как словаря,
    что позволяет добавлять произвольные ключи.
    """
    root: Dict[str, str] = Field(default_factory=get_default_styles)

    def get_styles_as_dict(self) -> Dict[str, str]:
        """Возвращает стили в виде словаря."""
        return self.root

    def __getattr__(self, name: str) -> str:
        # Позволяет обращаться к ключам как к атрибутам (например, theme.status_running)
        return self.root.get(name, "")

    def __setattr__(self, name: str, value: Any):
        # Позволяет изменять значения через атрибуты
        if name == "root":
            super().__setattr__(name, value)
        else:
            self.root[name] = value


class GeneralConfig(BaseModel):
    language: str = Field(
        default="ru_RU",
        description=locale_manager.get("config_manager.model_descriptions.language"),
    )
    active_theme_name: str = Field(
        default="default",
        description=locale_manager.get(
            "config_manager.model_descriptions.active_theme_name"
        ),
    )


class AppConfigModel(BaseModel):
    # ... (код этого класса без изменений) ...
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    environment_variables: Dict[str, str] = Field(
        default_factory=dict,
        description=locale_manager.get(
            "config_manager.model_descriptions.environment_variables"
        ),
    )
    # --- ИЗМЕНЕНИЕ: Тип и фабрика для themes ---
    themes: Dict[str, ConsoleStylesConfig] = Field(
        default_factory=lambda: {"default": ConsoleStylesConfig()}
    )
    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class ConfigManager:
    def __init__(self, config_path: pathlib.Path = CONFIG_FILE_PATH):
        self.config_path: pathlib.Path = config_path
        self.config: AppConfigModel = self._load_or_create_config()
        self._ensure_default_theme_exists()

    def _ensure_default_theme_exists(self):
        """Гарантирует, что тема 'default' всегда существует."""
        if "default" not in self.config.themes:
            self.config.themes["default"] = ConsoleStylesConfig()
        if (
            not self.config.general.active_theme_name
            or self.config.general.active_theme_name not in self.config.themes
        ):
            self.config.general.active_theme_name = "default"

    def get_active_theme(self) -> ConsoleStylesConfig:
        """Возвращает модель стилей для активной темы."""
        active_name = self.config.general.active_theme_name
        return self.config.themes.get(active_name, self.config.themes["default"])

    # 3. БЛОК: Метод _load_or_create_config (ИЗМЕНЕН)
    def _load_or_create_config(self) -> AppConfigModel:
        if self.config_path.exists() and self.config_path.is_file():
            try:
                config_data = toml.load(self.config_path)
                
                # --- УДАЛЕНА СТАРАЯ ЛОГИКА МИГРАЦИИ ---
                # Она больше не нужна, так как новая модель Pydantic
                # сама справится с парсингом словарей.

                model = AppConfigModel.model_validate(config_data)
                logger.info(
                    locale_manager.get(
                        "config_manager.log_info.config_loaded", path=self.config_path
                    )
                )

                if model.paths.python_interpreter:
                    model.paths.python_interpreter = resolve_path(
                        model.paths.python_interpreter,
                        base_dir=APPLICATION_ROOT_DIR,
                    )

                if model.ui.last_used_sets_collection_file:
                    model.ui.last_used_sets_collection_file = resolve_path(
                        model.ui.last_used_sets_collection_file,
                        base_dir=APPLICATION_ROOT_DIR,
                    )

                return model
            except (toml.TomlDecodeError, ValidationError, Exception) as e:
                logger.error(
                    locale_manager.get(
                        "config_manager.log_error.load_failed",
                        path=self.config_path,
                        error=e,
                    ),
                    exc_info=True,
                )
                logger.warning(
                    locale_manager.get(
                        "config_manager.log_warning.creating_default_config"
                    )
                )
                default_model_on_error = AppConfigModel()
                self._save_to_file(default_model_on_error)
                return default_model_on_error
        else:
            logger.info(
                locale_manager.get(
                    "config_manager.log_info.config_not_found", path=self.config_path
                )
            )
            default_model_on_create = AppConfigModel()
            self._save_to_file(default_model_on_create)
            return default_model_on_create


# ==============================================================================
    def _save_to_file(self, model_to_save: AppConfigModel) -> bool:
        try:
            config_copy_for_save = model_to_save.model_copy(deep=True)

            # --- ИЗМЕНЕНИЕ: Используем новую, гибкую функцию для путей окружения ---
            config_copy_for_save.paths.python_interpreter = find_best_relative_path(
                config_copy_for_save.paths.python_interpreter,
                base_dir=APPLICATION_ROOT_DIR,
            )

            config_copy_for_save.paths.additional_env_paths = [
                find_best_relative_path(p, base_dir=APPLICATION_ROOT_DIR)
                for p in config_copy_for_save.paths.additional_env_paths
            ]

            config_copy_for_save.paths.python_paths = [
                find_best_relative_path(p, base_dir=APPLICATION_ROOT_DIR)
                for p in config_copy_for_save.paths.python_paths
            ]
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

            # Путь к последней коллекции по-прежнему обрабатывается строгой функцией
            config_copy_for_save.ui.last_used_sets_collection_file = (
                to_relative_if_possible(
                    config_copy_for_save.ui.last_used_sets_collection_file,
                    base_dir=APPLICATION_ROOT_DIR,
                )
            )

            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            config_dict = config_copy_for_save.model_dump(mode="python")

            with open(self.config_path, "w", encoding="utf-8") as f:
                toml.dump(config_dict, f)
            logger.info(
                locale_manager.get(
                    "config_manager.log_info.config_saved", path=self.config_path
                )
            )
            return True
        except Exception as e:
            logger.error(
                locale_manager.get(
                    "config_manager.log_error.save_failed",
                    path=self.config_path,
                    error=e,
                ),
                exc_info=True,
            )
            return False

# ... (остальная часть файла без изменений) ...


    def save_config(self) -> bool:
        return self._save_to_file(self.config)

    @property
    def python_interpreter(self) -> pathlib.Path:
        return pathlib.Path(self.config.paths.python_interpreter).resolve()

    @python_interpreter.setter
    def python_interpreter(self, value: Union[str, pathlib.Path]):
        self.config.paths.python_interpreter = str(value)

    @property
    def additional_env_paths(self) -> List[str]:
        return [
            str(pathlib.Path(p).resolve())
            for p in self.config.paths.additional_env_paths
        ]

    @additional_env_paths.setter
    def additional_env_paths(self, value: List[str]):
        self.config.paths.additional_env_paths = value

    @property
    def python_paths(self) -> List[str]:
        """Возвращает список путей для PYTHONPATH, преобразованных в абсолютные."""
        return [str(pathlib.Path(p).resolve()) for p in self.config.paths.python_paths]

    @python_paths.setter
    def python_paths(self, value: List[str]):
        self.config.paths.python_paths = value

    @property
    def environment_variables(self) -> Dict[str, str]:
        """Возвращает словарь пользовательских переменных окружения."""
        return self.config.environment_variables

    @environment_variables.setter
    def environment_variables(self, value: Dict[str, str]):
        self.config.environment_variables = value

    @property
    def last_used_script_set(self) -> str:
        return self.config.ui.last_used_script_set

    @last_used_script_set.setter
    def last_used_script_set(self, value: str):
        self.config.ui.last_used_script_set = value

    @property
    def last_used_sets_collection_file(self) -> str:
        return self.config.ui.last_used_sets_collection_file

    @last_used_sets_collection_file.setter
    def last_used_sets_collection_file(self, value: str):
        self.config.ui.last_used_sets_collection_file = (
            str(pathlib.Path(value).resolve()) if value else ""
        )

    @property
    def log_level(self) -> ValidLogLevels:
        return self.config.logging.log_level

    @log_level.setter
    def log_level(self, value: str):
        self.config.logging.log_level = value

    @property
    def language(self) -> str:
        return self.config.general.language

    @language.setter
    def language(self, value: str):
        self.config.general.language = value
        
