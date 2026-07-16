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

# 1. Блок: Импорты (убраны RootModel и QObject)
# ==============================================================================
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ConfigDict,
    field_validator,
)

from .app_constants import APPLICATION_ROOT_DIR
from .path_utils import to_relative_if_possible, resolve_path, find_best_relative_path
from .locale_manager import LocaleManager

locale_manager = LocaleManager()
logger = logging.getLogger(f"PyScriptManager.{__name__}")

CONFIG_FILE_NAME = "config.toml"
CONFIG_FILE_PATH = APPLICATION_ROOT_DIR / CONFIG_FILE_NAME

# 2. Блок: Модели данных (без изменений, кроме удаления ConsoleStylesConfig)
# ==============================================================================

class PathsConfig(BaseModel):
    python_interpreter: str = Field(
        default_factory=lambda: str(pathlib.Path(sys.executable).resolve())
    )
    additional_env_paths: List[str] = Field(default_factory=list)
    python_paths: List[str] = Field(default_factory=list)

class UIConfig(BaseModel):
    last_used_script_set: str = Field(default="")
    last_used_sets_collection_file: str = Field(default="")

ValidLogLevels = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
RuntimeContextBackend = Literal["shared_memory", "file"]
RuntimeContextOverflowPolicy = Literal["fail_current_write"]
RuntimeContextCheckpointPolicy = Literal["after_each_script", "on_save_exit"]

class LoggingConfig(BaseModel):
    log_level: ValidLogLevels = Field(default="INFO")
    @field_validator("log_level")
    @classmethod
    def check_log_level_case_insensitive(cls, value: str) -> str:
        return value.upper()

# 3. Блок: Модели GeneralConfig и AppConfigModel (УПРОЩЕНЫ)
# ==============================================================================

class GeneralConfig(BaseModel):
    language: str = Field(default="ru_RU")
    # --- ИЗМЕНЕНИЕ ---
    # Это единственное поле, связанное с темами, которое осталось.
    # Оно хранит имя папки активной темы.
    active_theme_name: str = Field(default="default")


class RuntimeContextConfig(BaseModel):
    backend: RuntimeContextBackend = Field(default="shared_memory")
    shared_memory_min_size_mb: int = Field(default=16, ge=1)
    shared_memory_max_size_mb: int = Field(default=128, ge=1)
    shared_memory_overflow_policy: RuntimeContextOverflowPolicy = Field(
        default="fail_current_write"
    )
    checkpoint_policy: RuntimeContextCheckpointPolicy = Field(
        default="after_each_script"
    )

    @field_validator("checkpoint_policy", mode="before")
    @classmethod
    def migrate_checkpoint_policy(cls, value: str) -> str:
        if value == "on_exit":
            return "on_save_exit"
        return value

    @field_validator("shared_memory_max_size_mb")
    @classmethod
    def check_max_size(cls, value: int, info) -> int:
        min_value = info.data.get("shared_memory_min_size_mb", 1)
        if value < min_value:
            raise ValueError("shared_memory_max_size_mb must be >= shared_memory_min_size_mb")
        return value


class AppConfigModel(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    runtime_context: RuntimeContextConfig = Field(default_factory=RuntimeContextConfig)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    
    # --- ИЗМЕНЕНИЕ ---
    # Поле `themes` полностью удалено из конфигурации.
    
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

# 4. Блок: Класс ConfigManager (УПРОЩЕН)
# ==============================================================================

class ConfigManager:
    def __init__(self, config_path: pathlib.Path = CONFIG_FILE_PATH):
        self.config_path: pathlib.Path = config_path
        self.config: AppConfigModel = self._load_or_create_config()

    # --- ИЗМЕНЕНИЕ ---
    # Методы `_ensure_default_theme_exists` и `get_active_theme` полностью удалены.
    # Их ответственность перешла к новому ThemeManager.

    def _load_or_create_config(self) -> AppConfigModel:
        if self.config_path.exists() and self.config_path.is_file():
            try:
                config_data = toml.load(self.config_path)
                model = AppConfigModel.model_validate(config_data)
                
                # Логика разрешения путей не изменилась
                if model.paths.python_interpreter:
                    model.paths.python_interpreter = resolve_path(
                        model.paths.python_interpreter, base_dir=APPLICATION_ROOT_DIR
                    )
                if model.ui.last_used_sets_collection_file:
                    model.ui.last_used_sets_collection_file = resolve_path(
                        model.ui.last_used_sets_collection_file, base_dir=APPLICATION_ROOT_DIR
                    )
                return model
            except (toml.TomlDecodeError, ValidationError, Exception) as e:
                logger.error(f"Ошибка загрузки конфигурации '{self.config_path}': {e}", exc_info=True)
                default_model_on_error = AppConfigModel()
                self._save_to_file(default_model_on_error)
                return default_model_on_error
        else:
            default_model_on_create = AppConfigModel()
            self._save_to_file(default_model_on_create)
            return default_model_on_create

    def _save_to_file(self, model_to_save: AppConfigModel) -> bool:
        try:
            config_copy_for_save = model_to_save.model_copy(deep=True)

            config_copy_for_save.paths.python_interpreter = find_best_relative_path(
                config_copy_for_save.paths.python_interpreter, base_dir=APPLICATION_ROOT_DIR
            )
            config_copy_for_save.paths.additional_env_paths = [
                find_best_relative_path(p, base_dir=APPLICATION_ROOT_DIR)
                for p in config_copy_for_save.paths.additional_env_paths
            ]
            config_copy_for_save.paths.python_paths = [
                find_best_relative_path(p, base_dir=APPLICATION_ROOT_DIR)
                for p in config_copy_for_save.paths.python_paths
            ]
            config_copy_for_save.ui.last_used_sets_collection_file = to_relative_if_possible(
                config_copy_for_save.ui.last_used_sets_collection_file, base_dir=APPLICATION_ROOT_DIR
            )
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            config_dict = config_copy_for_save.model_dump(mode="python")

            config_text = self._format_config_toml(config_dict)
            with open(self.config_path, "w", encoding="utf-8") as f:
                f.write(config_text)
            return True
        except Exception as e:
            logger.error(f"Не удалось сохранить конфигурацию в '{self.config_path}': {e}", exc_info=True)
            return False

    def _format_config_toml(self, config_dict: Dict[str, Any]) -> str:
        config_text = toml.dumps(config_dict)
        if "[runtime_context]\n" in config_text:
            config_text = config_text.replace(
                "[runtime_context]\n",
                "[runtime_context]\n"
                "# Runtime context backend.\n"
                "# Possible values:\n"
                '# - "shared_memory": default runtime exchange through Python SharedMemory.\n'
                '# - "file": diagnostic fallback through .context.json.\n',
                1,
            )
        config_text = config_text.replace(
            "shared_memory_overflow_policy = \"fail_current_write\"\n",
            "\n"
            "# SharedMemory overflow behavior.\n"
            "# Possible values:\n"
            '# - "fail_current_write": reject the write when payload does not fit the segment.\n'
            "shared_memory_overflow_policy = \"fail_current_write\"\n",
            1,
        )
        checkpoint_comment = (
            "\n"
            "# When PySM writes live runtime context back to .context.json.\n"
            "# Possible values:\n"
            '# - "after_each_script": checkpoint after every completed script.\n'
            '# - "on_save_exit": checkpoint only when the collection is saved or PySM closes with Save.\n'
        )
        for value in ("after_each_script", "on_save_exit"):
            config_text = config_text.replace(
                f"checkpoint_policy = \"{value}\"\n",
                f"{checkpoint_comment}checkpoint_policy = \"{value}\"\n",
                1,
            )
        return config_text

    def save_config(self) -> bool:
        return self._save_to_file(self.config)

    # 5. Блок: Свойства доступа (ПОЛНЫЙ КОД)
    # ==============================================================================
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
