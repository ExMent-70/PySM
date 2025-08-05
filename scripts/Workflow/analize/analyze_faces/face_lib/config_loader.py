# analize/analyze_faces/face_lib/config_loader.py
"""
Модуль для загрузки, валидации и управления конфигурацией
этапа анализа лиц с использованием Pydantic.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# --- Блок 1: Pydantic Модели для валидации config.toml ---
# ==============================================================================

class PathsConfig(BaseModel):
    """Настройки путей, используемые на этапе анализа."""
    model_root: str = "models"

class ProviderConfig(BaseModel):
    """Настройки ONNX Runtime Provider."""
    provider_name: Optional[str] = None
    device_id: int = 0
    tensorRT_cache_path: str = "TensorRT_cache"

class ModelConfig(BaseModel):
    """Настройки моделей для анализа."""
    name: str = "antelopev2"
    det_thresh: float = 0.5
    det_size: List[int] = Field(default=[1280, 1280])
    gender_model: str = "FACEONNX/gender_efficientnet_b2.onnx"
    emotion_model: str = "FACEONNX/emotion_cnn.onnx"
    age_model: str = "FACEONNX/age_efficientnet_b2.onnx"
    beauty_model: str = "FACEONNX/beauty_resnet18.onnx"
    eyeblink_model: str = "FACEONNX/eye_blink_cnn.onnx"
    emotion_labels: List[str] = Field(default=["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"])

class TaskFlags(BaseModel):
    """Флаги для включения/отключения задач анализа."""
    analyze_gender: bool = True
    analyze_emotion: bool = True
    analyze_age: bool = True
    analyze_beauty: bool = False
    analyze_eyeblink: bool = True
    save_debug_kps: bool = False

class AppConfig(BaseModel):
    """Корневая модель конфигурации."""
    paths: PathsConfig = Field(default_factory=PathsConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    task_flags: TaskFlags = Field(default_factory=TaskFlags)


# --- Блок 2: Класс ConfigManager ---
# ==============================================================================

class ConfigManager:
    """Управляет загрузкой, валидацией и доступом к конфигурации."""

    def __init__(self, config_path: Path):
        """
        Инициализирует менеджер конфигурации.

        Args:
            config_path: Путь к файлу config.toml.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_and_validate()
        self._resolve_paths()

    def _load_and_validate(self) -> Dict[str, Any]:
        """Загружает TOML, валидирует с помощью Pydantic и возвращает как словарь."""
        if not self.config_path.is_file():
            logger.error(f"Файл конфигурации не найден: {self.config_path.resolve()}")
            raise FileNotFoundError(f"Config file not found: {self.config_path.resolve()}")
        try:
            config_data = toml.load(self.config_path)
            validated_config = AppConfig(**config_data).model_dump(mode="python")
            logger.info(f"Дополнительные настройки скрипта загружены из файла {self.config_path.name}.")
            return validated_config
        except ValidationError as e:
            logger.error(f"Ошибка валидации конфигурации в {self.config_path.name}:\n{e}")
            raise
        except Exception as e:
            logger.error(f"Не удалось загрузить или прочитать файл конфигурации: {e}")
            raise

    def _resolve_paths(self):
        """
        Преобразует относительные пути в конфиге в абсолютные,
        используя директорию самого config.toml как базовую.
        """
        base_dir = self.config_path.parent
        
        # Путь к корневой папке моделей
        model_root = Path(self.config["paths"]["model_root"])
        if not model_root.is_absolute():
            self.config["paths"]["model_root"] = str((base_dir / model_root).resolve())
        logger.debug(f"Абсолютный путь к моделям: {self.config['paths']['model_root']}")

        # Путь к кэшу TensorRT
        trt_cache = Path(self.config["provider"]["tensorRT_cache_path"])
        if not trt_cache.is_absolute():
            self.config["provider"]["tensorRT_cache_path"] = str((base_dir / trt_cache).resolve())
        logger.debug(f"Абсолютный путь к кэшу TensorRT: {self.config['provider']['tensorRT_cache_path']}")


    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Получает вложенное значение, используя точечную нотацию (e.g., 'model.det_thresh').
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

