import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

class PathsConfig(BaseModel):
    model_root: str = "../../../../_BIN"
    clip_model_onnx: str = "models/CLIP/ViT-B-32.onnx"

class ProviderConfig(BaseModel):
    provider_name: Optional[str] = None
    device_id: int = 0
    tensorRT_cache_path: str = "../../../../_BIN/TensorRT_cache"

class ModelParamsConfig(BaseModel):
    input_size: List[int] = Field(default=[224, 224])

class ClusteringConfig(BaseModel):
    min_samples: int = 2
    metric: str = "cosine"

class ClassificationConfig(BaseModel):
    match_threshold: float = 0.25

class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    model_params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)

class ConfigManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_and_validate()
        self._resolve_paths()

    def _load_and_validate(self) -> Dict[str, Any]:
        if not self.config_path.is_file():
            logger.error(f"Файл конфигурации не найден: {self.config_path.resolve()}")
            raise FileNotFoundError(f"Config file not found: {self.config_path.resolve()}")
        try:
            config_data = toml.load(self.config_path)
            validated_config = AppConfig(**config_data).model_dump(mode="python")
            print(f"Настройки кластеризации локаций загружены из файла {self.config_path.name}.")
            return validated_config
        except ValidationError as e:
            logger.error(f"Ошибка валидации конфигурации в {self.config_path.name}:\n{e}")
            raise
        except Exception as e:
            logger.error(f"Не удалось загрузить или прочитать файл конфигурации: {e}")
            raise

    def _resolve_paths(self):
        base_dir = self.config_path.parent
        model_root = Path(self.config["paths"]["model_root"])
        if not model_root.is_absolute():
            self.config["paths"]["model_root"] = str((base_dir / model_root).resolve())
        
        trt_cache = Path(self.config["provider"]["tensorRT_cache_path"])
        if not trt_cache.is_absolute():
            self.config["provider"]["tensorRT_cache_path"] = str((base_dir / trt_cache).resolve())

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default