from pathlib import Path
from typing import List, Optional
import logging

import toml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class PathsConfig(BaseModel):
    model_root: str = "../../../../_BIN"
    clip_model_onnx: str = "models/CLIP/ViT-L-14.onnx"
    tokenizer_path: str = "models/tokenizer/clip-vit-large-patch14"


class ProviderConfig(BaseModel):
    provider_name: Optional[str] = None
    device_id: int = 0
    tensorRT_cache_path: str = "../../../../_BIN/TensorRT_cache"


class ModelParamsConfig(BaseModel):
    input_size: List[int] = Field(default=[224, 224])
    mask_suffix: str = "_BiRefNet-portrait_output.jpg"


class ClusteringConfig(BaseModel):
    eps: float = 0.14
    min_samples: int = 2


class ClassificationConfig(BaseModel):
    match_threshold: float = 0.25
    prompts: List[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    model_params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)


class ConfigManager:
    def __init__(self, path: Path):
        self.path = path
        self.config: AppConfig = self._load()
        self._resolve_paths()

    def _load(self) -> AppConfig:
        if not self.path.exists():
            logger.warning(f"Config not found: {self.path}, using defaults")
            return AppConfig()

        try:
            data = toml.load(self.path)
            return AppConfig(**data)
        except ValidationError as e:
            raise RuntimeError(f"Config validation error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")

    def _resolve_paths(self):
        base = self.path.parent

        def resolve(p: str) -> str:
            path = Path(p)
            return str((base / path).resolve()) if not path.is_absolute() else str(path)

        self.config.paths.model_root = resolve(self.config.paths.model_root)
        self.config.provider.tensorRT_cache_path = resolve(
            self.config.provider.tensorRT_cache_path
        )