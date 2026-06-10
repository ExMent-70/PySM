from pathlib import Path
from typing import List, Optional, Literal
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


class ModelConfig(BaseModel):
    backend: str = "clip"
    name: str = "ViT-L-14"


class ClipConfig(BaseModel):
    name: str = "CLIP ViT-L-14"
    model_onnx: str = "models/CLIP/ViT-L-14.onnx"
    tokenizer_path: str = "models/tokenizer/clip-vit-large-patch14"
    input_size: List[int] = Field(default=[224, 224])


class Siglip2OnnxConfig(BaseModel):
    name: str = "SigLIP2 SO400M ONNX"
    model_dir: str = "models/SigLIP2/siglip2-so400m-patch14-384-ONNX"
    vision_model: str = "vision_model.onnx"
    text_model: str = "text_model.onnx"
    tokenizer_path: str = "models/tokenizer/siglip2-so400m-patch14-384"
    image_output: str = "last_hidden_state"
    spatial_strategy: str = "flatten_axis1_norm"
    input_size: List[int] = Field(default=[384, 384])


class ModelParamsConfig(BaseModel):
    input_size: List[int] = Field(default=[224, 224])
    mask_suffix: str = "_BiRefNet-portrait_output.jpg"


class ClusteringConfig(BaseModel):
    eps: float = 0.14
    min_samples: int = 2


class ClassificationConfig(BaseModel):
    match_threshold: float = 0.11
    prompts: List[str] = Field(default_factory=list)


class CacheConfig(BaseModel):
    mode: Literal["use", "refresh", "off"] = "use"


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    clip: ClipConfig = Field(default_factory=ClipConfig)
    siglip2_onnx: Siglip2OnnxConfig = Field(default_factory=Siglip2OnnxConfig)
    model_params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


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

        clip_model_path = Path(self.config.clip.model_onnx)
        if not clip_model_path.is_absolute():
            self.config.clip.model_onnx = str(
                (Path(self.config.paths.model_root) / clip_model_path).resolve()
            )

        clip_tokenizer_path = Path(self.config.clip.tokenizer_path)
        if not clip_tokenizer_path.is_absolute():
            self.config.clip.tokenizer_path = str(
                (Path(self.config.paths.model_root) / clip_tokenizer_path).resolve()
            )

        siglip2_onnx_model_dir = Path(self.config.siglip2_onnx.model_dir)
        if not siglip2_onnx_model_dir.is_absolute():
            self.config.siglip2_onnx.model_dir = str(
                (Path(self.config.paths.model_root) / siglip2_onnx_model_dir).resolve()
            )

        tokenizer_path = Path(self.config.siglip2_onnx.tokenizer_path)
        if not tokenizer_path.is_absolute():
            self.config.siglip2_onnx.tokenizer_path = str(
                (Path(self.config.paths.model_root) / tokenizer_path).resolve()
            )

        self.apply_backend_defaults()

    def apply_backend_defaults(self):
        backend = self.config.model.backend.lower()
        if backend == "clip":
            self.config.model.name = self.config.clip.name
            self.config.model_params.input_size = list(self.config.clip.input_size)
        elif backend == "siglip2_onnx":
            self.config.model.name = self.config.siglip2_onnx.name
            self.config.model_params.input_size = list(self.config.siglip2_onnx.input_size)
        else:
            raise RuntimeError(f"Unsupported model backend: {self.config.model.backend}")
