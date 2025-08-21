# rmbg_library/rmbg_config.py

import tomllib
import logging
from pathlib import Path
from typing import List, Optional, Union, Literal

from pydantic import (
    BaseModel,
    Field,
    DirectoryPath,
    FilePath,
    field_validator,
    PositiveInt,
    NonNegativeInt,
    conlist,
    confloat,
)

try:
    from pydantic.v1 import ValidationError
except ImportError:
    from pydantic import ValidationError
import sys

logger = logging.getLogger(__name__)


# 1. БЛОК: Базовые модели конфигурации (без изменений)
# ==============================================================================
class PathsConfig(BaseModel):
    input_dir: DirectoryPath = Field(..., description="Directory containing input images")
    output_dir: Path = Field(..., description="Directory where output images/masks will be saved")
    model_root: Path = Field(..., description="Root directory to store all downloaded models")

    @field_validator("input_dir", mode="before")
    @classmethod
    def check_input_dir(cls, v: Union[str, Path]) -> Path:
        if v is None: raise ValueError("Input directory path cannot be None")
        path = Path(v)
        if not path.is_dir(): raise ValueError(f"Input directory does not exist: {path}")
        return path

    @field_validator("output_dir", "model_root", mode="before")
    @classmethod
    def ensure_path_and_create(cls, v: Union[str, Path]) -> Path:
        if v is None: raise ValueError("Path cannot be None")
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

class ModelConfig(BaseModel):
    name: str = Field("RMBG-2.0", description="Model name (key from AVAILABLE_MODELS)")
    mask_blur: NonNegativeInt = 0
    mask_offset: int = 0
    background: Literal["Alpha", "Solid", "Original"] = "Alpha"
    background_color: conlist(item_type=NonNegativeInt, min_length=3, max_length=3) = [255, 255, 255]
    invert_output: bool = False
    save_options: List[Literal["image", "mask"]] = Field(["image", "mask"], description="Что сохранять: 'image', 'mask' или и то, и другое.")
    process_res: Optional[PositiveInt] = 1024

    sensitivity: confloat(ge=0.0, le=1.0) = 1.0 # Используем confloat для валидации

    @field_validator("background_color")
    @classmethod
    def check_color_range(cls, v: List[int]) -> List[int]:
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("Background color components must be between 0 and 255")
        return v

class PostprocessingConfigMixin(BaseModel):
    mask_blur: Optional[NonNegativeInt] = None
    mask_offset: Optional[int] = None
    invert_output: Optional[bool] = None
    background: Optional[Literal["Alpha", "Solid", "Original"]] = None
    background_color: Optional[conlist(item_type=NonNegativeInt, min_length=3, max_length=3)] = None

    @field_validator("background_color")
    @classmethod
    def check_opt_color_range(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is not None and not all(0 <= c <= 255 for c in v):
            raise ValueError("Background color components must be between 0 and 255")
        return v

# 2. БЛОК: Специфичные конфиги (УПРОЩЕНО)
# ==============================================================================
class RmbgSpecificConfig(PostprocessingConfigMixin):
    refine_foreground: Optional[bool] = False
    class Config: extra = "allow"

class BiRefNetSpecificConfig(PostprocessingConfigMixin):
    img_scale: Optional[int] = 0 # 0 будет означать "использовать default_res"
    class Config: extra = "allow"

class LoggingConfig(BaseModel):
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str = "rmbg_process.log"


# 3. БЛОК: Основная модель конфигурации (УПРОЩЕНО)
# ==============================================================================
class Config(BaseModel):
    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # Оставляем только два валидных процессора
    processor_type: Literal["rmbg", "birefnet"] = Field("rmbg")
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
    
    paths: PathsConfig
    model: ModelConfig
    logging: LoggingConfig
    
    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # Удаляем все специфичные секции, кроме rmbg и birefnet
    rmbg_specific: Optional[RmbgSpecificConfig] = Field(default_factory=dict)
    birefnet_specific: Optional[BiRefNetSpecificConfig] = Field(default_factory=dict)
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

# 4. БЛОК: Функция load_config (без изменений)
# ==============================================================================
def load_config(config_path: Union[str, Path]) -> Config:
    from .rmbg_logger import get_message
    config_path = Path(config_path)
    if not config_path.is_file():
        msg = get_message("ERROR_CONFIG_FILE_NOT_FOUND", config_path=str(config_path))
        print(f"ERROR: {msg}", file=sys.stderr)
        raise FileNotFoundError(msg)
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        config = Config(**data)
        logging.getLogger(__name__).info(get_message("INFO_CONFIG_LOADED", config_path=str(config_path)))
        return config
    except (tomllib.TOMLDecodeError, ValidationError, Exception) as e:
        error_type = "ERROR_CONFIG_VALIDATION" if isinstance(e, ValidationError) else "ERROR_CONFIG_LOAD"
        msg = get_message(error_type, config_path=str(config_path), exc=e)
        details = f"\nDetails:\n{e}" if isinstance(e, ValidationError) else ""
        print(f"ERROR: {msg}{details}", file=sys.stderr)
        logging.getLogger(__name__).exception(msg)
        raise