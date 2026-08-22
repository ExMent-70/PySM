"""Versioned configuration contract shared by RMBG Configurator and Process."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .path_contract import DEFAULT_MODEL_DIR, normalize_model_dir_value


RMBG_SCHEMA_VERSION = 1


class TaskType(str, Enum):
    BACKGROUND_REMOVAL = "background_removal"
    PROMPT_SEGMENTATION = "prompt_segmentation"
    MASK_REFINEMENT = "mask_refinement"


class ProfilePreset(str, Enum):
    GENERAL_HQ = "general_hq"
    PORTRAIT_HQ = "portrait_hq"
    TRANSPARENT_HQ = "transparent_hq"
    CUSTOM = "custom"


class ModelSelection(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


class ModelName(str, Enum):
    RMBG_2_0 = "rmbg_2_0"
    BIREFNET_GENERAL = "birefnet_general"
    BIREFNET_512X512 = "birefnet_512x512"
    BIREFNET_HR = "birefnet_hr"
    BIREFNET_PORTRAIT = "birefnet_portrait"
    BIREFNET_MATTING = "birefnet_matting"
    BIREFNET_HR_MATTING = "birefnet_hr_matting"
    BIREFNET_LITE = "birefnet_lite"
    BIREFNET_LITE_2K = "birefnet_lite_2k"
    BIREFNET_DYNAMIC = "birefnet_dynamic"
    BIREFNET_LITE_MATTING = "birefnet_lite_matting"
    LUCIDA = "lucida"


class DeviceName(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class PrecisionName(str, Enum):
    AUTO = "auto"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class RefinementMode(str, Enum):
    AUTO = "auto"
    NONE = "none"
    FAST = "fast"
    SDMATTE = "sdmatte"


class SDMatteVariant(str, Enum):
    STANDARD = "sdmatte"
    PLUS = "sdmatte_plus"


class BackgroundMode(str, Enum):
    """Composite background type; alpha/original remain for schema-v1 migration."""

    ALPHA = "alpha"
    SOLID = "solid"
    IMAGE = "image"
    ORIGINAL = "original"


class BackgroundFitMode(str, Enum):
    COVER = "cover"
    CONTAIN = "contain"
    STRETCH = "stretch"


class BackgroundPosition(str, Enum):
    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


class ImageFormat(str, Enum):
    PNG = "png"
    WEBP = "webp"
    JPEG = "jpg"


class StrictConfigModel(BaseModel):
    """Base class that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TaskConfig(StrictConfigModel):
    type: TaskType = TaskType.BACKGROUND_REMOVAL
    preset: ProfilePreset = ProfilePreset.GENERAL_HQ


class ModelConfig(StrictConfigModel):
    selection: ModelSelection = ModelSelection.AUTO
    name: ModelName | None = None
    model_dir: str = Field(default=DEFAULT_MODEL_DIR, min_length=1)
    process_resolution: int = Field(default=0, ge=0, le=4096)
    device: DeviceName = DeviceName.AUTO
    precision: PrecisionName = PrecisionName.AUTO
    unload_after_run: bool = True

    @field_validator("model_dir")
    @classmethod
    def normalize_model_dir(cls, value: str) -> str:
        """Keep the model store as an explicit non-empty context setting."""

        return normalize_model_dir_value(value)

    @field_validator("unload_after_run", mode="before")
    @classmethod
    def keep_safe_unload_policy(cls, value: Any) -> bool:
        """Keep the schema-v1 field while enforcing per-process cleanup."""

        return True

    @model_validator(mode="after")
    def validate_manual_selection(self) -> "ModelConfig":
        if self.selection == ModelSelection.MANUAL and self.name is None:
            raise ValueError("Для ручного выбора необходимо указать model.name.")
        return self


class SegmentationConfig(StrictConfigModel):
    prompt: str = ""
    threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    merge_instances: bool = True
    max_segments: int = Field(default=0, ge=0, le=1000)

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        return value.strip()


class MaskConfig(StrictConfigModel):
    sensitivity: float = Field(default=1.0, ge=0.0, le=1.0)
    blur: int = Field(default=0, ge=0, le=64)
    offset: int = Field(default=0, ge=-20, le=20)
    feather: int = Field(default=0, ge=0, le=64)
    fill_holes: bool = True
    max_hole_area: int = Field(default=4096, ge=0, le=1_000_000)
    remove_small_regions: bool = True
    min_region_area: int = Field(default=64, ge=0, le=1_000_000)
    invert: bool = False
    refinement: RefinementMode = RefinementMode.AUTO
    sdmatte_variant: SDMatteVariant = SDMatteVariant.STANDARD
    sdmatte_resolution: int = Field(default=1024, ge=256, le=2048)
    sdmatte_transparent_object: bool = True
    sdmatte_constraint: float = Field(default=0.9, ge=0.1, le=1.0)


class OutputConfig(StrictConfigModel):
    save_cutout: bool = True
    save_mask: bool = True
    save_composite: bool = False
    background_mode: BackgroundMode = BackgroundMode.SOLID
    background_color: str = "#FFFFFF"
    background_image: str = ""
    background_fit: BackgroundFitMode = BackgroundFitMode.COVER
    background_position: BackgroundPosition = BackgroundPosition.CENTER
    image_suffix: str = "_rmbg"
    mask_suffix: str = "_mask"
    composite_suffix: str = "_composite"
    image_format: ImageFormat = ImageFormat.PNG
    png_compress_level: int = Field(default=3, ge=0, le=9)
    jpeg_quality: int = Field(default=95, ge=1, le=100)

    @field_validator("background_color")
    @classmethod
    def validate_background_color(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 7 or not normalized.startswith("#"):
            raise ValueError("Цвет должен быть указан в формате #RRGGBB.")
        try:
            int(normalized[1:], 16)
        except ValueError as exc:
            raise ValueError("Цвет должен быть указан в формате #RRGGBB.") from exc
        return normalized

    @field_validator("background_image")
    @classmethod
    def validate_background_image(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if not normalized:
            return ""
        relative = PurePosixPath(normalized)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(":" in part for part in relative.parts)
        ):
            raise ValueError(
                "Фоновое изображение должно быть относительным путём внутри "
                "background_dir."
            )
        return relative.as_posix()

    @field_validator("image_suffix", "mask_suffix", "composite_suffix")
    @classmethod
    def validate_suffix(cls, value: str) -> str:
        if any(char in value for char in '<>:"/\\|?*'):
            raise ValueError("Суффикс содержит запрещённые для Windows символы.")
        return value


class PerformanceConfig(StrictConfigModel):
    batch_size: int = Field(default=1, ge=1, le=64)
    io_workers: int = Field(default=4, ge=1, le=32)
    max_loaded_models: int = Field(default=1, ge=1, le=4)
    allow_cpu_fallback: bool = False

    @field_validator("batch_size", "max_loaded_models", mode="before")
    @classmethod
    def keep_single_model_runtime(cls, value: Any) -> int:
        """Retain legacy schema fields while enforcing the real runtime."""

        return 1

    @field_validator("allow_cpu_fallback", mode="before")
    @classmethod
    def disable_unimplemented_cpu_fallback(cls, value: Any) -> bool:
        """Do not promise an OOM fallback that the processor cannot perform."""

        return False


AUTO_PROFILE_MODELS: Mapping[ProfilePreset, ModelName] = {
    ProfilePreset.GENERAL_HQ: ModelName.RMBG_2_0,
    ProfilePreset.PORTRAIT_HQ: ModelName.BIREFNET_PORTRAIT,
    ProfilePreset.TRANSPARENT_HQ: ModelName.LUCIDA,
    ProfilePreset.CUSTOM: ModelName.BIREFNET_GENERAL,
}


class RmbgSettings(StrictConfigModel):
    """Complete context payload consumed by both RMBG scripts."""

    schema_version: Literal[1] = RMBG_SCHEMA_VERSION
    profile_name: str = Field(default="Универсальный HQ", min_length=1, max_length=120)
    task: TaskConfig = Field(default_factory=TaskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @field_validator("profile_name")
    @classmethod
    def normalize_profile_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Название профиля не может быть пустым.")
        return normalized

    @model_validator(mode="after")
    def validate_cross_section_contract(self) -> "RmbgSettings":
        if not (
            self.output.save_cutout
            or self.output.save_mask
            or self.output.save_composite
        ):
            raise ValueError("Необходимо выбрать хотя бы один сохраняемый результат.")
        if (
            self.task.type == TaskType.PROMPT_SEGMENTATION
            and not self.segmentation.prompt
        ):
            raise ValueError("Для prompt-сегментации требуется текстовый запрос.")
        if self.output.save_composite and self.output.background_mode not in {
            BackgroundMode.SOLID,
            BackgroundMode.IMAGE,
        }:
            raise ValueError(
                "Для composite необходимо выбрать сплошной цвет или изображение."
            )
        if self.mask.refinement == RefinementMode.SDMATTE:
            if self.model.device == DeviceName.CPU:
                raise ValueError(
                    "SDMatte из ComfyUI-RMBG 3.1.0 требует CUDA; "
                    "выберите CUDA или Автоматически."
                )
            if self.model.precision == PrecisionName.BF16:
                raise ValueError(
                    "SDMatte поддерживает FP16/FP32, но не BF16."
                )
        return self

    def resolved_model_name(self) -> ModelName:
        """Return the explicit model or the model selected by the profile."""

        if self.model.selection == ModelSelection.MANUAL:
            assert self.model.name is not None
            return self.model.name
        return AUTO_PROFILE_MODELS[self.task.preset]

    def resolved_refinement(self) -> RefinementMode:
        """Resolve auto to the balanced CPU refinement without model downloads."""

        if self.mask.refinement == RefinementMode.AUTO:
            return RefinementMode.FAST
        return self.mask.refinement

    def to_context_value(self) -> dict[str, Any]:
        """Serialize the settings to a JSON-compatible PySM context value."""

        return self.model_dump(mode="json")

    def stable_hash(self) -> str:
        """Build a stable SHA-256 hash for manifests and cache keys."""

        payload = json.dumps(
            self.to_context_value(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def default_settings() -> RmbgSettings:
    """Return a fresh validated default configuration."""

    return RmbgSettings()


def parse_settings(payload: Any) -> RmbgSettings:
    """Validate a raw context payload and reject unsupported schema versions."""

    if not isinstance(payload, dict):
        raise TypeError("Конфигурация RMBG должна быть JSON-объектом.")
    version = payload.get("schema_version")
    if version != RMBG_SCHEMA_VERSION:
        raise ValueError(
            f"Неподдерживаемая schema_version={version!r}; "
            f"ожидается {RMBG_SCHEMA_VERSION}."
        )
    return RmbgSettings.model_validate(payload)
