"""Build an effective, serializable pipeline plan before loading ML libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config_schema import BackgroundMode, RmbgSettings, TaskType
from .model_registry import ModelRegistry


class UnsupportedTaskError(RuntimeError):
    """Raised when a valid future schema feature is not implemented yet."""


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    config_hash: str
    profile_name: str
    task_type: str
    model_id: str
    model_display_name: str
    process_resolution: int
    device: str
    precision: str
    refinement: str
    adapter_available: bool
    required_cli_inputs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "profile_name": self.profile_name,
            "task_type": self.task_type,
            "model_id": self.model_id,
            "model_display_name": self.model_display_name,
            "process_resolution": self.process_resolution,
            "device": self.device,
            "precision": self.precision,
            "refinement": self.refinement,
            "adapter_available": self.adapter_available,
            "required_cli_inputs": list(self.required_cli_inputs),
        }


def build_pipeline_plan(
    settings: RmbgSettings,
    registry: ModelRegistry,
) -> PipelinePlan:
    """Resolve profiles and runtime requirements without loading a model."""

    if settings.task.type != TaskType.BACKGROUND_REMOVAL:
        raise UnsupportedTaskError(
            f"Задача '{settings.task.type.value}' будет подключена следующим этапом."
        )

    model_id = settings.resolved_model_name()
    descriptor = registry.get(model_id)
    resolution = settings.model.process_resolution or descriptor.default_resolution
    if not descriptor.min_resolution <= resolution <= descriptor.max_resolution:
        raise ValueError(
            f"Разрешение {resolution} не поддерживается моделью "
            f"{descriptor.display_name}; допустимо "
            f"{descriptor.min_resolution}..{descriptor.max_resolution}."
        )

    required_inputs = ["input_dir", "output_dir"]
    if (
        settings.output.save_composite
        and settings.output.background_mode == BackgroundMode.IMAGE
    ):
        required_inputs.append("background_dir")

    return PipelinePlan(
        config_hash=settings.stable_hash(),
        profile_name=settings.profile_name,
        task_type=settings.task.type.value,
        model_id=model_id.value,
        model_display_name=descriptor.display_name,
        process_resolution=resolution,
        device=settings.model.device.value,
        precision=settings.model.precision.value,
        refinement=settings.resolved_refinement().value,
        adapter_available=registry.has_adapter(model_id),
        required_cli_inputs=tuple(required_inputs),
    )
