"""Lazy model registry that keeps heavy ML imports outside script startup."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .adapters.base import ModelAdapter
from .config_schema import ModelName


AdapterFactory = Callable[[], ModelAdapter]


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    model_id: ModelName
    display_name: str
    family: str
    repository: str
    weights_file: str
    model_script_file: str
    default_resolution: int
    min_resolution: int
    max_resolution: int
    license_name: str
    upstream_source: str
    quality_hint: str


class ModelRegistryError(RuntimeError):
    """Base registry error."""


class UnknownModelError(ModelRegistryError):
    """Raised for model identifiers that are not part of the subsystem contract."""


class ModelAdapterUnavailableError(ModelRegistryError):
    """Raised when a descriptor exists but its inference adapter is not installed."""


class ModelRegistry:
    """Descriptor registry with optional lazy factories for verified adapters."""

    def __init__(self, descriptors: Iterable[ModelDescriptor]) -> None:
        self._descriptors = {item.model_id: item for item in descriptors}
        self._factories: dict[ModelName, AdapterFactory] = {}

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        return tuple(self._descriptors.values())

    def get(self, model_id: ModelName | str) -> ModelDescriptor:
        normalized = ModelName(model_id)
        try:
            return self._descriptors[normalized]
        except KeyError as exc:
            raise UnknownModelError(f"Неизвестная модель: {model_id}") from exc

    def register_factory(
        self,
        model_id: ModelName | str,
        factory: AdapterFactory,
        *,
        replace: bool = False,
    ) -> None:
        normalized = ModelName(model_id)
        self.get(normalized)
        if normalized in self._factories and not replace:
            raise ModelRegistryError(
                f"Factory для модели '{normalized.value}' уже зарегистрирована."
            )
        self._factories[normalized] = factory

    def has_adapter(self, model_id: ModelName | str) -> bool:
        return ModelName(model_id) in self._factories

    def create(self, model_id: ModelName | str) -> ModelAdapter:
        normalized = ModelName(model_id)
        descriptor = self.get(normalized)
        try:
            adapter = self._factories[normalized]()
        except KeyError as exc:
            raise ModelAdapterUnavailableError(
                f"Адаптер '{descriptor.display_name}' ещё не подключён к runtime."
            ) from exc
        if adapter.model_id != normalized.value:
            raise ModelRegistryError(
                "Factory вернула адаптер с другим model_id: "
                f"expected={normalized.value}, actual={adapter.model_id}."
            )
        return adapter


BUILTIN_MODEL_DESCRIPTORS = (
    ModelDescriptor(
        model_id=ModelName.RMBG_2_0,
        display_name="RMBG-2.0",
        family="rmbg",
        repository="1038lab/RMBG-2.0",
        weights_file="model.safetensors",
        model_script_file="birefnet.py",
        default_resolution=1024,
        min_resolution=512,
        max_resolution=2048,
        license_name="Apache-2.0",
        upstream_source="py/AILab_RMBG.py",
        quality_hint="Универсальное высококачественное удаление фона.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_GENERAL,
        display_name="BiRefNet-general",
        family="birefnet",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet-general.safetensors",
        model_script_file="birefnet.py",
        default_resolution=1024,
        min_resolution=512,
        max_resolution=2048,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Универсальный BiRefNet-профиль.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_512X512,
        display_name="BiRefNet_512x512",
        family="birefnet",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet_512x512.safetensors",
        model_script_file="birefnet.py",
        default_resolution=512,
        min_resolution=256,
        max_resolution=1024,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Быстрый вариант для обработки при 512 px.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_HR,
        display_name="BiRefNet-HR",
        family="birefnet",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet-HR.safetensors",
        model_script_file="birefnet.py",
        default_resolution=2048,
        min_resolution=1024,
        max_resolution=2560,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Высокое разрешение для мелких деталей и сложных краёв.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_PORTRAIT,
        display_name="BiRefNet-portrait",
        family="birefnet",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet-portrait.safetensors",
        model_script_file="birefnet.py",
        default_resolution=1024,
        min_resolution=512,
        max_resolution=2048,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Портреты, люди и сложные края волос.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_MATTING,
        display_name="BiRefNet-matting",
        family="birefnet",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet-matting.safetensors",
        model_script_file="birefnet.py",
        default_resolution=1024,
        min_resolution=512,
        max_resolution=2048,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Мягкие и полупрозрачные границы общего назначения.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_HR_MATTING,
        display_name="BiRefNet-HR-matting",
        family="birefnet",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet-HR-matting.safetensors",
        model_script_file="birefnet.py",
        default_resolution=2048,
        min_resolution=1024,
        max_resolution=2560,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Высокодетальный matting для сложных полупрозрачных краёв.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_LITE,
        display_name="BiRefNet_lite",
        family="birefnet_lite",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet_lite.safetensors",
        model_script_file="birefnet_lite.py",
        default_resolution=1024,
        min_resolution=512,
        max_resolution=2048,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Облегчённая модель с меньшим расходом памяти.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_LITE_2K,
        display_name="BiRefNet_lite-2K",
        family="birefnet_lite",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet_lite-2K.safetensors",
        model_script_file="birefnet_lite.py",
        default_resolution=2048,
        min_resolution=1024,
        max_resolution=2560,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Облегчённая 2K-модель для детальных изображений.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_DYNAMIC,
        display_name="BiRefNet_dynamic",
        family="birefnet",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet_dynamic.safetensors",
        model_script_file="birefnet.py",
        default_resolution=1024,
        min_resolution=512,
        max_resolution=2048,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Динамический DIS-профиль для разных типов объектов.",
    ),
    ModelDescriptor(
        model_id=ModelName.BIREFNET_LITE_MATTING,
        display_name="BiRefNet_lite-matting",
        family="birefnet_lite",
        repository="1038lab/BiRefNet",
        weights_file="BiRefNet_lite-matting.safetensors",
        model_script_file="birefnet_lite.py",
        default_resolution=1024,
        min_resolution=512,
        max_resolution=2048,
        license_name="Apache-2.0",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Облегчённый matting-профиль.",
    ),
    ModelDescriptor(
        model_id=ModelName.LUCIDA,
        display_name="Lucida",
        family="birefnet",
        repository="1038lab/BiRefNet",
        weights_file="Lucida.safetensors",
        model_script_file="birefnet.py",
        default_resolution=1024,
        min_resolution=512,
        max_resolution=2048,
        license_name="MIT",
        upstream_source="py/AILab_BiRefNet.py",
        quality_hint="Прозрачные объекты, свечение, текст и иллюстрации.",
    ),
)


def create_model_registry() -> ModelRegistry:
    """Return a fresh registry so tests and scripts cannot share mutable factories."""

    return ModelRegistry(BUILTIN_MODEL_DESCRIPTORS)
