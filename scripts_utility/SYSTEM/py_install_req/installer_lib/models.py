# installer_lib/models.py

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

class PackageType(Enum):
    """Типы пакетов, требующие разного подхода к установке."""
    REGULAR = auto()
    TORCH = auto()
    ONNXRUNTIME = auto()
    INSIGHTFACE = auto()
    TRITON = auto()

@dataclass
class GpuInfo:
    """Информация о видеокарте."""
    name: str
    vendor: str
    generation: Optional[str] = None
    memory_mb: int = 0
    # Добавлены вычисляемые поля для подробного отчета
    backend: str = "cpu"
    tensorrt_support: bool = False
    compute_capability: str = "N/A"

@dataclass
class CudaInfo:
    """Информация о CUDA."""
    is_available: bool
    driver_version: Optional[str] = None
    # Рекомендуемая версия для установки, а не просто версия драйвера
    recommended_version: Optional[str] = None

@dataclass
class SystemInfo:
    """Собранная информация о системе."""
    os_type: str = "windows"
    gpu: Optional[GpuInfo] = None
    cuda: Optional[CudaInfo] = None

@dataclass
class PackageInfo:
    """Информация о распарсенном пакете."""
    name: str
    original_line: str
    package_type: PackageType
    version: Optional[str] = None
    extras: List[str] = field(default_factory=list)
    
    def to_spec(self) -> str:
        """Преобразует информацию в полную строку для pip/uv."""
        spec = self.name
        if self.extras:
            spec += f"[{','.join(self.extras)}]"
        if self.version:
            spec += self.version
        return spec

    def __repr__(self) -> str:
        """Кастомное представление для чистого вывода в логах."""
        return f"Package(name='{self.name}', spec='{self.to_spec()}')"

@dataclass
class InstallationPlan:
    """Пошаговый план установки, сгенерированный парсером."""
    regular_packages: List[PackageInfo] = field(default_factory=list)
    torch_packages: List[PackageInfo] = field(default_factory=list)
    onnx_packages: List[PackageInfo] = field(default_factory=list)
    insightface_packages: List[PackageInfo] = field(default_factory=list)
    triton_packages: List[PackageInfo] = field(default_factory=list)

    # Динамически определяемые параметры
    torch_index_url: Optional[str] = None
    onnx_package_name: Optional[str] = None

    def is_empty(self) -> bool:
        """Проверяет, содержит ли план хотя бы один пакет для установки."""
        return not any([
            self.regular_packages,
            self.torch_packages,
            self.onnx_packages,
            self.insightface_packages,
            self.triton_packages
        ])