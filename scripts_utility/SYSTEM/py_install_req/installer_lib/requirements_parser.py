# installer_lib/requirements_parser.py

import logging
import re
from pathlib import Path
from typing import List, Optional

from .models import InstallationPlan, PackageInfo, PackageType, SystemInfo
from .config import (
    TORCH_FAMILY, ONNXRUNTIME_FAMILY, INSIGHTFACE_FAMILY, TRITON_FAMILY,
    TORCH_INDEX_URLS, GPU_GENERATION_TO_CUDA_VERSION
)

# 1. Добавляем toml, т.к. теперь поддерживаем pyproject.toml
try:
    import toml
except ImportError:
    toml = None

class RequirementsParser:
    """
    Читает файл зависимостей (requirements.txt или pyproject.toml), 
    парсит его и создает интеллектуальный план установки.
    """
    def __init__(self, system_info: SystemInfo):
        self.system_info = system_info

    def parse(self, requirements_path: Path) -> InstallationPlan:
        """
        Главный метод. Выбирает стратегию парсинга в зависимости от имени файла.
        """
        # 2. НОВЫЙ БЛОК: Выбор стратегии парсинга.
        logging.info(f"Чтение файла: <i>{requirements_path}</i>")
        packages: List[PackageInfo] = []
        
        if requirements_path.name == 'pyproject.toml':
            if not toml:
                raise ImportError("Для парсинга pyproject.toml установите библиотеку 'toml' (`pip install toml`).")
            packages = self._parse_pyproject(requirements_path)
        else: # Считаем, что это requirements.txt-подобный файл
            packages = self._parse_requirements_txt(requirements_path)

        logging.info(f"Найдено <b>{len(packages)}</b> пакета(ов).<br>")
        return self._create_plan(packages)

    def _parse_requirements_txt(self, file_path: Path) -> List[PackageInfo]:
        """Парсит текстовый файл зависимостей."""
        # 3. Логика для requirements.txt, вынесенная в отдельный метод.
        packages = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parsed_package = self._parse_line(line)
                if parsed_package:
                    packages.append(parsed_package)
        return packages

    def _parse_pyproject(self, file_path: Path) -> List[PackageInfo]:
        """Парсит секцию [project.dependencies] из pyproject.toml."""
        # 4. НОВЫЙ МЕТОД: Логика для pyproject.toml.
        packages = []
        try:
            data = toml.load(file_path)
            dependencies = data.get("project", {}).get("dependencies", [])
            for dep_line in dependencies:
                parsed_package = self._parse_line(dep_line)
                if parsed_package:
                    packages.append(parsed_package)
        except Exception as e:
            logging.error(f"Ошибка при парсинге {file_path}: {e}")
        return packages

    def _parse_line(self, line: str) -> Optional[PackageInfo]:
        # ... этот метод не изменился ...
        original_line = line
        line = line.split('#')[0].strip()
        if not line or line.startswith('-'):
            return None
        match = re.match(r"([a-zA-Z0-9\-_.]+)(\[[a-zA-Z0-9\-_,]+\])?(.*)", line)
        if not match:
            return None
        name = match.group(1).strip()
        extras_str = match.group(2)
        version_spec = match.group(3).strip()
        extras = []
        if extras_str:
            extras = [e.strip() for e in extras_str.strip('[]').split(',')]
        return PackageInfo(
            name=name, original_line=original_line, package_type=self._classify_package(name),
            version=version_spec if version_spec else None, extras=extras
        )

    def _classify_package(self, name: str) -> PackageType:
        # ... этот метод не изменился ...
        name_lower = name.lower()
        if name_lower in TORCH_FAMILY: return PackageType.TORCH
        if name_lower in ONNXRUNTIME_FAMILY: return PackageType.ONNXRUNTIME
        if name_lower in INSIGHTFACE_FAMILY: return PackageType.INSIGHTFACE
        if name_lower in TRITON_FAMILY: return PackageType.TRITON
        return PackageType.REGULAR
    
    def _create_plan(self, packages: List[PackageInfo]) -> InstallationPlan:
        # ... этот метод не изменился ...
        plan = InstallationPlan()
        for pkg in packages:
            if pkg.package_type == PackageType.TORCH: plan.torch_packages.append(pkg)
            elif pkg.package_type == PackageType.ONNXRUNTIME: plan.onnx_packages.append(pkg)
            elif pkg.package_type == PackageType.INSIGHTFACE: plan.insightface_packages.append(pkg)
            elif pkg.package_type == PackageType.TRITON: plan.triton_packages.append(pkg)
            else: plan.regular_packages.append(pkg)
        plan.torch_index_url = self._get_torch_index_url()
        plan.onnx_package_name = self._get_onnx_package_name()
        logging.info("Сформирован план установки:")
        logging.info(f"  - URL для Torch: <b>{plan.torch_index_url}</b>")
        logging.info(f"  - Имя пакета ONNX: <b>{plan.onnx_package_name}</b>")
        return plan

    def _get_torch_index_url(self) -> str:
        # ... этот метод не изменился ...
        if self.system_info.gpu and self.system_info.gpu.vendor == "NVIDIA":
            gen = self.system_info.gpu.generation
            cuda_version_str = GPU_GENERATION_TO_CUDA_VERSION.get(gen, "")
            if not cuda_version_str and self.system_info.cuda and self.system_info.cuda.version:
                 cuda_version_str = self.system_info.cuda.version
            if cuda_version_str.startswith("12.8"): return TORCH_INDEX_URLS["12.8"]
            if cuda_version_str.startswith("12.4"): return TORCH_INDEX_URLS["12.4"]
            if cuda_version_str.startswith("12.1"): return TORCH_INDEX_URLS["12.1"]
            if cuda_version_str.startswith("11.8"): return TORCH_INDEX_URLS["11.8"]
        return TORCH_INDEX_URLS["cpu"]

    def _get_onnx_package_name(self) -> str:
        # ... этот метод не изменился ...
        if self.system_info.gpu:
            if self.system_info.gpu.vendor == "NVIDIA": return "onnxruntime-gpu"
            if self.system_info.gpu.vendor in ("AMD", "INTEL"): return "onnxruntime-directml"
        return "onnxruntime"