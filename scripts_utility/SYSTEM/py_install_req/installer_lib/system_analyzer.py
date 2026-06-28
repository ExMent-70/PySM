# installer_lib/system_analyzer.py

import logging
import platform
import re
import json
import os
from pathlib import Path
from typing import Optional, List

from .models import SystemInfo, GpuInfo, CudaInfo
from .utils import run_command, find_executable
from .config import (
    GPU_GENERATION_PATTERNS, GPU_GENERATION_TO_CUDA_VERSION,
    GPU_GENERATION_TO_COMPUTE_CAPABILITY, GPU_GENERATION_TENSORRT_SUPPORT
)

try:
    import wmi
except ImportError:
    wmi = None

class SystemAnalyzer:
    """
    Анализирует систему Windows для сбора информации о GPU и CUDA.
    """
    def __init__(self):
        """Инициализирует анализатор."""
        if platform.system() != "Windows":
            raise RuntimeError("Этот класс предназначен для работы только под Windows.")
        
        self.nvidia_smi_path = find_executable("nvidia-smi")
        
        self.wmi_conn = None
        if wmi:
            try:
                self.wmi_conn = wmi.WMI()
                logging.debug("Соединение WMI успешно установлено.")
            except Exception as e:
                logging.warning(f"Не удалось инициализировать WMI: {e}. Функционал WMI будет недоступен.")
        else:
            logging.warning("Библиотека WMI не найдена. Установите ее ('pip install WMI') для полной детекции GPU.")

    def analyze(self) -> SystemInfo:
        """
        Выполняет полный анализ системы, включая вычисление производных данных.
        """
        logging.debug("Анализ системы...")
        gpu_info = self._get_best_gpu_info()
        cuda_info = self._get_cuda_info(gpu_info)

        # Обогащаем GpuInfo вычисляемыми полями.
        if gpu_info and gpu_info.vendor == "NVIDIA":
            self._enrich_gpu_info(gpu_info)
        
        system_info = SystemInfo(
            os_type="windows",
            gpu=gpu_info,
            cuda=cuda_info
        )
        logging.debug("Анализ завершен.")
        return system_info
    
    def _enrich_gpu_info(self, gpu: GpuInfo):
        """Вычисляет и добавляет backend, CC и поддержку TRT на основе поколения GPU."""
        gen = gpu.generation or "unknown"
        gpu.backend = "cuda"
        gpu.compute_capability = GPU_GENERATION_TO_COMPUTE_CAPABILITY.get(gen, "N/A")
        gpu.tensorrt_support = gen in GPU_GENERATION_TENSORRT_SUPPORT

    def _get_command(self, base_command: str) -> str:
        """Возвращает полный путь к утилите, если он найден, иначе просто имя."""
        if base_command == "nvidia-smi" and self.nvidia_smi_path:
            return str(self.nvidia_smi_path)
        return base_command

    def _get_best_gpu_info(self) -> Optional[GpuInfo]:
        """Определяет наилучший доступный GPU, отдавая приоритет nvidia-smi."""
        gpu_from_smi = self._get_gpu_from_nvidia_smi()
        if gpu_from_smi:
            logging.debug(f"Обнаружена NVIDIA GPU через nvidia-smi: {gpu_from_smi.name}")
            return gpu_from_smi

        logging.debug("nvidia-smi недоступен или вернул ошибку. Переключаемся на WMI.")
        gpu_from_wmi = self._get_gpu_from_wmi()
        if gpu_from_wmi:
            logging.debug(f"Обнаружен GPU через WMI: {gpu_from_wmi.name}")
            return gpu_from_wmi
            
        logging.error("Не удалось определить GPU ни одним из доступных методов.")
        return None
    
    def _get_gpu_from_nvidia_smi(self) -> Optional[GpuInfo]:
        """Пытается получить информацию о GPU, используя утилиту nvidia-smi."""
        command = [
            self._get_command("nvidia-smi"),
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits"
        ]
        success, stdout, _ = run_command(command)
        if not success or not stdout.strip():
            return None
        
        try:
            line = stdout.strip().splitlines()[0]
            name, memory_mb = [item.strip() for item in line.split(',')]
            
            return GpuInfo(
                name=name,
                vendor="NVIDIA",
                memory_mb=int(memory_mb),
                generation=self._determine_gpu_generation(name)
            )
        except Exception as e:
            logging.warning(f"Ошибка парсинга вывода nvidia-smi: {e}")
            return None

    def _get_cuda_info(self, gpu: Optional[GpuInfo]) -> Optional[CudaInfo]:
        """Определяет информацию о CUDA, включая рекомендованную версию."""
        portable_path, portable_version = self._get_portable_cuda_info()

        if not gpu or gpu.vendor != "NVIDIA":
            return CudaInfo(
                is_available=False,
                portable_version=portable_version,
                portable_path=str(portable_path) if portable_path else None,
            )

        command = [self._get_command("nvidia-smi")]
        success, stdout, _ = run_command(command)

        if not success or not stdout:
            logging.warning("Команда nvidia-smi не вернула вывод. Считаем, что CUDA недоступна.")
            return CudaInfo(
                is_available=False,
                portable_version=portable_version,
                portable_path=str(portable_path) if portable_path else None,
                warnings=["nvidia-smi недоступен, поэтому CUDA wheel не выбирается автоматически."],
            )
            
        driver_version_match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+\.\d+)", stdout)
        driver_version = driver_version_match.group(1) if driver_version_match else None
        
        # Рекомендованная версия для установки берется из маппинга по поколению GPU.
        recommended_version = GPU_GENERATION_TO_CUDA_VERSION.get(gpu.generation, None)
        selected_version, selected_source, warnings = self._select_cuda_version(
            driver_version=driver_version,
            portable_version=portable_version,
            recommended_version=recommended_version,
        )
        
        return CudaInfo(
            is_available=True,
            driver_version=driver_version,
            recommended_version=recommended_version,
            portable_version=portable_version,
            portable_path=str(portable_path) if portable_path else None,
            selected_version=selected_version,
            selected_source=selected_source,
            warnings=warnings,
        )

    def _get_portable_cuda_info(self) -> tuple[Optional[Path], Optional[str]]:
        """Ищет portable CUDA PySM и читает ее версию из version.json."""
        candidates: List[Path] = []

        env_path = os.environ.get("PYSM_CUDA_PATH")
        if env_path:
            candidates.append(Path(env_path))

        repo_root = self._find_repo_root()
        if repo_root:
            candidates.append(repo_root.parent.parent / "ps_env" / "CUDA")

        candidates.append(Path(r"D:\PySM3_Codex\ps_env\CUDA"))

        for cuda_path in candidates:
            version_file = cuda_path / "version.json"
            if not version_file.is_file():
                continue
            try:
                with open(version_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                version = data.get("cuda", {}).get("version")
                if version:
                    return cuda_path, self._normalize_cuda_version(version)
            except Exception as e:
                logging.warning(f"Не удалось прочитать portable CUDA из {version_file}: {e}")

        return None, None

    def _find_repo_root(self) -> Optional[Path]:
        """Находит корень репозитория PySM относительно текущего файла."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "main.py").is_file() and (parent / "config.toml").is_file():
                return parent
        return None

    def _select_cuda_version(
        self,
        driver_version: Optional[str],
        portable_version: Optional[str],
        recommended_version: Optional[str],
    ) -> tuple[Optional[str], Optional[str], List[str]]:
        """Выбирает CUDA для wheel, не превышая возможности драйвера."""
        warnings: List[str] = []
        driver_tuple = self._version_tuple(driver_version)
        portable_tuple = self._version_tuple(portable_version)
        recommended_tuple = self._version_tuple(recommended_version)

        if portable_version:
            if driver_tuple and portable_tuple and portable_tuple <= driver_tuple:
                return portable_version, "portable", warnings
            if not driver_tuple:
                warnings.append(
                    "Portable CUDA найдена, но версия CUDA драйвера не определена; "
                    "GPU wheel не выбирается автоматически."
                )
            else:
                warnings.append(
                    f"Portable CUDA {portable_version} новее, чем поддерживает драйвер ({driver_version}); "
                    "она не будет выбрана автоматически."
                )

        if recommended_version:
            if driver_tuple and recommended_tuple and recommended_tuple <= driver_tuple:
                return recommended_version, "gpu_generation", warnings
            warnings.append(
                f"Рекомендованная CUDA {recommended_version} не подтверждена драйвером; "
                "будет выбран CPU wheel."
            )

        return None, None, warnings

    def _normalize_cuda_version(self, version: str) -> str:
        match = re.search(r"(\d+\.\d+)", version)
        return match.group(1) if match else version

    def _version_tuple(self, version: Optional[str]) -> Optional[tuple[int, int]]:
        if not version:
            return None
        match = re.search(r"(\d+)\.(\d+)", version)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _get_gpu_from_wmi(self) -> Optional[GpuInfo]:
        """Получает информацию о GPU через WMI и выбирает лучший."""
        if not self.wmi_conn:
            logging.debug("WMI недоступен.")
            return None
            
        try:
            wmi_gpus = self._get_gpus_from_wmi()
            if not wmi_gpus:
                logging.warning("WMI не вернул ни одного видеоадаптера.")
                return None
            
            wmi_gpus.sort(key=lambda g: {"NVIDIA": 0, "AMD": 1, "INTEL": 2}.get(g.vendor, 3))
            
            return wmi_gpus[0]
        except Exception as e:
            logging.error(f"Произошла ошибка при запросе к WMI: {e}")
            return None

    def _get_gpus_from_wmi(self) -> List[GpuInfo]:
        """Получает список GpuInfo всех видеоадаптеров из WMI."""
        gpus = []
        video_controllers = self.wmi_conn.Win32_VideoController()
        
        for controller in video_controllers:
            name = controller.Name
            if "Microsoft Basic Display Adapter" in name:
                continue
            
            vendor = self._determine_vendor_from_name(name)
            memory_bytes = controller.AdapterRAM or 0
            
            gpu = GpuInfo(
                name=name,
                vendor=vendor,
                memory_mb=int(memory_bytes / (1024 * 1024)),
                generation=self._determine_gpu_generation(name) if vendor == "NVIDIA" else None
            )
            gpus.append(gpu)
        return gpus
    
    def _determine_vendor_from_name(self, name: str) -> str:
        """Определяет производителя GPU по имени."""
        name_upper = name.upper()
        if "NVIDIA" in name_upper or "GEFORCE" in name_upper or "RTX" in name_upper:
            return "NVIDIA"
        if "AMD" in name_upper or "RADEON" in name_upper:
            return "AMD"
        if "INTEL" in name_upper:
            return "INTEL"
        return "UNKNOWN"

    def _determine_gpu_generation(self, gpu_name: str) -> str:
        """Определяет поколение GPU NVIDIA по имени."""
        gpu_name_upper = gpu_name.upper()
        for generation, patterns in GPU_GENERATION_PATTERNS.items():
            if any(pattern.upper() in gpu_name_upper for pattern in patterns):
                return generation
        return "unknown"
