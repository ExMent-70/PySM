# installer_lib/system_analyzer.py

import logging
import platform
import re
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
        if not gpu or gpu.vendor != "NVIDIA":
            return CudaInfo(is_available=False)

        command = [self._get_command("nvidia-smi")]
        success, stdout, _ = run_command(command)

        if not success or not stdout:
            logging.warning("Команда nvidia-smi не вернула вывод. Считаем, что CUDA недоступна.")
            return CudaInfo(is_available=False)
            
        driver_version_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", stdout)
        driver_version = driver_version_match.group(1) if driver_version_match else None
        
        # Рекомендованная версия для установки берется из маппинга по поколению GPU.
        recommended_version = GPU_GENERATION_TO_CUDA_VERSION.get(gpu.generation, None)
        
        return CudaInfo(
            is_available=True,
            driver_version=driver_version,
            recommended_version=recommended_version
        )

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