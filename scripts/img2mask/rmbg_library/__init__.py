# rmbg_library/__init__.py

import logging
import warnings # <--- 1. Добавляем импорт

# 2. БЛОК: Настройка окружения библиотеки (НОВЫЙ БЛОК)
# ==============================================================================
# Подавляем специфичное предупреждение от torch.meshgrid, которое вызывается
# старым кодом в моделях BEN/BEN2. Размещение здесь делает библиотеку
# самодостаточной и избавляет основной скрипт от этой обязанности.
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
    category=UserWarning
)

# 3. БЛОК: Импорты ключевых компонентов
# ==============================================================================
from .rmbg_config import load_config, Config
from .rmbg_logger import setup_logging, get_message
from .rmbg_utils import load_image, save_image, get_compute_device
from .rmbg_models import AVAILABLE_MODELS

# 4. БЛОК: Импорты классов-процессоров
# ==============================================================================
from .rmbg_rmbg import RmbgModuleProcessor
from .rmbg_birefnet import BiRefNetModuleProcessor

# 5. БЛОК: Импорты подмодулей и настройка логирования
# ==============================================================================
from . import rmbg_mask_ops as mask_ops
from . import rmbg_utils as rmbg_utils

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# 6. БЛОК: Экспорты `__all__`
# ==============================================================================
__all__ = [
    # Config & Logging
    "load_config", "Config", "setup_logging", "get_message",
    # Utils
    "load_image", "save_image", "get_compute_device",
    # Models Info
    "AVAILABLE_MODELS",
    # Main Processors/Facades
    "RmbgModuleProcessor", "BiRefNetModuleProcessor",
    # Mask Ops
    "rmbg_utils", "mask_ops",
]

__version__ = "1.2.1" # Финальная версия