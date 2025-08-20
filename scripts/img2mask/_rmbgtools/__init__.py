# ======================================================================================
# Файл: _rmbgtools/__init__.py
# ======================================================================================
import logging
import sys
import warnings

# --- Фильтрация назойливых предупреждений от сторонних библиотек ---
# Предупреждение от Transformers о 'device'
warnings.filterwarnings("ignore", message="The `device` argument is deprecated and will be removed in v5 of Transformers.", category=FutureWarning)
# Предупреждение от PyTorch о 'use_reentrant'
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*", category=UserWarning)
# Предупреждение от GroundingDINO о `torch.cuda.amp.autocast`
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.", category=FutureWarning)
# Предупреждение от huggingface_hub о 'local_dir_use_symlinks'
warnings.filterwarnings("ignore", message="`local_dir_use_symlinks` parameter is deprecated and will be ignored.", category=UserWarning)

# --- НОВОЕ ПРАВИЛО ---
# Подавление предупреждения от torch.meshgrid, которое вызывается внутри GroundingDINO
warnings.filterwarnings("ignore", message=".*torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated, please import via timm.models", category=FutureWarning)


# --- Настройка логгера ---
logger = logging.getLogger('_rmbgtools')
logger.setLevel(logging.INFO)
logger.propagate = False
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Версия библиотеки ---
__version__ = "1.0.0"

# --- Централизованные списки моделей ---
from .core.model_manager import MODEL_CONFIGS

RMBG_MODELS = [name for name, conf in MODEL_CONFIGS.items() if conf["type"] == "birefnet" and name.startswith("RMBG")]
BIREFNET_MODELS = [name for name, conf in MODEL_CONFIGS.items() if conf["type"] in ("birefnet", "birefnet_lite") and name.startswith("BiRefNet")]
ALL_REMOVER_MODELS = RMBG_MODELS + BIREFNET_MODELS

SAM2_MODELS = [name for name, conf in MODEL_CONFIGS.items() if conf["type"] == "sam2"]
DINO_MODELS = [name for name, conf in MODEL_CONFIGS.items() if conf["type"] == "dino"]

# --- Публичные импорты ---
from .remover import remove_background
from .segmenter import segment_by_text
from .config import load_config

# --- Управление видимостью через __all__ ---
__all__ = [
    # Функции-обертки
    "remove_background",
    "segment_by_text",
    "load_config",
    
    # Списки моделей
    "ALL_REMOVER_MODELS",
    "RMBG_MODELS",
    "BIREFNET_MODELS",
    "SAM2_MODELS",
    "DINO_MODELS",
]

logger.info(f"_rmbgtools v{__version__} initialized.")