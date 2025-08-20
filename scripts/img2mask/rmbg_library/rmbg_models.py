# rmbg_library/rmbg_models.py

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from huggingface_hub import hf_hub_download
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .rmbg_config import Config
from .rmbg_logger import get_message

logger = logging.getLogger(__name__)


# 1. БЛОК: Словарь AVAILABLE_MODELS (УПРОЩЕНО)
# ==============================================================================
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    # === Модели для процессора 'rmbg' (без изменений) ===
    "RMBG-2.0": {
        "type": "birefnet", "processor_module": "rmbg_rmbg", "repo_id": "1038lab/RMBG-2.0",
        "files": {
            "config_json": "config.json", "model_weights": "model.safetensors",
            "model_script": "birefnet.py", "config_script": "BiRefNet_config.py",
        },
        "cache_dir": "RMBG/RMBG-2.0",
    },
    "INSPYRENET": {
        "type": "inspyrenet_tb", "processor_module": "rmbg_rmbg",
        "repo_id": None, "files": {}, "cache_dir": "RMBG/INSPYRENET",
    },
    "BEN": {
        "type": "ben", "processor_module": "rmbg_rmbg", "repo_id": "1038lab/BEN",
        "files": {"model_script": "model.py", "model_weights": "BEN_Base.pth"},
        "cache_dir": "RMBG/BEN",
    },
    "BEN2": {
        "type": "ben2", "processor_module": "rmbg_rmbg", "repo_id": "1038lab/BEN2",
        "files": {"model_weights": "BEN2_Base.pth", "model_script": "BEN2.py"},
        "cache_dir": "RMBG/BEN2",
    },

    # --- НАЧАЛО ИСПРАВЛЕНИЙ: Сохраняем все ваши модели, но исправляем ключи `files` ---
    # === Модели для процессора 'birefnet' (обрабатываются rmbg_birefnet.py) ===
    "BiRefNet-general": {
        "type": "birefnet", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet-general.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "General purpose model", "default_res": 1024, "max_res": 2048, "min_res": 512
    },
    "BiRefNet_512x512": {
        "type": "birefnet", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet_512x512.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "Optimized for 512x512", "default_res": 512, "max_res": 1024, "min_res": 256, "force_res": True
    },
    "BiRefNet-HR": {
        "type": "birefnet", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet-HR.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "High resolution model", "default_res": 2048, "max_res": 2560, "min_res": 1024
    },
    "BiRefNet-portrait": {
        "type": "birefnet", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet-portrait.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "Optimized for portraits", "default_res": 1024, "max_res": 2048, "min_res": 512
    },
    "BiRefNet-matting": {
        "type": "birefnet", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet-matting.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "General purpose matting", "default_res": 1024, "max_res": 2048, "min_res": 512
    },
    "BiRefNet-HR-matting": {
        "type": "birefnet", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet-HR-matting.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "High resolution matting", "default_res": 2048, "max_res": 2560, "min_res": 1024
    },
    "BiRefNet_lite": {
        "type": "birefnet_lite", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet_lite.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet_lite.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "Lightweight version", "default_res": 1024, "max_res": 2048, "min_res": 512
    },
    "BiRefNet_lite-2K": {
        "type": "birefnet_lite", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet_lite.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet_lite-2K.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "Lightweight 2K version", "default_res": 2048, "max_res": 2560, "min_res": 1024
    },
    "BiRefNet_dynamic": {
        "type": "birefnet", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet_dynamic.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "Dynamic high-resolution", "default_res": 1024, "max_res": 2048, "min_res": 512
    },
    "BiRefNet_lite-matting": {
        "type": "birefnet_lite", "processor_module": "rmbg_birefnet", "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_script": "birefnet_lite.py",
            "config_script": "BiRefNet_config.py",
            "model_weights": "BiRefNet_lite-matting.safetensors",
            "config_json": "config.json"
        },
        "cache_dir": "RMBG/BiRefNet", "description": "Lightweight matting", "default_res": 1024, "max_res": 2048, "min_res": 512
    }
    # --- КОНЕЦ ИСПРАВЛЕНИЙ ---
}


# 2. БЛОК: Функции загрузки моделей (без изменений)
# ==============================================================================
def get_model_cache_dir(config_model_root: Path, model_info: Dict[str, Any]) -> Path:
    # ... (код без изменений) ...
    model_cache_subdir = model_info.get("cache_dir", "unknown_model")
    cache_dir = config_model_root / model_cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_model_files(model_name: str, config: "Config") -> Optional[Dict[str, Path]]:
    # ... (код без изменений, но теперь он будет работать только с оставшимися моделями) ...
    if model_name not in AVAILABLE_MODELS:
        logger.debug(f"Model '{model_name}' not in AVAILABLE_MODELS, handled elsewhere.")
        return None
    
    model_info = AVAILABLE_MODELS[model_name]
    repo_id = model_info.get("repo_id")
    files_to_download = model_info.get("files", {})
    model_cache_path = get_model_cache_dir(config.paths.model_root, model_info)

    if not repo_id or not files_to_download:
        logger.info(f"No repo_id or files for '{model_name}'. Skipping HF download.")
        return {"model_dir": model_cache_path, "no_hf_download": True}

    logger.info(get_message("INFO_DOWNLOADING_MODEL", model_name=model_name, cache_dir=str(model_cache_path)))
    downloaded_files: Dict[str, Path] = {"model_dir": model_cache_path}
    try:
        hf_cache_dir = config.paths.model_root / ".hf_cache"
        for logical_name, repo_filename in files_to_download.items():
            downloaded_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=repo_filename,
                cache_dir=hf_cache_dir,
                local_dir=model_cache_path,
                #local_dir_use_symlinks=False,
                #force_filename=repo_filename,
            )
            downloaded_files[logical_name] = Path(downloaded_path_str)
        logger.info(get_message("INFO_MODEL_FILES_DOWNLOADED", model_name=model_name))
        return downloaded_files
    except Exception as e:
        logger.error(get_message("ERROR_DOWNLOADING_MODEL", model_name=model_name, exc=e), exc_info=True)
        return None