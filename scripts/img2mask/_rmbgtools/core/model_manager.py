# ======================================================================================
# Блок 1: Импорты
# ======================================================================================
import os
import sys
import importlib.util
import types
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from .. import logger  # Импортируем настроенный логгер из __init__.py нашего пакета

# ======================================================================================
# Блок 2: Конфигурация моделей
# Централизованный словарь, описывающий все поддерживаемые модели.
# Структура:
#   "Название модели": {
#       "type": (str) Внутренний тип для группировки (например, все birefnet модели).
#       "repo_id": (str) ID репозитория на Hugging Face.
#       "files": (dict) Словарь файлов для скачивания. Ключ - логическое имя, значение - имя файла в репозитории.
#       "subfolder": (str) Имя подпапки внутри `model_dir` для хранения файлов этой модели.
#       "loader": (str) Имя метода в ModelManager, который будет загружать эту модель.
#       "model_class_name": (str, опционально) Имя класса модели для динамического импорта.
#       ... другие специфичные для модели параметры ...
#   }
# ======================================================================================
MODEL_CONFIGS = {
    # --- Модель из оригинального RMBG ---
    "RMBG-2.0": {
        "type": "birefnet",
        "repo_id": "1038lab/RMBG-2.0",
        "files": {
            "model_weights": "model.safetensors",
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "RMBG-2.0",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet",
    },

    # --- Семейство моделей BiRefNet ---
    "BiRefNet-general": {
        "type": "birefnet",
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet-general.safetensors",
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet",
    },
    "BiRefNet_512x512": {
        "type": "birefnet",
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet_512x512.safetensors",
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet",
    },
    "BiRefNet-HR": {
        "type": "birefnet",
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet-HR.safetensors",
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet",
    },
    "BiRefNet-portrait": {
        "type": "birefnet",
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet-portrait.safetensors",
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet",
    },
    "BiRefNet-matting": {
        "type": "birefnet",
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet-matting.safetensors",
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet",
    },
    "BiRefNet-HR-matting": {
        "type": "birefnet",
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet-HR-matting.safetensors",
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet",
    },
    "BiRefNet_lite": {
        "type": "birefnet_lite", # Отдельный тип, так как использует другой файл .py
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet_lite.safetensors",
            "model_script": "birefnet_lite.py", # Другой скрипт!
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet_lite", # Имя класса тоже другое
    },
    "BiRefNet_lite-2K": {
        "type": "birefnet_lite",
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet_lite-2K.safetensors",
            "model_script": "birefnet_lite.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet_lite",
    },
    "BiRefNet_dynamic": {
        "type": "birefnet",
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet_dynamic.safetensors",
            "model_script": "birefnet.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet",
    },
    "BiRefNet_lite-matting": {
        "type": "birefnet_lite",
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "model_weights": "BiRefNet_lite-matting.safetensors",
            "model_script": "birefnet_lite.py",
            "config_script": "BiRefNet_config.py",
        },
        "subfolder": "BiRefNet",
        "loader": "load_birefnet_family_model",
        "model_class_name": "BiRefNet_lite",
    },

    # --- Модели сегментации ---
    # --- Модели SAM2 ---
    "SAM2-Hiera-T": {
        "type": "sam2", "repo_id": "1038lab/sam2",
        "files": {"fp32_weights": "sam2.1_hiera_tiny.safetensors", "fp16_weights": "sam2.1_hiera_tiny-fp16.safetensors"},
        "subfolder": "SAM2", "loader": "load_sam2_model", "config_name": "sam2.1/sam2.1_hiera_t.yaml"
    },
    "SAM2-Hiera-S": {
        "type": "sam2", "repo_id": "1038lab/sam2",
        "files": {"fp32_weights": "sam2.1_hiera_small.safetensors", "fp16_weights": "sam2.1_hiera_small-fp16.safetensors"},
        "subfolder": "SAM2", "loader": "load_sam2_model", "config_name": "sam2.1/sam2.1_hiera_s.yaml"
    },
    "SAM2-Hiera-B+": {
        "type": "sam2", "repo_id": "1038lab/sam2",
        "files": {"fp32_weights": "sam2.1_hiera_base_plus.safetensors", "fp16_weights": "sam2.1_hiera_base_plus-fp16.safetensors"},
        "subfolder": "SAM2", "loader": "load_sam2_model", "config_name": "sam2.1/sam2.1_hiera_b+.yaml"
    },
    "SAM2-Hiera-L": {
        "type": "sam2", "repo_id": "1038lab/sam2",
        "files": {"fp32_weights": "sam2.1_hiera_large.safetensors", "fp16_weights": "sam2.1_hiera_large-fp16.safetensors"},
        "subfolder": "SAM2", "loader": "load_sam2_model", "config_name": "sam2.1/sam2.1_hiera_l.yaml"
    },

    # --- Модели GroundingDINO ---
    "GroundingDINO-T": {
        "type": "dino", "repo_id": "1038lab/GroundingDINO",
        "files": {"model_weights": "groundingdino_swint_ogc.safetensors", "config_script": "GroundingDINO_SwinT_OGC.cfg.py"},
        "subfolder": "GroundingDINO", "loader": "load_dino_model",
    },
    "GroundingDINO-B": {
        "type": "dino", "repo_id": "1038lab/GroundingDINO",
        "files": {"model_weights": "groundingdino_swinb_cogcoor.safetensors", "config_script": "GroundingDINO_SwinB.cfg.py"},
        "subfolder": "GroundingDINO", "loader": "load_dino_model",
    },


    # --- Модель для Inpainting ---
    "LaMa": {
        "type": "lama",
        "repo_id": "1038lab/Lama",
        "files": {
            "model_weights": "big-lama.pt"
        },
        "subfolder": "LaMa",
        "loader": "load_lama_model",
    }
}


# ======================================================================================
# Блок 3: Класс ModelManager (с ВОССТАНОВЛЕННЫМ методом)
# ======================================================================================
class ModelManager:
    def __init__(self, model_dir: str = "models", device: str = "auto"):
        self.model_dir = os.path.abspath(model_dir)
        self.hf_cache_dir = os.path.join(self.model_dir, ".hf_cache")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.hf_cache_dir, exist_ok=True)
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"ModelManager initialized. Model directory: {self.model_dir}, Device: {self.device}")
        self.loaded_models = {}

    # --- ВОТ ЭТОТ МЕТОД БЫЛ ПОТЕРЯН ---
    def _download_file(self, repo_id: str, filename: str, subfolder: str = "") -> str:
        """
        Скачивает один файл, если он отсутствует, используя двухуровневый кэш.
        """
        destination_folder = os.path.join(self.model_dir, subfolder)
        os.makedirs(destination_folder, exist_ok=True)
        destination_path = os.path.join(destination_folder, filename)

        if not os.path.exists(destination_path):
            logger.info(f"Downloading {filename} from {repo_id} to {destination_folder}...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=self.hf_cache_dir,
                    local_dir=destination_folder,
                )
                logger.info(f"Successfully downloaded {filename}.")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                raise
        return destination_path
    # --- КОНЕЦ ВОССТАНОВЛЕННОГО МЕТОДА ---

    def _download_model_files(self, model_name: str) -> dict:
        """
        Обеспечивает наличие файлов для модели на диске.
        """
        config = MODEL_CONFIGS[model_name]
        repo_id = config.get("repo_id")
        
        if not repo_id:
            raise NotImplementedError(f"Model {model_name} is configured as local-only, which is not yet supported.")

        local_model_dir = os.path.join(self.model_dir, config["subfolder"])
        os.makedirs(local_model_dir, exist_ok=True)
        
        downloaded_files = {}
        files_to_download = config.get("files", {})

        for logical_name, repo_filename in files_to_download.items():
            if not isinstance(repo_filename, str):
                continue
            
            # Вместо прямого вызова hf_hub_download, теперь используется внутренний метод
            downloaded_files[logical_name] = self._download_file(
                repo_id, repo_filename, subfolder=config["subfolder"]
            )

        downloaded_files["local_model_dir"] = local_model_dir
        return downloaded_files

    def get_model(self, model_name: str):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' is not supported.")
            
        config = MODEL_CONFIGS[model_name]
        
        file_paths = self._download_model_files(model_name)
        
        loader_method = getattr(self, config["loader"])
        
        logger.info(f"Loading {model_name} model to {self.device}...")
        model = loader_method(model_name, file_paths, config)
        
        self.loaded_models[model_name] = model
        logger.info(f"{model_name} model loaded successfully.")
        return model

    # ======================================================================================
    # Блок 4: Методы-загрузчики для каждой семьи моделей
    # ======================================================================================
    def load_birefnet_family_model(self, model_name, file_paths, config):
        """Загружает модели BiRefNet и RMBG-2.0, которые требуют динамической компиляции .py файлов."""
        model_code_path = file_paths["model_script"]
        config_code_path = file_paths["config_script"]
        weights_path = file_paths["model_weights"]
        
        # Создаем уникальное, изолированное пространство имен для каждого модуля,
        # чтобы избежать конфликтов, если, например, birefnet.py и birefnet_lite.py
        # имеют классы с одинаковыми именами.
        unique_package_name = f"rmbgtools.dynamic_models.{model_name.replace('-', '_').replace('.', '_')}"
        
        config_module_name = f"{unique_package_name}.BiRefNet_config"
        spec = importlib.util.spec_from_file_location(config_module_name, config_code_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules[config_module_name] = config_module
        spec.loader.exec_module(config_module)

        model_module_name = f"{unique_package_name}.birefnet"
        spec = importlib.util.spec_from_file_location(model_module_name, model_code_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[model_module_name] = model_module
        spec.loader.exec_module(model_module)

        # Создаем экземпляр класса модели, имя которого указано в конфиге
        model = getattr(model_module, config["model_class_name"])(config_module.BiRefNetConfig())
        state_dict = load_file(weights_path, device=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        
        # Очищаем sys.modules, чтобы не оставлять "мусор"
        del sys.modules[config_module_name]
        del sys.modules[model_module_name]
        return model


    def load_sam2_model(self, model_name, file_paths, config):
        # Импорты
        import sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        # 1. Логика выбора precision и скачивания (без изменений)
        is_cuda = self.device == 'cuda'
        precision = "fp16" if is_cuda else "fp32"
        weights_key = f"{precision}_weights"
        if weights_key not in config["files"]:
            logger.warning(f"{precision.upper()} weights not available for {model_name}. Falling back to FP32.")
            precision = "fp32"
            weights_key = "fp32_weights"

        logger.info(f"Selected {precision.upper()} precision for {model_name}.")
        weights_filename = config["files"][weights_key]
        weights_path = self._download_file(config["repo_id"], weights_filename, subfolder=config["subfolder"])

        # 2. Правильная работа с Hydra (самый простой и корректный вариант)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        # Жестко задаем относительный путь от этого файла (model_manager.py) до папки с конфигами
        config_path_relative = "../../sam2/configs"

        initialize(config_path=config_path_relative, version_base="1.2")
        
        cfg = compose(config_name=config["config_name"])
        OmegaConf.resolve(cfg)
        
        sam_model = instantiate(cfg.model, _recursive_=True)

        # 3. Загрузка весов и применение precision (без изменений)
        state_dict = load_file(weights_path, device=self.device)
        sam_model.load_state_dict(state_dict, strict=False)
        
        dtype = torch.float16 if precision == "fp16" else torch.float32
        sam_model.to(dtype=dtype, device=self.device).eval()

        predictor = SAM2ImagePredictor(sam_model)
        return predictor

    def load_dino_model(self, model_name, file_paths, config):
        """Загружает модель GroundingDINO."""
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict
        
        config_path = file_paths["config_script"]
        weights_path = file_paths["model_weights"]
        
        args = SLConfig.fromfile(config_path)
        model = build_model(args)
        # Сначала загружаем на CPU, чтобы избежать лишнего потребления VRAM, если модель большая
        checkpoint = load_file(weights_path, device="cpu")
        model.load_state_dict(clean_state_dict(checkpoint), strict=False)
        model.to(self.device).eval()
        return model

    def load_lama_model(self, model_name, file_paths, config):
        """Загружает JIT-скомпилированную модель LaMa."""
        weights_path = file_paths["model_weights"]
        model = torch.jit.load(weights_path, map_location=self.device)
        model.eval()
        return model

# ======================================================================================
# Блок 5: Тестовый блок для самопроверки модуля
# ======================================================================================
if __name__ == '__main__':
    # Добавляем корневую папку проекта в sys.path, чтобы `import sam2` сработал при прямом запуске
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    import logging
    from rmbgtools import logger

    print("--- Testing model_manager.py with real model loaders ---")
    logger.setLevel(logging.INFO)
    test_model_dir = os.path.join(project_root, "temp_test_models")

    # Очистка перед тестом для чистоты эксперимента
    if os.path.exists(test_model_dir):
        import shutil
        shutil.rmtree(test_model_dir)

    manager = ModelManager(model_dir=test_model_dir, device="auto")
    models_to_test = [
        "RMBG-2.0", 
        "BiRefNet-general", 
        "BiRefNet_lite", 
        "LaMa", 
        "GroundingDINO-T", 
        "SAM2-Hiera-T"
    ]

    all_tests_passed = True
    for model_name in models_to_test:
        try:
            print(f"\n--- Requesting model: {model_name} ---")
            model = manager.get_model(model_name)
            assert model is not None
            print(f"--- Model {model_name} test PASSED! ---")
        except Exception as e:
            logger.error(f"--- FAILED to test model {model_name}. Error: {e}", exc_info=True)
            all_tests_passed = False

    print("\n--- model_manager.py all tests finished ---")
    if all_tests_passed: 
        print("✅ All tests passed successfully!")
    else: 
        print("❌ Some tests failed.")