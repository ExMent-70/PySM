# rmbg_library/rmbg_birefnet.py
import logging
import time
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import importlib.util  # Для динамического импорта
import sys  # Для манипуляций с sys.path
import torchvision.transforms as transforms  # Для препроцессинга

# Импорт safetensors для загрузки весов
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    logging.error(
        "Библиотека 'safetensors' не найдена. Пожалуйста, установите ее: pip install safetensors"
    )
    load_safetensors = None  # Устанавливаем в None, если импорт не удался

# Импорты из других модулей библиотеки
from .rmbg_config import Config
from .rmbg_logger import get_message
from .rmbg_base_processor import BaseModelProcessor  # Базовый класс
from .rmbg_models import download_model_files, AVAILABLE_MODELS  # Утилиты для моделей
from . import rmbg_utils  # Утилиты для изображений

logger = logging.getLogger(__name__)


class BiRefNetModelProcessor(BaseModelProcessor):
    """
    Внутренний процессор для конкретных моделей BiRefNet
    (general, HR, lite, etc., определенных в AVAILABLE_MODELS).
    """

    def __init__(self, model_name: str, config: Config, device: torch.device):
        super().__init__(config, device)
        self.model_name = model_name
        # Устанавливаем имя процессора для логов
        self.processor_name = f"birefnet ({self.model_name})"
        self.model_files: Optional[Dict[str, Path]] = None
        self.model_module = None  # Модуль, импортированный из model_script
        self.config_module = None  # Модуль, импортированный из config_script
        # Используем специфичную секцию конфига [birefnet_specific] или общую [model] как fallback
        self.proc_config = (
            config.birefnet_specific if config.birefnet_specific else config.model
        )
        self.transform = None  # Трансформация для препроцессинга
        self.img_scale: int = (
            1024  # Разрешение для обработки, будет установлено в load()
        )

        if not load_safetensors:
            raise ImportError(
                "safetensors library is required for BiRefNet models but not found."
            )

        self.load()  # Загружаем модель при инициализации

    # Внутри rmbg_library/rmbg_birefnet.py -> класс BiRefNetModelProcessor

    # Внутри rmbg_library/rmbg_birefnet.py -> класс BiRefNetModelProcessor

    def load(self):
        """Loads a specific BiRefNet model (general, HR, lite, etc.) and necessary scripts."""
        import importlib.util
        import sys

        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError:
            logger.error("safetensors library not found.")
            raise

        logger.info(f"Loading model: {self.model_name} using BiRefNetModelProcessor")
        start_time = time.monotonic()

        # --- 1. Получаем информацию о модели и проверяем процессор ---
        model_info = AVAILABLE_MODELS.get(self.model_name)
        if not model_info or model_info.get("processor_module") != "rmbg_birefnet":
            raise ValueError(
                f"Model definition for '{self.model_name}' suitable for BiRefNetModuleProcessor not found."
            )

        # --- 2. Скачиваем/проверяем файлы ---
        try:
            logger.debug(f"Calling download_model_files for {self.model_name}")
            self.model_files = download_model_files(self.model_name, self.config)
            logger.debug(f"download_model_files returned: {self.model_files}")
        except Exception as e:
            logger.exception(
                f"Exception during download_model_files call for {self.model_name}: {e}"
            )
            self.model_files = None
        if (
            self.model_files is None
            or not isinstance(self.model_files, dict)
            or not self.model_files
        ):
            raise FileNotFoundError(
                f"Could not download or locate files for model '{self.model_name}'."
            )

        # --- 3. Проверяем наличие и получаем пути ---
        required_logical_keys = model_info.get("files", {}).keys()
        if not required_logical_keys:
            raise ValueError(f"No files defined for {self.model_name}")
        all_files_present = True
        paths = {}
        logger.debug(
            f"Проверка существования файлов для ключей: {list(required_logical_keys)}"
        )
        for key in required_logical_keys:
            file_path = self.model_files.get(key)
            if file_path and isinstance(file_path, Path) and file_path.exists():
                paths[key] = file_path
                logger.debug(f"File verified '{key}': {file_path}")
            elif key != "model_dir":
                all_files_present = False
                logger.error(f"File for key '{key}' not found: {file_path}")
                break
        if not all_files_present:
            raise FileNotFoundError(
                f"Missing required model files for {self.model_name}."
            )
        logger.info(f"Все необходимые файлы для {self.model_name} найдены.")

        model_script_path = paths.get("model_script")
        config_script_path = paths.get("config_script")
        weights_path = paths.get("model_weights")
        model_dir = self.model_files.get("model_dir")
        if not all([model_script_path, config_script_path, weights_path, model_dir]):
            missing = [
                k
                for k in ["model_script", "config_script", "model_weights", "model_dir"]
                if not paths.get(k)
            ]
            raise FileNotFoundError(
                f"Missing expected file paths ({', '.join(missing)}) for {self.model_name}"
            )

        # --- 4. Динамический импорт с исправлением ---
        original_sys_path = list(sys.path)
        config_module = None
        model_module = None
        try:
            if str(model_dir) not in sys.path:
                sys.path.insert(0, str(model_dir))
            config_module_name = Path(config_script_path).stem
            config_spec = importlib.util.spec_from_file_location(
                config_module_name, config_script_path
            )
            if config_spec is None or config_spec.loader is None:
                raise ImportError(f"Cannot create spec for {config_script_path}")
            self.config_module = importlib.util.module_from_spec(config_spec)
            sys.modules[config_module_name] = self.config_module
            config_spec.loader.exec_module(self.config_module)
            logger.debug(f"Config script '{config_module_name}' loaded.")

            model_module_name = Path(
                model_script_path
            ).stem  # "birefnet" или "birefnet_lite"
            try:  # Патчинг
                with open(model_script_path, "r", encoding="utf-8") as f:
                    model_content = f.read()
                original_content = model_content
                relative_import_original = f"from .{config_module_name}"
                corrected_import = f"from {config_module_name}"
                if relative_import_original in model_content:
                    model_content = model_content.replace(
                        relative_import_original, corrected_import
                    )
                    logger.debug(f"Patched relative import in {model_script_path}")
                timm_replacements = {
                    "from timm.models.layers": "from timm.layers",
                    "from timm.models.registry": "from timm.models",
                }
                for old, new in timm_replacements.items():
                    if old in model_content:
                        model_content = model_content.replace(old, new)
                        logger.debug(f"Patched timm import in {model_script_path}")

                # --- ИСПРАВЛЕНИЕ ЗДЕСЬ (СНОВА) ---
                if model_content != original_content:
                    # 'with' блок на новой строке с отступом
                    with open(model_script_path, "w", encoding="utf-8") as f:
                        f.write(model_content)
                    logger.debug(f"Finished patching imports in {model_script_path}.")
                else:
                    logger.debug(
                        f"No patching needed for imports in {model_script_path}."
                    )
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            except Exception as patch_err:
                logger.warning(
                    f"Could not patch imports in {model_script_path}: {patch_err}."
                )

            model_spec = importlib.util.spec_from_file_location(
                model_module_name, model_script_path
            )
            if model_spec is None or model_spec.loader is None:
                raise ImportError(f"Cannot create spec for {model_script_path}")
            self.model_module = importlib.util.module_from_spec(model_spec)
            model_spec.loader.exec_module(self.model_module)
            logger.debug(f"Model script '{model_module_name}' loaded.")
        except Exception as import_err:
            logger.exception(...)
            raise ImportError() from import_err
        finally:
            sys.path = original_sys_path
            logger.debug("Restored original sys.path")

        # --- 5. Инстанцирование и загрузка весов ---
        try:
            expected_class_name = "BiRefNet"  # Всегда ищем 'BiRefNet'
            ModelClass = getattr(self.model_module, expected_class_name, None)
            if not ModelClass:
                raise AttributeError(
                    f"Could not find class '{expected_class_name}' in {model_script_path}."
                )
            logger.debug(f"Using model class: {expected_class_name}")
            ModelConfigClass = getattr(self.config_module, "BiRefNet_config", None)
            model_config_obj = ModelConfigClass() if ModelConfigClass else None
            net = ModelClass(model_config_obj) if model_config_obj else ModelClass()

            logger.debug(f"Загрузка весов из Safetensors: {weights_path}")
            state_dict = load_safetensors(weights_path, device="cpu")
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            net.load_state_dict(state_dict)
            logger.info("Веса модели успешно загружены.")
            net.eval().to(self.device)
            if self.device.type == "cuda":
                try:  # Проверка FP16
                    capability = torch.cuda.get_device_capability(self.device)
                    if capability[0] >= 7:
                        net.half()
                        logger.info("Model converted to FP16.")
                    else:
                        logger.warning(
                            f"GPU compute capability ({capability[0]}.{capability[1]}) < 7.0, FP16 not used."
                        )
                except Exception as e:
                    logger.warning(f"Could not convert model to FP16: {e}")
            self.model = net

            # --- 6. Настройка препроцессинга ---
            default_res = model_info.get("default_res", 1024)
            config_res = None
            if (
                self.proc_config
                and isinstance(self.proc_config, dict)
                and "img_scale" in self.proc_config
            ):
                config_res = self.proc_config["img_scale"]
            elif self.proc_config and hasattr(self.proc_config, "img_scale"):
                config_res = self.proc_config.img_scale
            self.img_scale = config_res if config_res is not None else default_res
            logger.debug(
                f"Initial img_scale: {self.img_scale} (config: {config_res}, default: {default_res})"
            )
            if model_info.get("force_res", False):
                if self.img_scale != 512:
                    logger.warning(...)
                    self.img_scale = 512
            elif self.img_scale % 32 != 0:
                self.img_scale = max(32, round(self.img_scale / 32) * 32)
                logger.debug(f"Adjusted img_scale to {self.img_scale}")

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            # transforms импортирован в начале файла
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (self.img_scale, self.img_scale),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            logger.info(
                f"Using image scale for preprocessing: {self.img_scale}x{self.img_scale}"
            )
            logger.info(
                get_message(
                    "INFO_MODEL_LOADED",
                    model_name=self.model_name,
                    time=time.monotonic() - start_time,
                    processor="birefnet",
                )
            )

        except Exception as e:
            logger.exception(get_message("ERROR_MODEL_LOAD", ...))
            self.model = None
            raise

    def _do_process(self, image: Image.Image, **kwargs) -> Optional[Image.Image]:
        """Processes an image using the loaded BiRefNet model."""
        if self.model is None or self.transform is None:
            logger.error(
                f"Model/transform not loaded for {self.model_name}. Cannot process."
            )
            return None
        # Используем имя процессора из атрибута для логгирования
        logger.debug(
            f"Processing with {self.processor_name}..."
        )  # processor_name установлен в __init__ базового класса
        original_size = image.size
        image_rgb = image.convert("RGB")

        # Препроцессинг
        inputs = self.transform(image_rgb).unsqueeze(0).to(self.device)
        # Конвертируем в half если модель в half
        if hasattr(self.model, "dtype") and self.model.dtype == torch.float16:
            if inputs.dtype != torch.float16:  # Предотвращаем ненужное преобразование
                inputs = inputs.half()
                logger.debug("Input tensor converted to FP16 for model.")

        # Инференс
        try:
            with torch.no_grad():
                outputs = self.model(inputs)
                # Извлечение маски (предполагаем, что последний выход - лучший)
                if isinstance(outputs, (list, tuple)) and outputs:
                    scaled_pred = outputs[-1]
                elif isinstance(outputs, torch.Tensor):
                    scaled_pred = outputs
                else:
                    logger.error("Unexpected output structure from model.")
                    return None

                # Постобработка: sigmoid, убрать батч, на CPU, в PIL, ресайз
                mask_tensor = torch.sigmoid(scaled_pred.float()).squeeze(
                    0
                )  # CHW, float32
                mask_pil_scaled = rmbg_utils.tensor_to_mask_pil(
                    mask_tensor.cpu()
                )  # Ожидает CHW или B1HW tensor на CPU
                mask_pil_resized = mask_pil_scaled.resize(original_size, Image.LANCZOS)
            return mask_pil_resized
        except Exception as e:
            logger.exception(
                get_message(
                    "ERROR_MODEL_INFERENCE",
                    model_name=self.model_name,
                    processor="birefnet",
                    exc=e,
                )
            )
            return None

    # release метод наследуется из BaseModelProcessor и должен работать


class BiRefNetModuleProcessor:
    """Facade for models defined in AILab_BiRefNet.py."""

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.model_name = config.model.name
        self.processor: Optional[BiRefNetModelProcessor] = None
        self.processor_name = "birefnet"  # Общее имя для фасада
        self._load_internal_processor()

    def _load_internal_processor(self):
        """Loads the specific BiRefNet processor based on model_name."""
        model_info = AVAILABLE_MODELS.get(self.model_name)
        if not model_info or model_info.get("processor_module") != "rmbg_birefnet":
            logger.error(
                f"Model '{self.model_name}' is not defined or not designated for BiRefNetModuleProcessor."
            )
            # Можно вызвать исключение, чтобы main.py поймал ошибку инициализации
            raise ValueError(
                f"Model '{self.model_name}' cannot be handled by BiRefNetModuleProcessor."
            )

        try:
            # Используем BiRefNetModelProcessor для всех моделей этого модуля
            self.processor = BiRefNetModelProcessor(
                self.model_name, self.config, self.device
            )
        except Exception as e:
            # Ошибка уже залогирована внутри load()
            self.processor = None
            # Передаем исключение дальше, чтобы main.py знал о провале
            raise RuntimeError(
                f"Failed to initialize internal processor for {self.model_name}"
            ) from e

    def process(self, image: Image.Image, **kwargs) -> Optional[Image.Image]:
        """Delegates processing to the internal BiRefNetModelProcessor."""
        if (
            self.processor
        ):  # Процессор должен быть загружен, если __init__ не вызвал исключение
            start_time = time.monotonic()
            # Передаем kwargs дальше, если они понадобятся в будущем
            mask = self.processor.process(image, **kwargs)
            elapsed = time.monotonic() - start_time
            if mask is not None:
                logger.debug(
                    f"BiRefNetModuleProcessor ({self.model_name}) completed in {elapsed:.2f}s"
                )
            # Возвращаем результат (маску или None)
            return mask
        else:
            # Эта ветка не должна достигаться, если __init__ работает правильно
            logger.error(f"Internal processor for '{self.model_name}' was not loaded.")
            return None

    def release(self):
        """Releases resources held by the internal processor."""
        if self.processor:
            self.processor.release()
            self.processor = None
        else:
            logger.debug(
                "No internal processor to release for BiRefNetModuleProcessor."
            )
