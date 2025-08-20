# rmbg_library/rmbg_rmbg.py
import logging
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import importlib.util
import sys
import torchvision.transforms as transforms
import threading

# --- Необходимые импорты ---
from .rmbg_config import Config
from .rmbg_logger import get_message
from .rmbg_base_processor import BaseModelProcessor
from .rmbg_models import download_model_files, AVAILABLE_MODELS
from . import rmbg_utils

# --- Импорты для конкретных моделей ---
# Попытка импорта transparent-background для Inspyrenet
try:
    import transparent_background

    TRANSPARENT_BG_AVAILABLE = True
except ImportError:
    logging.warning(
        "Библиотека 'transparent-background' не найдена. pip install transparent_background. Поддержка INSPYRENET будет недоступна."
    )
    TRANSPARENT_BG_AVAILABLE = False

# Импорт safetensors для RMBG-2.0
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    logging.error(
        "Библиотека 'safetensors' не найдена. Пожалуйста, установите ее: pip install safetensors"
    )
    load_safetensors = None

logger = logging.getLogger(__name__)

# --- Processor Classes ---


class RmbgProcessorRMBG2(BaseModelProcessor):
    """Processor specifically for RMBG-2.0 (uses BiRefNet)."""

    def __init__(self, model_name: str, config: Config, device: torch.device):
        super().__init__(config, device)
        self.model_name = model_name
        self.processor_name = "rmbg (RMBG-2.0)"
        self.model_files: Optional[Dict[str, Path]] = None
        self.birefnet_module = None
        self.config_module = None
        self.transform = None
        self.proc_config = config.rmbg_specific
        if not load_safetensors:
            raise ImportError("safetensors library is required for RMBG-2.0")
        self.load()

    # Внутри rmbg_library/rmbg_rmbg.py -> класс RmbgProcessorRMBG2

    def load(self):
        """Loads the RMBG-2.0 (BiRefNet) model and necessary scripts."""
        import importlib.util
        import sys

        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError:
            logger.error(
                "Библиотека 'safetensors' не найдена. Пожалуйста, установите ее: pip install safetensors"
            )
            raise

        logger.info(f"Loading model: {self.model_name} using RmbgProcessorRMBG2")
        start_time = time.monotonic()
        try:
            logger.debug(f"Calling download_model_files for {self.model_name}")
            self.model_files = download_model_files(self.model_name, self.config)
            logger.debug(f"download_model_files returned: {self.model_files}")
        except Exception as e:
            logger.exception(f"... {self.model_name}: {e}")
            self.model_files = None
        if (
            self.model_files is None
            or not isinstance(self.model_files, dict)
            or self.model_files.get("is_rembg")
        ):
            raise FileNotFoundError(
                f"Could not download or locate files for model '{self.model_name}'."
            )

        required_logical_keys = [
            "model_script",
            "config_script",
            "model_weights",
            "config_json",
        ]
        all_files_present = True
        paths = {}
        logger.debug(
            f"Проверка существования файлов для ключей: {required_logical_keys}"
        )
        for key in required_logical_keys:
            file_path = self.model_files.get(key)
            if file_path and file_path.exists():
                paths[key] = file_path
                logger.debug(f"File verified '{key}': {file_path}")
            else:
                all_files_present = False
                logger.error(f"File for key '{key}' not found: {file_path}")
                break
        if not all_files_present:
            raise FileNotFoundError(
                f"Missing required files for {self.model_name}. Check logs."
            )
        logger.info(f"Все необходимые файлы для {self.model_name} найдены.")
        model_script_path = paths["model_script"]
        config_script_path = paths["config_script"]
        weights_path = paths["model_weights"]
        model_dir = self.model_files["model_dir"]

        original_sys_path = list(sys.path)
        config_module = None
        model_module = None
        try:  # Динамический импорт с патчингом
            if str(model_dir) not in sys.path:
                sys.path.insert(0, str(model_dir))
            config_module_name = Path(config_script_path).stem
            config_spec = importlib.util.spec_from_file_location(
                config_module_name, config_script_path
            )
            if config_spec is None or config_spec.loader is None:
                raise ImportError(f"Cannot create spec for {config_script_path}")
            config_module = importlib.util.module_from_spec(config_spec)
            sys.modules[config_module_name] = config_module
            config_spec.loader.exec_module(config_module)
            logger.debug(f"Config script '{config_module_name}' loaded.")

            model_module_name = Path(model_script_path).stem
            try:  # Патчинг
                with open(model_script_path, "r", encoding="utf-8") as f:
                    model_content = f.read()
                original_content = model_content
                relative_import_original = f"from .{config_module_name}"
                relative_import_corrected = f"from {config_module_name}"
                if relative_import_original in model_content:
                    model_content = model_content.replace(
                        relative_import_original, relative_import_corrected
                    )
                    logger.info(f"Patched relative import in {model_script_path}")
                timm_replacements = {
                    "from timm.models.layers": "from timm.layers",
                    "from timm.models.registry": "from timm.models",
                }
                patched_timm = False
                for old_import, new_import in timm_replacements.items():
                    if old_import in model_content:
                        model_content = model_content.replace(old_import, new_import)
                        logger.info(
                            f"Patched timm import: '{old_import}' -> '{new_import}'"
                        )
                        patched_timm = True

                # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
                if model_content != original_content:
                    with open(model_script_path, "w", encoding="utf-8") as f:
                        f.write(model_content)
                    logger.debug(f"Finished patching imports in {model_script_path}")
                else:
                    logger.debug(f"No imports needed patching in {model_script_path}")
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
            model_module = importlib.util.module_from_spec(model_spec)
            model_spec.loader.exec_module(model_module)
            logger.debug(f"Model script '{model_module_name}' loaded.")
        except Exception as import_err:
            logger.exception(
                f"Error during dynamic import for {self.model_name}: {import_err}"
            )
            raise ImportError() from import_err
        finally:
            sys.path = original_sys_path

        try:  # Инстанцирование и загрузка весов
            ModelClass = getattr(model_module, "BiRefNet", None)
            if not ModelClass:
                raise AttributeError("Could not find BiRefNet class.")
            ModelConfigClass = getattr(config_module, "BiRefNet_config", None)
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
            self.model = net

            img_scale = 1024
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_scale, img_scale)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            logger.info(
                get_message(
                    "INFO_MODEL_LOADED",
                    model_name=self.model_name,
                    time=time.monotonic() - start_time,
                    processor=self.processor_name,
                )
            )
        except Exception as e:
            logger.exception(
                get_message(
                    "ERROR_MODEL_LOAD",
                    model_name=self.model_name,
                    processor="rmbg",
                    exc=e,
                )
            )
            self.model = None
            raise

    def _do_process(self, image: Image.Image, **kwargs) -> Optional[Image.Image]:
        # ... (Код метода process остается БЕЗ ИЗМЕНЕНИЙ по сравнению с последней версией) ...
        if self.model is None or self.transform is None:
            logger.error("Model/transform not loaded.")
            return None
        logger.debug(f"Processing with {self.processor_name}...")
        original_size = image.size
        image_rgb = image.convert("RGB")
        inputs = self.transform(image_rgb).unsqueeze(0).to(self.device)
        try:
            with torch.no_grad():
                outputs = self.model(inputs)
                if isinstance(outputs, (list, tuple)):
                    scaled_pred = outputs[-1]
                elif isinstance(outputs, torch.Tensor):
                    scaled_pred = outputs
                else:
                    logger.error("Unexpected output structure")
                    return None
                mask_tensor = torch.sigmoid(scaled_pred).squeeze(0)  # CHW
            mask_pil_scaled = rmbg_utils.tensor_to_mask_pil(mask_tensor.cpu())
            mask_pil_resized = mask_pil_scaled.resize(original_size, Image.LANCZOS)
            return mask_pil_resized
        except Exception as e:
            logger.exception(
                get_message(
                    "ERROR_MODEL_INFERENCE",
                    model_name=self.model_name,
                    processor=self.processor_name,
                    exc=e,
                )
            )
            return None


# --- INSPYRENET Processor (Исправлено на transparent-background) ---
class RmbgProcessorInspyrenet(BaseModelProcessor):
    """Processor for INSPYRENET model using transparent-background library."""

    def __init__(self, model_name: str, config: Config, device: torch.device):
        super().__init__(config, device)
        self.model_name = model_name  # Имя будет 'INSPYRENET'
        self.processor_name = "rmbg (INSPYRENET/transparent-background)"
        # self.model_files не нужен, т.к. модель грузится библиотекой
        self.model: Optional[transparent_background.Remover] = (
            None  # Тип модели - Remover
        )
        self.proc_config = config.rmbg_specific
        self.params = {}  # Для хранения параметров из конфига

        if not TRANSPARENT_BG_AVAILABLE:
            raise ImportError(
                "Библиотека 'transparent-background' не найдена. Установите: pip install transparent_background"
            )
        self.load()

    def load(self):
        """Loads the transparent-background Remover."""
        logger.info(f"Loading model: {self.model_name} using transparent-background")
        start_time = time.monotonic()
        try:
            # Параметры для Remover можно задать при инициализации
            # TODO: Определить, какие параметры из transparent-background нужно вынести в config.toml -> [rmbg_specific]
            # Например: mode='base', jit=False, device=str(self.device) ?
            # Проверяем документацию transparent-background
            # model_params = {k: v for k, v in self.proc_config if k in ['mode', 'jit', 'device', ...]} if self.proc_config else {}
            # model_params['device'] = str(self.device) # Попробуем передать устройство

            # !!! Важно: transparent-background может сама решать, использовать GPU или CPU.
            # Передача 'device' может не поддерживаться или работать не так, как ожидается.
            # Лучше позволить ей определить автоматически или использовать ее переменные окружения, если они есть.
            logger.debug(
                "Initializing transparent_background.Remover() (device selection is internal to the library)"
            )
            self.model = (
                transparent_background.Remover()
            )  # Используем конструктор по умолчанию

            # Сохраняем параметры обработки из общего конфига [model]
            self.params["process_res"] = (
                self.config.model.process_res if self.config.model.process_res else 1024
            )  # Размер для обработки

            logger.info(
                get_message(
                    "INFO_MODEL_LOADED",
                    model_name=self.model_name,
                    time=time.monotonic() - start_time,
                    processor=self.processor_name,
                )
            )

        except Exception as e:
            logger.exception(
                get_message(
                    "ERROR_MODEL_LOAD",
                    model_name=self.model_name,
                    processor=self.processor_name,
                    exc=e,
                )
            )
            self.model = None
            raise

    def _do_process(self, image: Image.Image, **kwargs) -> Optional[Image.Image]:
        """Processes image using transparent-background."""
        if self.model is None:
            logger.error("transparent-background Remover not loaded.")
            return None
        logger.debug(f"Processing with {self.processor_name}...")
        try:
            orig_image = image  # PIL image
            w, h = orig_image.size

            # Resize для обработки (логика из оригинального AILab_RMBG)
            process_res = self.params.get("process_res", 1024)
            aspect_ratio = h / w
            new_w = process_res
            new_h = int(process_res * aspect_ratio)
            if new_h <= 0:
                new_h = process_res  # Предохранитель
            logger.debug(f"Resizing for processing: ({w}x{h}) -> ({new_w}x{new_h})")
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)

            # Обработка
            # model.process возвращает RGBA изображение
            # TODO: Проверить, какие параметры принимает .process() (type='rgba'?)
            foreground_rgba = self.model.process(resized_image, type="rgba")

            # Изменение размера обратно
            foreground_resized = foreground_rgba.resize((w, h), Image.LANCZOS)

            # Извлекаем маску из альфа-канала
            mask = foreground_resized.split()[-1]

            return mask  # Возвращаем маску PIL L mode

        except Exception as e:
            logger.exception(
                get_message(
                    "ERROR_MODEL_INFERENCE",
                    model_name=self.model_name,
                    processor=self.processor_name,
                    exc=e,
                )
            )
            return None

    def release(self):
        # transparent-background Remover может не требовать явного освобождения
        if self.model:
            logger.info(f"Releasing {self.processor_name}.")
        self.model = None
        super().release()  # Вызываем базовый для очистки кэша CUDA


class RmbgProcessorBEN(BaseModelProcessor):
    """Processor for BEN model."""

    def __init__(self, model_name: str, config: Config, device: torch.device):
        super().__init__(config, device)
        self.model_name = model_name
        self.processor_name = "rmbg (BEN)"
        self.model_files: Optional[Dict[str, Path]] = None
        self.model_module = None
        self.proc_config = config.rmbg_specific
        self.inference_lock = threading.Lock() # Блокировка для потокобезопасности
        self.load()

    def load(self):
        logger.info(f"Loading model: {self.model_name} using RmbgProcessorBEN")
        start_time = time.monotonic()
        try:
            self.model_files = download_model_files(self.model_name, self.config)
            if self.model_files is None:
                raise FileNotFoundError("Could not download files.")
            required_keys = ["model_script", "model_weights"]
            paths = {}
            for key in required_keys:
                file_path = self.model_files.get(key)
                if not (file_path and file_path.exists()):
                    raise FileNotFoundError(f"Missing file for key '{key}': {file_path}")
                paths[key] = file_path
                
            model_script_path = paths["model_script"]
            weights_path = paths["model_weights"]
            model_dir = self.model_files["model_dir"]

            original_sys_path = list(sys.path)
            try:
                if str(model_dir) not in sys.path:
                    sys.path.insert(0, str(model_dir))
                model_module_name = Path(model_script_path).stem
                model_spec = importlib.util.spec_from_file_location(model_module_name, model_script_path)
                if not (model_spec and model_spec.loader):
                    raise ImportError(f"Cannot create spec for {model_script_path}")
                self.model_module = importlib.util.module_from_spec(model_spec)
                model_spec.loader.exec_module(self.model_module)
                logger.debug(f"Script '{model_module_name}' loaded.")
            finally:
                sys.path = original_sys_path

            ModelClass = getattr(self.model_module, "BEN_Base", None)
            if not ModelClass:
                raise AttributeError("Could not find BEN_Base class in model.py.")
            net = ModelClass()
            net.eval().to(self.device)

            logger.debug(f"Loading weights via model.loadcheckpoints from: {weights_path}")
            net.loadcheckpoints(str(weights_path))
            logger.info("Model weights loaded successfully via loadcheckpoints.")
            self.model = net
            logger.info(get_message("INFO_MODEL_LOADED", model_name=self.model_name, time=time.monotonic() - start_time, processor=self.processor_name))
        except Exception as e:
            logger.exception(f"Failed to load BEN model: {e}")
            self.model = None
            raise

    def _do_process(self, image: Image.Image, **kwargs) -> Optional[Image.Image]:
        if self.model is None:
            logger.error("BEN model not loaded.")
            return None
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image
        
        try:
            with self.inference_lock:
                # 1. Сбрасываем состояние модели
                if hasattr(self.model, 'p_poses'): self.model.p_poses = []
                if hasattr(self.model, 'pools'): self.model.pools = []
                
                # 2. Вызываем inference с одним PIL.Image
                mask_pil, _ = self.model.inference(image_rgb)
            
            if not isinstance(mask_pil, Image.Image):
                logger.error("BEN model inference did not return a PIL Image as mask.")
                return None
            return mask_pil.convert("L")
        except Exception as e:
            logger.exception(get_message("ERROR_MODEL_INFERENCE", model_name=self.model_name, processor=self.processor_name, exc=e))
            return None


# --- BEN2 Processor (ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ) ---
class RmbgProcessorBEN2(BaseModelProcessor):
    """Processor for BEN2 model."""

    def __init__(self, model_name: str, config: Config, device: torch.device):
        super().__init__(config, device)
        self.model_name = model_name
        self.processor_name = "rmbg (BEN2)"
        self.model_files: Optional[Dict[str, Path]] = None
        self.model_module = None
        self.proc_config = config.rmbg_specific
        self.inference_lock = threading.Lock() # Блокировка для потокобезопасности
        self.load()

    def load(self):
        logger.info(f"Loading model: {self.model_name} using RmbgProcessorBEN2")
        start_time = time.monotonic()
        try:
            self.model_files = download_model_files(self.model_name, self.config)
            if self.model_files is None:
                raise FileNotFoundError("Could not download files.")
            required_keys = ["model_script", "model_weights"]
            paths = {}
            for key in required_keys:
                file_path = self.model_files.get(key)
                if not (file_path and file_path.exists()):
                    raise FileNotFoundError(f"Missing file for key '{key}': {file_path}")
                paths[key] = file_path

            model_script_path = paths["model_script"]
            weights_path = paths["model_weights"]
            model_dir = self.model_files["model_dir"]

            original_sys_path = list(sys.path)
            try:
                if str(model_dir) not in sys.path:
                    sys.path.insert(0, str(model_dir))
                model_module_name = Path(model_script_path).stem
                model_spec = importlib.util.spec_from_file_location(model_module_name, model_script_path)
                if not (model_spec and model_spec.loader):
                    raise ImportError(f"Cannot create spec for {model_script_path}")
                self.model_module = importlib.util.module_from_spec(model_spec)
                model_spec.loader.exec_module(self.model_module)
                logger.debug(f"Script '{model_module_name}' loaded.")
            finally:
                sys.path = original_sys_path

            ModelClass = getattr(self.model_module, "BEN_Base", None)
            if not ModelClass:
                raise AttributeError("Could not find BEN_Base class in BEN2.py.")
            net = ModelClass()
            net.eval().to(self.device)

            logger.debug(f"Loading weights via model.loadcheckpoints from: {weights_path}")
            net.loadcheckpoints(str(weights_path))
            logger.info("Model weights loaded successfully via loadcheckpoints.")
            self.model = net
            logger.info(get_message("INFO_MODEL_LOADED", model_name=self.model_name, time=time.monotonic() - start_time, processor=self.processor_name))
        except Exception as e:
            logger.exception(f"Failed to load BEN2 model: {e}")
            self.model = None
            raise

    def _do_process(self, image: Image.Image, **kwargs) -> Optional[Image.Image]:
        if self.model is None:
            logger.error("BEN2 model not loaded.")
            return None
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image
        
        try:
            refine: bool = self.proc_config.refine_foreground if self.proc_config else False
            
            with self.inference_lock:
                # 1. Сбрасываем состояние модели
                if hasattr(self.model, 'p_poses'): self.model.p_poses = []
                if hasattr(self.model, 'pools'): self.model.pools = []
            
                # 2. Вызываем inference со списком из одного PIL.Image
                foreground_list = self.model.inference([image_rgb], refine_foreground=refine)

            if not isinstance(foreground_list, list) or not foreground_list:
                logger.error("BEN2 model inference did not return a list of images.")
                return None
            foreground_pil = foreground_list[0]

            if not isinstance(foreground_pil, Image.Image):
                logger.error("BEN2 model inference did not return a PIL Image.")
                return None

            if foreground_pil.mode == "RGBA":
                return foreground_pil.split()[-1]
            elif foreground_pil.mode == "L":
                return foreground_pil
            else:
                logger.error(f"BEN2 inference returned unexpected image mode '{foreground_pil.mode}'.")
                return None
        except Exception as e:
            logger.exception(get_message("ERROR_MODEL_INFERENCE", model_name=self.model_name, processor=self.processor_name, exc=e))
            return None


# --- Facade for this module ---
class RmbgModuleProcessor:
    """Facade to select and run models originally from AILab_RMBG.py."""

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.model_name = config.model.name
        # Ссылка на экземпляр конкретного процессора (RMBG2, Inspyrenet, BEN, BEN2)
        self.processor: Optional[BaseModelProcessor] = None
        self.processor_name = "rmbg"  # Имя фасада
        self._load_internal_processor()

    def _load_internal_processor(self):
        """Loads the specific processor based on model_name."""
        model_info = AVAILABLE_MODELS.get(self.model_name)
        processor_class = None

        # Определяем класс процессора на основе имени или типа модели
        if self.model_name == "INSPYRENET":  # Особый случай для transparent-background
            model_type = "inspyrenet_tb"
            processor_class = RmbgProcessorInspyrenet
        elif model_info and model_info.get("processor_module") == "rmbg_rmbg":
            model_type = model_info.get("type")
            if model_type == "birefnet":
                processor_class = RmbgProcessorRMBG2
            elif model_type == "inspyrenet":
                processor_class = RmbgProcessorInspyrenet  # Если бы был rembg вариант
            elif model_type == "ben":
                processor_class = RmbgProcessorBEN
            elif model_type == "ben2":
                processor_class = RmbgProcessorBEN2
            else:
                logger.error(
                    f"Unknown model type '{model_type}' within RmbgModuleProcessor for '{self.model_name}'."
                )
                return
        else:
            # Если модель не найдена или не предназначена для этого модуля
            # Попытка использовать rembg как fallback? Или ошибка? Вызываем ошибку.
            raise ValueError(
                f"Model '{self.model_name}' is not defined in AVAILABLE_MODELS or not handled by RmbgModuleProcessor."
            )

        if processor_class:
            try:
                # Создаем экземпляр нужного процессора
                self.processor = processor_class(
                    self.model_name, self.config, self.device
                )
                # Проверяем, что у него есть метод _do_process, который будет вызван базовым process
                if not hasattr(self.processor, "_do_process"):
                    logger.critical(
                        f"Internal processor {processor_class.__name__} lacks a '_do_process' method."
                    )
                    self.processor = None  # Считаем загрузку неуспешной
            except Exception as e:
                # Ошибка уже должна была быть залогирована при инициализации процессора
                self.processor = None  # Убедимся, что процессор None при ошибке
                # Можно перевызвать исключение, чтобы main.py его поймал
                # raise RuntimeError(f"Failed to initialize internal processor {processor_class.__name__}") from e
        else:
            logger.error(
                f"No specific processor class found for model '{self.model_name}' (type {model_type})."
            )

    # --- ДОБАВЛЕН МЕТОД process ---
    def process(
        self, image: Image.Image, **kwargs
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Delegates processing to the loaded internal processor and applies common postprocessing.
        Matches the signature expected by the BaseModelProcessor's process method.
        """
        if self.processor is None:
            logger.error(
                f"Internal processor for '{self.model_name}' was not loaded successfully. Cannot process."
            )
            return None, None  # Возвращаем None для изображения и маски

        # Вызываем метод process базового класса (который вызовет _do_process внутреннего процессора
        # и затем применит общую постобработку _apply_common_postprocessing)
        # Передаем **kwargs дальше на случай, если они нужны
        try:
            # Базовый класс ожидает, что self.processor реализует _do_process
            # Мы вызываем метод process самого self.processor, который унаследован от BaseModelProcessor
            return self.processor.process(image, **kwargs)
        except NotImplementedError:
            logger.error(
                f"'_do_process' method not implemented in {self.processor.__class__.__name__}"
            )
            return None, None
        except Exception as e:
            logger.exception(
                f"Error during processing call to internal processor {self.processor.__class__.__name__}: {e}"
            )
            return None, None

    # --- КОНЕЦ ДОБАВЛЕНИЯ ---

    def release(self):
        """Releases resources held by the internal processor."""
        if self.processor and hasattr(self.processor, "release"):
            self.processor.release()
            self.processor = None
        else:
            logger.debug(
                f"No internal processor to release for {self.processor_name} ({self.model_name})."
            )
