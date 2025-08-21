# rmbg_library/rmbg_base_processor.py
import logging
import torch
from typing import Optional, Any, Tuple
from PIL import Image

# Импорт Config из rmbg_config для type hinting
# Используем TYPE_CHECKING для избежания циклического импорта при проверке типов
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rmbg_config import Config
    # Импортируем утилиты только для type hinting, если они нужны в базовом классе
    # from . import rmbg_utils, mask_ops

# Импортируем ImageOps здесь, если _apply_common_postprocessing останется тут
from PIL import ImageOps

logger = logging.getLogger(__name__)


class BaseModelProcessor:
    """
    Base class for all model processor modules.
    Defines the common interface and basic initialization.
    """

    def __init__(self, config: "Config", device: torch.device):
        """
        Initializes the base processor.

        Args:
            config: The loaded application configuration object.
            device: The torch device (e.g., 'cuda', 'cpu') to use.
        """
        self.config = config
        self.device = device
        self.model: Optional[Any] = None  # Holds the loaded model object
        self.model_name: str = "BaseModel"  # Should be overridden by subclasses
        self.processor_name: str = "base"  # Should be overridden by subclasses
        # Ссылка на специфичную секцию конфига (устанавливается в дочерних классах)
        # Тип Any, так как дочерние классы будут использовать разные специфичные конфиги
        self.proc_config: Optional[Any] = None
        logger.debug(f"Initializing {self.__class__.__name__} for device {self.device}")

    def load(self):
        """
        Loads the necessary model files and initializes the model.
        Subclasses MUST implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the 'load' method."
        )

    def _get_postproc_param(self, param_name: str, default_value: Any) -> Any:
        """Helper to get postprocessing param, checking specific then general config."""
        # 1. Проверяем специфичный конфиг процессора (self.proc_config)
        # Убедимся, что self.proc_config не None и имеет атрибут
        specific_value = None
        if self.proc_config and hasattr(self.proc_config, param_name):
            specific_value = getattr(self.proc_config, param_name)

        if (
            specific_value is not None
        ):  # Используем, если значение явно задано (не None)
            logger.debug(
                f"Using '{param_name}' = {specific_value} from specific config [{self.processor_name}]"
            )
            return specific_value

        # 2. Если не нашли или там None, берем из общей секции [model]
        if hasattr(self.config.model, param_name):
            general_value = getattr(self.config.model, param_name)
            # Не проверяем на None здесь, т.к. общие параметры имеют дефолтные значения
            logger.debug(
                f"Using '{param_name}' = {general_value} from general [model] config"
            )
            return general_value

        # 3. Если нет нигде (маловероятно из-за ModelConfig), возвращаем значение по умолчанию
        logger.debug(f"Using default '{param_name}' = {default_value}")
        return default_value


    def _apply_common_postprocessing(
        self,
        raw_mask: Optional[Image.Image],
        original_image: Image.Image,
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Применяет общие шаги постобработки (sensitivity, blur, offset, invert, background).
        """
        # Импортируем утилиты здесь, чтобы избежать циклических зависимостей на уровне модуля
        from . import rmbg_utils, mask_ops

        if raw_mask is None:
            logger.warning(
                f"Получена пустая маска в постобработке для {self.processor_name}. Постобработка невозможна."
            )
            return None, None

        try:
            # Работаем с копией, чтобы не изменять оригинальную сырую маску
            processed_mask = raw_mask.copy()

            # --- 1. Применение sensitivity (если значение не по умолчанию) ---
            sensitivity = self._get_postproc_param("sensitivity", 1.0)
            if sensitivity != 1.0:
                logger.debug(f"Применение sensitivity: {sensitivity}")
                # Конвертируем в тензор для математических операций
                mask_tensor = rmbg_utils.mask_pil_to_tensor(processed_mask)
                
                # Применяем формулу
                factor = 1.0 + (1.0 - sensitivity)
                mask_tensor = mask_tensor * factor
                
                # Обрезаем значения и конвертируем обратно
                mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
                processed_mask = rmbg_utils.tensor_to_mask_pil(mask_tensor)

            # --- 2. Применение blur и offset ---
            mask_blur = self._get_postproc_param("mask_blur", 0)
            mask_offset = self._get_postproc_param("mask_offset", 0)
            
            # Передаем уже обработанную (sensitivity) маску дальше
            processed_mask = mask_ops.apply_postprocessing(
                processed_mask, mask_blur, mask_offset
            )

            # --- 3. Применение инверсии ---
            invert_output = self._get_postproc_param("invert_output", False)
            if invert_output:
                logger.debug("Инвертирование маски.")
                processed_mask = ImageOps.invert(processed_mask.convert("L"))

            # --- 4. Генерация финального изображения на основе фона ---
            background_mode = self._get_postproc_param("background", "Alpha")
            
            if background_mode == "Original":
                # Если фон "Original", просто возвращаем исходное изображение
                final_image = original_image.copy()
                logger.debug("Возвращаем оригинальное изображение как финальное.")
            else:
                # Для 'Alpha' и 'Solid' применяем маску
                bg_color_tuple = (0, 0, 0, 0) # По умолчанию для 'Alpha' (прозрачный)
                if background_mode == "Solid":
                    background_color_list = self._get_postproc_param("background_color", [255, 255, 255])
                    bg_color_tuple = tuple(background_color_list) + (255,)
                    logger.debug(f"Применение сплошного фона: {bg_color_tuple}")
                else:
                    logger.debug("Применение альфа-канала (прозрачный фон).")
                
                final_image = rmbg_utils.apply_mask_to_image(
                    original_image, processed_mask, background_color=bg_color_tuple
                )

            # Возвращаем финальное изображение и обработанную маску
            return final_image, processed_mask

        except Exception as e:
            logger.exception(
                f"Ошибка во время общей постобработки для {self.processor_name}: {e}"
            )
            return None, None

    def process(
        self, image: Image.Image, **kwargs
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Processes the input PIL image using subclass logic and applies common postprocessing.

        Args:
            image: Input PIL Image.
            **kwargs: Additional keyword arguments passed to _do_process.

        Returns:
            A tuple containing:
            - final_image (Optional[PIL.Image]): Image with background applied/original.
            - processed_mask (Optional[PIL.Image]): Final mask after blur/offset/invert (L mode).
            Returns (None, None) if processing or postprocessing fails.
        """
        # Шаг 1: Вызов _do_process из дочернего класса для генерации raw_mask
        try:
            raw_mask = self._do_process(image, **kwargs)
        except NotImplementedError:
            logger.error(
                f"'_do_process' method not implemented in {self.__class__.__name__}"
            )
            return None, None
        except Exception as e:
            logger.exception(
                f"Error during core processing (_do_process) in {self.__class__.__name__}: {e}"
            )
            return None, None

        # Шаг 2: Применение общей постобработки к raw_mask
        # Передаем оригинальное изображение для возможности вернуть его при background="Original"
        final_image, processed_mask = self._apply_common_postprocessing(raw_mask, image)

        return final_image, processed_mask

    def _do_process(self, image: Image.Image, **kwargs) -> Optional[Image.Image]:
        """
        Core processing logic to be implemented by subclasses.
        Should generate and return the raw mask (PIL Image, L mode) or None on failure.
        """
        # Этот метод должен быть переопределен в каждом конкретном процессоре
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the '_do_process' method."
        )

    def release(self):
        """
        Releases model resources (e.g., clear GPU memory).
        Subclasses can override this to add specific cleanup steps,
        but should call super().release() as well.
        """
        processor_display_name = f"{self.model_name} ({self.__class__.__name__})"
        if self.model is not None:
            logger.debug(f"Releasing model for {processor_display_name}...")
            # Явно удаляем ссылку на модель, чтобы помочь сборщику мусора
            del self.model
            self.model = None
        else:
            logger.debug(f"No active model to release for {processor_display_name}.")

        # Очищаем кэш CUDA, если используется GPU
        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        if self.device.type == "cuda":
            try:
                # Блок try...except с правильными отступами
                torch.cuda.empty_cache()
                logger.debug(
                    f"Cleared CUDA cache after releasing {processor_display_name}."
                )
            except Exception as e:
                logger.warning(
                    f"Could not clear CUDA cache for {processor_display_name}: {e}"
                )
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        logger.info(f"Модель {processor_display_name} выгружена из памяти.")
