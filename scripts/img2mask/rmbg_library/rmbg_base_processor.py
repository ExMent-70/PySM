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
        original_image: Image.Image,  # Передаем оригинал для apply_mask_to_image
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Applies common postprocessing steps (blur, offset, invert, background)."""
        # Импортируем утилиты здесь, чтобы избежать циклических зависимостей на уровне модуля
        from . import rmbg_utils, mask_ops

        if raw_mask is None:
            logger.warning(
                f"Received None mask in postprocessing for {self.processor_name}. Cannot apply postprocessing."
            )
            # Возвращаем None для обоих результатов, если нет сырой маски
            return None, None

        try:
            # Получаем параметры постобработки
            mask_blur = self._get_postproc_param("mask_blur", 0)
            mask_offset = self._get_postproc_param("mask_offset", 0)
            invert_output = self._get_postproc_param("invert_output", False)
            background_mode = self._get_postproc_param("background", "Alpha")
            # background_color берется из специфичного или общего конфига
            background_color_list = self._get_postproc_param(
                "background_color", [255, 255, 255]
            )

            # Применяем blur и offset
            logger.debug(
                f"Applying postprocessing: blur={mask_blur}, offset={mask_offset}, invert={invert_output}, bg='{background_mode}'"
            )
            processed_mask = mask_ops.apply_postprocessing(
                raw_mask, mask_blur, mask_offset
            )

            # Применяем инверсию
            if invert_output:
                logger.debug("Inverting mask.")
                processed_mask = ImageOps.invert(
                    processed_mask.convert("L")
                )  # Убедимся что L режим

            # Генерируем финальное изображение
            if background_mode == "Alpha":
                bg_color_tuple = (0, 0, 0, 0)  # Прозрачный
                logger.debug("Applying Alpha background.")
            elif background_mode == "Solid":
                # Убедимся, что цвет корректный
                if (
                    not isinstance(background_color_list, list)
                    or len(background_color_list) != 3
                ):
                    logger.warning(
                        f"Invalid background_color list: {background_color_list}. Using white fallback."
                    )
                    background_color_list = [255, 255, 255]
                bg_color_tuple = tuple(background_color_list) + (
                    255,
                )  # Добавляем альфа
                logger.debug(f"Applying Solid background: {bg_color_tuple}")
            elif background_mode == "Original":
                bg_color_tuple = None  # Сигнал не применять фон
                logger.debug("Keeping Original background.")
            else:
                bg_color_tuple = (0, 0, 0, 0)  # По умолчанию Alpha
                logger.warning(
                    f"Invalid background mode '{background_mode}', using Alpha."
                )

            if bg_color_tuple is None:
                # Если фон "Original", возвращаем оригинальное изображение
                final_image = original_image.copy()
                logger.debug("Returning original image as final image.")
            else:
                # Применяем обработанную маску к ОРИГИНАЛЬНОМУ изображению
                logger.debug("Applying processed mask to original image.")
                final_image = rmbg_utils.apply_mask_to_image(
                    original_image, processed_mask, background_color=bg_color_tuple
                )

            # Возвращаем финальное изображение и обработанную маску
            return final_image, processed_mask

        except Exception as e:
            logger.exception(
                f"Error during common postprocessing for {self.processor_name}: {e}"
            )
            # Возвращаем None при ошибке постобработки, чтобы main.py мог это обработать
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

        logger.info(f"Resources released for {processor_display_name}.")
