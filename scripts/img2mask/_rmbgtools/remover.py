# _rmbgtools/remover.py

# ======================================================================================
# Блок 1: Импорты
# ======================================================================================
import os
from typing import Union, List, Tuple, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import torch
from torchvision import transforms

from .core.model_manager import ModelManager, MODEL_CONFIGS
from .utils.image_utils import (
    tensor_to_pil, pil_to_tensor, apply_mask_to_image,
    composite_with_color_background
)
from .core.processing import refine_foreground
from .tools import MaskEnhancer
from . import logger

# ======================================================================================
# Блок 3: Базовый класс
# ======================================================================================
class BaseRemover:
    def __init__(self, model_dir: str, device: str):
        self.model_manager = ModelManager(model_dir=model_dir, device=device)
        self.device = self.model_manager.device

    def _preprocess(self, image: Image.Image, resolution: int) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def _postprocess(self, preds: torch.Tensor, original_size: Tuple[int, int]) -> Image.Image:
        h, w = original_size[1], original_size[0]
        preds = torch.nn.functional.interpolate(
            preds, size=(h, w), mode='bicubic', align_corners=False
        )
        mask_tensor = preds.squeeze()
        return tensor_to_pil(mask_tensor)

    def _create_and_save_output(
        self, image, enhanced_mask, image_path, output_dir,
        refine_foreground_flag, background, background_color
    ) -> Tuple[str, str]:
        if refine_foreground_flag:
            image_tensor = pil_to_tensor(image).to(self.device)
            mask_tensor = pil_to_tensor(enhanced_mask).to(self.device)
            refined_tensor = refine_foreground(image_tensor, mask_tensor)
            rgba_image = tensor_to_pil(refined_tensor)
        else:
            rgba_image = apply_mask_to_image(image, enhanced_mask)

        if background == "Alpha":
            result_image = rgba_image
        else:
            result_image = composite_with_color_background(rgba_image, background_color)
            
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        img_path = os.path.join(output_dir, f"{base_name}_removed.png")
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        result_image.save(img_path)
        enhanced_mask.save(mask_path)
        
        return img_path, mask_path

    def _run_parallel(
        self,
        images: List[str],
        output_dir: str,
        worker_kwargs: dict,
        num_threads: int
    ) -> Iterator[Optional[Tuple[str, str]]]:
        """Запускает параллельную обработку и возвращает итератор по результатам."""
        os.makedirs(output_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_path = {
                executor.submit(self._process_single_image, path, output_dir, **worker_kwargs): path
                for path in images
            }
            # Превращаем метод в генератор, который отдает результаты по мере их готовности
            for future in as_completed(future_to_path):
                yield future.result()

    def remove(self, images: Union[str, List[str]], output_dir: str, **kwargs) -> Iterator[Optional[Tuple[str, str]]]:
        """Основной метод, который запускает обработку и возвращает генератор."""
        image_paths = [images] if isinstance(images, str) else images
        # Этот метод теперь также является генератором
        yield from self._run_parallel(image_paths, output_dir, kwargs, kwargs.get('num_threads', 4))

# ======================================================================================
# Блок 4: Класс RMBGRemover
# ======================================================================================
class RMBGRemover(BaseRemover):
    def __init__(self, model_dir: str = "models", device: str = "auto"):
        super().__init__(model_dir, device)
        self.model = self.model_manager.get_model("RMBG-2.0")

    def _process_single_image(self, image_path: str, output_dir: str, **kwargs) -> Optional[Tuple[str, str]]:
        try:
            image = Image.open(image_path)
            orig_size = image.size
            
            input_tensor = self._preprocess(image, kwargs['process_res'])
            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid()
            
            mask_pil = self._postprocess(preds, orig_size)
            
            enhanced_mask = MaskEnhancer.enhance(
                mask=mask_pil,
                sensitivity=kwargs['sensitivity'],
                blur=kwargs['mask_blur'],
                offset=kwargs['mask_offset'],
                invert=kwargs['invert_output'],
                smooth=kwargs['smooth'],
                fill_holes=kwargs['fill_holes']
            )

            return self._create_and_save_output(
                image, enhanced_mask, image_path, output_dir,
                kwargs['refine_foreground'], kwargs['background'], kwargs['background_color']
            )
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}", exc_info=True)
            return None

# ======================================================================================
# Блок 5: Класс BiRefNetRemover
# ======================================================================================
class BiRefNetRemover(BaseRemover):
    def __init__(self, model_name: str, model_dir: str = "models", device: str = "auto"):
        super().__init__(model_dir, device)
        self.model_name = model_name
        self.model = self.model_manager.get_model(model_name)
        model_config = MODEL_CONFIGS[model_name]
        self.process_res = model_config.get("default_res", 1024)

    def _process_single_image(self, image_path: str, output_dir: str, **kwargs) -> Optional[Tuple[str, str]]:
        try:
            image = Image.open(image_path)
            orig_size = image.size
            
            input_tensor = self._preprocess(image, self.process_res)
            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid()

            mask_pil = self._postprocess(preds, orig_size)
            
            enhanced_mask = MaskEnhancer.enhance(
                mask=mask_pil,
                blur=kwargs['mask_blur'],
                offset=kwargs['mask_offset'],
                invert=kwargs['invert_output'],
                smooth=kwargs['smooth'],
                fill_holes=kwargs['fill_holes']
            )

            return self._create_and_save_output(
                image, enhanced_mask, image_path, output_dir,
                kwargs['refine_foreground'], kwargs['background'], kwargs['background_color']
            )
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}", exc_info=True)
            return None

# ======================================================================================
# Блок 6: Функция-обертка
# ======================================================================================
def remove_background(
    images: Union[str, List[str]],
    output_dir: str,
    model_name: str = "RMBG-2.0",
    model_dir: str = "models",
    device: str = "auto",
    **kwargs
) -> Iterator[Optional[Tuple[str, str]]]:
    """
    Удобная функция-обертка для удаления фона.
    """
    constructor_kwargs = {"model_dir": model_dir, "device": device}
    
    if "RMBG" in model_name:
        RemoverClass = RMBGRemover
    elif "BiRefNet" in model_name:
        RemoverClass = BiRefNetRemover
        constructor_kwargs["model_name"] = model_name
    else:
        raise ValueError(f"Unsupported model for background removal: {model_name}")

    remover = RemoverClass(**constructor_kwargs)
    return remover.remove(images=images, output_dir=output_dir, **kwargs)