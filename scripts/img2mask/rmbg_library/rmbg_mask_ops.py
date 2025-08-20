# Содержимое rmbg_mask_ops.py остается прежним
import logging
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2
from scipy.ndimage import gaussian_filter
from typing import Union
from .rmbg_utils import ensure_opencv_mask

logger = logging.getLogger(__name__)


# blur_mask, expand_mask, shrink_mask, offset_mask, invert_mask, combine_masks, mask_to_image, image_to_mask, apply_postprocessing
# ... (код этих функций остается прежним) ...
def blur_mask(
    mask: Union[Image.Image, np.ndarray], radius: int
) -> Union[Image.Image, np.ndarray]:
    if radius <= 0:
        return mask
    is_pil = isinstance(mask, Image.Image)
    if is_pil:
        return mask.filter(ImageFilter.GaussianBlur(radius=radius))
    elif isinstance(mask, np.ndarray):
        mask_float = mask.astype(np.float32) if mask.dtype != np.float32 else mask
        sigma = max(0.5, radius / 2.0)
        blurred_np = gaussian_filter(mask_float, sigma=sigma)
        if mask.dtype == np.uint8:
            blurred_np = np.clip(blurred_np, 0, 255).astype(np.uint8)
        return blurred_np
    else:
        logger.error("Unsupported type for blur_mask.")
        return mask


def expand_mask(
    mask: Union[Image.Image, np.ndarray], pixels: int
) -> Union[Image.Image, np.ndarray]:
    if pixels <= 0:
        return mask
    mask_np = ensure_opencv_mask(mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1)
    )
    expanded_np = cv2.dilate(mask_np, kernel, iterations=1)
    if isinstance(mask, Image.Image):
        return Image.fromarray(expanded_np, mode="L")
    else:
        return (
            expanded_np.astype(mask.dtype)
            if isinstance(mask, np.ndarray)
            else expanded_np
        )


def shrink_mask(
    mask: Union[Image.Image, np.ndarray], pixels: int
) -> Union[Image.Image, np.ndarray]:
    if pixels <= 0:
        return mask
    mask_np = ensure_opencv_mask(mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1)
    )
    shrunk_np = cv2.erode(mask_np, kernel, iterations=1)
    if isinstance(mask, Image.Image):
        return Image.fromarray(shrunk_np, mode="L")
    else:
        return (
            shrunk_np.astype(mask.dtype) if isinstance(mask, np.ndarray) else shrunk_np
        )


def offset_mask(
    mask: Union[Image.Image, np.ndarray], pixels: int
) -> Union[Image.Image, np.ndarray]:
    if pixels == 0:
        return mask
    elif pixels > 0:
        return expand_mask(mask, pixels)
    else:
        return shrink_mask(mask, abs(pixels))


def invert_mask(mask: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
    if isinstance(mask, Image.Image):
        mask_l = mask.convert("L") if mask.mode != "L" else mask
        return ImageOps.invert(mask_l)
    elif isinstance(mask, np.ndarray):
        if mask.dtype == np.uint8:
            return 255 - mask
        elif mask.dtype == np.bool_:
            return ~mask
        elif mask.max() <= 1.0 and mask.min() >= 0.0:
            return 1.0 - mask
        else:
            return 255 - mask.astype(np.uint8)  # Fallback
    else:
        logger.error("Unsupported type for invert_mask.")
        return mask


def combine_masks(
    mask1: Union[Image.Image, np.ndarray],
    mask2: Union[Image.Image, np.ndarray],
    mode: str = "add",
) -> Union[Image.Image, np.ndarray]:
    is_pil_output = isinstance(mask1, Image.Image)
    m1 = ensure_opencv_mask(mask1)
    m2 = ensure_opencv_mask(mask2)
    if m1.shape != m2.shape:
        logger.warning(
            f"Mask shapes differ ({m1.shape} vs {m2.shape}). Resizing second mask."
        )
        m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mode == "add":
        result_np = cv2.bitwise_or(m1, m2)
    elif mode == "subtract":
        result_np = cv2.bitwise_and(m1, cv2.bitwise_not(m2))
    elif mode == "multiply":
        result_np = cv2.bitwise_and(m1, m2)
    elif mode == "difference":
        result_np = cv2.bitwise_xor(m1, m2)
    else:
        logger.error(f"Unsupported combine_masks mode: {mode}.")
        return mask1
    if is_pil_output:
        return Image.fromarray(result_np, mode="L")
    else:
        return (
            result_np.astype(mask1.dtype)
            if isinstance(mask1, np.ndarray)
            else result_np
        )


def mask_to_image(
    mask: Union[Image.Image, np.ndarray],
) -> Union[Image.Image, np.ndarray]:
    is_pil = isinstance(mask, Image.Image)
    if is_pil:
        mask_l = mask.convert("L") if mask.mode != "L" else mask
        return mask_l.convert("RGB")
    elif isinstance(mask, np.ndarray):
        mask_np = ensure_opencv_mask(mask)
        return np.stack((mask_np,) * 3, axis=-1)
    else:
        logger.error("Unsupported type for mask_to_image.")
        return mask


def image_to_mask(
    image: Union[Image.Image, np.ndarray], threshold: int = 128
) -> Union[Image.Image, np.ndarray]:
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        mask_l = image.convert("L")
        mask_thresholded = mask_l.point(
            lambda p: 255 if p >= threshold else 0, mode="L"
        )
        return mask_thresholded
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            gray_np = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY,
            )
        elif image.ndim == 2:
            gray_np = image
        else:
            logger.error(f"Unsupported numpy shape for image_to_mask: {image.shape}")
            return image
        _, mask_np = cv2.threshold(
            gray_np.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY
        )
        return mask_np
    else:
        logger.error("Unsupported type for image_to_mask.")
        return image


def apply_postprocessing(
    mask: Union[Image.Image, np.ndarray], blur_radius: int, offset_pixels: int
) -> Image.Image:
    is_pil_input = isinstance(mask, Image.Image)
    current_mask = mask
    if blur_radius > 0:
        logger.debug(f"Applying blur with radius {blur_radius}")
        current_mask = blur_mask(current_mask, blur_radius)
    if offset_pixels != 0:
        logger.debug(f"Applying offset with pixels {offset_pixels}")
        current_mask = offset_mask(current_mask, offset_pixels)
    if not isinstance(current_mask, Image.Image):
        current_mask = Image.fromarray(ensure_opencv_mask(current_mask), mode="L")
    elif current_mask.mode != "L":
        current_mask = current_mask.convert("L")
    return current_mask

def fast_refine(mask: Image.Image) -> Image.Image:
    """
    Применяет быстрое уточнение краев маски, смягчая переходы.
    """
    try:
        mask_np_float = np.array(mask.convert("L")).astype(np.float32) / 255.0
        
        # 1. Бинарная маска для основы
        thresh = 0.45
        mask_binary = (mask_np_float > thresh).astype(np.float32)
        
        # 2. Легкое размытие краев
        edge_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0)
        
        # 3. Определение "зон перехода"
        transition_mask = (mask_np_float > 0.05) & (mask_np_float < 0.95)
        
        # 4. Смешивание
        alpha = 0.85
        mask_refined_np = np.where(transition_mask,
                                   alpha * mask_np_float + (1 - alpha) * edge_blur,
                                   mask_binary)
        
        # 5. Дополнительное легкое смягчение в самих зонах перехода
        edge_region = (mask_np_float > 0.2) & (mask_np_float < 0.8)
        mask_refined_np = np.where(edge_region,
                                   mask_refined_np * 0.98,
                                   mask_refined_np)
        
        # 6. Конвертация обратно в PIL Image
        final_mask_np = np.clip(mask_refined_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(final_mask_np, mode="L")
        
    except Exception as e:
        logger.error(f"Error during fast_refine: {e}. Returning original mask.")
        return mask