# Содержимое rmbg_utils.py остается прежним
import torch
import numpy as np
from PIL import Image, ImageOps
import logging
from pathlib import Path
import cv2
from typing import Union

logger = logging.getLogger(__name__)


# --- Tensor Conversion ---
# numpy_to_tensor, pil_to_tensor, tensor_to_numpy, tensor_to_pil, tensor_to_mask_pil, mask_pil_to_tensor
def numpy_to_tensor(image_np: np.ndarray) -> torch.Tensor:
    if image_np.dtype == np.uint8:
        image_np = image_np.astype(np.float32) / 255.0
    elif image_np.max() > 1.0:  # Если float, но не в диапазоне 0-1
        logger.warning(
            "Input NumPy array is float but max value > 1.0. Assuming range 0-255 and scaling."
        )
        image_np = image_np.astype(np.float32) / 255.0
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=2)
    tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).unsqueeze(0)
    return tensor


def pil_to_tensor(image_pil: Image.Image) -> torch.Tensor:
    if image_pil.mode == "L":
        image_pil = image_pil.convert("RGB")
    elif image_pil.mode == "RGBA":
        background = Image.new("RGB", image_pil.size, (255, 255, 255))
        background.paste(image_pil, mask=image_pil.split()[3])
        image_pil = background
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).unsqueeze(0)
    return tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor[0]
    image_np = tensor.detach().cpu().numpy().transpose((1, 2, 0))
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    if image_np.shape[2] == 1:
        image_np = image_np.squeeze(axis=2)
    return image_np


def tensor_to_pil(tensor: torch.Tensor, mode: str = "RGB") -> Image.Image:
    image_np = tensor_to_numpy(tensor)
    pil_mode = (
        "L"
        if image_np.ndim == 2 or image_np.shape[2] == 1
        else ("RGBA" if image_np.shape[2] == 4 else "RGB")
    )
    img = Image.fromarray(
        image_np.squeeze(axis=2)
        if pil_mode == "L" and image_np.ndim == 3
        else image_np,
        mode=pil_mode,
    )
    return img if pil_mode == mode else img.convert(mode)


def tensor_to_mask_pil(mask_tensor: torch.Tensor) -> Image.Image:
    if mask_tensor.ndim not in [3, 4]:
        raise ValueError(f"Mask tensor must have 3 or 4 dims, got {mask_tensor.ndim}")
    if mask_tensor.ndim == 4:
        mask_tensor = (
            mask_tensor[0] if mask_tensor.shape[0] == 1 else mask_tensor[0]
        )  # Warn?
    if mask_tensor.shape[0] != 1:
        mask_tensor = mask_tensor[0:1, :, :]  # Warn?
    mask_np = np.clip(
        mask_tensor.detach().cpu().numpy().squeeze() * 255, 0, 255
    ).astype(np.uint8)
    return Image.fromarray(mask_np, mode="L")


def mask_pil_to_tensor(mask_pil: Image.Image) -> torch.Tensor:
    if mask_pil.mode != "L":
        mask_pil = mask_pil.convert("L")
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    return mask_tensor


# --- Image/Mask IO ---
def load_image(image_path: Union[str, Path]) -> Image.Image | None:
    try:
        img = Image.open(image_path)
        img.load()
        return img
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: Union[Image.Image, np.ndarray], output_path: Union[str, Path]):
    from .rmbg_logger import (
        get_message,
    )  # Late import to avoid circular dependency issues

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(image, np.ndarray):
            pil_mode = (
                "L"
                if image.ndim == 2 or image.shape[2] == 1
                else ("RGBA" if image.shape[2] == 4 else "RGB")
            )
            image_pil = Image.fromarray(
                image.squeeze(axis=2) if pil_mode == "L" and image.ndim == 3 else image,
                mode=pil_mode,
            )
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            logger.error("Unsupported image type for saving.")
            return
        image_pil.save(output_path)
        logger.debug(get_message("DEBUG_IMAGE_SAVED", output_path=str(output_path)))
    except Exception as e:
        logger.error(
            get_message("ERROR_SAVING_OUTPUT", output_path=str(output_path), exc=e)
        )


# --- Mask Operations Helpers ---
def ensure_opencv_mask(mask: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask.convert("L"))
    elif isinstance(mask, np.ndarray):
        mask_np = mask
        if mask_np.dtype != np.uint8:
            mask_np = (
                (mask_np * 255).astype(np.uint8)
                if mask_np.max() <= 1.0 and mask_np.min() >= 0.0
                else mask_np.astype(np.uint8)
            )
        if mask_np.ndim == 3 and mask_np.shape[2] == 1:
            mask_np = mask_np.squeeze(axis=2)
        elif mask_np.ndim != 2:
            raise ValueError(
                f"Unsupported NumPy mask shape for OpenCV: {mask_np.shape}"
            )
    else:
        raise TypeError("Input mask must be a PIL Image or NumPy array")
    return mask_np


def apply_mask_to_image(
    image: Image.Image, mask: Image.Image, background_color: tuple = (0, 0, 0, 0)
) -> Image.Image:
    image_rgba = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
    mask_l = mask.convert("L") if mask.mode != "L" else mask
    if image_rgba.size != mask_l.size:
        mask_l = mask_l.resize(image_rgba.size, Image.LANCZOS)
    image_rgba.putalpha(mask_l)
    if background_color == (0, 0, 0, 0):
        return image_rgba
    else:
        bg_color_tuple = (
            tuple(background_color[:3]) + (255,)
            if len(background_color) >= 3
            else (0, 0, 0, 255)
        )
        background = Image.new("RGBA", image_rgba.size, bg_color_tuple)
        background.alpha_composite(image_rgba)
        return background.convert("RGB")


def add_alpha_channel(image: Image.Image, alpha_value: int = 255) -> Image.Image:
    if image.mode == "RGBA":
        return image
    image_rgb = image.convert("RGB") if image.mode != "RGB" else image
    alpha = Image.new("L", image_rgb.size, alpha_value)
    image_rgb.putalpha(alpha)
    return image_rgb

def get_compute_device() -> torch.device:
    """
    Определяет и возвращает наиболее подходящее устройство для вычислений (CUDA, MPS, CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Проверка для macOS (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")