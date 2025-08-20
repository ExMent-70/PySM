# ======================================================================================
# Блок 1: Импорты и аннотации типов
# ======================================================================================
from typing import Union, Tuple
import numpy as np
import torch
from PIL import Image

# ======================================================================================
# Блок 2: Функции конвертации Tensor <-> PIL
# Эти функции являются ядром для взаимодействия между PyTorch (где происходят
# вычисления моделей) и Pillow (стандартный формат для работы с изображениями в Python).
# ======================================================================================
def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """
    Конвертирует тензор PyTorch в изображение Pillow (PIL.Image).
    
    Принимает на вход тензоры разной формы (батч [B, C, H, W], одно изображение [C, H, W]
    или даже [H, W, C]) и корректно преобразует их в PIL Image.

    Args:
        image (torch.Tensor): Входной тензор. Ожидаются значения в диапазоне [0.0, 1.0].

    Returns:
        Image.Image: Изображение в формате PIL.
    """
    if image.ndim == 4:
        image = image[0]  # Извлекаем первое изображение из батча

    # Перемещаем тензор на CPU, конвертируем в NumPy, денормализуем (умножаем на 255)
    # и преобразуем в беззнаковый 8-битный целочисленный тип, который ожидает PIL.
    # np.clip гарантирует, что значения останутся в диапазоне [0, 255].
    np_image = np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)
    
    # PyTorch работает с форматом [Каналы, Высота, Ширина] (C, H, W).
    # NumPy/PIL работают с форматом [Высота, Ширина, Каналы] (H, W, C).
    # Эта проверка определяет, нужно ли менять порядок измерений.
    if np_image.shape[0] in {1, 3, 4}:
        np_image = np.transpose(np_image, (1, 2, 0))

    # Для одноканальных изображений (масок) убираем лишнее измерение канала,
    # так как PIL ожидает для режима 'L' двумерный массив.
    if np_image.shape[-1] == 1:
        np_image = np_image.squeeze(axis=-1)
        
    return Image.fromarray(np_image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Конвертирует изображение Pillow (PIL.Image) в тензор PyTorch.

    Преобразует изображение в тензор с плавающей точкой, нормализует значения
    к диапазону [0.0, 1.0] и приводит к формату [B, C, H, W], который
    ожидают модели PyTorch.

    Args:
        image (Image.Image): Входное изображение PIL.

    Returns:
        torch.Tensor: Тензор в формате [1, C, H, W].
    """
    # Конвертируем изображение в массив NumPy и нормализуем
    np_image = np.array(image).astype(np.float32) / 255.0
    
    # Если изображение одноканальное (маска), у него нет измерения для каналов.
    # Добавляем его, чтобы получить форму (H, W, 1).
    if np_image.ndim == 2:
        np_image = np_image[..., np.newaxis]
        
    # Меняем порядок измерений с (H, W, C) на (C, H, W) и добавляем
    # измерение для батча в начало, получая [1, C, H, W].
    tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
    return tensor

# ======================================================================================
# Блок 3: Функции для работы с масками и цветом
# Утилиты, реализующие общую логику наложения масок и композитинга.
# ======================================================================================
def apply_mask_to_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Применяет одноканальную маску к изображению, создавая RGBA-изображение.
    
    Оригинальное изображение становится RGB-частью, а маска - альфа-каналом.

    Args:
        image (Image.Image): Исходное RGB-изображение.
        mask (Image.Image): Одноканальная ('L') маска. Белый цвет - непрозрачно, черный - прозрачно.

    Returns:
        Image.Image: Изображение в формате RGBA.
    """
    # Гарантируем, что маска имеет тот же размер, что и изображение
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    
    # Конвертируем изображение в RGBA, чтобы у него появился альфа-канал
    img_rgba = image.convert('RGBA')
    # Убеждаемся, что маска одноканальная
    mask_l = mask.convert('L')
    
    # Вставляем маску в альфа-канал изображения
    img_rgba.putalpha(mask_l)
    return img_rgba


def hex_to_rgba(hex_color: str, default_alpha: int = 255) -> Tuple[int, int, int, int]:
    """
    Конвертирует HEX-код цвета (например, '#FF0000' или '#FF000080') в кортеж RGBA.
    
    Args:
        hex_color (str): Цвет в формате HEX.
        default_alpha (int): Значение альфа-канала по умолчанию, если оно не указано в HEX.

    Returns:
        Tuple[int, int, int, int]: Кортеж (R, G, B, A).
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:  # Формат RRGGBB
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        a = default_alpha
    elif len(hex_color) == 8:  # Формат RRGGBBAA
        r, g, b, a = (int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
    else:
        raise ValueError("Invalid HEX color format. Use #RRGGBB or #RRGGBBAA.")
    return r, g, b, a


def composite_with_color_background(
    foreground: Image.Image, 
    background_color: Union[str, Tuple[int, int, int, int]]
) -> Image.Image:
    """
    Накладывает RGBA-изображение (передний план) на сплошной цветной фон.
    
    Args:
        foreground (Image.Image): Изображение переднего плана с альфа-каналом (RGBA).
        background_color (Union[str, Tuple]): Цвет фона в формате HEX или кортежа RGBA.
        
    Returns:
        Image.Image: Итоговое RGB-изображение без альфа-канала.
    """
    if foreground.mode != 'RGBA':
        # Функция ожидает, что альфа-канал уже применен.
        foreground = foreground.convert('RGBA')
        
    if isinstance(background_color, str):
        rgba_bg_color = hex_to_rgba(background_color)
    else:
        rgba_bg_color = background_color
        
    # Создаем сплошное изображение нужного цвета и размера
    bg_image = Image.new('RGBA', foreground.size, rgba_bg_color)
    
    # Выполняем альфа-композитинг: накладываем передний план на фон
    composite_image = Image.alpha_composite(bg_image, foreground)
    
    # Возвращаем в формате RGB, так как прозрачность больше не нужна
    return composite_image.convert('RGB')

# ======================================================================================
# Блок 4: Тестовый блок для самопроверки
# ======================================================================================
if __name__ == '__main__':
    print("--- Testing image_utils.py ---")

    # 1. Создаем тестовые данные
    test_pil_image_rgb = Image.new('RGB', (128, 128), color='red')
    test_tensor_chw = torch.rand(1, 3, 64, 64)
    test_mask_l = Image.new('L', (128, 128), color=128) # Полупрозрачная маска

    # 2. Тест pil_to_tensor
    converted_tensor = pil_to_tensor(test_pil_image_rgb)
    print(f"PIL to Tensor -> Original PIL size: {test_pil_image_rgb.size}, Converted Tensor shape: {converted_tensor.shape}")
    assert converted_tensor.shape == (1, 3, 128, 128)

    # 3. Тест tensor_to_pil
    converted_pil = tensor_to_pil(test_tensor_chw)
    print(f"Tensor to PIL -> Original Tensor shape: {test_tensor_chw.shape}, Converted PIL size: {converted_pil.size}")
    assert converted_pil.size == (64, 64)

    # 4. Тест apply_mask_to_image
    masked_image_rgba = apply_mask_to_image(test_pil_image_rgb, test_mask_l)
    print(f"Apply mask -> Result mode: {masked_image_rgba.mode}")
    assert masked_image_rgba.mode == 'RGBA'
    # Проверяем, что альфа-канал соответствует маске
    assert masked_image_rgba.getpixel((10, 10))[3] == 128 
    print("Apply mask to image test passed.")

    # 5. Тест composite_with_color_background
    composite_img_rgb = composite_with_color_background(masked_image_rgba, "#00FF00") # Зеленый фон
    print(f"Composite with color -> Result mode: {composite_img_rgb.mode}")
    assert composite_img_rgb.mode == 'RGB'
    # Пиксель фона должен быть зеленым
    assert composite_img_rgb.getpixel((0, 0)) == (0, 255, 0)
    # Пиксель объекта должен быть смесью красного и зеленого (из-за полупрозрачности)
    # (127, 128, 0) - результат смешивания red=(255,0,0) и green=(0,255,0) с alpha=128/255
    assert composite_img_rgb.getpixel((50, 50)) == (127, 128, 0)
    print("Composite with color background test passed.")
    
    print("\n✅ --- image_utils.py all tests passed! ---")