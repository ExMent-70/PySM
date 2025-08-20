# ======================================================================================
# Блок 1: Импорты
# ======================================================================================
import torch
import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

# ======================================================================================
# Блок 2: Функции обработки масок
# ======================================================================================
def enhance_mask(
    mask: Image.Image, 
    blur_radius: int = 0, 
    offset: int = 0
) -> Image.Image:
    """
    Улучшает маску, применяя размытие и смещение границ.
    """
    if mask.mode != 'L':
        mask = mask.convert('L')
        
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    if offset != 0:
        op_filter = ImageFilter.MaxFilter if offset > 0 else ImageFilter.MinFilter
        for _ in range(abs(offset)):
            mask = mask.filter(op_filter(3))
            
    return mask

def invert_mask(mask: Image.Image) -> Image.Image:
    """Инвертирует (обращает) цвета маски."""
    if mask.mode != 'L':
        mask = mask.convert('L')
    return Image.fromarray(255 - np.array(mask))

# ======================================================================================
# Блок 3: НОВАЯ ФУНКЦИЯ - Улучшение переднего плана
# Адаптировано из оригинального кода AILab_BiRefNet.py
# ======================================================================================
def refine_foreground(image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
    """
    Уточняет цвета на границах вырезанного объекта, чтобы сделать переход к
    прозрачному фону более плавным и естественным (Fast Foreground Color Estimation).

    Args:
        image_tensor (torch.Tensor): Тензор исходного изображения [B, C, H, W] со значениями [0, 1].
        mask_tensor (torch.Tensor): Тензор маски [B, 1, H, W] со значениями [0, 1].

    Returns:
        torch.Tensor: Тензор RGBA переднего плана [B, 4, H, W].
    """
    # Убедимся, что тензоры на CPU и в формате float32 для обработки
    image_np = image_tensor.cpu().float().numpy()
    mask_np = mask_tensor.cpu().float().numpy()
    
    refined_batch = []
    for i in range(image_np.shape[0]):
        # Извлекаем данные для одного изображения:
        # image: [C, H, W] -> [H, W, C]
        img_hwc = np.transpose(image_np[i], (1, 2, 0))
        # mask: [1, H, W] -> [H, W]
        mask_hw = mask_np[i, 0]

        # --- Начало логики уточнения ---
        
        # 1. Создаем бинарную маску для определения четких краев
        threshold = 0.5
        mask_binary = (mask_hw > threshold).astype(np.float32)
        
        # 2. Слегка размываем бинарную маску, чтобы получить плавные края.
        # Используем scipy.ndimage для консистентности с другими функциями.
        # Сигма 2.5 подобрана, чтобы быть похожей на (5,5) ядро в cv2.
        edge_blur = ndimage.gaussian_filter(mask_binary, sigma=2.5)
        
        # 3. Определяем "переходную зону" - область полупрозрачности
        transition_mask = np.logical_and(mask_hw > 0.05, mask_hw < 0.95)
        
        # 4. В переходной зоне смешиваем исходную "мягкую" маску с размытой "жесткой"
        alpha = 0.85
        mask_refined = np.where(
            transition_mask,
            alpha * mask_hw + (1 - alpha) * edge_blur,
            mask_binary
        )
        
        # --- Конец логики уточнения ---

        # 5. Формируем RGBA изображение
        # Убедимся, что форма (H, W, C)
        h, w, c = img_hwc.shape
        refined_rgba = np.zeros((h, w, 4), dtype=np.float32)
        
        refined_rgba[..., :3] = img_hwc
        # Присваиваем уточненную маску в альфа-канал.
        # mask_refined должен иметь форму (H, W), что совпадает с refined_rgba[..., 3]
        refined_rgba[..., 3] = mask_refined
        
        # 6. Конвертируем обратно в [B, C, H, W] тензор
        refined_tensor = torch.from_numpy(refined_rgba).permute(2, 0, 1).unsqueeze(0)
        refined_batch.append(refined_tensor)
        
    return torch.cat(refined_batch, dim=0)

# ======================================================================================
# Блок 4: НОВЫЕ функции постобработки
# ======================================================================================
def smooth_mask(mask: Image.Image, sigma: float) -> Image.Image:
    """
    Сглаживает края бинарной маски с помощью Гауссова фильтра.
    Это помогает убрать "ступенчатость" краев.

    Args:
        mask (Image.Image): Входная маска PIL.
        sigma (float): Сила сглаживания (стандартное отклонение для Гаусса).

    Returns:
        Image.Image: Сглаженная маска.
    """
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    mask_np = np.array(mask)
    # Создаем бинарную маску, чтобы сглаживать только четкие края
    binary_mask = (mask_np > 127).astype(np.float32) 
    
    # Применяем Гауссов фильтр
    smoothed_mask_np = ndimage.gaussian_filter(binary_mask, sigma=sigma)
    
    # Применяем порог, чтобы снова сделать маску бинарной, но уже с гладкими краями
    final_mask_np = (smoothed_mask_np > 0.5).astype(np.uint8) * 255
    
    return Image.fromarray(final_mask_np)

def fill_holes_in_mask(mask: Image.Image) -> Image.Image:
    """
    Находит и заливает "дыры" (черные области) внутри замкнутых белых контуров на маске.

    Args:
        mask (Image.Image): Входная маска PIL.

    Returns:
        Image.Image: Маска с заполненными отверстиями.
    """
    if mask.mode != 'L':
        mask = mask.convert('L')
        
    mask_np = np.array(mask)
    # Бинаризуем маску для поиска контуров
    _, thresh_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    
    # Находим все контуры
    contours, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Создаем пустое изображение для рисования залитых контуров
    filled_mask = np.zeros_like(mask_np)
    
    if hierarchy is not None:
        # Иерархия в OpenCV имеет формат [Next, Previous, First_Child, Parent]
        # Дыры - это контуры, у которых есть родитель (т.е. они находятся внутри другого контура)
        for i in range(len(contours)):
            # Заливаем все внешние контуры
            if hierarchy[0][i][3] == -1: # Если нет родителя
                cv2.drawContours(filled_mask, [contours[i]], 0, 255, -1)
                
    return Image.fromarray(filled_mask)


# ======================================================================================
# Блок 4: Тестовый блок
# ======================================================================================
if __name__ == '__main__':
    # ... (существующие тесты для enhance_mask и invert_mask)
    pass # Тестирование refine_foreground будет происходить в remover.py, где есть реальные данные

    # Тестовый блок для проверки функциональности
    print("--- Testing processing.py ---")

    # 1. Создаем тестовую маску
    test_mask = Image.new('L', (100, 100), 0)
    # Рисуем белый квадрат в центре
    for x in range(25, 75):
        for y in range(25, 75):
            test_mask.putpixel((x, y), 255)
    
    # test_mask.save("test_mask_before.png")

    # 2. Тест с расширением (offset > 0)
    expanded_mask = enhance_mask(test_mask, offset=5)
    # expanded_mask.save("test_mask_expanded.png")
    # Проверяем, что центральный пиксель остался белым
    assert expanded_mask.getpixel((50, 50)) == 255
    # Проверяем, что пиксель за пределами исходного квадрата стал белым
    assert expanded_mask.getpixel((20, 50)) == 255
    print("Mask expansion test passed.")
    
    # 3. Тест с сужением (offset < 0)
    eroded_mask = enhance_mask(test_mask, offset=-5)
    # eroded_mask.save("test_mask_eroded.png")
    # Проверяем, что пиксель на границе исходного квадрата стал черным
    assert eroded_mask.getpixel((25, 50)) == 0
    print("Mask erosion test passed.")
    
    # 4. Тест с размытием
    blurred_mask = enhance_mask(test_mask, blur_radius=5)
    # blurred_mask.save("test_mask_blurred.png")
    # Проверяем, что центральный пиксель все еще белый
    assert blurred_mask.getpixel((50, 50)) == 255
    # Проверяем, что пиксель на границе стал серым (не 0 и не 255)
    assert 0 < blurred_mask.getpixel((25, 50)) < 255
    print("Mask blur test passed.")

    # 5. Тест инвертирования
    inverted_mask = invert_mask(test_mask)
    # inverted_mask.save("test_mask_inverted.png")
    # Центральный пиксель должен стать черным
    assert inverted_mask.getpixel((50, 50)) == 0
    # Угловой пиксель должен стать белым
    assert inverted_mask.getpixel((0, 0)) == 255
    print("Mask inversion test passed.")

    
    print("\n--- Testing new processing functions ---")
    
    # 1. Тест fill_holes_in_mask
    hole_mask = Image.new('L', (100, 100), 0)
    # Рисуем белый "бублик"
    for x in range(10, 90):
        for y in range(10, 90):
            hole_mask.putpixel((x, y), 255)
    for x in range(30, 70):
        for y in range(30, 70):
            hole_mask.putpixel((x, y), 0) # Дырка
            
    # hole_mask.save("./test_results/hole_mask_before.png")
    filled = fill_holes_in_mask(hole_mask)
    # filled.save("./test_results/hole_mask_after.png")
    
    assert filled.getpixel((50, 50)) == 255 # Центр дырки должен стать белым
    assert filled.getpixel((5, 5)) == 0     # Внешняя область должна остаться черной
    print("Fill holes test passed.")

    # 2. Тест smooth_mask
    jagged_mask = Image.new('L', (100, 100), 0)
    # Рисуем простой квадрат
    for x in range(25, 75):
        for y in range(25, 75):
            jagged_mask.putpixel((x, y), 255)

    smoothed = smooth_mask(jagged_mask, sigma=5)
    # smoothed.save("./test_results/smoothed_mask.png")
    # Проверяем, что углы скруглились (пиксель на углу стал черным)
    assert smoothed.getpixel((25, 25)) == 0 
    print("Smooth mask test passed.")

    print("--- processing.py all tests passed! ---")
    
