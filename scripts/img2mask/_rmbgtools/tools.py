# ======================================================================================
# Блок 1: Импорты
# ======================================================================================
from PIL import Image
from typing import Optional

# Импортируем наши низкоуровневые функции
from .core.processing import (
    enhance_mask as apply_blur_and_offset,
    invert_mask,
    smooth_mask,
    fill_holes_in_mask
)

# ======================================================================================
# Блок 2: Класс MaskEnhancer
# ======================================================================================
class MaskEnhancer:
    """
    Предоставляет статические методы для постобработки и улучшения масок.
    """
    @staticmethod
    def enhance(
        mask: Image.Image,
        sensitivity: Optional[float] = None,
        blur: int = 0,
        offset: int = 0,
        smooth: float = 0.0,
        fill_holes: bool = False,
        invert: bool = False
    ) -> Image.Image:
        """
        Последовательно применяет различные фильтры и операции к маске.

        Args:
            mask (Image.Image): Входная маска.
            sensitivity (float, optional): Порог бинаризации (0-1). Если None, не применяется.
            blur (int): Радиус Гауссова размытия.
            offset (int): Смещение краев (расширение/сужение).
            smooth (float): Сигма для Гауссова сглаживания краев.
            fill_holes (bool): Если True, заполняет дыры в маске.
            invert (bool): Если True, инвертирует финальную маску.

        Returns:
            Image.Image: Улучшенная маска.
        """
        
        # Шаг 1: Конвертация в 'L' для консистентности
        processed_mask = mask.convert('L')

        # Шаг 2: Чувствительность (бинаризация)
        if sensitivity is not None:
            threshold = int(255 * sensitivity)
            processed_mask = processed_mask.point(lambda p: 255 if p > threshold else 0)

        # Шаг 3: Сглаживание краев
        if smooth > 0:
            processed_mask = smooth_mask(processed_mask, sigma=smooth)
            
        # Шаг 4: Заполнение дыр
        if fill_holes:
            processed_mask = fill_holes_in_mask(processed_mask)
            
        # Шаг 5: Размытие и смещение (старая функция)
        if blur > 0 or offset != 0:
            processed_mask = apply_blur_and_offset(processed_mask, blur_radius=blur, offset=offset)
            
        # Шаг 6: Инвертирование
        if invert:
            processed_mask = invert_mask(processed_mask)
            
        return processed_mask

# ======================================================================================
# Блок 3: Функция-обертка
# ======================================================================================
def enhance_mask_func(mask: Image.Image, **kwargs) -> Image.Image:
    """Удобная обертка для вызова MaskEnhancer.enhance."""
    return MaskEnhancer.enhance(mask, **kwargs)

# ======================================================================================
# Блок 4: Тестовый блок
# ======================================================================================
if __name__ == '__main__':
    print("--- Testing tools.py (MaskEnhancer) ---")
    
    # Создаем тестовую маску с дыркой и "рваным" краем
    test_mask = Image.new('L', (200, 200), 0)
    # Рисуем "бублик"
    for x in range(20, 180):
        for y in range(20, 180):
            if 40 < x < 160 and 40 < y < 160:
                continue # Дырка
            if (x // 10) % 2 == 0: # Рваный край
                test_mask.putpixel((x, y), 255)
    
    os.makedirs("./test_results", exist_ok=True)
    test_mask.save("./test_results/enhancer_before.png")
    
    # Применяем все эффекты
    enhanced = MaskEnhancer.enhance(
        test_mask, 
        smooth=2.0, 
        fill_holes=True,
        offset=5,
        blur=2
    )
    
    enhanced.save("./test_results/enhancer_after.png")
    
    # Проверки
    assert enhanced.getpixel((100, 100)) == 255 # Дырка заполнилась
    assert enhanced.getpixel((21, 21)) > 0      # Край расширился и сгладился
    print("MaskEnhancer test finished. Check ./test_results/ for visual comparison.")