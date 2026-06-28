# analize/cluster_editor/_lib/image_processing.py
# -*- coding: utf-8 -*-

"""
Модуль обработки изображений.
Содержит функции цветокоррекции и генерации водяных знаков.
Используется как в GUI (превью), так и в Multiprocessing воркерах (экспорт).
Зависимости: Только Pillow (PIL). Никакого Qt.
"""

import random
from typing import Dict, List, Tuple, Any
try:
    from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageFilter, ImageChops
except ImportError:
    Image = None

def apply_color_corrections(image: Image.Image, factors: Dict[str, float]) -> Image.Image:
    """
    Применяет полный стек цветокоррекции:
    1. Basic (Brightness, Contrast, Color, Sharpness)
    2. Advanced (Temperature, Tint)
    3. Levels (Capture One Style: Black/White points)
    
    Корректно обрабатывает RGBA изображения.
    """
    if not image or not Image: return image
    
    # 1. Base Corrections
    if factors.get("brightness", 1.0) != 1.0:
        image = ImageEnhance.Brightness(image).enhance(factors["brightness"])
    if factors.get("contrast", 1.0) != 1.0:
        image = ImageEnhance.Contrast(image).enhance(factors["contrast"])
    if factors.get("color", 1.0) != 1.0:
        image = ImageEnhance.Color(image).enhance(factors["color"])
    if factors.get("sharpness", 1.0) != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(factors["sharpness"])

    # 2. Advanced Corrections (Temp / Tint / Levels)
    # Проверяем, нужно ли вообще делать тяжелые операции с пикселями
    temp = factors.get("temperature", 0.0)
    tint = factors.get("tint", 0.0)
    black_val = int(factors.get("black_point", 0))
    white_val = int(factors.get("white_point", 0))
    
    if temp == 0.0 and tint == 0.0 and black_val == 0 and white_val == 0:
        return image

    # Распаковка каналов
    has_alpha = False
    alpha_channel = None
    
    if image.mode == 'RGBA':
        has_alpha = True
        r, g, b, alpha_channel = image.split()
    elif image.mode == 'RGB':
        r, g, b = image.split()
    else:
        # Для L или CMYK конвертируем в RGB
        image = image.convert('RGB')
        r, g, b = image.split()

    # --- Temperature & Tint ---
    if temp != 0.0 or tint != 0.0:
        # Temp: Warm (+R, -B), Cool (-R, +B)
        t_val = temp * 0.002 
        if temp > 0:
            r = r.point(lambda i: i * (1 + t_val))
            b = b.point(lambda i: i * (1 - t_val))
        else:
            r = r.point(lambda i: i * (1 + t_val))
            b = b.point(lambda i: i * (1 - t_val))
            
        # Tint: Magenta (+R+B or -G), Green (+G)
        tn_val = tint * 0.002
        g = g.point(lambda i: i * (1 - tn_val))

    # --- Levels (Capture One Style) ---
    if black_val != 0 or white_val != 0:
        in_black, in_white = 0, 255
        out_black, out_white = 0, 255

        # Logic: 
        # Black Slider < 0 -> Increase Input Black (Crush shadows)
        # Black Slider > 0 -> Increase Output Black (Lift shadows / Fade)
        if black_val < 0: in_black = int(abs(black_val) * 0.5) 
        else: out_black = int(black_val * 0.5)

        # White Slider > 0 -> Decrease Input White (Clip highlights)
        # White Slider < 0 -> Decrease Output White (Dim highlights)
        if white_val > 0: in_white = 255 - int(white_val * 0.5)
        else: out_white = 255 - int(abs(white_val) * 0.5)

        def levels_map(x):
            if x <= in_black: return out_black
            if x >= in_white: return out_white
            # Линейная интерполяция
            res = out_black + (x - in_black) * (out_white - out_black) / (in_white - in_black)
            return int(res)
            
        r = r.point(levels_map)
        g = g.point(levels_map)
        b = b.point(levels_map)

    # Сборка обратно
    if has_alpha:
        image = Image.merge("RGBA", (r, g, b, alpha_channel))
    else:
        image = Image.merge("RGB", (r, g, b))

    return image


def create_watermark_layer(base_size: Tuple[int, int], faces_bboxes: List[List[float]], settings: Dict[str, Any], student_name: str) -> Any:
    """
    Генерирует слой (PIL Image RGBA) с водяными знаками (полосы + маски лиц + текст).
    """
    if not Image: return None
    
    width, height = base_size
    stripes = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stripes)
    
    base_period = int(width * 0.05)
    base_period = max(40, base_period)

    # Извлечение параметров
    stripe_alpha = int(settings.get("wm_stripe_alpha", 45))
    mask_fill = int(settings.get("wm_mask_fill", 20))
    pad_w_coeff = float(settings.get("wm_pad_w", 0.1))
    pad_h_coeff = float(settings.get("wm_pad_h", 0.2))
    text_content = str(settings.get("wm_text", "ВЫБОР ФОТОГРАФИИ")) + " - " + student_name
    text_alpha = int(settings.get("wm_text_alpha", 150))

    # 1. Подготовка текста (один раз)
    rotated_txt_img = None
    if ImageFont and text_content:
        font_size = int(max(12, height * 0.02)) 
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try: font = ImageFont.load_default()
            except: font = None
        
        if font:
            dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
            try:
                left, top, right, bottom = dummy_draw.textbbox((0, 0), text_content, font=font)
                text_w, text_h = right - left, bottom - top
            except AttributeError:
                text_w, text_h = dummy_draw.textsize(text_content, font=font)
            
            txt_img = Image.new('RGBA', (text_w, text_h + 10), (255, 255, 255, 0))
            d_txt = ImageDraw.Draw(txt_img)
            d_txt.text((0, 0), text_content, font=font, fill=(255, 255, 255, text_alpha))
            rotated_txt_img = txt_img.rotate(90, expand=True)
            rw, rh = rotated_txt_img.size

    # 2. Рисование полос
    current_x = 0
    # Используем фиксированный seed для воспроизводимости в превью и экспорте?
    # Пока оставим random, но в идеале передавать seed. 
    # В данном случае random вызывается локально, так что полосы будут немного разными каждый раз, но это не критично.
    
    while current_x < width:
        stripe_width = random.randint(int(base_period * 0.6), int(base_period * 0.9))
        gap_width = random.randint(int(base_period * 0.2), int(base_period * 0.4))
        
        r = random.randint(200, 255)
        g = random.randint(200, 255)
        b = random.randint(200, 255)
        
        draw.rectangle([current_x, 0, current_x + stripe_width, height], fill=(r, g, b, stripe_alpha))
        
        # Вставка текста на полосу
        if rotated_txt_img and stripe_width > 10:
            center_x = current_x + (stripe_width // 2)
            paste_x = int(center_x - (rw / 2))
            
            if height > rh + 20:
                paste_y = random.randint(10, height - rh - 10)
            else:
                paste_y = 0
            
            stripes.paste(rotated_txt_img, (paste_x, paste_y), rotated_txt_img)

        current_x += stripe_width + gap_width

    # 3. Маска лиц
    if not faces_bboxes:
        return stripes

    mask = Image.new("L", (width, height), 255)
    draw_mask = ImageDraw.Draw(mask)
    
    for bbox in faces_bboxes:
        if len(bbox) != 4: continue
        x1, y1, x2, y2 = bbox
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        
        face_w = x2 - x1
        face_h = y2 - y1
        
        pad_w = face_w * pad_w_coeff
        pad_h = face_h * pad_h_coeff
        
        mx1 = max(0, x1 - pad_w)
        my1 = max(0, y1 - pad_h)
        mx2 = min(width, x2 + pad_w)
        my2 = min(height, y2 + pad_h)
        
        draw_mask.ellipse((mx1, my1, mx2, my2), fill=mask_fill)

    # 4. Blur mask
    # Динамический радиус блюра (зависит от размера изображения)
    blur_radius = max(5, int(min(width, height) * 0.0015))
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # 5. Composite
    r_ch, g_ch, b_ch, a_ch = stripes.split()
    final_alpha = ImageChops.multiply(a_ch, mask)
    stripes.putalpha(final_alpha)
    
    return stripes
