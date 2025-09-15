# 1. БЛОК: Файл _lib/data_models.py (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ==============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_models.py
==============
Модуль, определяющий типизированные структуры данных (dataclasses) для проекта.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class Face:
    """
    Представляет только необходимые для редактора данные одного лица.
    Все остальные поля из JSON хранятся в `extra_data`.
    """
    bbox: List[float]
    
    # Поля для кластеризации по лицам
    cluster_label: Optional[int] = None
    child_name: Optional[str] = None
    
    # Словарь для хранения всех остальных полей из JSON
    extra_data: Dict[str, Any] = field(default_factory=dict)

    # Поля, добавляемые во время выполнения для удобства UI
    filename: str = ""
    effective_name: str = ""


@dataclass
class ImageRecord:
    """Представляет полную запись для одного изображения."""
    filename: str
    image_type: str  # 'portrait' или 'group'
    faces: List[Face]
    original_shape: Tuple[int, int]
    
    # Поля для кластеризации по локациям
    location_cluster: Optional[int] = None
    location_name: Optional[str] = None
    
    # Поле для отслеживания изменений
    is_changed: bool = False