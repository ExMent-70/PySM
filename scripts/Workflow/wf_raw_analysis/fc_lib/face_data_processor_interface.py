# fc_lib/face_data_processor_interface.py
# --- ИЗМЕНЕНИЙ НЕ ТРЕБУЕТСЯ ---

from typing import Protocol, Dict, Any, Optional
import numpy as np
from .fc_config import ConfigManager  # Используем для type hinting

# Импортируем insightface.Face как псевдоним, если он нужен для type hinting
# или используем Any
try:
    from insightface.app.common import Face as InsightFaceObject
except ImportError:
    InsightFaceObject = Any  # Fallback, если insightface не установлен


class FaceDataProcessorInterface(Protocol):
    """Интерфейс для модулей, обрабатывающих данные одного лица."""

    def __init__(self, config: ConfigManager):
        """Инициализация с доступом к общей конфигурации."""
        self.config = config
        ...

    @property
    def is_enabled(self) -> bool:
        """Свойство, указывающее, включен ли этот обработчик в конфиге."""
        # Реализация должна читать нужный флаг из self.config
        ...

    def process_face_data(
        self, face_data_bundle: Dict[str, Any], face_data_dict: Dict[str, Any]
    ) -> None:
        """
        Обрабатывает данные одного лица и ДОБАВЛЯЕТ результаты в face_data_dict.

        Args:
            face_data_bundle: Словарь с исходными данными для анализа (изображения, точки, эмбеддинг).
                              Содержит ключи типа 'full_image', 'face_crop_initial',
                              'landmarks_2d_106', 'landmarks_3d_68', 'embedding' и т.д.
            face_data_dict: Словарь с основными данными лица (из insightface, но БЕЗ эмбеддинга),
                             который нужно модифицировать, добавляя результаты анализа.
                             Ключи добавляемых полей должны быть уникальны
                             (рекомендуется использовать префиксы).
        """
        ...


# ПРИМЕРНЫЕ КЛЮЧИ в face_data_bundle (передаются из FaceDetector):
# 'full_image': np.ndarray          # Оригинальное полное изображение (BGR)
# 'face_crop_initial': np.ndarray   # Кроп лица до второго get (с паддингом, BGR)
# --- 'face_object_final': InsightFaceObject # Объект лица от второго get (InsightFace) --- # Убрали, т.к. все нужные поля передаются отдельно
# 'landmarks_2d_106': Optional[np.ndarray] # 106 точек (коорд. оригинала)
# 'landmarks_3d_68': Optional[np.ndarray]  # 68 точек (коорд. оригинала)
# 'pose': Optional[np.ndarray]        # Поза (коорд. оригинала)
# 'embedding': Optional[np.ndarray]   # Эмбеддинг лица <--- ДОБАВЛЕНО СЮДА
# 'filename': str                   # Имя исходного файла
# 'face_index': int                 # Индекс лица в файле
# 'config': ConfigManager           # Ссылка на общий конфиг

# ПРИМЕРНЫЕ КЛЮЧИ, которые УЖЕ ЕСТЬ в face_data_dict (создается в FaceDetector):
# 'bbox': Optional[List[float]]
# 'kps': Optional[List[List[float]]]
# 'det_score': float
# 'landmark_3d_68': Optional[List[List[float]]]
# 'pose': Optional[List[float]]
# 'landmark_2d_106': Optional[List[List[float]]]
# 'gender_insight': Optional[int]
# 'age_insight': Optional[int]
# --- 'embedding': Optional[List[float]] --- # УБРАНО ОТСЮДА
# 'original_bbox': Optional[List[float]]
# 'cluster_label': None # Инициализируются None
# 'child_name': None
# 'matched_portrait_cluster_label': None
# 'matched_child_name': None
# 'match_distance': None
# 'keypoint_analysis': None
# ... сюда обработчики добавляют свои поля (например, gender_faceonnx) ...
