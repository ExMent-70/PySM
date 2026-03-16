# analize/analyze_faces/face_lib/coordinate_transformer.py
"""
Инкапсулирует сложную, многоступенчатую логику преобразования
координат лиц из одной системы в другую.
Этот модуль критически важен для точности данных.
"""

import logging
from typing import Dict, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """
    Класс для управления всеми этапами преобразования координат.
    Он хранит состояние всех промежуточных преобразований для одного лица.
    """

    def __init__(self, original_shape: tuple, resized_shape_for_det: tuple):
        """
        Инициализирует трансформер с параметрами первичного масштабирования.

        Args:
            original_shape: Форма (H, W) исходного изображения.
            resized_shape_for_det: Форма (H, W) изображения после масштабирования для детектора.
        """
        self.original_h, self.original_w = original_shape
        self.resized_h, self.resized_w = resized_shape_for_det

        # Коэффициенты для пересчета из resized_shape_for_det в original_shape
        self.scale_x_det = self.original_w / self.resized_w if self.resized_w > 0 else 1.0
        self.scale_y_det = self.original_h / self.resized_h if self.resized_h > 0 else 1.0

        # Параметры для обратного преобразования из финального кропа
        self.crop_params: Dict[str, Any] = {}
        logger.debug(
            f"CoordinateTransformer создан. Original: {original_shape}, Resized: {resized_shape_for_det}"
        )

    def recalculate_initial_bbox(self, bbox_from_resized: np.ndarray) -> Optional[np.ndarray]:
        """
        Пересчитывает bbox из системы координат масштабированного изображения
        в систему координат исходного изображения.
        """
        # --- ИЗМЕНЕНИЕ: Добавлена обработка ошибок на случай некорректных данных ---
        try:
            return (
                bbox_from_resized * np.array([self.scale_x_det, self.scale_y_det, self.scale_x_det, self.scale_y_det])
            ).astype(np.float32)
        except (TypeError, ValueError) as e:
            logger.warning(f"Ошибка пересчета initial_bbox: {e}")
            return None
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    def store_final_analysis_params(
        self,
        padded_face_shape: Tuple[int, int],
        new_cropped_shape: Tuple[int, int],
        crop_origin_coords: Tuple[int, int],
        # --- ИЗМЕНЕНИЕ: Добавлен недостающий параметр для корректного расчета масштаба ---
        original_crop_shape: Tuple[int, int]
    ) -> None:
        """
        Сохраняет все необходимые параметры для финального обратного преобразования.
        Эта информация собирается на этапе подготовки кропа для вторичного анализа.

        Args:
            padded_face_shape: Форма (H, W) квадратного холста, на который помещен кроп.
            new_cropped_shape: Форма (H, W) кропа после его масштабирования.
            crop_origin_coords: Координаты (x1, y1) левого верхнего угла кропа в исходном изображении.
            original_crop_shape: Форма (H, W) кропа до его масштабирования.
        """
        # --- ИЗМЕНЕНИЕ: Весь блок обернут в try...except для надежности ---
        try:
            target_h, target_w = padded_face_shape
            new_crop_h, new_crop_w = new_cropped_shape
            x1_padded, y1_padded = crop_origin_coords
            orig_crop_h, orig_crop_w = original_crop_shape

            self.crop_params['x1_padded'] = x1_padded
            self.crop_params['y1_padded'] = y1_padded
            self.crop_params['x_offset'] = (target_w - new_crop_w) // 2
            self.crop_params['y_offset'] = (target_h - new_crop_h) // 2
            
            # Вычисляем масштаб, с которым оригинальный кроп был вписан в квадратный холст
            self.crop_params['scale_crop_x'] = orig_crop_w / new_crop_w if new_crop_w > 0 else 1.0
            self.crop_params['scale_crop_y'] = orig_crop_h / new_crop_h if new_crop_h > 0 else 1.0
            
            logger.debug(f"Параметры для финальной трансформации сохранены: {self.crop_params}")
        except Exception as e:
            logger.error(f"Критическая ошибка при сохранении параметров трансформации: {e}", exc_info=True)
            # Сбрасываем параметры в случае ошибки, чтобы избежать неверных вычислений в дальнейшем
            self.crop_params = {}
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---


    def recalculate_final_coords_to_original(self, face_from_crop: Any) -> Dict[str, Optional[np.ndarray]]:
        """
        Выполняет финальное преобразование координат из проанализированного
        кропа (padded_face) в систему координат исходного изображения.
        Эта логика в точности повторяет оригинальную.
        """
        p = self.crop_params
        if not all(k in p for k in ['x1_padded', 'y1_padded', 'x_offset', 'y_offset', 'scale_crop_x', 'scale_crop_y']):
            logger.error("Параметры трансформации не были полностью подготовлены. Преобразование невозможно.")
            return {
                "bbox": None, "kps": None,
                "landmark_2d_106": None, "landmark_3d_68": None
            }

        recalculated: Dict[str, Optional[np.ndarray]] = {}
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        # Возвращаем декларативный словарь для описания атрибутов
        coord_attrs = {
            "bbox": {"dims": 1, "shape": (4,)}, 
            "kps": {"dims": 2, "shape_part": (2,)}, 
            "landmark_2d_106": {"dims": 2, "shape_part": (2,)}, 
            "landmark_3d_68": {"dims": 2, "shape_part": (3,)}
        }

        for attr, props in coord_attrs.items():
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            relative_coords = getattr(face_from_crop, attr, None)
            if relative_coords is None:
                recalculated[attr] = None
                continue

            try:
                relative_coords_np = np.array(relative_coords, dtype=np.float32)
                final_coords: Optional[np.ndarray] = None

                if attr == "bbox":
                    if relative_coords_np.shape != props["shape"]:
                        raise ValueError(f"Неверная форма bbox: {relative_coords_np.shape}, ожидалась {props['shape']}")
                    
                    final_coords = np.array([
                        p['x1_padded'] + (relative_coords_np[0] - p['x_offset']) * p['scale_crop_x'],
                        p['y1_padded'] + (relative_coords_np[1] - p['y_offset']) * p['scale_crop_y'],
                        p['x1_padded'] + (relative_coords_np[2] - p['x_offset']) * p['scale_crop_x'],
                        p['y1_padded'] + (relative_coords_np[3] - p['y_offset']) * p['scale_crop_y'],
                    ], dtype=np.float32)
                else:
                    if relative_coords_np.ndim != props["dims"] or relative_coords_np.shape[1] < props["shape_part"][0]:
                        raise ValueError(f"Неверная форма для {attr}: {relative_coords_np.shape}, ожидалась (N, >={props['shape_part'][0]})")

                    final_coords = relative_coords_np.copy()
                    final_coords[:, 0] = p['x1_padded'] + (relative_coords_np[:, 0] - p['x_offset']) * p['scale_crop_x']
                    final_coords[:, 1] = p['y1_padded'] + (relative_coords_np[:, 1] - p['y_offset']) * p['scale_crop_y']
                
                recalculated[attr] = final_coords

            except (TypeError, IndexError, ValueError) as e:
                logger.warning(f"Ошибка при пересчете атрибута '{attr}': {e}. Значение не будет установлено.")
                recalculated[attr] = None
        
        return recalculated