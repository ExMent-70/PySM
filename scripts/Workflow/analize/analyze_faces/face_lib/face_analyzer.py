# analize/analyze_faces/face_lib/face_analyzer.py
"""
Основной модуль, содержащий класс FaceAnalyzer, который оркестрирует
весь процесс детекции и анализа лиц на изображениях.
"""

import logging
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from .config_loader import ConfigManager
from .onnx_manager import ONNXModelManager
from .attribute_analyzer import AttributeAnalyzer
from .coordinate_transformer import CoordinateTransformer
from .face_data_processor_interface import FaceDataProcessorInterface

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """
    Главный класс-оркестратор. Инициализирует все необходимые компоненты
    и выполняет анализ лиц на предоставленных изображениях.
    """

    def __init__(self, config_manager: ConfigManager, output_dir_override: Path):
        """
        Инициализирует анализатор, создавая все необходимые зависимости.
        """
        self.config_manager = config_manager
        # Явно сохраняем путь для вывода, он понадобится для отладки
        self.output_dir = output_dir_override
        logger.debug("Инициализация FaceAnalyzer...")

        # Извлечение параметров из конфига
        self.det_thresh = self.config_manager.get('model.det_thresh', 0.25)
        self.det_size = tuple(self.config_manager.get('model.det_size', [1280, 1280]))
        self.save_debug_kps = self.config_manager.get('task_flags.save_debug_kps', False)
        
        self.onnx_manager = ONNXModelManager(self.config_manager.get('provider', {}))
        self.attribute_analyzer = AttributeAnalyzer(self.config_manager, self.onnx_manager)
        self.analyzer: FaceAnalysis = self._initialize_insightface()
        
        if self.save_debug_kps:
            logger.info("Сохранение отладочных изображений с ключевыми точками ВКЛЮЧЕНО.")

    def _initialize_insightface(self) -> FaceAnalysis:
        """Изолирует инициализацию insightface.app.FaceAnalysis."""
        logger.info("Инициализация модели Insightface.FaceAnalysis. Консоль будет временно заблокирована...")

        provider_name = self.onnx_manager.provider_name
        provider_options = self.onnx_manager.provider_options
        model_root = self.config_manager.get('paths.model_root')
        model_name = self.config_manager.get('model.name')

        try:
            print("PYSM_CONSOLE_BLOCK_START")
            app = FaceAnalysis(
                name=model_name,
                root=model_root,
                providers=[provider_name],
                provider_options=provider_options
            )

            ctx_id = 0 if "ExecutionProvider" in provider_name and "CPU" not in provider_name else -1
            app.prepare(ctx_id=ctx_id, det_thresh=self.det_thresh)
            print("PYSM_CONSOLE_BLOCK_END")
            logger.info(f"Модель <b>Insightface.FaceAnalysis</b> успешно инициализирован с провайдером <b>{provider_name}</b><br>")
            return app
        except Exception as e:
            logger.critical(f"Критическая ошибка при инициализации insightface: {e}", exc_info=True)
            raise
        finally:
            print("PYSM_CONSOLE_BLOCK_END")


    def analyze_image(self, image_path: Path) -> Tuple[Optional[List[Dict]], Optional[List[np.ndarray]], Optional[Tuple[int, int]]]:
        """
        Выполняет полный цикл анализа для одного изображения.

        Returns:
            Кортеж (список метаданных, список эмбеддингов, форма_изображения) или (None, None, None).
        """
        original_shape: Optional[Tuple[int, int]] = None
        try:
            # Блок 1: Надежная загрузка изображения с Unicode-путями
            try:
                with open(image_path, "rb") as f:
                    img_buffer = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            except Exception as read_err:
                logger.error(f"Ошибка чтения или декодирования файла {image_path.name}: {read_err}")
                return None, None, None

            if img is None:
                logger.error(f"Не удалось загрузить изображение (imdecode вернул None): {image_path.name}")
                return None, None, None

            original_shape = img.shape[:2]
            
            # Масштабирование для первичной детекции
            scale = min(self.det_size[1] / original_shape[0], self.det_size[0] / original_shape[1])
            if scale < 1.0:
                new_h, new_w = int(original_shape[0] * scale), int(original_shape[1] * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img
                new_h, new_w = original_shape

            # Первичная детекция
            initial_faces = self.analyzer.get(img_resized)
            if not initial_faces:
                logger.info(f"Лица не найдены на изображении: {image_path.name}")
                return None, None, original_shape
            
            # Фильтрация по порогу уверенности
            initial_faces = [f for f in initial_faces if f.det_score >= self.det_thresh]
            if not initial_faces:
                logger.info(f"Лица с недостаточной уверенностью отброшены на: {image_path.name}")
                return None, None, original_shape
                
            processed_face_data_list: List[Dict] = []
            processed_face_embeddings_list: List[np.ndarray] = []

            for idx, face_initial in enumerate(initial_faces):
                result = self._process_single_face(
                    face_initial=face_initial,
                    full_image=img,
                    resized_shape_for_det=(new_h, new_w),
                    filename=image_path.name,
                    face_index=idx
                )
                if result:
                    face_data, face_embedding = result
                    processed_face_data_list.append(face_data)
                    processed_face_embeddings_list.append(face_embedding)
            
            if not processed_face_data_list:
                return None, None, original_shape
                
            return processed_face_data_list, processed_face_embeddings_list, original_shape

        except Exception as e:
            logger.error(f"Не удалось обработать изображение {image_path.name}: {e}", exc_info=True)
            # Возвращаем shape даже при ошибке, если он был определен
            return None, None, original_shape
            

    def _process_single_face(
        self, face_initial: Any, full_image: np.ndarray, resized_shape_for_det: tuple, filename: str, face_index: int
    ) -> Optional[Tuple[Dict, np.ndarray]]:
        """Обрабатывает одно найденное на изображении лицо."""
        try:
            # 1. Трансформация координат
            transformer = CoordinateTransformer(full_image.shape[:2], resized_shape_for_det)
            original_bbox = transformer.recalculate_initial_bbox(face_initial.bbox)
            if original_bbox is None:
                logger.warning(f"Не удалось пересчитать bbox для лица #{face_index} на {filename}")
                return None

            # 2. Подготовка кропа для вторичного анализа
            x1, y1, x2, y2 = map(int, original_bbox)
            bbox_w, bbox_h = x2 - x1, y2 - y1
            padding_x, padding_y = int(bbox_w * 0.45), int(bbox_h * 0.45)
            
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(full_image.shape[1], x2 + padding_x)
            crop_y2 = min(full_image.shape[0], y2 + padding_y)

            cropped_face = full_image[crop_y1:crop_y2, crop_x1:crop_x2]
            if cropped_face.size == 0: return None
            
            original_crop_shape = cropped_face.shape[:2]
            
            # Масштабирование и вписывание в квадратный холст 112x112
            target_size = (112, 112)
            orig_h, orig_w = original_crop_shape
            scale = min(target_size[0] / orig_h, target_size[1] / orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            resized_face = cv2.resize(cropped_face, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            padded_face_for_analysis = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            y_offset = (target_size[0] - new_h) // 2
            x_offset = (target_size[1] - new_w) // 2
            padded_face_for_analysis[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_face
            
            # 3. Сохранение параметров трансформации
            transformer.store_final_analysis_params(
                padded_face_shape=target_size,
                new_cropped_shape=(new_h, new_w),
                crop_origin_coords=(crop_x1, crop_y1),
                original_crop_shape=original_crop_shape
            )

            # 4. Вторичный анализ
            final_faces = self.analyzer.get(padded_face_for_analysis)
            if not final_faces: return None
            final_face = final_faces[0]
            
            embedding = getattr(final_face, "embedding", None)
            if embedding is None: return None

            # 5. Финальный пересчет координат
            final_coords = transformer.recalculate_final_coords_to_original(final_face)

            # 6. Сборка словаря метаданных
            face_data_dict = self._convert_face_to_dict(final_face, final_coords)
            face_data_dict['original_bbox'] = original_bbox.tolist()

            # 7. Анализ атрибутов
            if self.attribute_analyzer.is_enabled:
                face_data_bundle = {
                    "full_image": full_image,
                    "face_crop": cropped_face,
                    "filename": filename,
                    "face_index": face_index
                }
                self.attribute_analyzer.process_face_data(face_data_bundle, face_data_dict)

            # 8. Отладочная визуализация
            if self.save_debug_kps:
                self._save_debug_image(full_image.copy(), face_data_dict, filename, face_index)
                
            return face_data_dict, embedding

        except Exception as e:
            logger.error(f"Ошибка при обработке лица #{face_index} на {filename}: {e}", exc_info=True)
            return None

    def _convert_face_to_dict(self, face: Any, final_coords: Dict[str, Optional[np.ndarray]]) -> Dict[str, Any]:
        """Преобразует объект insightface.Face и финальные координаты в словарь."""
        data = {
            "det_score": float(face.det_score) if hasattr(face, "det_score") else 0.0,
            "gender_insight": int(face.gender) if hasattr(face, "gender") and face.gender is not None else None,
            "age_insight": int(face.age) if hasattr(face, "age") and face.age is not None else None,
            "pose": face.pose.tolist() if hasattr(face, 'pose') and face.pose is not None else None,
        }
        for key, value in final_coords.items():
            data[key] = value.tolist() if value is not None else None
        
        return data

# [...] Код класса FaceAnalyzer без изменений до этого метода

    def _save_debug_image(self, image: np.ndarray, face_data: Dict, filename: str, face_index: int):
        """Сохраняет отладочное изображение с нанесенными ключевыми точками."""
        try:
            landmarks = face_data.get("landmark_2d_106")
            if not landmarks:
                logger.debug(f"Нет ландмарков для сохранения отладочного изображения {filename}.")
                return

            for (x, y) in landmarks:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)

            debug_dir = self.output_dir / "debug_kps"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            filename_stem = Path(filename).stem
            output_path = debug_dir / f"{filename_stem}_face{face_index}_kps.jpg"
            
            # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            # Блок 1: Надежное сохранение файла с Unicode-путем
            # Сначала кодируем изображение в JPEG формат в памяти
            success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                # Затем записываем байтовый буфер на диск стандартными средствами Python
                with open(output_path, "wb") as f:
                    f.write(buffer)
            else:
                logger.warning(f"cv2.imencode не смог закодировать отладочное изображение: {output_path.name}")
            # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

        except Exception as e:
            logger.warning(f"Не удалось сохранить отладочное изображение для {filename}: {e}", exc_info=True)

# [...] Остальной код класса без изменений

    def shutdown(self):
        """Освобождает ресурсы."""
        logger.info("Освобождение ресурсов...")
        if self.onnx_manager:
            self.onnx_manager.shutdown()
        if hasattr(self, 'analyzer'):
            del self.analyzer
        gc.collect()
        logger.info(" - ресурсы Insightface.FaceAnalysis освобождены.")