# analize/analyze_faces/face_lib/face_analyzer.py

import logging
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
#import datetime
from insightface.app import FaceAnalysis



from .config_loader import ConfigManager
from _common.onnx_manager import ONNXModelManager, suppress_output
from .attribute_analyzer import AttributeAnalyzer
from .coordinate_transformer import CoordinateTransformer
from .face_data_processor_interface import FaceDataProcessorInterface

logger = logging.getLogger(__name__)

from _common import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
)

class FaceAnalyzer:
    """
    Главный класс-оркестратор. Инициализирует все необходимые компоненты
    и выполняет анализ лиц на предоставленных изображениях.
    """

    def __init__(self, config_manager: ConfigManager, output_dir_override: Path):
        self.config_manager = config_manager
        self.output_dir = output_dir_override
        logger.debug("Инициализация FaceAnalyzer...")

        # Извлечение параметров
        self.det_thresh = self.config_manager.get('model.det_thresh', 0.25)
        self.det_size = tuple(self.config_manager.get('model.det_size', [1280, 1280]))
        self.save_debug_kps = self.config_manager.get('task_flags.save_debug_kps', False)
        
        self.onnx_manager = ONNXModelManager(self.config_manager.get('provider', {}))
        self.attribute_analyzer = AttributeAnalyzer(self.config_manager, self.onnx_manager)
        
        # Легкая инициализация (без прогрева)
        self.analyzer: FaceAnalysis = self._initialize_insightface()
        
        if self.save_debug_kps:
            logger.info("Сохранение отладочных изображений с ключевыми точками ВКЛЮЧЕНО.")

    def _initialize_insightface(self) -> FaceAnalysis:
        """
        Базовая загрузка объекта FaceAnalysis. 
        Тяжелая компиляция перенесена в prepare_models.
        """
        provider_name = self.onnx_manager.provider_name
        provider_options = self.onnx_manager.provider_options
        model_root = self.config_manager.get('paths.model_root')
        model_name = self.config_manager.get('model.name')

        # Проверка на необходимость скачивания
        root_path = Path(model_root)
        if not (root_path / "models" / model_name).exists() and not (root_path / model_name).exists():
            logger.info(f"Модель <b>Insightface</b> не найдена локально и будет загружена с GitHub.")

        try:
            # Блокируем вывод лишнего спама от библиотеки (C++ stdout/stderr)
            with suppress_output():
                app = FaceAnalysis(
                    name=model_name,
                    root=model_root,
                    providers=[provider_name],
                    provider_options=provider_options
                )
                ctx_id = 0 if "ExecutionProvider" in provider_name and "CPU" not in provider_name else -1
                app.prepare(ctx_id=ctx_id, det_thresh=self.det_thresh)
            
            logger.info(f"Объект <b>Insightface</b> создан (провайдер: {provider_name})")
            return app
        except Exception as e:
            logger.critical(f"Ошибка инициализации Insightface: {e}", exc_info=True)
            raise


    def prepare_models(self):
            """
            Явный прогрев (Warmup) и компиляция всех моделей.
            """
            import time
            #print(f"{datetime.datetime.now()} - prepare_models: строка - Запуск компиляции и прогрева моделей AI")
            logger.info("<br><b>Подготовка подсистемы AI и загрузка моделей...</b>")

            if "Tensorrt" in self.onnx_manager.provider_name:
                status = self.onnx_manager.check_trt_cache_status()
                if "Кеш не найден" in status:
                     logger.warning(status)
            
            # 1. Прогрев Детектора (через штатный вызов get, чтобы учесть препроцессинг)
            target_h, target_w = self.det_size
            logger.info(f"<br><b><i>Загрузка детектора лиц (Input: {target_w}x{target_h})</i></b>")
           
            try:
                t0 = time.time()
                # Создаем картинку под размер детекции
                dummy_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                self.analyzer.get(dummy_img)
                #logger.info(f"{icon_ok} Детектор лиц готов ({time.time()-t0:.2f} сек).")
                logger.info(f"{icon_ok} Детектор лиц готов")
            except Exception as e:
                logger.warning(f"{icon_warning} Ошибка загрузки детектора: {e}")

            # 2. Прогрев ОСТАЛЬНЫХ моделей InsightFace (Recognition, Landmarks, etc.)
            # Перебираем все загруженные модели внутри объекта analyzer
            if hasattr(self.analyzer, 'models') and isinstance(self.analyzer.models, dict):
                logger.info(f"<br><b><i>Загрузка вспомогательных моделей InsightFace...</i></b>")
                
                for name, model in self.analyzer.models.items():
                    if name == 'detection': continue # Детектор уже прогрели выше
                    
                    try:
                        if hasattr(model, 'session'):
                            t0 = time.time()
                            session = model.session
                            inputs = session.get_inputs()
                            
                            # Формируем dummy-вход на основе метаданных модели
                            feed_dict = {}
                            for inp in inputs:
                                shape = inp.shape
                                # Обработка динамических размеров (обычно batch_size)
                                safe_shape = []
                                for dim in shape:
                                    if isinstance(dim, str) or dim is None or dim < 1:
                                        safe_shape.append(1) # Заменяем динамику на 1
                                    else:
                                        safe_shape.append(dim)
                                
                                # Для ландмарков и рекогнишна входы обычно (1, 3, H, W)
                                # Если shape не определился корректно, берем стандартный 112x112 или 192x192
                                if len(safe_shape) == 4 and safe_shape[2] == 1 and safe_shape[3] == 1:
                                    # Иногда бывает shape [1, 3, -1, -1], ставим дефолт
                                    safe_shape = [1, 3, 112, 112]
                                    if 'landmark' in name: safe_shape = [1, 3, 192, 192]
                                
                                feed_dict[inp.name] = np.zeros(safe_shape, dtype=np.float32)
                            
                            # Запуск
                            session.run(None, feed_dict)
                            #logger.info(f"{icon_ok} модель '{name}' готова ({time.time()-t0:.2f} сек).")
                            logger.info(f"{icon_ok} модель <i>{name}</i> готова")
                    except Exception as e:
                        logger.warning(f"{icon_warning}️ не удалось прогреть модель '{name}': {e}")

            # 3. Прогрев наших кастомных атрибутов
            if self.attribute_analyzer.is_enabled:
                logger.info("<br><b><i>Загрузка моделей атрибутов...</i></b>")
                self.attribute_analyzer.preload_models()
            
            logger.info("<b>Все модели AI готовы к работе</b><br>")

    def analyze_image(self, image: np.ndarray, filename: str) -> Tuple[Optional[List[Dict]], Optional[List[np.ndarray]], Optional[Tuple[int, int]]]:
        """
        Выполняет полный цикл анализа.
        Использует Padding (Letterbox) для обеспечения фиксированного размера входа детектора.
        """
        original_shape: Optional[Tuple[int, int]] = None
        try:
            if image is None: return None, None, None
            original_shape = image.shape[:2]
            
            # --- ЛОГИКА РЕСАЙЗА С ПАДДИНГОМ ---
            # 1. Вычисляем масштаб
            target_h, target_w = self.det_size
            scale = min(target_h / original_shape[0], target_w / original_shape[1])
            
            if scale < 1.0:
                new_h, new_w = int(original_shape[0] * scale), int(original_shape[1] * scale)
                img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = image
                new_h, new_w = original_shape
                scale = 1.0

            # 2. Создаем жесткий холст фиксированного размера (черный квадрат)
            # Это критически важно для TensorRT, чтобы избежать перекомпиляции под разные Aspect Ratio
            det_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # 3. Размещаем изображение в левом верхнем углу (0,0)
            # Это упрощает пересчет координат, так как оффсет равен 0
            det_img[:new_h, :new_w, :] = img_resized

            # --- АНАЛИЗ ---
            # Подаем в модель всегда 1280x1280 (или то, что в конфиге)
            #print(f"{datetime.datetime.now()} - Анализ лица - face_analyzer.py/analyze_image/initial_faces = self.analyzer.get(img_resized)")

            initial_faces = self.analyzer.get(det_img)
            
            if not initial_faces:
                return None, None, original_shape
            
            initial_faces = [f for f in initial_faces if f.det_score >= self.det_thresh]
            if not initial_faces:
                return None, None, original_shape
                
            processed_face_data_list: List[Dict] = []
            processed_face_embeddings_list: List[np.ndarray] = []

            for idx, face_initial in enumerate(initial_faces):
                # Важно: CoordinateTransformer инициализируем размерами САМОЙ КАРТИНКИ (new_h, new_w),
                # а не всего черного квадрата. Так как лицо не может быть найдено в черной зоне,
                # координаты будут корректны относительно (0,0).
                result = self._process_single_face(
                    face_initial=face_initial,
                    full_image=image,
                    resized_shape_for_det=(new_h, new_w), 
                    filename=filename,
                    face_index=idx
                )
                if result:
                    face_data, face_embedding = result
                    
                    # --- ВАЖНО: Присваиваем face_index (Immutable Index) ---
                    # Индекс присваивается последовательно только успешно обработанным лицам.
                    # Он соответствует индексу в списке processed_face_embeddings_list,
                    # и соответственно, позиции в файле векторов (.npy).
                    face_data['face_index'] = len(processed_face_data_list)
                    
                    processed_face_data_list.append(face_data)
                    processed_face_embeddings_list.append(face_embedding)
            
            if not processed_face_data_list:
                return None, None, original_shape
                
            return processed_face_data_list, processed_face_embeddings_list, original_shape

        except Exception as e:
            logger.error(f"Не удалось обработать изображение {filename}: {e}", exc_info=True)
            return None, None, original_shape



    def _process_single_face(self, face_initial: Any, full_image: np.ndarray, resized_shape_for_det: tuple, filename: str, face_index: int) -> Optional[Tuple[Dict, np.ndarray]]:
        # ... (Код метода _process_single_face БЕЗ ИЗМЕНЕНИЙ) ...
        try:
            transformer = CoordinateTransformer(full_image.shape[:2], resized_shape_for_det)
            original_bbox = transformer.recalculate_initial_bbox(face_initial.bbox)
            if original_bbox is None: return None

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
            target_size = (112, 112)
            orig_h, orig_w = original_crop_shape
            scale = min(target_size[0] / orig_h, target_size[1] / orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            resized_face = cv2.resize(cropped_face, (new_w, new_h), interpolation=cv2.INTER_AREA)
            padded_face_for_analysis = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            y_offset, x_offset = (target_size[0] - new_h) // 2, (target_size[1] - new_w) // 2
            padded_face_for_analysis[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_face
            
            transformer.store_final_analysis_params(target_size, (new_h, new_w), (crop_x1, crop_y1), original_crop_shape)

            final_faces = self.analyzer.get(padded_face_for_analysis)
            if not final_faces: return None
            final_face = final_faces[0]
            
            embedding = getattr(final_face, "embedding", None)
            if embedding is None: return None

            final_coords = transformer.recalculate_final_coords_to_original(final_face)
            face_data_dict = self._convert_face_to_dict(final_face, final_coords)
            
            # --- ИЗМЕНЕНИЕ: .astype(float) перед округлением для чистого JSON ---
            face_data_dict['original_bbox'] = np.round(original_bbox.astype(float), 4).tolist()

            if self.attribute_analyzer.is_enabled:
                face_data_bundle = {"full_image": full_image, "face_crop": cropped_face, "filename": filename, "face_index": face_index}
                self.attribute_analyzer.process_face_data(face_data_bundle, face_data_dict)

            if self.save_debug_kps:
                self._save_debug_image(full_image.copy(), face_data_dict, filename, face_index)
                
            return face_data_dict, embedding
        except Exception as e:
            logger.error(f"Ошибка при обработке лица #{face_index} на {filename}: {e}", exc_info=True)
            return None

    def _convert_face_to_dict(self, face: Any, final_coords: Dict[str, Optional[np.ndarray]]) -> Dict[str, Any]:
        # --- ИЗМЕНЕНИЕ: .astype(float) перед округлением для чистого JSON ---
        
        # Det Score
        score = float(face.det_score) if hasattr(face, "det_score") else 0.0
        score = round(score, 4)

        data = {
            "det_score": score,
            "gender_insight": int(face.gender) if hasattr(face, "gender") and face.gender is not None else None,
            "age_insight": int(face.age) if hasattr(face, "age") and face.age is not None else None,
            # Pose (Euler angles)
            "pose": np.round(face.pose.astype(float), 4).tolist() if hasattr(face, 'pose') and face.pose is not None else None,
        }
        
        # Coordinates (bbox, kps, landmarks) - rounding numpy arrays
        for key, value in final_coords.items():
            if value is not None:
                # Преобразование в float64 перед округлением
                data[key] = np.round(value.astype(float), 4).tolist()
            else:
                data[key] = None
            
        return data

    def _save_debug_image(self, image: np.ndarray, face_data: Dict, filename: str, face_index: int):
        # ... (Код БЕЗ ИЗМЕНЕНИЙ) ...
        try:
            landmarks = face_data.get("landmark_2d_106")
            if not landmarks: return
            for (x, y) in landmarks:
                if np.isfinite(x) and np.isfinite(y): cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
            debug_dir = self.output_dir / "debug_kps"
            debug_dir.mkdir(parents=True, exist_ok=True)
            output_path = debug_dir / f"{Path(filename).stem}_face{face_index}_kps.jpg"
            success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                with open(output_path, "wb") as f: f.write(buffer)
        except Exception: pass

    def shutdown(self):
        logger.info("<b>Освобождение ресурсов...</b>")
        if self.onnx_manager: self.onnx_manager.shutdown()
        if hasattr(self, 'analyzer'): del self.analyzer
        gc.collect()
        logger.info(" - ресурсы Insightface.FaceAnalysis освобождены.")