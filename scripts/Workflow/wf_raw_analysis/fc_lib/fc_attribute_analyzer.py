# fc_lib/fc_attribute_analyzer.py

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable
import time

import cv2
import numpy as np
import onnxruntime as ort

from .fc_config import ConfigManager
from .fc_onnx_manager import ONNXModelManager
from .fc_messages import get_message
from .face_data_processor_interface import FaceDataProcessorInterface

logger = logging.getLogger(__name__)

# --- Константы ---
LEFT_EYE_INDICES = [35, 36, 37, 38, 39, 40, 41, 42]
RIGHT_EYE_INDICES = [89, 90, 91, 92, 93, 94, 95, 96]
IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BEAUTY_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32).reshape(3, 1, 1)


# --- Вспомогательные функции ---
def softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0 or np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return np.full_like(x, 1.0 / x.size if x.size > 0 else 0, dtype=float)
    try:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        result = e_x / np.sum(e_x, axis=-1, keepdims=True)
        if np.any(np.isnan(result)):
            return np.full_like(x, 1.0 / x.size if x.size > 0 else 0, dtype=float)
        return result
    except Exception as e:
        logger.error(f"Ошибка softmax: {e}", exc_info=True)
        return np.full_like(x, 1.0 / x.size if x.size > 0 else 0, dtype=float)


def get_eye_bbox_from_landmarks(
    landmarks: Optional[List[List[float]]],
    eye_indices: List[int],
    padding_ratio: float = 0.15,
) -> Optional[Tuple[int, int, int, int]]:
    if not landmarks:
        return None
    max_req_index = max(eye_indices)
    if len(landmarks) <= max_req_index:
        return None
    try:
        eye_points_list = []
        for i in eye_indices:
            if 0 <= i < len(landmarks):
                point = landmarks[i]
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                eye_points_list.append(point[:2])
        if len(eye_points_list) < 3:
            return None
        eye_points = np.array(eye_points_list, dtype=np.float32)
        min_x, max_x = np.min(eye_points[:, 0]), np.max(eye_points[:, 0])
        min_y, max_y = np.min(eye_points[:, 1]), np.max(eye_points[:, 1])
        width, height = max_x - min_x, max_y - min_y
        if width <= 0 or height <= 0:
            return None
        pad_w, pad_h = int(width * padding_ratio), int(height * padding_ratio)
        x1, y1 = int(min_x - pad_w), int(min_y - pad_h)
        x2, y2 = int(max_x + pad_w), int(max_y + pad_h)
        if x1 >= x2:
            x2 = x1 + 1
        if y1 >= y2:
            y2 = y1 + 1
        return x1, y1, x2, y2
    except Exception as e:
        logger.error(f"Ошибка вычисления BBox глаза: {e}", exc_info=True)
        return None


class AttributeAnalyzer(FaceDataProcessorInterface):
    """
    Класс для анализа атрибутов лица. Не создает свои сессии, а использует
    переданный ONNXModelManager для их получения.
    """

    def __init__(self, config: ConfigManager):
        """
        Инициализация. Только собирает информацию о моделях, не загружая их.
        """
        self.config = config
        self._enabled = False
        self.enabled_tasks = {
            "gender": self.config.get("task", "analyze_gender", False),
            "emotion": self.config.get("task", "analyze_emotion", False),
            "age": self.config.get("task", "analyze_age", False),
            "beauty": self.config.get("task", "analyze_beauty", False),
            "eyeblink": self.config.get("task", "analyze_eyeblink", False),
        }
        self._enabled = any(self.enabled_tasks.values())

        self.models_info: Dict[str, Dict[str, Any]] = {}
        self.model_root = Path(self.config.get("paths", "model_root", "../BIN"))
        if self.is_enabled:
            logger.debug("[AttributeAnalyzer] Инициализация (сбор информации о моделях)...")
            model_config = self.config.get("model", {})
            model_filenames = {
                "gender": model_config.get("gender_model_filename"),
                "emotion": model_config.get("emotion_model_filename"),
                "age": model_config.get("age_model_filename"),
                "beauty": model_config.get("beauty_model_filename"),
                "eyeblink": model_config.get("eyeblink_model_filename"),
            }

            task_list = ""
            for task_name, enabled in self.enabled_tasks.items():
                if not enabled:
                    continue
                filename = model_filenames.get(task_name)
                if not filename:
                    logger.warning(f"[AttributeAnalyzer] Имя файла модели для '{task_name}' не указано.")
                    continue

                model_path = (self.model_root / filename).resolve()
                if not model_path.is_file():
                    logger.error(f"[AttributeAnalyzer] Файл модели для '{task_name}' не найден: {model_path}")
                    continue
                
                input_shape, is_nchw, input_name, out_names = (None, None, None, None)
                if task_name == 'gender':
                    input_shape, is_nchw, input_name, out_names = (3, 224, 224), True, 'input', ['output']
                elif task_name == 'emotion':
                    input_shape, is_nchw, input_name, out_names = (1, 48, 48), True, 'input.1', ['97']
                elif task_name == 'age':
                    input_shape, is_nchw, input_name, out_names = (3, 224, 224), True, 'input', ['output']
                elif task_name == 'beauty':
                    input_shape, is_nchw, input_name, out_names = (3, 224, 224), True, 'input', ['output']
                elif task_name == 'eyeblink':
                    input_shape, is_nchw, input_name, out_names = (1, 26, 34), False, 'input_3', ['activation_5']

                task_list += str(f"'{task_name}', ")

                if input_shape:
                    self.models_info[task_name] = {
                        "path": model_path,
                        "input_name": input_name,
                        "output_names": out_names,
                        "input_shape_tuple": input_shape,
                        "is_nchw": is_nchw,
                    }
                    logger.debug(f"[AttributeAnalyzer] Собрана информация для модели '{task_name}'.")

            logger.info(f"[AttributeAnalyzer] Загружены модели: {task_list}")
        else:
            logger.info("[AttributeAnalyzer] Анализатор атрибутов отключен.")


    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _preprocess_gender(
        self, image: np.ndarray, target_shape: tuple, is_nchw: bool
    ) -> Optional[np.ndarray]:
        _c, target_h, target_w = target_shape
        if not is_nchw:
            logger.error("_preprocess_gender ожидает is_nchw=True")
            return None
        try:
            img_resized = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_normalized -= IMAGENET_MEAN_RGB
            img_normalized /= IMAGENET_STD_RGB
            img_blob = np.expand_dims(img_normalized.transpose(2, 0, 1), axis=0)
            return img_blob.astype(np.float32)
        except Exception as e:
            logger.error(f"Ошибка _preprocess_gender: {e}")
            return None

    def _preprocess_emotion(
        self, image: np.ndarray, target_shape: tuple, is_nchw: bool
    ) -> Optional[np.ndarray]:
        _c, target_h, target_w = target_shape
        if not is_nchw:
            logger.error("_preprocess_emotion ожидает is_nchw=True")
            return None
        try:
            img_resized = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            img_gray = (
                cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                if len(img_resized.shape) == 3
                else img_resized
            )
            img_normalized = img_gray.astype(np.float32) / 256.0
            img_blob = img_normalized.reshape(1, 1, target_h, target_w)
            return img_blob.astype(np.float32)
        except Exception as e:
            logger.error(f"Ошибка _preprocess_emotion: {e}")
            return None

    def _preprocess_age(
        self, image: np.ndarray, target_shape: tuple, is_nchw: bool
    ) -> Optional[np.ndarray]:
        return self._preprocess_gender(image, target_shape, is_nchw)

    def _preprocess_beauty(
        self, image: np.ndarray, target_shape: tuple, is_nchw: bool
    ) -> Optional[np.ndarray]:
        _c, target_h, target_w = target_shape
        if not is_nchw:
            logger.error("_preprocess_beauty ожидает is_nchw=True")
            return None
        try:
            img_resized = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            processed_image = img_resized.astype(np.float32).transpose(2, 0, 1)
            processed_image -= BEAUTY_MEAN_BGR
            processed_image /= 255.0
            img_blob = np.expand_dims(processed_image, axis=0)
            return img_blob.astype(np.float32)
        except Exception as e:
            logger.error(f"Ошибка _preprocess_beauty: {e}")
            return None

    def _preprocess_eyeblink(
        self, image: np.ndarray, target_shape: tuple, is_nchw: bool
    ) -> Optional[np.ndarray]:
        _c, target_h, target_w = target_shape
        if is_nchw:
            logger.error("_preprocess_eyeblink ожидает is_nchw=False")
            return None
        try:
            img_resized = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            img_gray = (
                cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                if len(img_resized.shape) == 3
                else img_resized
            )
            img_normalized = img_gray.astype(np.float32) / 255.0
            img_blob = img_normalized.reshape(1, target_h, target_w, 1)
            return img_blob.astype(np.float32)
        except Exception as e:
            logger.error(f"Ошибка _preprocess_eyeblink: {e}")
            return None

    def _get_preprocess_func(self, task_name: str) -> Optional[Callable]:
        mapping = {
            "gender": self._preprocess_gender,
            "emotion": self._preprocess_emotion,
            "age": self._preprocess_age,
            "beauty": self._preprocess_beauty,
            "eyeblink": self._preprocess_eyeblink,
        }
        return mapping.get(task_name)

    def _postprocess_gender(self, outputs: List[np.ndarray]) -> Optional[str]:
        try:
            scores = outputs[0].flatten()
            prediction_index = np.argmax(scores)
            return "Male" if prediction_index == 0 else "Female"
        except Exception as e:
            logger.error(f"Ошибка _postprocess_gender: {e}")
            return None

    def _postprocess_emotion(self, outputs: List[np.ndarray]) -> Optional[str]:
        try:
            scores = outputs[0].flatten()
            labels = self.config.get("model", "emotion_labels")
            if labels and scores.shape == (len(labels),):
                probabilities = softmax(scores)
                prediction_index = np.argmax(probabilities)
                return labels[prediction_index]
            else:
                logger.warning(
                    f"Форма выхода Emotion {scores.shape} или метки {labels} некорректны."
                )
                return None
        except Exception as e:
            logger.error(f"Ошибка _postprocess_emotion: {e}")
            return None

    def _postprocess_age(self, outputs: List[np.ndarray]) -> Optional[int]:
        try:
            age_value = outputs[0].flatten()[0]
            estimated_age = int(round(age_value))
            return estimated_age if 0 <= estimated_age <= 120 else None
        except Exception as e:
            logger.error(f"Ошибка _postprocess_age: {e}")
            return None

    def _postprocess_beauty(self, outputs: List[np.ndarray]) -> Optional[float]:
        try:
            beauty_score = outputs[0].flatten()[0]
            if np.isnan(beauty_score) or np.isinf(beauty_score):
                logger.warning(f"Получено некорректное значение beauty_score: {beauty_score}")
                return None
            return float(beauty_score)
        except Exception as e:
            logger.error(f"Ошибка _postprocess_beauty: {e}")
            return None

    def _postprocess_eyeblink(
        self, outputs: List[np.ndarray]
    ) -> Optional[Tuple[str, float]]:
        try:
            score = outputs[0].flatten()[0]
            if np.isnan(score) or np.isinf(score):
                logger.warning(f"Получено некорректное значение eyeblink_score: {score}")
                return None
            threshold = self.config.get("model", "eyeblink_threshold", 0.5)
            labels = self.config.get("model", "eyeblink_labels", ["Closed", "Open"])
            if len(labels) != 2:
                logger.error("eyeblink_labels в конфиге должен содержать 2 элемента. Используются ['Closed', 'Open'].")
                labels = ["Closed", "Open"]
            state = labels[1] if score > threshold else labels[0]
            return state, float(score)
        except Exception as e:
            logger.error(f"Ошибка _postprocess_eyeblink: {e}")
            return None

    def _get_postprocess_func(self, task_name: str) -> Optional[Callable]:
        mapping = {
            "gender": self._postprocess_gender,
            "emotion": self._postprocess_emotion,
            "age": self._postprocess_age,
            "beauty": self._postprocess_beauty,
            "eyeblink": self._postprocess_eyeblink,
        }
        return mapping.get(task_name)

    def _run_single_analysis(
        self,
        face_or_eye_image: np.ndarray,
        task_name: str,
        session: ort.InferenceSession,
        model_meta: Dict[str, Any],
    ) -> Any:
        if face_or_eye_image is None or face_or_eye_image.size == 0:
            return None
            
        start_time = time.time()
        preprocess_func = self._get_preprocess_func(task_name)
        postprocess_func = self._get_postprocess_func(task_name)

        if not preprocess_func or not postprocess_func:
            logger.error(f"Не найдены функции pre/post-processing для задачи '{task_name}'.")
            return None

        input_blob = preprocess_func(
            face_or_eye_image.copy(), model_meta["input_shape_tuple"], model_meta["is_nchw"]
        )
        if input_blob is None:
            logger.error(f"Ошибка предобработки для задачи '{task_name}'.")
            return None

        try:
            outputs = session.run(
                model_meta["output_names"], {model_meta["input_name"]: input_blob}
            )
        except Exception as e:
            logger.error(f"Ошибка выполнения модели '{task_name}': {e}", exc_info=True)
            logger.error(f"Форма входа ({model_meta['input_name']}): {input_blob.shape}")
            return None
            
        result = postprocess_func(outputs)
        duration = time.time() - start_time
        logger.debug(
            f"Анализ '{task_name}' завершен за {duration:.4f} сек. Результат: {result}"
        )
        return result

    def process_face_data(
        self, face_data_bundle: Dict[str, Any], face_data_dict: Dict[str, Any]
    ) -> None:
        """
        Основной метод. Получает ONNX-менеджер из бандла и запрашивает у него сессии.
        """
        face_data_dict["gender_faceonnx"] = None
        face_data_dict["emotion_faceonnx"] = None
        face_data_dict["age_faceonnx"] = None
        face_data_dict["beauty_faceonnx"] = None
        face_data_dict["left_eye_state"] = None
        face_data_dict["left_eye_score"] = None
        face_data_dict["right_eye_state"] = None
        face_data_dict["right_eye_score"] = None

        onnx_manager: Optional[ONNXModelManager] = face_data_bundle.get("onnx_manager")

        if not self.is_enabled or onnx_manager is None:
            logger.debug("AttributeAnalyzer отключен или не получил ONNX-менеджер.")
            return

        face_crop_initial = face_data_bundle.get("face_crop_initial")
        full_original_image = face_data_bundle.get("full_image")
        landmarks_final_orig_coords_list = face_data_dict.get("landmark_2d_106")

        if face_crop_initial is None:
            logger.warning(
                "[AttributeAnalyzer] Отсутствует 'face_crop_initial'. Анализ атрибутов лица пропущен."
            )
            return

        img_h, img_w = 0, 0
        if full_original_image is not None:
            img_h, img_w = full_original_image.shape[:2]
        else:
            logger.warning(
                "[AttributeAnalyzer] Отсутствует 'full_original_image'. Анализ моргания будет пропущен."
            )

        try:
            face_tasks = ["gender", "emotion", "age", "beauty"]
            
            for task_name in face_tasks:
                model_info = self.models_info.get(task_name)
                if not model_info:
                    continue
                
                session = onnx_manager.get_session(model_info["path"])
                if not session:
                    continue

                task_result = self._run_single_analysis(
                    face_crop_initial, task_name, session, model_info
                )
                if task_result is not None:
                    face_data_dict[f"{task_name}_faceonnx"] = task_result

            model_info_eyeblink = self.models_info.get("eyeblink")
            if model_info_eyeblink:
                session_eyeblink = onnx_manager.get_session(model_info_eyeblink["path"])
                if session_eyeblink:
                    left_eye_img, right_eye_img = None, None
                    if full_original_image is not None and landmarks_final_orig_coords_list:
                        left_bbox = get_eye_bbox_from_landmarks(landmarks_final_orig_coords_list, LEFT_EYE_INDICES)
                        if left_bbox:
                            lx1, ly1, lx2, ly2 = left_bbox
                            lx1, ly1 = max(0, lx1), max(0, ly1)
                            lx2, ly2 = min(img_w, lx2), min(img_h, ly2)
                            if lx1 < lx2 and ly1 < ly2:
                                left_eye_img = full_original_image[ly1:ly2, lx1:lx2]
                        
                        right_bbox = get_eye_bbox_from_landmarks(landmarks_final_orig_coords_list, RIGHT_EYE_INDICES)
                        if right_bbox:
                            rx1, ry1, rx2, ry2 = right_bbox
                            rx1, ry1 = max(0, rx1), max(0, ry1)
                            rx2, ry2 = min(img_w, rx2), min(img_h, ry2)
                            if rx1 < rx2 and ry1 < ry2:
                                right_eye_img = full_original_image[ry1:ry2, rx1:rx2]

                    if left_eye_img is not None and left_eye_img.size > 0:
                        left_eye_res = self._run_single_analysis(
                            left_eye_img, "eyeblink", session_eyeblink, model_info_eyeblink
                        )
                        if left_eye_res:
                            face_data_dict["left_eye_state"] = left_eye_res[0]
                            face_data_dict["left_eye_score"] = float(left_eye_res[1])

                    if right_eye_img is not None and right_eye_img.size > 0:
                        right_eye_res = self._run_single_analysis(
                           right_eye_img, "eyeblink", session_eyeblink, model_info_eyeblink
                        )
                        if right_eye_res:
                            face_data_dict["right_eye_state"] = right_eye_res[0]
                            face_data_dict["right_eye_score"] = float(right_eye_res[1])

        except Exception as e:
            filename = face_data_bundle.get("filename", "unknown_file")
            face_index = face_data_bundle.get("face_index", "N/A")
            logger.error(f"Критическая ошибка в AttributeAnalyzer для {filename}[{face_index}]: {e}", exc_info=True)
            face_data_dict["attribute_analysis_error"] = str(e)

        logger.debug(f"AttributeAnalyzer обновил поля в face_data_dict.")