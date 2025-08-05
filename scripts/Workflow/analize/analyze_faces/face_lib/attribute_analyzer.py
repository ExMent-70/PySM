# analize/analyze_faces/face_lib/attribute_analyzer.py
"""
Модуль для анализа вторичных атрибутов лица (пол, возраст, эмоции, глаза)
с использованием ONNX моделей.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable

import cv2
import numpy as np

from .config_loader import ConfigManager
from .face_data_processor_interface import FaceDataProcessorInterface
from .onnx_manager import ONNXModelManager

logger = logging.getLogger(__name__)


# --- Блок 1: Вспомогательные функции и константы ---
# ==============================================================================
LEFT_EYE_INDICES = [35, 36, 37, 38, 39, 40, 41, 42]
RIGHT_EYE_INDICES = [89, 90, 91, 92, 93, 94, 95, 96]
IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BEAUTY_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32).reshape(3, 1, 1)

def softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return np.array([])
    try:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    except Exception:
        return np.full_like(x, 1.0 / x.size if x.size > 0 else 0)

def get_eye_bbox_from_landmarks(
    landmarks: Optional[np.ndarray],
    eye_indices: List[int],
    padding_ratio: float = 0.15,
) -> Optional[Tuple[int, int, int, int]]:
    if landmarks is None or len(landmarks) < max(eye_indices) + 1: return None
    try:
        eye_points = landmarks[eye_indices]
        min_x, max_x = int(np.min(eye_points[:, 0])), int(np.max(eye_points[:, 0]))
        min_y, max_y = int(np.min(eye_points[:, 1])), int(np.max(eye_points[:, 1]))
        width, height = max_x - min_x, max_y - min_y
        if width <= 0 or height <= 0: return None
        pad_w, pad_h = int(width * padding_ratio), int(height * padding_ratio)
        return min_x - pad_w, min_y - pad_h, max_x + pad_w, max_y + pad_h
    except Exception as e:
        logger.error(f"Ошибка вычисления BBox глаза: {e}", exc_info=True)
        return None

# --- Блок 2: Класс AttributeAnalyzer ---
# ==============================================================================
class AttributeAnalyzer:
    _is_enabled: bool

    def __init__(self, config_manager: ConfigManager, onnx_manager: ONNXModelManager):
        # (Код __init__ и _collect_models_info без изменений)
        self.config_manager = config_manager
        self.onnx_manager = onnx_manager
        self.model_config = self.config_manager.get("model", {})
        self.task_flags = self.config_manager.get("task_flags", {})
        self.model_root = Path(self.config_manager.get("paths.model_root"))
        self._is_enabled = any(self.task_flags.values())
        self.models_info: Dict[str, Dict[str, Any]] = self._collect_models_info()

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    def _collect_models_info(self) -> Dict[str, Any]:
        # (Логика _collect_models_info без изменений)
        if not self.is_enabled: return {}
        info = {}
        model_meta = {
            "gender": {"input_shape": (3, 224, 224), "input_name": "input", "output_names": ["output"]},
            "emotion": {"input_shape": (1, 48, 48), "input_name": "input.1", "output_names": ["97"]},
            "age": {"input_shape": (3, 224, 224), "input_name": "input", "output_names": ["output"]},
            "beauty": {"input_shape": (3, 224, 224), "input_name": "input", "output_names": ["output"]},
            "eyeblink": {"input_shape": (1, 26, 34), "input_name": "input_3", "output_names": ["activation_5"]},
        }
        for task_name, meta in model_meta.items():
            if self.task_flags.get(f"analyze_{task_name}"):
                model_filename = self.model_config.get(f"{task_name}_model")
                if not model_filename:
                    logger.warning(f"Имя файла модели для '{task_name}' не указано.")
                    continue
                model_path = self.model_root / model_filename
                if not model_path.is_file():
                    logger.error(f"Файл модели для '{task_name}' не найден: {model_path}")
                    continue
                info[task_name] = {"path": model_path, **meta}
        logger.info(f"AttributeAnalyzer сконфигурирован для анализа: <b>{list(info.keys())}</b>")
        return info

    # --- Методы предобработки с ВОССТАНОВЛЕННОЙ обработкой ошибок ---
    def _preprocess_gender_age(self, image: np.ndarray, target_shape: tuple) -> Optional[np.ndarray]:
        try:
            _c, h, w = target_shape
            img_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            blob = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB
            return np.expand_dims(blob.transpose(2, 0, 1), axis=0).astype(np.float32)
        except Exception as e:
            logger.error(f"Ошибка _preprocess_gender_age: {e}")
            return None

    def _preprocess_emotion(self, image: np.ndarray, target_shape: tuple) -> Optional[np.ndarray]:
        try:
            _c, h, w = target_shape
            img_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            return (img_gray.astype(np.float32) / 256.0).reshape(1, 1, h, w)
        except Exception as e:
            logger.error(f"Ошибка _preprocess_emotion: {e}")
            return None

    def _preprocess_beauty(self, image: np.ndarray, target_shape: tuple) -> Optional[np.ndarray]:
        try:
            _c, h, w = target_shape
            img_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            blob = img_resized.astype(np.float32).transpose(2, 0, 1) - BEAUTY_MEAN_BGR
            return np.expand_dims(blob, axis=0)
        except Exception as e:
            logger.error(f"Ошибка _preprocess_beauty: {e}")
            return None

    def _preprocess_eyeblink(self, image: np.ndarray, target_shape: tuple) -> Optional[np.ndarray]:
        try:
            _c, h, w = target_shape
            img_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            return (img_gray.astype(np.float32) / 255.0).reshape(1, h, w, 1)
        except Exception as e:
            logger.error(f"Ошибка _preprocess_eyeblink: {e}")
            return None

    def _get_preprocess_func(self, task_name: str) -> Optional[Callable]:
        mapping = {
            "gender": self._preprocess_gender_age,
            "age": self._preprocess_gender_age,
            "beauty": self._preprocess_beauty,
            "emotion": self._preprocess_emotion,
            "eyeblink": self._preprocess_eyeblink,
        }
        return mapping.get(task_name)

    # --- Методы постобработки с ВОССТАНОВЛЕННОЙ обработкой ошибок ---
    def _postprocess_gender(self, outputs: List[np.ndarray]) -> Optional[str]:
        try:
            return "Male" if np.argmax(outputs[0].flatten()) == 0 else "Female"
        except Exception as e:
            logger.error(f"Ошибка _postprocess_gender: {e}")
            return None

    def _postprocess_emotion(self, outputs: List[np.ndarray]) -> Optional[str]:
        try:
            labels = self.model_config.get("emotion_labels", [])
            scores = outputs[0].flatten()
            return labels[np.argmax(softmax(scores))]
        except Exception as e:
            logger.error(f"Ошибка _postprocess_emotion: {e}")
            return None

    def _postprocess_age(self, outputs: List[np.ndarray]) -> Optional[int]:
        try:
            age = int(round(outputs[0].flatten()[0]))
            return age if 0 <= age <= 120 else None
        except Exception as e:
            logger.error(f"Ошибка _postprocess_age: {e}")
            return None

    def _postprocess_beauty(self, outputs: List[np.ndarray]) -> Optional[float]:
        try:
            score = float(outputs[0].flatten()[0])
            return score if not (np.isnan(score) or np.isinf(score)) else None
        except Exception as e:
            logger.error(f"Ошибка _postprocess_beauty: {e}")
            return None

    def _postprocess_eyeblink(self, outputs: List[np.ndarray]) -> Optional[Tuple[str, float]]:
        try:
            score = float(outputs[0].flatten()[0])
            if np.isnan(score) or np.isinf(score): return None
            state = "Open" if score > 0.5 else "Closed"
            return state, score
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

    def _run_analysis(self, image: np.ndarray, task: str) -> Any:
        if image is None or image.size == 0: return None
        model_info = self.models_info.get(task)
        if not model_info: return None

        session = self.onnx_manager.get_session(model_info["path"])
        
        if not session: return None

        preprocess_func = self._get_preprocess_func(task)
        if not preprocess_func: return None
        
        input_blob = preprocess_func(image, model_info["input_shape"])
        if input_blob is None:
            logger.error(f"Предобработка для '{task}' вернула None.")
            return None
            
        try:
            outputs = session.run(model_info["output_names"], {model_info["input_name"]: input_blob})
            
            postprocess_func = self._get_postprocess_func(task)
            if not postprocess_func: return None
            
            return postprocess_func(outputs)
        except Exception as e:
            logger.error(f"Ошибка выполнения модели '{task}': {e}", exc_info=True)
        return None

    def process_face_data(self, face_data_bundle: Dict[str, Any], face_data_dict: Dict[str, Any]):
        if not self.is_enabled: return
        
        try:
            face_crop = face_data_bundle.get("face_crop")
            if face_crop is None: return

            for task in ["gender", "emotion", "age", "beauty"]:
                if self.task_flags.get(f"analyze_{task}"):
                    result = self._run_analysis(face_crop, task)
                    face_data_dict[f"{task}_faceonnx"] = result
            
            if self.task_flags.get("analyze_eyeblink"):
                full_image = face_data_bundle.get("full_image")
                landmarks = face_data_dict.get("landmark_2d_106")
                if full_image is not None and landmarks is not None:
                    landmarks_np = np.array(landmarks)
                    img_h, img_w = full_image.shape[:2]

                    left_bbox = get_eye_bbox_from_landmarks(landmarks_np, LEFT_EYE_INDICES)
                    if left_bbox:
                        x1, y1, x2, y2 = left_bbox
                        left_eye_img = full_image[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                        res = self._run_analysis(left_eye_img, "eyeblink")
                        if res: face_data_dict["left_eye_state"], face_data_dict["left_eye_score"] = res

                    right_bbox = get_eye_bbox_from_landmarks(landmarks_np, RIGHT_EYE_INDICES)
                    if right_bbox:
                        x1, y1, x2, y2 = right_bbox
                        right_eye_img = full_image[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                        res = self._run_analysis(right_eye_img, "eyeblink")
                        if res: face_data_dict["right_eye_state"], face_data_dict["right_eye_score"] = res
        except Exception as e:
            filename = face_data_bundle.get("filename", "unknown")
            face_idx = face_data_bundle.get("face_index", -1)
            logger.error(f"Критическая ошибка в AttributeAnalyzer.process_face_data для {filename}[{face_idx}]: {e}", exc_info=True)