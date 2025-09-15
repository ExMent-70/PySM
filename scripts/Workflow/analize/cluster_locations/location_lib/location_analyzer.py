# location_lib/location_analyzer.py

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
import onnxruntime as ort

from .config_loader import ConfigManager
from _common.onnx_manager import ONNXModelManager

# --- НАЧАЛО ИЗМЕНЕНИЙ ---
# Попытка импортировать transformers и установить флаг
try:
    from transformers import CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
# --- КОНЕЦ ИЗМЕНЕНИЙ ---

logger = logging.getLogger(__name__)

IMAGENET_MEAN_RGB = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGENET_STD_RGB = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

class LocationAnalyzer:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.input_size = tuple(self.config_manager.get("model_params.input_size"))

        self.onnx_manager = ONNXModelManager(self.config_manager.get('provider', {}))

        self.session: ort.InferenceSession
        self.input_names: Dict[str, str] = {}
        self.output_names: Dict[str, str] = {}
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        self.tokenizer = None # Инициализируем токенизатор как None
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        
        self.session, self.input_names, self.output_names = self._initialize_clip_model()


    def _initialize_clip_model(self) -> Tuple[ort.InferenceSession, Dict[str, str], Dict[str, str]]:
        model_root = Path(self.config_manager.get("paths.model_root"))
        clip_model_path = model_root / self.config_manager.get("paths.clip_model_onnx")
        
        print("PYSM_CONSOLE_BLOCK_START")       

        session = self.onnx_manager.get_session(clip_model_path)
        
        print("PYSM_CONSOLE_BLOCK_END")        

        if not session:
            raise RuntimeError(f"Не удалось инициализировать модель CLIP из {clip_model_path}")
        
        available_inputs = [inp.name for inp in session.get_inputs()]
        
        required_input_names = {
            "pixel_values": ["pixel_values", "image"],
            "input_ids": ["input_ids", "text"],
            "attention_mask": ["attention_mask"]
        }
        
        found_input_names = {}
        for key, possible_names in required_input_names.items():
            found_name = next((name for name in possible_names if name in available_inputs), None)
            if not found_name:
                raise RuntimeError(
                    f"Критическая ошибка: Не удалось найти обязательный входной тензор для '{key}' в модели CLIP. "
                    f"Доступные входы: {available_inputs}"
                )
            found_input_names[key] = found_name
        
        available_outputs = [out.name for out in session.get_outputs()]
        found_output_names = {
            "image_embeds": next((name for name in ["image_embeds", "image_features", "embedding"] if name in available_outputs), None),
            "text_embeds": next((name for name in ["text_embeds", "text_features"] if name in available_outputs), None)
        }

        if not all(found_output_names.values()):
            raise RuntimeError(
                f"Не удалось найти все необходимые выходные тензоры в модели CLIP. "
                f"Требуются: 'image_embeds', 'text_embeds'. "
                f"Найдено: {found_output_names}. "
                f"Доступные выходы: {available_outputs}"
            )

        logger.debug(f"Входные тензоры модели CLIP определены: <b>{found_input_names}</b>")
        logger.debug(f"Выходные тензоры модели CLIP определены: <b>{found_output_names}</b>")

        return session, found_input_names, found_output_names

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        resized_img = cv2.resize(image, self.input_size, interpolation=cv2.INTER_CUBIC)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        normalized_img = rgb_img.astype(np.float32) / 255.0
        standardized_img = (normalized_img - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB
        final_tensor = np.expand_dims(standardized_img.transpose(2, 0, 1), axis=0).astype(np.float32)
        return final_tensor

    def get_image_embedding(self, mask_path: Path) -> Optional[Tuple[Path, np.ndarray]]:
        if not self.session: return None
        try:
            original_filename_stem = mask_path.name.replace("_BiRefNet-portrait_output.jpg", "")
            original_filename = f"{original_filename_stem}.jpg"
            original_photo_path = mask_path.parent.parent / original_filename

            if not original_photo_path.exists():
                logger.warning(f"Не найден оригинальный файл {original_filename} для маски {mask_path.name}")
                return None
            
            with open(mask_path, "rb") as f:
                img_buffer = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)

            if img is None:
                logger.error(f"Не удалось загрузить изображение-маску: {mask_path.name}")
                return None

            input_tensor = self._preprocess_image(img)
            
            dummy_text_input = np.zeros((1, 77), dtype=np.int64)
            input_feed = {
                self.input_names["pixel_values"]: input_tensor,
                self.input_names["input_ids"]: dummy_text_input,
                self.input_names["attention_mask"]: dummy_text_input
            }
            
            embedding = self.session.run([self.output_names["image_embeds"]], input_feed)[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return (original_photo_path, embedding.flatten())

        except Exception as e:
            logger.error(f"Не удалось обработать маску {mask_path.name}: {e}", exc_info=True)
            return None

    # --- НАЧАЛО ИЗМЕНЕНИЙ: Реализация метода для вычисления эмбеддингов текста ---
    def get_text_embeddings(self, prompts: List[str]) -> np.ndarray:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Библиотека 'transformers' не найдена. Пожалуйста, установите ее командой: pip install transformers")
        
        if not self.tokenizer:
            print("Инициализация токенизатора CLIP (единоразово)...")
            # Используем стандартную предобученную модель, совместимую с ViT-B/32
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        print(f"Токенизация и вычисление эмбеддингов для <b>{len(prompts)}</b> текстовых описаний...")
        
        # 1. Токенизация
        inputs = self.tokenizer(
            prompts, 
            padding="max_length",
            max_length=self.tokenizer.model_max_length, # Используем длину из токенизатора (обычно 77)
            truncation=True,
            return_tensors="np"
        )
        
        # 2. Подготовка входа для ONNX
        # Нулевой тензор для изображения, так как нас интересует только выход для текста
        batch_size = len(prompts)
        dummy_pixel_values = np.zeros((batch_size, 3, *self.input_size), dtype=np.float32)
        
        input_feed = {
            self.input_names["pixel_values"]: dummy_pixel_values,
            self.input_names["input_ids"]: inputs["input_ids"].astype(np.int64),
            self.input_names["attention_mask"]: inputs["attention_mask"].astype(np.int64)
        }

        # 3. Вычисление
        text_embeddings = self.session.run([self.output_names["text_embeds"]], input_feed)[0]
        
        # 4. Нормализация
        text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        return text_embeddings
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    def shutdown(self):
        #print("Освобождение ресурсов...")
        if self.onnx_manager:
            self.onnx_manager.shutdown()