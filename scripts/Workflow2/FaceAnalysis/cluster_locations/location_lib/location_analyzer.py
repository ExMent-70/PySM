# cluster_locations/location_lib/location_analyzer.py

import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union

import cv2
import numpy as np
import onnxruntime as ort
import requests

from .config_loader import ConfigManager
from _common.onnx_manager import ONNXModelManager, suppress_output

from _common import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
) 

try:
    from transformers import CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

IMAGENET_MEAN_RGB = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGENET_STD_RGB = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

MODEL_URLS = {
    "ViT-L-14.onnx": "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/Rqfo48sIngLSFg",
    "ViT-B-32.onnx": "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/OL0eh6Pe-0MnHw"
}

class LocationAnalyzer:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.input_size = tuple(self.config_manager.get("model_params.input_size"))
        self.mask_suffix = self.config_manager.get("model_params.mask_suffix", "_BiRefNet-portrait_output.jpg")

        self.onnx_manager = ONNXModelManager(self.config_manager.get('provider', {}))
        
        self.model_root = Path(self.config_manager.get("paths.model_root"))
        self.tokenizer_rel_path = self.config_manager.get("paths.tokenizer_path")
        
        self.session: ort.InferenceSession
        self.input_names: Dict[str, str] = {}
        self.output_names: Dict[str, str] = {}
        self.tokenizer = None
        
        self.session, self.input_names, self.output_names = self._initialize_clip_model()

    def _initialize_clip_model(self) -> Tuple[ort.InferenceSession, Dict[str, str], Dict[str, str]]:
        clip_model_path = self.model_root / self.config_manager.get("paths.clip_model_onnx")
        
        if not clip_model_path.exists():
            self._download_model_if_needed(clip_model_path)

        with suppress_output():
            session = self.onnx_manager.get_session(clip_model_path)

        if not session:
            raise RuntimeError(f"{icon_error} Не удалось инициализировать модель CLIP из {clip_model_path}")
        
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
                raise RuntimeError(f"{icon_error} Не найден входной тензор для '{key}'. Доступно: {available_inputs}")
            found_input_names[key] = found_name
        
        available_outputs = [out.name for out in session.get_outputs()]
        found_output_names = {
            "image_embeds": next((name for name in ["image_embeds", "image_features", "embedding"] if name in available_outputs), None),
            "text_embeds": next((name for name in ["text_embeds", "text_features"] if name in available_outputs), None)
        }

        if not all(found_output_names.values()):
            raise RuntimeError(f"{icon_error} Не найдены выходные тензоры 'image_embeds'/'text_embeds'. Доступно: {available_outputs}")

        return session, found_input_names, found_output_names

    def _download_model_if_needed(self, local_path: Path):
        filename = local_path.name
        url = MODEL_URLS.get(filename)
        if not url:
            logger.error(f"{icon_error} Файл модели {filename} не найден локально и для него нет ссылки для скачивания.")
            raise FileNotFoundError(f"{icon_error} Model file not found: {local_path}")
            
        logger.info(f"<br>{icon_warning} Модель {filename} не найдена. Начинаю скачивание...")
        try:
            from pysm_lib.pysm_progress_reporter import tqdm
            local_path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            with open(local_path, 'wb') as f, tqdm(desc="Скачивание модели "+filename, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    size = f.write(chunk)
                    bar.update(size)
            logger.info(f"{icon_save} Модель успешно скачана и сохранена в {local_path}<br>")
        except Exception as e:
            logger.error(f"{icon_error} Ошибка при скачивании модели {filename}: {e}")
            if local_path.exists(): local_path.unlink()
            raise

    def _resolve_original_path(self, path: Path, input_is_mask: bool) -> Optional[Path]:
        """
        Если на входе маска -> ищем оригинал.
        Если на входе оригинал -> просто возвращаем его.
        """
        if not input_is_mask:
            return path

        try:
            # Логика восстановления из маски
            original_filename = path.name.replace(self.mask_suffix, "") + ".jpg"
            if original_filename == path.name + ".jpg":
                pass # Защита, если суффикс не найден

            original_photo_path = path.parent.parent / original_filename
            
            if not original_photo_path.exists():
                 # Попытка для JPEG
                 alt_filename = path.name.replace(self.mask_suffix, "") + ".jpeg"
                 original_photo_path = path.parent.parent / alt_filename
                 if not original_photo_path.exists():
                    return None
            return original_photo_path
        except Exception:
            return None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        resized_img = cv2.resize(image, self.input_size, interpolation=cv2.INTER_CUBIC)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        normalized_img = rgb_img.astype(np.float32) / 255.0
        standardized_img = (normalized_img - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB
        final_tensor = np.expand_dims(standardized_img.transpose(2, 0, 1), axis=0).astype(np.float32)
        return final_tensor

    def _load_and_preprocess_single(self, path: Path, input_is_mask: bool) -> Optional[Tuple[Path, np.ndarray]]:
        """
        Универсальный загрузчик.
        :param path: Путь к файлу (маске или оригиналу).
        :param input_is_mask: Флаг, указывающий, является ли path маской.
        """
        try:
            # 1. Определяем "истинный" путь к оригиналу (для отчетов и JSON)
            original_path = self._resolve_original_path(path, input_is_mask)
            
            # Если мы работаем с масками, но не нашли оригинал — это проблема данных
            if input_is_mask and not original_path:
                return None
            
            # Если работаем с оригиналами напрямую, original_path = path
            if not original_path: 
                original_path = path

            # 2. Читаем изображение (то, которое передали в path)
            with open(path, "rb") as f:
                img_buffer = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)

            if img is None:
                logger.error(f"{icon_error} Не удалось декодировать: {path.name}")
                return None

            tensor = self._preprocess_image(img)
            return (original_path, tensor)

        except Exception as e:
            logger.error(f"{icon_error} Ошибка обработки {path.name}: {e}")
            return None

    def get_image_embeddings_batch(self, paths: List[Path], max_workers: int = 4, input_is_mask: bool = True) -> List[Tuple[Path, np.ndarray]]:
        """
        Пакетная обработка.
        :param paths: Список путей к файлам.
        :param input_is_mask: Если True, считаем файлы масками и ищем оригиналы. Если False, считаем оригиналами.
        """
        if not self.session or not paths:
            return []

        valid_original_paths = []
        batch_tensors = []

        # Обертка для map
        def process_wrapper(p):
            return self._load_and_preprocess_single(p, input_is_mask=input_is_mask)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_wrapper, paths)
            
            for res in results:
                if res is not None:
                    path, tensor = res
                    valid_original_paths.append(path)
                    batch_tensors.append(tensor)

        if not batch_tensors:
            return []

        input_tensor_batch = np.vstack(batch_tensors)
        batch_size = input_tensor_batch.shape[0]
        dummy_text_input = np.zeros((batch_size, 77), dtype=np.int64)
        
        input_feed = {
            self.input_names["pixel_values"]: input_tensor_batch,
            self.input_names["input_ids"]: dummy_text_input,
            self.input_names["attention_mask"]: dummy_text_input
        }
        
        try:
            with suppress_output():
                embeddings_batch = self.session.run([self.output_names["image_embeds"]], input_feed)[0]
            
            norms = np.linalg.norm(embeddings_batch, axis=1, keepdims=True)
            normalized_embeddings = embeddings_batch / norms
            
            result_list = []
            for i, path in enumerate(valid_original_paths):
                result_list.append((path, normalized_embeddings[i].flatten()))
                
            return result_list

        except Exception as e:
            logger.error(f"{icon_error} Ошибка при выполнении инференса батча: {e}", exc_info=True)
            return []

    def get_text_embeddings(self, prompts: List[str]) -> np.ndarray:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("{icon_error} Библиотека 'transformers' не найдена.")
        
        if not self.tokenizer:
            self._init_tokenizer()

        inputs = self.tokenizer(
            prompts, 
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np"
        )
        
        batch_size = len(prompts)
        dummy_pixel_values = np.zeros((batch_size, 3, *self.input_size), dtype=np.float32)
        
        input_feed = {
            self.input_names["pixel_values"]: dummy_pixel_values,
            self.input_names["input_ids"]: inputs["input_ids"].astype(np.int64),
            self.input_names["attention_mask"]: inputs["attention_mask"].astype(np.int64)
        }

        with suppress_output():
            text_embeddings = self.session.run([self.output_names["text_embeds"]], input_feed)[0]
            
        text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        return text_embeddings

    def _init_tokenizer(self):
        local_tokenizer_path = self.model_root / self.tokenizer_rel_path
        hf_model_name = "openai/clip-vit-large-patch14" 
        
        try:
            if local_tokenizer_path.exists() and any(local_tokenizer_path.iterdir()):
                logger.info(f"Загрузка токенизатора из кеша...")
                self.tokenizer = CLIPTokenizer.from_pretrained(str(local_tokenizer_path))
            else:
                logger.info(f"Скачивание токенизатора {hf_model_name}...")
                with suppress_output():
                    self.tokenizer = CLIPTokenizer.from_pretrained(hf_model_name)
                    self.tokenizer.save_pretrained(str(local_tokenizer_path))
        except Exception as e:
            logger.critical(f"Ошибка загрузки токенизатора: {e}")
            raise

    def shutdown(self):
        if self.onnx_manager:
            self.onnx_manager.shutdown()