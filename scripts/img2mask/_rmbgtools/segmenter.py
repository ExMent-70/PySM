# _rmbgtools/segmenter.py

# ======================================================================================
# Блок 1: Импорты
# ======================================================================================
import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from .core.model_manager import ModelManager
from .utils.image_utils import apply_mask_to_image, composite_with_color_background
from .tools import MaskEnhancer
from . import logger

import sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util import box_ops
from groundingdino.datasets.transforms import Compose, RandomResize, ToTensor, Normalize
from contextlib import nullcontext

# ======================================================================================
# Блок 2: Класс TextSegmenter
# ======================================================================================
class TextSegmenter:
    def __init__(
        self,
        sam_model_name: str = "SAM2-Hiera-T",
        dino_model_name: str = "GroundingDINO-T",
        model_dir: str = "models",
        device: str = "auto"
    ):
        self.model_manager = ModelManager(model_dir=model_dir, device=device)
        self.device = self.model_manager.device
        logger.info("Loading models for TextSegmenter...")
        self.sam_model = self.model_manager.get_model(sam_model_name).model
        self.dino_model = self.model_manager.get_model(dino_model_name)
        logger.info("TextSegmenter models loaded successfully.")
        
        self.dino_transform = Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _process_single_image(
        self, 
        image_path: str, 
        output_dir: str, 
        prompt: str,
        **kwargs
    ) -> Tuple[str, str]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                image = Image.open(image_path)
                image_rgb = image.convert("RGB")
                
                image_tensor_dino, _ = self.dino_transform(image_rgb, None)
                image_tensor_dino = image_tensor_dino.unsqueeze(0).to(self.device)
                
                text_prompt = prompt.lower().strip()
                if not text_prompt.endswith("."): text_prompt += "."
                
                with torch.no_grad():
                    outputs = self.dino_model(image_tensor_dino, captions=[text_prompt])
                
                logits = outputs["pred_logits"].sigmoid()[0]
                boxes = outputs["pred_boxes"][0]
                filt_mask = logits.max(dim=1)[0] > kwargs['threshold']
                boxes_filt = boxes[filt_mask]
                
                if boxes_filt.shape[0] == 0:
                    return "not_found", image_path

                H, W = image.size[1], image.size[0]
                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_filt) * torch.tensor([W, H, W, H], device=self.device)
                boxes_xyxy_np = boxes_xyxy.cpu().numpy()

                local_sam_predictor = SAM2ImagePredictor(self.sam_model)
                precision = next(local_sam_predictor.model.parameters()).dtype
                autocast_context = torch.autocast(self.device, dtype=precision) if self.device == 'cuda' else nullcontext()
                
                with autocast_context:
                    local_sam_predictor.set_image(image_rgb)
                    all_masks = []
                    for box in boxes_xyxy_np:
                        with torch.no_grad():
                            masks, _, _ = local_sam_predictor.predict(box=box, multimask_output=False)
                        all_masks.append(masks)
                
                if not all_masks: final_mask_np = np.zeros((H, W), dtype=np.uint8)
                else:
                    combined_mask = np.zeros_like(all_masks[0][0], dtype=np.float32)
                    for mask_group in all_masks: combined_mask = np.maximum(combined_mask, mask_group[0])
                    final_mask_np = (combined_mask * 255).astype(np.uint8)

                final_mask_pil = Image.fromarray(final_mask_np, mode="L")
                
                enhanced_mask = MaskEnhancer.enhance(
                    mask=final_mask_pil,
                    blur=kwargs['mask_blur'],
                    offset=kwargs['mask_offset'],
                    invert=kwargs['invert_output'],
                    smooth=kwargs['smooth'],
                    fill_holes=kwargs['fill_holes']
                )
                
                rgba_image = apply_mask_to_image(image, enhanced_mask)
                
                result_image = composite_with_color_background(rgba_image, kwargs['background_color']) if kwargs['background'] == "Color" else rgba_image
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                img_path = os.path.join(output_dir, f"{base_name}_segmented.png")
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                result_image.save(img_path)
                enhanced_mask.save(mask_path)
                
                return img_path, mask_path
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                return "error", image_path

    def segment(
        self, 
        images: Union[str, List[str]], 
        prompt: str,
        output_dir: str,
        num_threads: int = 4,
        **kwargs
    ) -> Iterator[Tuple[str, str]]:
        image_paths = [images] if isinstance(images, str) else images
        os.makedirs(output_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_path = {
                executor.submit(self._process_single_image, path, output_dir, prompt, **kwargs): path
                for path in image_paths
            }
            for future in as_completed(future_to_path):
                yield future.result()

# ======================================================================================
# Блок 3: Функция-обертка
# ======================================================================================
def segment_by_text(
    images: Union[str, List[str]],
    prompt: str,
    output_dir: str,
    sam_model_name: str = "SAM2-Hiera-T",
    dino_model_name: str = "GroundingDINO-T",
    model_dir: str = "models",
    device: str = "auto",
    **kwargs
) -> Iterator[Tuple[str, str]]:
    constructor_kwargs = {"sam_model_name": sam_model_name, "dino_model_name": dino_model_name, "model_dir": model_dir, "device": device}
    segmenter = TextSegmenter(**constructor_kwargs)
    
    all_kwargs = {"num_threads": kwargs.get("num_threads", 4), **kwargs}
    return segmenter.segment(images=images, prompt=prompt, output_dir=output_dir, **all_kwargs)