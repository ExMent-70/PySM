from pathlib import Path
from typing import Tuple

import onnxruntime as ort

from ...model_loader import ModelLoader


class ClipModelLoader(ModelLoader):
    def __init__(self, model_path: Path, provider: dict):
        super().__init__(model_path, provider)
        self.inputs, self.outputs = self._map_tensors(self.session)

    def _map_tensors(self, session: ort.InferenceSession) -> Tuple[dict, dict]:
        inputs = [i.name for i in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]

        def find(candidates, pool):
            for c in candidates:
                if c in pool:
                    return c
            raise RuntimeError(f"Missing tensor {candidates}")

        return (
            {
                "pixel": find(["pixel_values", "image"], inputs),
                "ids": find(["input_ids"], inputs),
                "mask": find(["attention_mask"], inputs),
            },
            {
                "image": find(["image_embeds", "embedding"], outputs),
                "text": find(["text_embeds"], outputs),
            },
        )
